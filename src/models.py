"""CV framework and model training for DSC 148 NYC Airbnb price prediction."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from preprocess import Preprocessor, get_target
from features import FeatureEngineer

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / 'raw files'
SUBMISSIONS_DIR = ROOT / 'submissions'
SEED = 42
N_FOLDS = 5


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw CSVs and convert price from currency string to float."""
    train = pd.read_csv(DATA_DIR / 'train.csv', low_memory=False)
    test  = pd.read_csv(DATA_DIR / 'test.csv',  low_memory=False)
    train['price'] = pd.to_numeric(
        train['price'].astype(str).str.replace(r'[$,]', '', regex=True), errors='coerce'
    )
    return train, test


def _build_matrices(
    train_df: pd.DataFrame,
    *transform_dfs: pd.DataFrame,
) -> tuple[pd.DataFrame, ...]:
    """
    Fit Preprocessor + FeatureEngineer on train_df; apply to any additional dfs.
    Both transformers receive the raw df so FeatureEngineer can access text/location columns
    that Preprocessor drops.
    Returns (X_train, X_df1, X_df2, ...).
    """
    prep = Preprocessor()
    fe   = FeatureEngineer()

    X_train = pd.concat([
        prep.fit_transform(train_df),
        fe.fit_transform(train_df, train_df['price']),
    ], axis=1)

    extras = tuple(
        pd.concat([prep.transform(df), fe.transform(df)], axis=1)
        for df in transform_dfs
    )
    return (X_train,) + extras


def build_feature_matrix(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Fit on full train_df. Returns (X_train, X_test, y_train)."""
    X_train, X_test = _build_matrices(train_df, test_df)  # type: ignore[misc]
    return X_train, X_test, get_target(train_df)


def cross_validate(
    model,
    raw_train_df: pd.DataFrame,
    n_folds: int = N_FOLDS,
    seed: int = SEED,
    verbose: bool = True,
) -> dict:
    """
    5-fold CV with MAE on the original price scale.
    Preprocessor + FeatureEngineer are re-fit inside each fold to prevent leakage.
    Works with any sklearn-compatible estimator.

    Returns dict with keys: mae_mean, mae_std, fold_maes.
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_maes = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(raw_train_df), 1):
        fold_train = raw_train_df.iloc[train_idx]
        fold_val   = raw_train_df.iloc[val_idx]

        X_tr, X_val = _build_matrices(fold_train, fold_val)  # type: ignore[misc]
        y_tr  = get_target(fold_train)
        y_val = get_target(fold_val)

        model.fit(X_tr, y_tr)
        log_preds = np.clip(model.predict(X_val), -10, 11)
        preds = np.expm1(log_preds).clip(0)
        mae   = mean_absolute_error(np.expm1(y_val), preds)
        fold_maes.append(mae)
        if verbose:
            print(f'  Fold {fold}/{n_folds}  MAE: ${mae:.2f}')

    result = {
        'mae_mean': float(np.mean(fold_maes)),
        'mae_std':  float(np.std(fold_maes)),
        'fold_maes': fold_maes,
    }
    if verbose:
        print(f'  → CV MAE: ${result["mae_mean"]:.2f} ± ${result["mae_std"]:.2f}')
    return result


def oof_predict(
    model,
    raw_train_df: pd.DataFrame,
    test_df: pd.DataFrame | None = None,
    n_folds: int = N_FOLDS,
    seed: int = SEED,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    OOF predictions on train set + averaged test predictions (price scale).
    test_df predictions come from the same fold-fitted transformers — no leakage.
    Returns (oof_preds, test_preds).  test_preds is None when test_df is not provided.
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof_preds = np.zeros(len(raw_train_df))
    test_preds_folds: list[np.ndarray] = []
    fold_maes = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(raw_train_df), 1):
        fold_train = raw_train_df.iloc[train_idx]
        fold_val   = raw_train_df.iloc[val_idx]

        if test_df is not None:
            X_tr, X_val, X_te = _build_matrices(fold_train, fold_val, test_df)  # type: ignore[misc]
        else:
            X_tr, X_val = _build_matrices(fold_train, fold_val)  # type: ignore[misc]

        y_tr  = get_target(fold_train)
        y_val = get_target(fold_val)

        model.fit(X_tr, y_tr)
        oof_preds[val_idx] = np.expm1(np.clip(model.predict(X_val), -10, 11)).clip(0)
        mae = mean_absolute_error(np.expm1(y_val), oof_preds[val_idx])
        fold_maes.append(mae)

        if test_df is not None:
            test_preds_folds.append(np.expm1(np.clip(model.predict(X_te), -10, 11)).clip(0))

        if verbose:
            print(f'  Fold {fold}/{n_folds}  MAE: ${mae:.2f}')

    if verbose:
        print(f'  → OOF MAE: ${np.mean(fold_maes):.2f} ± ${np.std(fold_maes):.2f}')

    test_preds = np.mean(test_preds_folds, axis=0) if test_preds_folds else None
    return oof_preds, test_preds


def make_submission(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    test_ids: pd.Series,
    filename: str,
) -> Path:
    """Train on full X_train, predict X_test, save submission CSV."""
    model.fit(X_train, y_train)
    preds = np.expm1(np.clip(model.predict(X_test), -10, 11)).clip(0)

    SUBMISSIONS_DIR.mkdir(exist_ok=True)
    path = SUBMISSIONS_DIR / filename
    pd.DataFrame({'Id': test_ids, 'Predicted': preds}).to_csv(path, index=False)
    print(f'Saved: {path}  ({len(preds)} rows)')
    return path


def plot_feature_importance(
    model,
    feature_names: list,
    top_n: int = 30,
    filename: str = 'lgbm_importance.png',
) -> None:
    importances = model.feature_importances_
    idx = np.argsort(importances)[-top_n:]

    fig, ax = plt.subplots(figsize=(8, top_n * 0.32 + 1))
    ax.barh(range(top_n), importances[idx])
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in idx], fontsize=8)
    ax.set_xlabel('Importance (gain)')
    ax.set_title(f'LightGBM — top {top_n} features')
    plt.tight_layout()

    SUBMISSIONS_DIR.mkdir(exist_ok=True)
    out = SUBMISSIONS_DIR / filename
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'Feature importance plot: {out}')


def build_lgbm() -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=127,
        min_child_samples=20,
        colsample_bytree=0.8,
        subsample=0.8,
        subsample_freq=1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=SEED,
        n_jobs=-1,
        verbose=-1,
    )


def build_xgb() -> xgb.XGBRegressor:
    return xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        tree_method='hist',
        random_state=SEED,
        n_jobs=-1,
        verbosity=0,
    )


def tune_lgbm(
    raw_train_df: pd.DataFrame,
    n_trials: int = 50,
    n_folds: int = 3,
    seed: int = SEED,
) -> dict:
    """Optuna search over LightGBM hyperparameters. Returns best params dict."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        model = lgb.LGBMRegressor(
            n_estimators=200,
            num_leaves=trial.suggest_int('num_leaves', 31, 255),
            learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            min_child_samples=trial.suggest_int('min_child_samples', 5, 100),
            colsample_bytree=trial.suggest_float('colsample_bytree', 0.4, 1.0),
            subsample=trial.suggest_float('subsample', 0.5, 1.0),
            subsample_freq=1,
            reg_alpha=trial.suggest_float('reg_alpha', 1e-4, 2.0, log=True),
            reg_lambda=trial.suggest_float('reg_lambda', 1e-4, 2.0, log=True),
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )
        return cross_validate(model, raw_train_df, n_folds=n_folds, seed=seed, verbose=False)['mae_mean']

    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def build_lgbm_tuned(params: dict) -> lgb.LGBMRegressor:
    """LightGBM with tuned params at full 500 estimators."""
    return lgb.LGBMRegressor(
        n_estimators=500,
        subsample_freq=1,
        random_state=SEED,
        n_jobs=-1,
        verbose=-1,
        **params,
    )


def tune_xgb(
    raw_train_df: pd.DataFrame,
    n_trials: int = 50,
    n_folds: int = 3,
    seed: int = SEED,
) -> dict:
    """Optuna search over XGBoost hyperparameters. Returns best params dict."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=trial.suggest_int('max_depth', 3, 9),
            learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            subsample=trial.suggest_float('subsample', 0.5, 1.0),
            colsample_bytree=trial.suggest_float('colsample_bytree', 0.4, 1.0),
            reg_alpha=trial.suggest_float('reg_alpha', 1e-4, 2.0, log=True),
            reg_lambda=trial.suggest_float('reg_lambda', 1e-4, 2.0, log=True),
            min_child_weight=trial.suggest_int('min_child_weight', 1, 20),
            tree_method='hist',
            random_state=seed,
            n_jobs=-1,
            verbosity=0,
        )
        return cross_validate(model, raw_train_df, n_folds=n_folds, seed=seed, verbose=False)['mae_mean']

    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def build_xgb_tuned(params: dict) -> xgb.XGBRegressor:
    """XGBoost with tuned params at full 500 estimators."""
    return xgb.XGBRegressor(
        n_estimators=500,
        tree_method='hist',
        random_state=SEED,
        n_jobs=-1,
        verbosity=0,
        **params,
    )


def tune_catboost(
    raw_train_df: pd.DataFrame,
    n_trials: int = 50,
    n_folds: int = 3,
    seed: int = SEED,
) -> dict:
    """Optuna search over CatBoost hyperparameters. Returns best params dict."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        model = CatBoostRegressor(
            iterations=200,
            depth=trial.suggest_int('depth', 4, 10),
            learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            l2_leaf_reg=trial.suggest_float('l2_leaf_reg', 1e-4, 10.0, log=True),
            bagging_temperature=trial.suggest_float('bagging_temperature', 0.0, 1.0),
            random_strength=trial.suggest_float('random_strength', 1e-3, 10.0, log=True),
            random_seed=seed,
            thread_count=-1,
            verbose=0,
        )
        return cross_validate(model, raw_train_df, n_folds=n_folds, seed=seed, verbose=False)['mae_mean']

    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def build_catboost_tuned(params: dict) -> CatBoostRegressor:
    """CatBoost with tuned params at full 500 iterations."""
    return CatBoostRegressor(
        iterations=500,
        random_seed=SEED,
        thread_count=-1,
        verbose=0,
        **params,
    )


def build_mlp() -> Pipeline:
    return Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPRegressor(
            hidden_layer_sizes=(256, 128),
            activation='relu',
            solver='adam',
            alpha=0.01,
            batch_size=512,
            learning_rate='adaptive',
            learning_rate_init=0.0003,
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
            random_state=SEED,
            verbose=False,
        )),
    ])


def build_catboost() -> CatBoostRegressor:
    return CatBoostRegressor(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        random_seed=SEED,
        thread_count=-1,
        verbose=0,
    )


if __name__ == '__main__':
    print('Loading data...')
    train_df, test_df = load_data()
    print(f'Train: {train_df.shape}  Test: {test_df.shape}')

    # Task 6/7 — LightGBM
    print('\n=== LightGBM (5-fold CV) ===')
    cv_lgbm = cross_validate(build_lgbm(), train_df)

    # Task 8 — XGBoost
    print('\n=== XGBoost (5-fold CV) ===')
    cv_xgb = cross_validate(build_xgb(), train_df)

    # Task 8 — CatBoost
    print('\n=== CatBoost (5-fold CV) ===')
    cv_cat = cross_validate(build_catboost(), train_df)

    # Build full feature matrix once for all final models
    print('\n=== Building full feature matrix ===')
    X_train, X_test, y_train = build_feature_matrix(train_df, test_df)
    print(f'X_train: {X_train.shape}  X_test: {X_test.shape}')

    print('\n=== Saving submissions ===')
    lgbm_final = build_lgbm()
    make_submission(lgbm_final, X_train, y_train, X_test, test_df['id'], 'lgbm_v1.csv')
    plot_feature_importance(lgbm_final, list(X_train.columns))

    make_submission(build_xgb(), X_train, y_train, X_test, test_df['id'], 'xgb_v1.csv')
    make_submission(build_catboost(), X_train, y_train, X_test, test_df['id'], 'catboost_v1.csv')

    # Task 9 — MLP Neural Network
    print('\n=== MLP Neural Network (5-fold CV) ===')
    cv_mlp = cross_validate(build_mlp(), train_df)
    make_submission(build_mlp(), X_train, y_train, X_test, test_df['id'], 'mlp_v1.csv')

    # Comparison table
    results = [
        ('LightGBM', cv_lgbm),
        ('XGBoost',  cv_xgb),
        ('CatBoost', cv_cat),
        ('MLP',      cv_mlp),
    ]
    print('\n=== Model Comparison ===')
    print(f"{'Model':<12} {'CV MAE':>10} {'Std':>8}")
    print('-' * 32)
    for name, cv in results:
        print(f"{name:<12} ${cv['mae_mean']:>8.2f} ± ${cv['mae_std']:.2f}")
