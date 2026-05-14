"""Optuna hyperparameter tuning for all tree models — Task 10."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from models import (
    load_data, build_feature_matrix, make_submission, cross_validate,
    tune_lgbm, build_lgbm_tuned,
    tune_xgb, build_xgb_tuned,
    tune_catboost, build_catboost_tuned,
)

N_TRIALS = 50
PARAMS_OUT = Path(__file__).parent.parent / 'submissions' / 'best_params.json'

print('Loading data...')
train_df, test_df = load_data()
print(f'Train: {train_df.shape}  Test: {test_df.shape}')

# ------------------------------------------------------------------
# Optuna search — 50 trials, 3-fold CV each
# ------------------------------------------------------------------

print(f'\n=== Tuning LightGBM ({N_TRIALS} trials × 3-fold) ===')
lgbm_params = tune_lgbm(train_df, n_trials=N_TRIALS)
print(f'  Best params: {lgbm_params}')

print(f'\n=== Tuning XGBoost ({N_TRIALS} trials × 3-fold) ===')
xgb_params = tune_xgb(train_df, n_trials=N_TRIALS)
print(f'  Best params: {xgb_params}')

print(f'\n=== Tuning CatBoost ({N_TRIALS} trials × 3-fold) ===')
cb_params = tune_catboost(train_df, n_trials=N_TRIALS)
print(f'  Best params: {cb_params}')

# Save params so we can skip retuning next session
PARAMS_OUT.parent.mkdir(exist_ok=True)
PARAMS_OUT.write_text(json.dumps({'lgbm': lgbm_params, 'xgb': xgb_params, 'catboost': cb_params}, indent=2))
print(f'\nBest params saved → {PARAMS_OUT}')

# ------------------------------------------------------------------
# Final 5-fold CV + submissions with tuned models
# ------------------------------------------------------------------

print('\n=== Building full feature matrix ===')
X_train, X_test, y_train = build_feature_matrix(train_df, test_df)
print(f'X_train: {X_train.shape}  X_test: {X_test.shape}')

configs = [
    ('LightGBM tuned', lgbm_params, build_lgbm_tuned, 'lgbm_tuned_v1.csv'),
    ('XGBoost tuned',  xgb_params,  build_xgb_tuned,  'xgb_tuned_v1.csv'),
    ('CatBoost tuned', cb_params,   build_catboost_tuned, 'catboost_tuned_v1.csv'),
]

print('\n=== Final 5-fold CV + submissions ===')
results = []
for name, params, builder, fname in configs:
    print(f'\n--- {name} ---')
    cv = cross_validate(builder(params), train_df)
    make_submission(builder(params), X_train, y_train, X_test, test_df['id'], fname)
    results.append((name, cv))

# Comparison table
print('\n=== Tuned Model Comparison ===')
print(f"{'Model':<20} {'CV MAE':>10} {'Std':>8}")
print('-' * 42)
for name, cv in results:
    print(f"{name:<20} ${cv['mae_mean']:>8.2f} ± ${cv['mae_std']:.2f}")
