# Task Checklist — DSC 148 Kaggle

## Phase 1: EDA and Baseline
- [x] Task 1: EDA notebook (`notebooks/01_eda.ipynb`)
- [x] Task 2: Preprocessing pipeline (`src/preprocess.py`)
- [ ] CHECKPOINT: EDA clean, preprocessing handles all 64 columns

## Phase 2: Feature Engineering
- [x] Task 3: Location and property features (`src/features.py`)
- [x] Task 4: Host and review features (`src/features.py`)
- [x] Task 5: Text features (`src/features.py`)
- [x] CHECKPOINT: Full feature matrix, no leakage, train/test aligned

## Phase 3: Modeling
- [x] Task 6: CV framework (`src/models.py`)
- [x] Task 7: LightGBM baseline + submission (`submissions/lgbm_v1.csv`)
- [x] Task 8: XGBoost and CatBoost comparison
- [x] Task 9: Neural Network method
- [x] Task 10: Hyperparameter tuning (all model)#edited from best model
- [x] Task 11: Ensemble + final submission (`submissions/ensemble_final.csv`)
- [ ] CHECKPOINT: Best submission on Kaggle leaderboard

## Results Log
| Version | Model | CV MAE | LB MAE | Notes |
|---------|-------|--------|--------|-------|
| baseline | Mean ($145.18) | - | - | instructor baseline |
| v1_linear | Linear Regression | - | - | 5 features, borough OHE |
| lgbm_v1 | LightGBM | $37.99 ± $0.74 | - | 146 features, default params, 5-fold CV |
| xgb_v1 | XGBoost | $38.17 ± $0.77 | - | 146 features, default params, 5-fold CV |
| catboost_v1 | CatBoost | $38.98 ± $0.74 | - | 146 features, default params, 5-fold CV |
| mlp_v1 | MLP (256→128) | $51.75 ± $2.89 | - | 146 features, sklearn MLPRegressor + StandardScaler |
| lgbm_tuned_v1 | LightGBM tuned | $37.87 ± $0.73 | - | Optuna 50 trials |
| xgb_tuned_v1 | XGBoost tuned | $37.95 ± $0.73 | - | Optuna 50 trials |
| catboost_tuned_v1 | CatBoost tuned | $38.54 ± $0.82 | - | Optuna 50 trials |
| ensemble_final | LGBM+XGB+CB blend | $37.53 | - | Weights: 0.433/0.364/0.204, MLP=0 |
