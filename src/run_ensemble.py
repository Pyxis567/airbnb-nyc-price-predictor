"""Weighted-average ensemble of tuned models — Task 11.
Run after run_tuning.py has produced submissions/best_params.json.
OOF predictions are cached to submissions/oof_cache.npz so the slow
per-model CV only runs once; subsequent runs skip straight to blending.
"""
import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_error

sys.path.insert(0, str(Path(__file__).parent))

from models import (
    load_data, oof_predict,
    build_lgbm_tuned, build_xgb_tuned, build_catboost_tuned, build_mlp,
)
from preprocess import get_target

ROOT = Path(__file__).parent.parent
PARAMS_FILE  = ROOT / 'submissions' / 'best_params.json'
CACHE_FILE   = ROOT / 'submissions' / 'oof_cache.npz'
SUBMISSIONS_DIR = ROOT / 'submissions'

if not PARAMS_FILE.exists():
    sys.exit(f'ERROR: {PARAMS_FILE} not found — run src/run_tuning.py first.')

print('Loading data...')
train_df, test_df = load_data()
print(f'Train: {train_df.shape}  Test: {test_df.shape}')

params = json.loads(PARAMS_FILE.read_text())
y_true = np.expm1(get_target(train_df))

configs = [
    ('LGBM tuned',     build_lgbm_tuned(params['lgbm'])),
    ('XGB tuned',      build_xgb_tuned(params['xgb'])),
    ('CatBoost tuned', build_catboost_tuned(params['catboost'])),
    ('MLP',            build_mlp()),
]
names = [name for name, _ in configs]

# ------------------------------------------------------------------
# Load cache or compute OOF predictions model-by-model
# ------------------------------------------------------------------
if CACHE_FILE.exists():
    print(f'\nLoading cached OOF predictions from {CACHE_FILE}')
    cache = np.load(CACHE_FILE)
    oof_matrix  = cache['oof_matrix']
    test_matrix = cache['test_matrix']
    cached_names = list(cache['names'])
    print(f'  Cached models: {cached_names}')
else:
    cached_names = []
    oof_matrix  = np.zeros((len(train_df), 0))
    test_matrix = np.zeros((len(test_df),  0))

missing = [n for n in names if n not in cached_names]

for name, model in configs:
    if name not in missing:
        continue
    print(f'\n--- {name} (OOF + test preds) ---')
    oof, test_preds = oof_predict(model, train_df, test_df)
    print(f'  OOF MAE: ${mean_absolute_error(y_true, oof):.2f}')

    oof_matrix  = np.column_stack([oof_matrix,  oof])        if oof_matrix.shape[1]  else oof[:, None]
    test_matrix = np.column_stack([test_matrix, test_preds]) if test_matrix.shape[1] else test_preds[:, None]
    cached_names.append(name)

    # Save after every model so progress is never lost
    SUBMISSIONS_DIR.mkdir(exist_ok=True)
    np.savez(
        CACHE_FILE,
        oof_matrix=oof_matrix,
        test_matrix=test_matrix,
        names=cached_names,
    )
    print(f'  Progress saved → {CACHE_FILE}')

# Re-order columns to match configs order
order = [cached_names.index(n) for n in names]
oof_matrix  = oof_matrix[:, order]
test_matrix = test_matrix[:, order]

# ------------------------------------------------------------------
# Find optimal blend weights via OOF MAE minimization
# ------------------------------------------------------------------
print('\n=== Individual OOF MAEs ===')
for name, col in zip(names, oof_matrix.T):
    print(f'  {name:<20}: ${mean_absolute_error(y_true, col):.2f}')

n_models = len(configs)

def _mae_objective(weights: np.ndarray) -> float:
    w = weights / weights.sum()
    return mean_absolute_error(y_true, oof_matrix @ w)

result = minimize(
    _mae_objective,
    x0=np.ones(n_models) / n_models,
    method='SLSQP',
    bounds=[(0.0, 1.0)] * n_models,
    constraints=[{'type': 'eq', 'fun': lambda w: w.sum() - 1.0}],
    options={'ftol': 1e-9, 'maxiter': 1000},
)
best_weights = result.x / result.x.sum()

print('\n=== Optimal Ensemble Weights ===')
for name, w in zip(names, best_weights):
    print(f'  {name:<20}: {w:.4f}')
print(f'  Ensemble OOF MAE: ${result.fun:.2f}')

# ------------------------------------------------------------------
# Final predictions + submission
# ------------------------------------------------------------------
ensemble_preds = (test_matrix @ best_weights).clip(0)

out = SUBMISSIONS_DIR / 'ensemble_final.csv'
pd.DataFrame({'Id': test_df['id'], 'Predicted': ensemble_preds}).to_csv(out, index=False)
print(f'\nSaved: {out}  ({len(ensemble_preds)} rows)')

weight_record = {name: float(w) for name, w in zip(names, best_weights)}
weight_record['ensemble_oof_mae'] = float(result.fun)
(SUBMISSIONS_DIR / 'ensemble_weights.json').write_text(json.dumps(weight_record, indent=2))
print(f'Weights saved → {SUBMISSIONS_DIR / "ensemble_weights.json"}')
