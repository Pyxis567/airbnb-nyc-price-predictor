# Tests for src/models.py -- builder functions, CV pipeline, and tuning.
# Run with:
#   conda run -p C:/Users/Xh321/Miniforge3/envs/dsc80 pytest tests/test_models.py -v
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_regression
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import lightgbm as lgb

from models import (
    build_lgbm, build_xgb, build_catboost, build_mlp,
    build_lgbm_tuned, build_xgb_tuned, build_catboost_tuned,
    cross_validate, load_data,
)


# ---------------------------------------------------------------------------
# Builder unit tests — check type and key hyperparameters
# ---------------------------------------------------------------------------

class TestBuildLGBM:
    def test_returns_lgbm_regressor(self):
        m = build_lgbm()
        assert isinstance(m, lgb.LGBMRegressor)

    def test_n_estimators(self):
        assert build_lgbm().n_estimators == 500

    def test_seed(self):
        assert build_lgbm().random_state == 42


class TestBuildXGB:
    def test_returns_xgb_regressor(self):
        m = build_xgb()
        assert isinstance(m, XGBRegressor)

    def test_n_estimators(self):
        assert build_xgb().n_estimators == 500

    def test_seed(self):
        assert build_xgb().random_state == 42

    def test_hist_method(self):
        assert build_xgb().tree_method == 'hist'

    def test_verbosity_silent(self):
        assert build_xgb().verbosity == 0


class TestBuildCatBoost:
    def test_returns_catboost_regressor(self):
        m = build_catboost()
        assert isinstance(m, CatBoostRegressor)

    def test_iterations(self):
        assert build_catboost().get_param('iterations') == 500

    def test_seed(self):
        assert build_catboost().get_param('random_seed') == 42

    def test_verbose_off(self):
        assert build_catboost().get_param('verbose') == 0


# ---------------------------------------------------------------------------
# Smoke test — both models fit and predict on a tiny synthetic dataset
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def tiny_dataset():
    X, y = make_regression(n_samples=200, n_features=10, noise=0.1, random_state=42)
    return pd.DataFrame(X, columns=[f'f{i}' for i in range(10)]), pd.Series(y)


class TestModelsSmoke:
    def test_xgb_fit_predict(self, tiny_dataset):
        X, y = tiny_dataset
        m = build_xgb()
        m.fit(X, y)
        preds = m.predict(X)
        assert len(preds) == len(y)
        assert not np.any(np.isnan(preds))

    def test_catboost_fit_predict(self, tiny_dataset):
        X, y = tiny_dataset
        m = build_catboost()
        m.fit(X, y)
        preds = m.predict(X)
        assert len(preds) == len(y)
        assert not np.any(np.isnan(preds))


# ---------------------------------------------------------------------------
# Task 9 — MLP Neural Network
# ---------------------------------------------------------------------------

class TestBuildMLP:
    def test_returns_pipeline(self):
        from sklearn.pipeline import Pipeline
        assert isinstance(build_mlp(), Pipeline)

    def test_has_scaler(self):
        from sklearn.preprocessing import StandardScaler
        assert isinstance(build_mlp().named_steps['scaler'], StandardScaler)

    def test_has_mlp_step(self):
        from sklearn.neural_network import MLPRegressor
        assert isinstance(build_mlp().named_steps['mlp'], MLPRegressor)

    def test_seed(self):
        assert build_mlp().named_steps['mlp'].random_state == 42

    def test_early_stopping_enabled(self):
        assert build_mlp().named_steps['mlp'].early_stopping is True

    def test_fit_predict_smoke(self, tiny_dataset):
        X, y = tiny_dataset
        m = build_mlp()
        m.fit(X, y)
        preds = m.predict(X)
        assert len(preds) == len(y)
        assert not np.any(np.isnan(preds))


# ---------------------------------------------------------------------------
# Task 10 — Tuned model builders
# ---------------------------------------------------------------------------

LGBM_SAMPLE_PARAMS = {
    'num_leaves': 63,
    'learning_rate': 0.05,
    'min_child_samples': 20,
    'colsample_bytree': 0.8,
    'subsample': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
}

XGB_SAMPLE_PARAMS = {
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'min_child_weight': 5,
}

CATBOOST_SAMPLE_PARAMS = {
    'depth': 6,
    'learning_rate': 0.05,
    'l2_leaf_reg': 3.0,
    'bagging_temperature': 0.5,
    'random_strength': 1.0,
}


class TestBuildLGBMTuned:
    def test_returns_lgbm(self):
        assert isinstance(build_lgbm_tuned(LGBM_SAMPLE_PARAMS), lgb.LGBMRegressor)

    def test_n_estimators_500(self):
        assert build_lgbm_tuned(LGBM_SAMPLE_PARAMS).n_estimators == 500

    def test_params_applied(self):
        assert build_lgbm_tuned(LGBM_SAMPLE_PARAMS).num_leaves == 63

    def test_fit_predict_smoke(self, tiny_dataset):
        X, y = tiny_dataset
        m = build_lgbm_tuned(LGBM_SAMPLE_PARAMS)
        m.fit(X, y)
        preds = m.predict(X)
        assert len(preds) == len(y)
        assert not np.any(np.isnan(preds))


class TestBuildXGBTuned:
    def test_returns_xgb(self):
        assert isinstance(build_xgb_tuned(XGB_SAMPLE_PARAMS), XGBRegressor)

    def test_n_estimators_500(self):
        assert build_xgb_tuned(XGB_SAMPLE_PARAMS).n_estimators == 500

    def test_params_applied(self):
        assert build_xgb_tuned(XGB_SAMPLE_PARAMS).max_depth == 6

    def test_fit_predict_smoke(self, tiny_dataset):
        X, y = tiny_dataset
        m = build_xgb_tuned(XGB_SAMPLE_PARAMS)
        m.fit(X, y)
        preds = m.predict(X)
        assert len(preds) == len(y)
        assert not np.any(np.isnan(preds))


class TestBuildCatBoostTuned:
    def test_returns_catboost(self):
        assert isinstance(build_catboost_tuned(CATBOOST_SAMPLE_PARAMS), CatBoostRegressor)

    def test_iterations_500(self):
        assert build_catboost_tuned(CATBOOST_SAMPLE_PARAMS).get_param('iterations') == 500

    def test_params_applied(self):
        assert build_catboost_tuned(CATBOOST_SAMPLE_PARAMS).get_param('depth') == 6

    def test_fit_predict_smoke(self, tiny_dataset):
        X, y = tiny_dataset
        m = build_catboost_tuned(CATBOOST_SAMPLE_PARAMS)
        m.fit(X, y)
        preds = m.predict(X)
        assert len(preds) == len(y)
        assert not np.any(np.isnan(preds))
