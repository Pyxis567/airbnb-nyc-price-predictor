# Airbnb NYC Price Predictor
✅ Core model complete — actively experimenting with new approaches and ensembles.

Predicts nightly rental prices for New York City Airbnb listings using a tuned ensemble of gradient boosting models. Built for the UCSD Spring 2026 DSC 148 Kaggle competition (metric: MAE).

**Best result: CV MAE $37.53** — a 74% reduction from the $145.18 mean-value baseline.

> Built with [Claude Code](https://claude.ai/code) and [agent-skills](https://github.com/addyosmani/agent-skills).

---

## Results

| Model | CV MAE |
|---|---|
| Mean baseline (predict mean) | $145.18 |
| LightGBM (default) | $37.99 ± $0.74 |
| XGBoost (default) | $38.17 ± $0.77 |
| CatBoost (default) | $38.98 ± $0.74 |
| MLP (256→128) | $51.75 ± $2.89 |
| LightGBM (Optuna-tuned) | $37.87 ± $0.73 |
| XGBoost (Optuna-tuned) | $37.95 ± $0.73 |
| CatBoost (Optuna-tuned) | $38.54 ± $0.82 |
| **Ensemble (LGBM + XGB + CB)** | **$37.53** |

Ensemble weights (SLSQP-optimized on OOF predictions): LightGBM 43.3%, XGBoost 36.4%, CatBoost 20.4%.

---

## Dataset

- **Train:** 33,538 listings, 64 columns including `price`
- **Test:** 17,337 listings, same columns minus `price`
- **Target:** nightly price in USD (right-skewed; modeled on `log1p` scale)
- Source: NYC Airbnb listings from the Kaggle competition

---

## Project Structure

```
├── raw files/              # original competition CSVs (not tracked)
├── notebooks/
│   └── 01_eda.ipynb        # exploratory data analysis
├── src/
│   ├── preprocess.py       # Preprocessor class — cleaning, imputation, encoding
│   ├── features.py         # FeatureEngineer class — 146 engineered features
│   ├── models.py           # CV framework, model training, OOF prediction
│   ├── run_tuning.py       # Optuna hyperparameter search for all models
│   ├── run_ensemble.py     # blend OOF predictions, optimize weights, write submission
│   └── run_mlp.py          # MLP training script
├── submissions/            # output CSVs and cached OOF predictions
│   ├── ensemble_final.csv  # final submission
│   ├── best_params.json    # tuned hyperparameters
│   └── oof_cache.npz       # cached out-of-fold predictions for ensemble
└── tasks/
    ├── plan.md
    └── todo.md
```

---

## Pipeline

1. **Preprocessing** (`src/preprocess.py`) — strips currency/percent strings, parses dates into numeric features, one-hot encodes categoricals, imputes missing values, adds missingness flags for sparse columns (`square_feet`, review scores).

2. **Feature engineering** (`src/features.py`) — 146 features across five groups:
   - *Location:* smoothed target-encoding of neighbourhood, borough one-hot
   - *Property:* amenity flags (top amenities parsed from JSON-like strings), room type, accommodates, bedrooms/bathrooms
   - *Host:* superhost flag, days since joining, response rate, listing count
   - *Reviews:* score aggregates, `has_reviews` flag, reviews per month
   - *Text:* character-length of listing text fields, keyword flags (luxury, cozy, etc.), TF-IDF top terms

3. **Modeling** (`src/models.py`) — 5-fold CV framework; `Preprocessor` and `FeatureEngineer` are re-fit inside each fold to prevent target-encoding leakage. OOF predictions are cached for ensemble weight optimization.

4. **Tuning** (`src/run_tuning.py`) — 50 Optuna trials per model (TPE sampler, 3-fold inner CV).

5. **Ensemble** (`src/run_ensemble.py`) — SLSQP-optimized blend weights on OOF predictions; final predictions averaged over 5 folds on test set.

---

## Setup

```bash
# create and activate environment
conda create -n airbnb python=3.12
conda activate airbnb
pip install pandas numpy scikit-learn lightgbm xgboost catboost optuna matplotlib

# run the full pipeline
python src/models.py        # trains all models, saves OOF cache
python src/run_tuning.py    # Optuna tuning (~2-3 hrs total)
python src/run_ensemble.py  # optimizes blend weights, writes final submission
```

---

## Tech Stack

Python 3.12 · pandas · scikit-learn · LightGBM · XGBoost · CatBoost · Optuna · NumPy

---

## Acknowledgements

This project was built with [Claude Code](https://claude.ai/code) and [agent-skills](https://github.com/addyosmani/agent-skills) — a structured set of AI agent workflows covering planning, incremental implementation, test-driven development, and code review.
