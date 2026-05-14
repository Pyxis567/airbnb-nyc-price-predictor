# Implementation Plan: NYC Airbnb Price Prediction

## Overview
Build a regression model to predict nightly Airbnb prices in NYC using 64 features across text, host, location, property, booking, and review categories. Evaluation metric is MAE. The target is skewed so log1p transformation is recommended.

## Architecture Decisions
- Use log1p(price) as training target; expm1 to convert predictions back
- Start with tree-based models (LightGBM/XGBoost) — handle missing values and categoricals natively
- Add text features (TF-IDF or length/keyword features) as a second pass
- Ensemble top models as a final step

## Dependency Graph
```
raw files/train.csv, test.csv
        │
        ├── EDA (understand distributions, missingness, correlations)
        │       │
        │       └── Preprocessing pipeline (clean strings, encode cats, impute)
        │               │
        │               ├── Feature engineering (dates, text, amenities)
        │               │       │
        │               │       └── Model training (LightGBM, XGBoost, CatBoost)
        │               │               │
        │               │               └── Submission CSV → Kaggle
        │               │
        │               └── Cross-validation framework (MAE scoring)
```

---

## Phase 1: EDA and Baseline Understanding

### Task 1: Exploratory Data Analysis
**Description:** Load train.csv and test.csv, understand distributions, missingness, and correlations with price.

**Acceptance criteria:**
- [ ] Price distribution plotted (raw and log1p)
- [ ] Missing value counts per column documented
- [ ] Top correlating numeric features with price identified
- [ ] Borough-level price distribution visualized

**Verification:** Notebook runs end-to-end without errors

**Files:** `notebooks/01_eda.ipynb`

**Scope:** Small

---

### Task 2: Preprocessing Pipeline
**Description:** Build a reusable preprocessing function that cleans all messy fields and returns a numeric feature matrix.

**Acceptance criteria:**
- [ ] Currency strings (`$1,200.00`) converted to float
- [ ] Percentage strings (`97%`) converted to float
- [ ] Date fields parsed → days since epoch or year/month features
- [ ] Boolean strings (`t`/`f`) → 0/1
- [ ] Categoricals (room_type, property_type, neighbourhood_group_cleansed, cancellation_policy) encoded
- [ ] Missing values imputed (median for numeric, "missing" sentinel for categorical)
- [ ] Returns same-shape feature matrix for train and test

**Verification:** No NaN values in output; train and test have identical column count

**Files:** `src/preprocess.py`

**Scope:** Medium

---

### Checkpoint: Phase 1
- [ ] EDA notebook runs clean
- [ ] Preprocessing pipeline handles all 64 columns without errors
- [ ] Train and test feature matrices are aligned

---

## Phase 2: Feature Engineering

### Task 3: Location and Property Features
**Description:** Engineer features from location (borough, neighbourhood, zipcode) and property attributes (room type, amenities, capacity).

**Acceptance criteria:**
- [ ] Neighbourhood target-encoded using training price mean
- [ ] Amenities list parsed — top 20 amenities one-hot encoded
- [ ] Capacity score = accommodates / (bedrooms + 1)
- [ ] square_feet missingness flag added

**Files:** `src/features.py`

**Scope:** Small

---

### Task 4: Host and Review Features
**Description:** Engineer features from host history and review patterns.

**Acceptance criteria:**
- [ ] host_since → days since host joined as of latest date in dataset
- [ ] first_review, last_review → days since first/last review
- [ ] review_missing flag for listings with no reviews
- [ ] host_response_rate and host_acceptance_rate cleaned and imputed
- [ ] Composite review score = mean of all review_scores_* columns

**Files:** `src/features.py`

**Scope:** Small

---

### Task 5: Text Features
**Description:** Extract simple features from listing text fields.

**Acceptance criteria:**
- [ ] Character length of: name, summary, description, house_rules
- [ ] Keyword flags: "luxury", "cozy", "entire", "private", "studio"
- [ ] TF-IDF (top 50 terms) on description column

**Files:** `src/features.py`

**Scope:** Small

---

### Checkpoint: Phase 2
- [ ] Feature matrix includes location, property, host, review, and text features
- [ ] Train/test shapes are consistent
- [ ] No data leakage (no target encoding on full train without CV)

---

## Phase 3: Modeling

### Task 6: Cross-Validation Framework
**Description:** Set up 5-fold CV with MAE scoring on log1p(price).

**Acceptance criteria:**
- [ ] 5-fold CV returns mean MAE on original price scale
- [ ] Reproducible with fixed random seed (42)
- [ ] Works with any sklearn-compatible estimator

**Files:** `src/models.py`

**Scope:** Small

---

### Task 7: LightGBM Baseline
**Description:** Train LightGBM on full feature set, evaluate with CV, generate submission.

**Acceptance criteria:**
- [ ] CV MAE lower than simple linear regression baseline
- [ ] Submission CSV saved to `submissions/lgbm_v1.csv`
- [ ] Feature importances plotted

**Files:** `src/models.py`, `notebooks/03_modeling.ipynb`

**Scope:** Small

---

### Task 8: XGBoost and CatBoost Comparison
**Description:** Train XGBoost and CatBoost, compare CV MAE against LightGBM.

**Acceptance criteria:**
- [ ] All three models CV MAE recorded in a comparison table
- [ ] Best single model identified
- [ ] Submissions saved for each

**Files:** `notebooks/03_modeling.ipynb`

**Scope:** Small

---

### Task 9: Hyperparameter Tuning
**Description:** Tune the best model using Optuna or grid search.

**Acceptance criteria:**
- [ ] At least 50 trials run
- [ ] Best params saved
- [ ] CV MAE improves over default params

**Files:** `notebooks/03_modeling.ipynb`

**Scope:** Medium

---

### Task 10: Ensemble
**Description:** Blend predictions from top 2-3 models.

**Acceptance criteria:**
- [ ] Weighted average of LightGBM, XGBoost, CatBoost predictions
- [ ] Ensemble CV MAE beats best single model
- [ ] Final submission saved to `submissions/ensemble_final.csv`

**Files:** `notebooks/03_modeling.ipynb`

**Scope:** Small

---

### Checkpoint: Phase 3 — Final
- [ ] All submissions saved in `submissions/`
- [ ] Best submission uploaded to Kaggle
- [ ] CV MAE and leaderboard MAE recorded in this plan

---

## Risks and Mitigations
| Risk | Impact | Mitigation |
|------|--------|------------|
| Target leakage in neighbourhood encoding | High | Use CV-based target encoding only |
| log1p transform produces negative expm1 values | Medium | Clip predictions to 0 before submitting |
| Overfitting on text features | Medium | Use TF-IDF max_features limit; validate on holdout |
| Missing square_feet (>90% missing) | Low | Use missingness flag instead of imputing |

## Open Questions
- Are external datasets (e.g. NYC borough shapefiles, transit data) allowed?
- What is the Kaggle daily submission limit?
