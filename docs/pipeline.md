# Pipeline Reference

## Usage

```python
from src.preprocess import Preprocessor, get_target
from src.features import FeatureEngineer
import pandas as pd

train = pd.read_csv('raw files/train.csv', low_memory=False)
test  = pd.read_csv('raw files/test.csv',  low_memory=False)

prep = Preprocessor()
feat = FeatureEngineer()

X_train = pd.concat([prep.fit_transform(train), feat.fit_transform(train, train['price'])], axis=1)
X_test  = pd.concat([prep.transform(test),      feat.transform(test)], axis=1)
y_train = get_target(train)  # log1p(price)
```

---

## `src/preprocess.py` — Cleans raw data into a numeric matrix

Takes the raw CSV and makes it model-ready. Everything here is **cleaning**, not feature creation.

| Step | What it does |
|------|-------------|
| Drop | `host_acceptance_rate` (100% missing), `experiences_offered` (constant), all text/id/location columns handled by `features.py` |
| Currency → float | `extra_people`: `$30.00` → `30.0` |
| Percentage → float | `host_response_rate`: `97%` → `97.0` |
| Boolean → 0/1 | 7 columns (`host_is_superhost`, etc.): `t`/`f`/NaN → 1/0/0 |
| Date → days | `host_since`, `first_review`, `last_review` → days since 2019-01-01 |
| Missingness flags | `has_square_feet`, `has_reviews` — created before imputation |
| Impute | Median for all numeric columns (fit on train only, applied to test) |
| Ordinal encode | `host_response_time`: 4 levels → 0–3 |
| One-hot encode | `room_type`, `property_type`, borough, `cancellation_policy`, `bed_type` |

**Output:** 63 numeric columns, zero NaNs, identical shape for train and test.

### Key classes / functions

```python
Preprocessor()
    .fit_transform(train_df) -> pd.DataFrame   # fit on train, return X_train
    .transform(test_df)      -> pd.DataFrame   # apply to test

get_target(train_df) -> pd.Series              # returns log1p(price)
```

---

## `src/features.py` — Engineers new signals from raw columns

Takes the same raw CSV and **creates new predictive features** that don't exist directly in the data. Run in parallel with the preprocessor and concatenated on top.

| Feature | Description |
|---------|-------------|
| `neighbourhood_encoded` | 220 neighbourhood names → 1 smoothed number (mean price per neighbourhood, pulled toward global mean for rare neighbourhoods via smoothing factor k=5) |
| `amenity_*` (×20) | Parses `{Wifi,"Air conditioning",...}` string → binary flag for each of the top 20 most common amenities |
| `capacity_score` | `accommodates / (bedrooms + 1)` — beds-per-person signal |
| `composite_review_score` | Row-wise mean of all 7 `review_scores_*` columns; NaN-aware (uses only available scores); 0 when no reviews |
| `review_span_days` | Days between `first_review` and `last_review` — how long a listing has been accumulating reviews (maturity signal) |
| `len_name`, `len_summary`, `len_description`, `len_house_rules` | Character count of each text field; 0 when missing |
| `kw_luxury`, `kw_cozy`, `kw_entire`, `kw_private`, `kw_studio` | Binary flag: 1 if keyword appears anywhere in the listing text fields |
| `tfidf_*` (×50) | Top 50 TF-IDF terms from `description` column (vocabulary fit on training data only) |

**Output:** 82 numeric columns, zero NaNs, identical shape for train and test.

### Key classes / functions

```python
FeatureEngineer(n_amenities=20, n_tfidf=50)
    .fit_transform(train_df, price_series) -> pd.DataFrame  # fit on train, return features
    .transform(test_df)                    -> pd.DataFrame  # apply to test
```

### No-leakage guarantees
- Target encoding: neighbourhood means computed on training price only
- TF-IDF vocabulary: fit on training descriptions only; unseen test words are ignored
- Imputation medians: all fit on training data only

---

## Combined output

| Source | Columns |
|--------|---------|
| `preprocess.py` | 63 |
| `features.py` | 82 |
| **Total** | **145** |

All 145 columns are numeric, contain no NaNs, and are identical in structure for train and test.
