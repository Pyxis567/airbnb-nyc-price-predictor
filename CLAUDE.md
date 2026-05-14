# DSC 148 — Kaggle: NYC Airbnb Price Prediction

## Agent Skills (always use by default)
- Use `/agent-skills:plan` before starting any new task or phase
- Use `/agent-skills:build` when implementing each task
- Use `/agent-skills:test` to verify correctness after each implementation

## Competition
- **Platform:** Kaggle — UCSD Spring 2026 DSC148: Data Mining
- **Task:** Supervised regression — predict nightly rental price (USD) for NYC Airbnb listings
- **Evaluation metric:** Mean Absolute Error (MAE) — lower is better
- **Submission format:** CSV with two columns: `Id`, `Predicted`

## Project Structure
```
ucsd-spring-2026-dsc-148/
├── CLAUDE.md                   ← you are here
├── raw files/                  ← original competition data, do not modify
│   ├── train.csv               ← 33,538 rows, 64 columns, includes `price`
│   ├── test.csv                ← 17,337 rows, same columns minus `price`
│   ├── simple_baseline.ipynb   ← starter notebook from instructor
│   ├── mean_value_baseline.csv
│   └── simple_linear_regression_baseline.csv
├── competition description/
│   ├── Kaggle Overview.pdf
│   └── Kaggle Data.pdf
├── notebooks/                  ← working notebooks (EDA, preprocessing, modeling)
├── src/                        ← reusable Python modules
├── submissions/                ← saved submission CSVs
└── tasks/
    ├── plan.md                 ← implementation plan
    └── todo.md                 ← task checklist
```

## Data
- **Read with:** `pd.read_csv('raw files/train.csv', low_memory=False)` — column 30 (zipcode) has mixed types
- **Target:** `price` — float, mean $145.18, median $109, range $0–$1,999, right-skewed
- **Target transform:** use `log1p(price)` during training, `expm1(pred)` before submitting
- **Clip negatives** in final predictions to 0 before submitting

## Feature Groups (64 columns total)
| Group | Key columns |
|-------|-------------|
| Text | name, summary, space, description, neighborhood_overview, notes, transit, access, interaction, house_rules |
| Host | host_since, host_response_time, host_response_rate, host_is_superhost, host_listings_count, host_identity_verified |
| Location | neighbourhood_cleansed, neighbourhood_group_cleansed, zipcode (5 boroughs: Manhattan, Brooklyn, Queens, Bronx, Staten Island) |
| Property | property_type, room_type, accommodates, bathrooms, bedrooms, beds, bed_type, amenities, square_feet |
| Booking | guests_included, extra_people, minimum_nights, maximum_nights, instant_bookable, cancellation_policy |
| Reviews | number_of_reviews, first_review, last_review, review_scores_*, reviews_per_month |

## Key EDA Findings
- `host_acceptance_rate` — 100% missing, drop entirely
- `square_feet` — 99% missing, add binary `has_square_feet` flag, impute with 0
- Review score fields — ~23% missing (listings with no reviews); add `has_reviews` flag
- Manhattan has significantly higher prices than other boroughs
- `room_type` strongly separates price (Entire home >> Private room >> Shared room)
- Text field lengths weakly correlate with price — worth including as cheap features

## Key Preprocessing Notes
- `price`, `extra_people` are currency strings — strip `$` and `,` before converting to float
- `host_response_rate`, `host_acceptance_rate` are percentage strings — strip `%`
- `host_since`, `first_review`, `last_review` are date strings — extract year, month, days-since features
- `amenities` is a JSON-like string list — parse and one-hot encode top amenities
- `square_feet` is mostly missing — missingness flag may be more useful than imputation
- Review score fields have many NaNs for new listings — missingness is informative
- Boolean-like columns (`instant_bookable`, `host_is_superhost`, etc.) need t/f → 0/1

## Baseline Performance
| Model | MAE (approx) |
|-------|-------------|
| Mean value | ~$XX (predicts $145.18 for all) |
| Simple linear regression | uses 5 numeric + 5 borough one-hot features |

## Environment
- **Conda env:** `dsc80` at `C:\Users\Xh321\Miniforge3\envs\dsc80`
- **Python:** 3.12.12
- **Key packages:** pandas 2.2.3, numpy 2.1.1, scikit-learn 1.5.2, xgboost 3.2.0, lightgbm 4.6.0, catboost 1.2.10, optuna 4.8.0, torch 2.11.0
- **Run scripts with:** `conda run -p C:\Users\Xh321\Miniforge3\envs\dsc80 python <script>`
- **Default Python (`python`) uses system Python 3.14 — always prefix with `conda run` for this project**

## Workflow
```bash
# EDA
conda run -p C:\Users\Xh321\Miniforge3\envs\dsc80 jupyter notebook notebooks/01_eda.ipynb

# Run a model script
conda run -p C:\Users\Xh321\Miniforge3\envs\dsc80 python src/models.py

# Submit — upload submissions/submission_vN.csv to Kaggle
```

## Rules
- External data: check Kaggle rules tab
- Max submissions per day: check Kaggle rules tab
- Team size: check Kaggle rules tab
