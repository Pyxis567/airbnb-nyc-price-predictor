import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

# All dates measured relative to this reference point (just after the dataset period)
REFERENCE_DATE = pd.Timestamp('2019-01-01')

# Columns to drop entirely before any feature work
DROP_COLS = [
    'id', 'host_id', 'host_name', 'host_location', 'host_about',
    'host_acceptance_rate',       # 100% missing
    'host_neighbourhood',         # redundant with neighbourhood_cleansed
    'experiences_offered',        # constant ('none' in all rows)
    'city', 'state', 'country_code', 'country', 'market',  # all NYC, near-constant
    'zipcode',                    # 485 missing; neighbourhood_cleansed is better
    'neighbourhood_cleansed',     # 220 categories — target-encoded in features.py
    # Text fields — feature-engineered separately in features.py
    'name', 'summary', 'space', 'description', 'neighborhood_overview',
    'notes', 'transit', 'access', 'interaction', 'house_rules', 'amenities',
]

BOOL_COLS = [
    'host_is_superhost', 'host_has_profile_pic', 'host_identity_verified',
    'instant_bookable', 'is_business_travel_ready',
    'require_guest_profile_picture', 'require_guest_phone_verification',
]

DATE_COLS = ['host_since', 'first_review', 'last_review']

# One-hot encoded categoricals
OHE_COLS = ['room_type', 'property_type', 'neighbourhood_group_cleansed',
            'cancellation_policy', 'bed_type']

# Ordinal encoded: host_response_time has a natural order
RESPONSE_TIME_ORDER = {
    'within an hour': 0,
    'within a few hours': 1,
    'within a day': 2,
    'a few days or more': 3,
}

NUMERIC_COLS = [
    'accommodates', 'bathrooms', 'bedrooms', 'beds', 'guests_included',
    'minimum_nights', 'maximum_nights', 'number_of_reviews',
    'host_listings_count', 'calculated_host_listings_count',
    'review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness',
    'review_scores_checkin', 'review_scores_communication',
    'review_scores_location', 'review_scores_value', 'reviews_per_month',
    'square_feet', 'extra_people', 'host_response_rate', 'host_verifications',
    # date-derived columns — added dynamically after _clean
    'host_since_days', 'first_review_days', 'last_review_days',
    'host_response_time',
]

# property_type: group rare types as 'Other' to limit OHE width
TOP_PROPERTY_TYPES = [
    'Apartment', 'House', 'Townhouse', 'Loft', 'Condominium',
    'Serviced apartment', 'Guest suite',
]


class Preprocessor:
    """
    Fit on training data, then apply identically to test data.

    Usage:
        prep = Preprocessor()
        X_train = prep.fit_transform(train_df)
        X_test  = prep.transform(test_df)
        y_train = get_target(train_df)
    """

    def __init__(self):
        self._medians = {}
        self._ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        self._cat_fill = {}

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._clean(df)

        # Fit medians on training data only; fall back to 0 if all values are NaN
        for col in NUMERIC_COLS:
            if col in df.columns:
                median = df[col].median()
                self._medians[col] = median if not np.isnan(median) else 0.0

        # Fit OHE on training data only
        cat_df = self._prep_ohe_input(df, fit=True)
        self._ohe.fit(cat_df)

        return self._transform(df)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._clean(df)
        return self._transform(df)

    # ------------------------------------------------------------------

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

        # Drop target if present so output is always features-only
        if 'price' in df.columns:
            df = df.drop(columns=['price'])

        # Currency strings → float
        if 'extra_people' in df.columns:
            df['extra_people'] = pd.to_numeric(
                df['extra_people'].astype(str).str.replace(r'[$,]', '', regex=True),
                errors='coerce',
            )

        # Percentage string → float
        if 'host_response_rate' in df.columns:
            df['host_response_rate'] = pd.to_numeric(
                df['host_response_rate'].astype(str).str.replace('%', '', regex=False),
                errors='coerce',
            )

        # host_verifications: count items in the list string, e.g. "['email', 'phone']" → 2
        if 'host_verifications' in df.columns:
            raw = df['host_verifications'].fillna('[]').astype(str)
            # Empty list "[]" → 0; non-empty → count commas + 1
            df['host_verifications'] = raw.apply(
                lambda s: 0 if s.strip() in ('[]', '') else s.count(',') + 1
            )

        # Boolean strings → 0/1; missing → 0
        for col in BOOL_COLS:
            if col in df.columns:
                df[col] = df[col].map({'t': 1, 'f': 0}).fillna(0).astype(int)

        # Missingness flags — before imputation so they capture true missingness
        if 'square_feet' in df.columns:
            df['has_square_feet'] = df['square_feet'].notna().astype(int)

        if 'review_scores_rating' in df.columns:
            df['has_reviews'] = df['review_scores_rating'].notna().astype(int)

        # Date fields → days since REFERENCE_DATE (larger = older)
        for col in DATE_COLS:
            if col in df.columns:
                parsed = pd.to_datetime(df[col], errors='coerce')
                df[f'{col}_days'] = (REFERENCE_DATE - parsed).dt.days
                df = df.drop(columns=[col])

        # Ordinal encode host_response_time
        if 'host_response_time' in df.columns:
            df['host_response_time'] = df['host_response_time'].map(RESPONSE_TIME_ORDER)

        # Group rare property types before OHE
        if 'property_type' in df.columns:
            df['property_type'] = df['property_type'].where(
                df['property_type'].isin(TOP_PROPERTY_TYPES), other='Other'
            )

        return df

    def _prep_ohe_input(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        cat_df = df[OHE_COLS].copy()
        for col in OHE_COLS:
            if fit:
                mode = cat_df[col].mode()
                self._cat_fill[col] = mode[0] if not mode.empty else 'missing'
            cat_df[col] = cat_df[col].fillna(self._cat_fill.get(col, 'missing'))
        return cat_df

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # Impute all numeric columns with training medians
        for col in NUMERIC_COLS:
            if col in df.columns and col in self._medians:
                df[col] = df[col].fillna(self._medians[col])

        # OHE categoricals
        cat_df = self._prep_ohe_input(df)
        ohe_arr = self._ohe.transform(cat_df)
        ohe_cols = self._ohe.get_feature_names_out(OHE_COLS)
        ohe_df = pd.DataFrame(ohe_arr, columns=ohe_cols, index=df.index)

        df = df.drop(columns=[c for c in OHE_COLS if c in df.columns])
        df = pd.concat([df, ohe_df], axis=1)

        return df


def get_target(train_df: pd.DataFrame) -> pd.Series:
    """Return log1p-transformed price as the training target."""
    return np.log1p(train_df['price'])
