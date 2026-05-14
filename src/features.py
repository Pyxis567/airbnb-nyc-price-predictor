import csv
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Smoothing factor for target encoding — how much to pull towards the global mean
# Higher = more regularization for rare neighbourhoods
TARGET_ENCODE_SMOOTHING = 5

TEXT_LEN_COLS = ['name', 'summary', 'description', 'house_rules']
KEYWORDS = ['luxury', 'cozy', 'entire', 'private', 'studio']

REVIEW_SCORE_COLS = [
    'review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness',
    'review_scores_checkin', 'review_scores_communication',
    'review_scores_location', 'review_scores_value',
]


def _parse_amenities(s) -> set:
    """Parse Airbnb amenity string '{Wifi,"Air conditioning",Kitchen}' into a set of names."""
    if pd.isna(s) or str(s).strip() in ('{}', ''):
        return set()
    inner = str(s).strip().strip('{}')
    try:
        items = next(csv.reader([inner], quotechar='"'))
    except Exception:
        items = inner.split(',')
    return {item.strip().lower() for item in items if item.strip()}


def _amenity_col(name: str) -> str:
    """Normalise an amenity name to a safe column name."""
    name = name.lower()
    name = re.sub(r'[^a-z0-9]+', '_', name)
    return f'amenity_{name.strip("_")}'


class FeatureEngineer:
    """
    Engineers location, property, host, review, and text features from the raw DataFrame.

    Fit on training data only; transform applies the same mapping to test data.

    Usage:
        fe = FeatureEngineer()
        X_train_feat = fe.fit_transform(train_df, train_df['price'])
        X_test_feat  = fe.transform(test_df)
    """

    def __init__(self, n_amenities: int = 20, n_tfidf: int = 50):
        self.n_amenities = n_amenities
        self.n_tfidf = n_tfidf
        self._global_mean: float = 0.0
        self._neighbourhood_map: dict = {}
        self._top_amenities: list = []
        self._tfidf: TfidfVectorizer | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_transform(self, df: pd.DataFrame, price: pd.Series) -> pd.DataFrame:
        df = df.copy()

        self._fit_target_encoding(df, price)
        self._fit_amenities(df)
        self._fit_tfidf(df)

        return self._build_features(df)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        return self._build_features(df)

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def _fit_target_encoding(self, df: pd.DataFrame, price: pd.Series) -> None:
        self._global_mean = float(price.mean())
        stats = (
            pd.DataFrame({'neighbourhood': df['neighbourhood_cleansed'], 'price': price.values})
            .groupby('neighbourhood')['price']
            .agg(['mean', 'count'])
        )
        k = TARGET_ENCODE_SMOOTHING
        # Smoothed mean: pull rare neighbourhoods towards global mean
        stats['encoded'] = (
            (stats['count'] * stats['mean'] + k * self._global_mean)
            / (stats['count'] + k)
        )
        self._neighbourhood_map = stats['encoded'].to_dict()

    def _fit_tfidf(self, df: pd.DataFrame) -> None:
        corpus = df['description'].fillna('').astype(str).tolist() if 'description' in df.columns else ['']
        self._tfidf = TfidfVectorizer(
            max_features=self.n_tfidf,
            strip_accents='unicode',
            stop_words='english',
            lowercase=True,
        )
        try:
            self._tfidf.fit(corpus)
        except ValueError:
            # Corpus is all empty / stop-words — add a sentinel so vocab is never empty.
            # The sentinel never appears in real text so it contributes zero weight.
            self._tfidf.fit(corpus + ['sentinel_placeholder_token'])

    def _fit_amenities(self, df: pd.DataFrame) -> None:
        from collections import Counter
        counts = Counter()
        for raw in df['amenities'].dropna():
            counts.update(_parse_amenities(raw))
        self._top_amenities = [name for name, _ in counts.most_common(self.n_amenities)]

    # ------------------------------------------------------------------
    # Feature construction
    # ------------------------------------------------------------------

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        parts = []

        parts.append(self._neighbourhood_features(df))
        parts.append(self._amenity_features(df))
        parts.append(self._property_features(df))
        parts.append(self._review_features(df))
        parts.append(self._text_features(df))

        result = pd.concat(parts, axis=1)

        # Drop target if raw train df was passed
        if 'price' in result.columns:
            result = result.drop(columns=['price'])

        return result

    def _neighbourhood_features(self, df: pd.DataFrame) -> pd.DataFrame:
        encoded = (
            df['neighbourhood_cleansed']
            .map(self._neighbourhood_map)
            .fillna(self._global_mean)
        )
        return pd.DataFrame({'neighbourhood_encoded': encoded.values}, index=df.index)

    def _amenity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        parsed = df['amenities'].apply(_parse_amenities)
        data = {
            _amenity_col(name): parsed.apply(lambda s: int(name in s))
            for name in self._top_amenities
        }
        return pd.DataFrame(data, index=df.index)

    def _property_features(self, df: pd.DataFrame) -> pd.DataFrame:
        bedrooms = pd.to_numeric(df['bedrooms'], errors='coerce').fillna(0)
        accommodates = pd.to_numeric(df['accommodates'], errors='coerce').fillna(1)
        capacity_score = accommodates / (bedrooms + 1)
        return pd.DataFrame({'capacity_score': capacity_score.values}, index=df.index)

    def _review_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Composite review score: mean of available review_scores_* columns
        available = [c for c in REVIEW_SCORE_COLS if c in df.columns]
        scores = df[available].apply(pd.to_numeric, errors='coerce')
        # nanmean per row; rows with all NaN → 0
        composite = scores.mean(axis=1, skipna=True).fillna(0.0)

        # Review span: days between first and last review (listing maturity)
        first = pd.to_datetime(df['first_review'], errors='coerce') if 'first_review' in df.columns else pd.NaT
        last  = pd.to_datetime(df['last_review'],  errors='coerce') if 'last_review'  in df.columns else pd.NaT
        span  = (last - first).dt.days.fillna(0).clip(lower=0)

        return pd.DataFrame({
            'composite_review_score': composite.values,
            'review_span_days':       span.values,
        }, index=df.index)

    def _text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        result = {}

        # Character lengths — missing text → 0
        for col in TEXT_LEN_COLS:
            src = df[col].fillna('').astype(str) if col in df.columns else pd.Series([''] * len(df), index=df.index)
            result[f'len_{col}'] = src.str.len().values

        # Keyword flags — search across all text columns, case-insensitive
        all_text = pd.Series([''] * len(df), index=df.index)
        for col in TEXT_LEN_COLS:
            if col in df.columns:
                all_text = all_text + ' ' + df[col].fillna('').astype(str)
        all_text = all_text.str.lower()

        for kw in KEYWORDS:
            result[f'kw_{kw}'] = all_text.str.contains(kw, regex=False).astype(int).values

        # TF-IDF on description
        corpus = df['description'].fillna('').astype(str).tolist() if 'description' in df.columns else [''] * len(df)
        tfidf_matrix = self._tfidf.transform(corpus).toarray()
        tfidf_cols = [f'tfidf_{t}' for t in self._tfidf.get_feature_names_out()]
        for i, col in enumerate(tfidf_cols):
            result[col] = tfidf_matrix[:, i]

        return pd.DataFrame(result, index=df.index)
