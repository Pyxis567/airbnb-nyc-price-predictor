"""
Tests for src/features.py (Task 3 & 4)

Run with:
    conda run -n dsc80 pytest tests/test_features.py -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import numpy as np
import pandas as pd
from features import FeatureEngineer

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'raw files')

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def raw_train():
    return pd.read_csv(os.path.join(DATA_DIR, 'train.csv'), low_memory=False)

@pytest.fixture(scope='module')
def raw_test():
    return pd.read_csv(os.path.join(DATA_DIR, 'test.csv'), low_memory=False)

@pytest.fixture(scope='module')
def fitted(raw_train, raw_test):
    fe = FeatureEngineer()
    X_train = fe.fit_transform(raw_train, raw_train['price'])
    X_test  = fe.transform(raw_test)
    return fe, X_train, X_test


def make_row(**overrides):
    base = dict(
        neighbourhood_cleansed='Williamsburg',
        accommodates=4, bedrooms=1.0, beds=2.0,
        amenities='{Wifi,"Air conditioning",Kitchen}',
        square_feet=np.nan,
        number_of_reviews=10,
        host_since='2015-01-01',
        first_review='2016-01-01',
        last_review='2018-06-01',
        review_scores_rating=95.0,
        review_scores_accuracy=10.0,
        review_scores_cleanliness=10.0,
        review_scores_checkin=10.0,
        review_scores_communication=10.0,
        review_scores_location=10.0,
        review_scores_value=10.0,
        reviews_per_month=1.5,
        host_response_rate='100%',
        price=150.0,
        # text columns — required for TF-IDF and length/keyword features
        name='Cozy apartment in Brooklyn',
        summary='Great place to stay near subway',
        description='Spacious bright apartment near subway station perfect location',
        house_rules='No smoking no pets quiet hours',
    )
    base.update(overrides)
    return pd.DataFrame([base])


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

class TestIntegration:

    def test_no_nan_in_X_train(self, fitted):
        _, X_train, _ = fitted
        nans = X_train.isnull().sum()
        assert nans.sum() == 0, f"NaNs remain:\n{nans[nans > 0]}"

    def test_no_nan_in_X_test(self, fitted):
        _, _, X_test = fitted
        nans = X_test.isnull().sum()
        assert nans.sum() == 0, f"NaNs remain:\n{nans[nans > 0]}"

    def test_columns_match(self, fitted):
        _, X_train, X_test = fitted
        assert list(X_train.columns) == list(X_test.columns)

    def test_row_counts(self, fitted, raw_train, raw_test):
        _, X_train, X_test = fitted
        assert len(X_train) == len(raw_train)
        assert len(X_test)  == len(raw_test)

    def test_all_numeric(self, fitted):
        _, X_train, _ = fitted
        non_num = X_train.select_dtypes(exclude='number').columns.tolist()
        assert non_num == [], f"Non-numeric columns: {non_num}"

    def test_price_not_in_output(self, fitted):
        _, X_train, X_test = fitted
        assert 'price' not in X_train.columns
        assert 'price' not in X_test.columns

    def test_amenity_columns_present(self, fitted):
        _, X_train, _ = fitted
        amenity_cols = [c for c in X_train.columns if c.startswith('amenity_')]
        assert len(amenity_cols) == 20, f"Expected 20 amenity cols, got {len(amenity_cols)}"

    def test_capacity_score_present(self, fitted):
        _, X_train, _ = fitted
        assert 'capacity_score' in X_train.columns

    def test_neighbourhood_encoded_present(self, fitted):
        _, X_train, _ = fitted
        assert 'neighbourhood_encoded' in X_train.columns

    def test_neighbourhood_encoded_is_numeric(self, fitted):
        _, X_train, _ = fitted
        assert pd.api.types.is_numeric_dtype(X_train['neighbourhood_encoded'])


# ---------------------------------------------------------------------------
# Unit tests — target encoding
# ---------------------------------------------------------------------------

class TestTargetEncoding:

    def test_known_neighbourhood_gets_encoded_value(self):
        train = pd.concat([
            make_row(neighbourhood_cleansed='SoHo', price=200.0),
            make_row(neighbourhood_cleansed='SoHo', price=300.0),
            make_row(neighbourhood_cleansed='Bronx', price=80.0),
        ], ignore_index=True)
        fe = FeatureEngineer()
        X = fe.fit_transform(train, train['price'])
        # SoHo encoded value should be close to 250 (with smoothing towards global mean ~193)
        soho_encoded = X.loc[train['neighbourhood_cleansed'] == 'SoHo', 'neighbourhood_encoded'].iloc[0]
        assert 150 < soho_encoded < 300, f"Unexpected SoHo encoded value: {soho_encoded}"

    def test_unseen_neighbourhood_gets_global_mean(self):
        train = make_row(neighbourhood_cleansed='Harlem', price=150.0)
        test  = make_row(neighbourhood_cleansed='MysteryPlace').drop(columns=['price'])
        fe = FeatureEngineer()
        fe.fit_transform(train, train['price'])
        X_test = fe.transform(test)
        global_mean = 150.0
        # Should fall back to something near global mean (within 50)
        encoded = X_test['neighbourhood_encoded'].iloc[0]
        assert abs(encoded - global_mean) < 50, f"Unseen neighbourhood encoded as {encoded}"

    def test_no_leakage_test_uses_train_means(self):
        train = make_row(neighbourhood_cleansed='Harlem', price=200.0)
        test  = make_row(neighbourhood_cleansed='Harlem').drop(columns=['price'])
        fe = FeatureEngineer()
        fe.fit_transform(train, train['price'])
        X_test = fe.transform(test)
        # Test encoding uses train price mean, not test price
        encoded = X_test['neighbourhood_encoded'].iloc[0]
        assert encoded > 0


# ---------------------------------------------------------------------------
# Unit tests — amenities
# ---------------------------------------------------------------------------

class TestAmenities:

    def test_wifi_detected(self):
        row = make_row(amenities='{Wifi,Kitchen}')
        fe = FeatureEngineer()
        fe.fit_transform(row, row['price'])
        X = fe.transform(row.drop(columns=['price']))
        assert 'amenity_wifi' in X.columns
        assert X['amenity_wifi'].iloc[0] == 1

    def test_missing_amenity_is_zero(self):
        train = make_row(amenities='{Wifi,Kitchen,"Air conditioning"}')
        test  = make_row(amenities='{Kitchen}').drop(columns=['price'])
        fe = FeatureEngineer()
        fe.fit_transform(train, train['price'])
        X_test = fe.transform(test)
        assert X_test['amenity_wifi'].iloc[0] == 0

    def test_empty_amenities_all_zeros(self):
        row = make_row(amenities='{}')
        fe = FeatureEngineer()
        fe.fit_transform(row, row['price'])
        X = fe.transform(row.drop(columns=['price']))
        amenity_cols = [c for c in X.columns if c.startswith('amenity_')]
        assert X[amenity_cols].sum(axis=1).iloc[0] == 0

    def test_missing_amenities_all_zeros(self):
        row = make_row(amenities=np.nan)
        fe = FeatureEngineer()
        fe.fit_transform(row, row['price'])
        X = fe.transform(row.drop(columns=['price']))
        amenity_cols = [c for c in X.columns if c.startswith('amenity_')]
        assert X[amenity_cols].sum(axis=1).iloc[0] == 0

    def test_exactly_top_n_amenity_cols(self):
        row = make_row(amenities='{Wifi,Kitchen,"Air conditioning",TV,Heating}')
        fe = FeatureEngineer(n_amenities=3)
        fe.fit_transform(row, row['price'])
        X = fe.transform(row.drop(columns=['price']))
        amenity_cols = [c for c in X.columns if c.startswith('amenity_')]
        assert len(amenity_cols) == 3

    def test_unseen_amenity_in_test_ignored(self):
        train = make_row(amenities='{Wifi,Kitchen}')
        test  = make_row(amenities='{Wifi,"Hot tub"}').drop(columns=['price'])
        fe = FeatureEngineer()
        fe.fit_transform(train, train['price'])
        X_test = fe.transform(test)
        # "Hot tub" was not in train top amenities — no new column created
        assert 'amenity_hot_tub' not in X_test.columns


# ---------------------------------------------------------------------------
# Unit tests — capacity score
# ---------------------------------------------------------------------------

class TestReviewFeatures:

    def test_composite_review_score_present(self):
        row = make_row()
        fe = FeatureEngineer()
        X = fe.fit_transform(row, row['price'])
        assert 'composite_review_score' in X.columns

    def test_composite_review_score_mean_of_scores(self):
        row = make_row(
            review_scores_rating=100.0, review_scores_accuracy=9.0,
            review_scores_cleanliness=8.0, review_scores_checkin=10.0,
            review_scores_communication=10.0, review_scores_location=9.0,
            review_scores_value=9.0,
        )
        fe = FeatureEngineer()
        X = fe.fit_transform(row, row['price'])
        # Mean of [100, 9, 8, 10, 10, 9, 9] = 155/7 ≈ 22.14 — not normalized,
        # just raw mean of the 7 columns as-is
        expected = np.mean([100.0, 9.0, 8.0, 10.0, 10.0, 9.0, 9.0])
        assert X['composite_review_score'].iloc[0] == pytest.approx(expected)

    def test_composite_score_zero_when_all_missing(self):
        row = make_row(
            review_scores_rating=np.nan, review_scores_accuracy=np.nan,
            review_scores_cleanliness=np.nan, review_scores_checkin=np.nan,
            review_scores_communication=np.nan, review_scores_location=np.nan,
            review_scores_value=np.nan,
        )
        fe = FeatureEngineer()
        X = fe.fit_transform(row, row['price'])
        assert X['composite_review_score'].iloc[0] == 0.0

    def test_composite_score_ignores_nan_columns(self):
        # Only some scores present — composite should use available ones only
        row = make_row(
            review_scores_rating=90.0, review_scores_accuracy=np.nan,
            review_scores_cleanliness=np.nan, review_scores_checkin=np.nan,
            review_scores_communication=np.nan, review_scores_location=np.nan,
            review_scores_value=np.nan,
        )
        fe = FeatureEngineer()
        X = fe.fit_transform(row, row['price'])
        assert X['composite_review_score'].iloc[0] == pytest.approx(90.0)

    def test_review_span_days_present(self):
        row = make_row()
        fe = FeatureEngineer()
        X = fe.fit_transform(row, row['price'])
        assert 'review_span_days' in X.columns

    def test_review_span_days_correct(self):
        row = make_row(first_review='2016-01-01', last_review='2018-01-01')
        fe = FeatureEngineer()
        X = fe.fit_transform(row, row['price'])
        expected = (pd.Timestamp('2018-01-01') - pd.Timestamp('2016-01-01')).days
        assert X['review_span_days'].iloc[0] == expected

    def test_review_span_days_zero_when_no_reviews(self):
        row = make_row(first_review=np.nan, last_review=np.nan)
        fe = FeatureEngineer()
        X = fe.fit_transform(row, row['price'])
        assert X['review_span_days'].iloc[0] == 0

    def test_no_nan_in_review_features(self):
        row = make_row(
            review_scores_rating=np.nan, first_review=np.nan, last_review=np.nan
        )
        fe = FeatureEngineer()
        X = fe.fit_transform(row, row['price'])
        assert not X['composite_review_score'].isna().any()
        assert not X['review_span_days'].isna().any()


class TestTextFeatures:

    def test_length_columns_present(self):
        row = make_row()
        fe = FeatureEngineer()
        X = fe.fit_transform(row, row['price'])
        for col in ['len_name', 'len_summary', 'len_description', 'len_house_rules']:
            assert col in X.columns, f"Missing column: {col}"

    def test_length_correct(self):
        row = make_row()
        # make_row doesn't set name/summary/description/house_rules — add them
        row['name'] = 'Nice place'          # 10 chars
        row['description'] = 'Great spot!'  # 11 chars
        row['summary'] = 'Cozy'             # 4 chars
        row['house_rules'] = 'No smoking'   # 10 chars
        fe = FeatureEngineer()
        X = fe.fit_transform(row, row['price'])
        assert X['len_name'].iloc[0] == 10
        assert X['len_description'].iloc[0] == 11
        assert X['len_summary'].iloc[0] == 4
        assert X['len_house_rules'].iloc[0] == 10

    def test_length_zero_when_missing(self):
        row = make_row()
        row['name'] = np.nan
        row['description'] = np.nan
        fe = FeatureEngineer()
        X = fe.fit_transform(row, row['price'])
        assert X['len_name'].iloc[0] == 0
        assert X['len_description'].iloc[0] == 0

    def test_keyword_columns_present(self):
        row = make_row()
        fe = FeatureEngineer()
        X = fe.fit_transform(row, row['price'])
        for kw in ['kw_luxury', 'kw_cozy', 'kw_entire', 'kw_private', 'kw_studio']:
            assert kw in X.columns, f"Missing keyword column: {kw}"

    def test_keyword_flag_is_1_when_present(self):
        row = make_row()
        row['description'] = 'This is a luxury apartment in Manhattan'
        fe = FeatureEngineer()
        X = fe.fit_transform(row, row['price'])
        assert X['kw_luxury'].iloc[0] == 1

    def test_keyword_flag_is_0_when_absent(self):
        row = make_row()
        row['description'] = 'Nice clean place near subway'
        fe = FeatureEngineer()
        X = fe.fit_transform(row, row['price'])
        assert X['kw_luxury'].iloc[0] == 0

    def test_keyword_case_insensitive(self):
        row = make_row()
        row['description'] = 'LUXURY penthouse with views'
        fe = FeatureEngineer()
        X = fe.fit_transform(row, row['price'])
        assert X['kw_luxury'].iloc[0] == 1

    def test_keyword_zero_when_text_missing(self):
        row = make_row()
        row['description'] = np.nan
        fe = FeatureEngineer()
        X = fe.fit_transform(row, row['price'])
        assert X['kw_luxury'].iloc[0] == 0

    def test_tfidf_columns_present(self):
        row = make_row()
        row['description'] = 'spacious clean bright apartment near subway'
        fe = FeatureEngineer(n_tfidf=10)
        X = fe.fit_transform(row, row['price'])
        tfidf_cols = [c for c in X.columns if c.startswith('tfidf_')]
        # max_features is an upper bound — actual count <= n_tfidf
        assert 0 < len(tfidf_cols) <= 10

    def test_tfidf_no_nan(self):
        row = make_row()
        row['description'] = 'cozy private studio in Brooklyn'
        fe = FeatureEngineer(n_tfidf=5)
        X = fe.fit_transform(row, row['price'])
        tfidf_cols = [c for c in X.columns if c.startswith('tfidf_')]
        assert not X[tfidf_cols].isnull().any().any()

    def test_tfidf_missing_description_all_zeros(self):
        row = make_row()
        row['description'] = np.nan
        fe = FeatureEngineer(n_tfidf=5)
        X = fe.fit_transform(row, row['price'])
        tfidf_cols = [c for c in X.columns if c.startswith('tfidf_')]
        assert X[tfidf_cols].sum(axis=1).iloc[0] == 0.0

    def test_tfidf_no_leakage_uses_train_vocab(self):
        train = make_row()
        train['description'] = 'cozy bright apartment'
        test = make_row().drop(columns=['price'])
        test['description'] = 'luxury penthouse suite'  # entirely different words
        fe = FeatureEngineer(n_tfidf=5)
        X_train = fe.fit_transform(train, train['price'])
        X_test  = fe.transform(test)
        train_tfidf = [c for c in X_train.columns if c.startswith('tfidf_')]
        test_tfidf  = [c for c in X_test.columns  if c.startswith('tfidf_')]
        # test must have the exact same columns as train — no new columns for unseen words
        assert train_tfidf == test_tfidf

    def test_no_nan_in_all_text_features(self, fitted):
        _, X_train, _ = fitted
        text_cols = [c for c in X_train.columns
                     if c.startswith(('len_', 'kw_', 'tfidf_'))]
        assert len(text_cols) > 0
        assert X_train[text_cols].isnull().sum().sum() == 0


class TestCapacityScore:

    def test_basic_calculation(self):
        row = make_row(accommodates=4, bedrooms=1.0)
        fe = FeatureEngineer()
        X = fe.fit_transform(row, row['price'])
        assert X['capacity_score'].iloc[0] == pytest.approx(4 / (1 + 1))

    def test_zero_bedrooms_no_division_error(self):
        row = make_row(accommodates=2, bedrooms=0.0)
        fe = FeatureEngineer()
        X = fe.fit_transform(row, row['price'])
        assert X['capacity_score'].iloc[0] == pytest.approx(2 / (0 + 1))

    def test_missing_bedrooms_handled(self):
        row = make_row(accommodates=3, bedrooms=np.nan)
        fe = FeatureEngineer()
        X = fe.fit_transform(row, row['price'])
        assert not np.isnan(X['capacity_score'].iloc[0])
