"""
Tests for src/preprocess.py

Run with:
    conda run -n dsc80 pytest tests/test_preprocess.py -v
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import pandas as pd
import numpy as np
from preprocess import (
    Preprocessor, get_target,
    BOOL_COLS, DROP_COLS, OHE_COLS, RESPONSE_TIME_ORDER, TOP_PROPERTY_TYPES,
)

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
    prep = Preprocessor()
    X_train = prep.fit_transform(raw_train)
    X_test = prep.transform(raw_test)
    y_train = get_target(raw_train)
    return prep, X_train, X_test, y_train


def make_minimal_row(**overrides):
    """Build a one-row DataFrame with all required columns set to safe defaults."""
    base = {
        'id': 1, 'host_id': 100,
        'host_name': 'Alice', 'host_since': '2015-06-01',
        'host_location': 'New York', 'host_about': 'Hi',
        'host_response_time': 'within an hour',
        'host_response_rate': '100%',
        'host_acceptance_rate': np.nan,
        'host_is_superhost': 't',
        'host_neighbourhood': 'Harlem',
        'host_listings_count': 1,
        'host_verifications': "['email', 'phone']",
        'host_has_profile_pic': 't',
        'host_identity_verified': 't',
        'neighbourhood_cleansed': 'Harlem',
        'neighbourhood_group_cleansed': 'Manhattan',
        'city': 'New York', 'state': 'NY', 'zipcode': '10031',
        'market': 'New York', 'country_code': 'US', 'country': 'United States',
        'property_type': 'Apartment', 'room_type': 'Entire home/apt',
        'accommodates': 4, 'bathrooms': 1.0, 'bedrooms': 1.0,
        'beds': 1.0, 'bed_type': 'Real Bed',
        'amenities': "{'TV','Wifi'}",
        'square_feet': np.nan,
        'price': 150.0,
        'guests_included': 1, 'extra_people': '$20.00',
        'minimum_nights': 1, 'maximum_nights': 365,
        'number_of_reviews': 10,
        'first_review': '2016-01-01', 'last_review': '2018-12-01',
        'review_scores_rating': 95.0, 'review_scores_accuracy': 10.0,
        'review_scores_cleanliness': 10.0, 'review_scores_checkin': 10.0,
        'review_scores_communication': 10.0, 'review_scores_location': 10.0,
        'review_scores_value': 10.0,
        'instant_bookable': 'f', 'is_business_travel_ready': 'f',
        'cancellation_policy': 'flexible',
        'require_guest_profile_picture': 'f',
        'require_guest_phone_verification': 'f',
        'calculated_host_listings_count': 1,
        'reviews_per_month': 1.5,
        'experiences_offered': 'none',
        'name': 'Nice place', 'summary': 'Great spot',
        'space': 'Cozy', 'description': 'Lovely',
        'neighborhood_overview': 'Great area',
        'notes': '', 'transit': 'Subway nearby',
        'access': 'Full', 'interaction': 'Available',
        'house_rules': 'No smoking', 'host_about': 'Love hosting',
    }
    base.update(overrides)
    return pd.DataFrame([base])


# ---------------------------------------------------------------------------
# Integration tests (real data)
# ---------------------------------------------------------------------------

class TestIntegration:

    def test_no_nan_in_X_train(self, fitted):
        _, X_train, _, _ = fitted
        assert X_train.isnull().sum().sum() == 0, \
            f"NaNs remain in X_train: {X_train.isnull().sum()[X_train.isnull().sum() > 0]}"

    def test_no_nan_in_X_test(self, fitted):
        _, _, X_test, _ = fitted
        assert X_test.isnull().sum().sum() == 0, \
            f"NaNs remain in X_test: {X_test.isnull().sum()[X_test.isnull().sum() > 0]}"

    def test_columns_match_train_test(self, fitted):
        _, X_train, X_test, _ = fitted
        assert list(X_train.columns) == list(X_test.columns)

    def test_train_row_count(self, fitted, raw_train):
        _, X_train, _, _ = fitted
        assert len(X_train) == len(raw_train)

    def test_test_row_count(self, fitted, raw_test):
        _, _, X_test, _ = fitted
        assert len(X_test) == len(raw_test)

    def test_price_not_in_output(self, fitted):
        _, X_train, X_test, _ = fitted
        assert 'price' not in X_train.columns
        assert 'price' not in X_test.columns

    def test_dropped_cols_absent(self, fitted):
        _, X_train, _, _ = fitted
        for col in DROP_COLS:
            assert col not in X_train.columns, f"Dropped column '{col}' still present"

    def test_ohe_columns_present(self, fitted):
        _, X_train, _, _ = fitted
        assert 'room_type_Entire home/apt' in X_train.columns
        assert 'neighbourhood_group_cleansed_Manhattan' in X_train.columns
        assert 'cancellation_policy_flexible' in X_train.columns

    def test_missingness_flags_present(self, fitted):
        _, X_train, _, _ = fitted
        assert 'has_square_feet' in X_train.columns
        assert 'has_reviews' in X_train.columns

    def test_date_columns_present(self, fitted):
        _, X_train, _, _ = fitted
        assert 'host_since_days' in X_train.columns
        assert 'first_review_days' in X_train.columns
        assert 'last_review_days' in X_train.columns

    def test_target_shape_and_no_nan(self, fitted):
        _, X_train, _, y_train = fitted
        assert len(y_train) == len(X_train)
        assert y_train.isnull().sum() == 0

    def test_target_is_log1p(self, raw_train, fitted):
        _, _, _, y_train = fitted
        expected = np.log1p(raw_train['price'])
        np.testing.assert_array_almost_equal(y_train.values, expected.values)

    def test_all_numeric(self, fitted):
        _, X_train, _, _ = fitted
        non_numeric = X_train.select_dtypes(exclude='number').columns.tolist()
        assert non_numeric == [], f"Non-numeric columns in output: {non_numeric}"


# ---------------------------------------------------------------------------
# Unit tests (synthetic data — fast, isolated)
# ---------------------------------------------------------------------------

class TestCurrencyCleaning:

    def test_extra_people_dollar_string(self):
        df = make_minimal_row(extra_people='$30.00')
        prep = Preprocessor()
        X = prep.fit_transform(df)
        assert X['extra_people'].iloc[0] == 30.0

    def test_extra_people_zero(self):
        df = make_minimal_row(extra_people='$0.00')
        prep = Preprocessor()
        X = prep.fit_transform(df)
        assert X['extra_people'].iloc[0] == 0.0

    def test_extra_people_missing_imputed(self):
        train = make_minimal_row(extra_people='$20.00')
        test = make_minimal_row(extra_people=np.nan)
        test = test.drop(columns=['price'])
        prep = Preprocessor()
        prep.fit_transform(train)
        X_test = prep.transform(test)
        assert not np.isnan(X_test['extra_people'].iloc[0])


class TestPercentageCleaning:

    def test_response_rate_percent_string(self):
        df = make_minimal_row(host_response_rate='87%')
        prep = Preprocessor()
        X = prep.fit_transform(df)
        assert X['host_response_rate'].iloc[0] == 87.0

    def test_response_rate_missing_imputed(self):
        train = make_minimal_row(host_response_rate='100%')
        test = make_minimal_row(host_response_rate=np.nan)
        test = test.drop(columns=['price'])
        prep = Preprocessor()
        prep.fit_transform(train)
        X_test = prep.transform(test)
        assert not np.isnan(X_test['host_response_rate'].iloc[0])


class TestBooleanCleaning:

    def test_true_maps_to_1(self):
        df = make_minimal_row(host_is_superhost='t')
        prep = Preprocessor()
        X = prep.fit_transform(df)
        assert X['host_is_superhost'].iloc[0] == 1

    def test_false_maps_to_0(self):
        df = make_minimal_row(host_is_superhost='f')
        prep = Preprocessor()
        X = prep.fit_transform(df)
        assert X['host_is_superhost'].iloc[0] == 0

    def test_missing_bool_maps_to_0(self):
        df = make_minimal_row(host_is_superhost=np.nan)
        prep = Preprocessor()
        X = prep.fit_transform(df)
        assert X['host_is_superhost'].iloc[0] == 0


class TestHostVerifications:

    def test_two_verifications(self):
        df = make_minimal_row(host_verifications="['email', 'phone']")
        prep = Preprocessor()
        X = prep.fit_transform(df)
        assert X['host_verifications'].iloc[0] == 2

    def test_empty_list(self):
        df = make_minimal_row(host_verifications='[]')
        prep = Preprocessor()
        X = prep.fit_transform(df)
        assert X['host_verifications'].iloc[0] == 0

    def test_one_verification(self):
        df = make_minimal_row(host_verifications="['email']")
        prep = Preprocessor()
        X = prep.fit_transform(df)
        assert X['host_verifications'].iloc[0] == 1

    def test_missing_verifications(self):
        df = make_minimal_row(host_verifications=np.nan)
        prep = Preprocessor()
        X = prep.fit_transform(df)
        assert X['host_verifications'].iloc[0] == 0


class TestMissingnessFlags:

    def test_has_square_feet_when_present(self):
        df = make_minimal_row(square_feet=500.0)
        prep = Preprocessor()
        X = prep.fit_transform(df)
        assert X['has_square_feet'].iloc[0] == 1

    def test_has_square_feet_when_missing(self):
        df = make_minimal_row(square_feet=np.nan)
        prep = Preprocessor()
        X = prep.fit_transform(df)
        assert X['has_square_feet'].iloc[0] == 0

    def test_has_reviews_when_present(self):
        df = make_minimal_row(review_scores_rating=95.0)
        prep = Preprocessor()
        X = prep.fit_transform(df)
        assert X['has_reviews'].iloc[0] == 1

    def test_has_reviews_when_missing(self):
        df = make_minimal_row(review_scores_rating=np.nan,
                              first_review=np.nan, last_review=np.nan,
                              reviews_per_month=np.nan)
        prep = Preprocessor()
        X = prep.fit_transform(df)
        assert X['has_reviews'].iloc[0] == 0


class TestDateFeatures:

    def test_host_since_produces_numeric(self):
        df = make_minimal_row(host_since='2015-01-01')
        prep = Preprocessor()
        X = prep.fit_transform(df)
        assert 'host_since_days' in X.columns
        assert X['host_since_days'].iloc[0] == (pd.Timestamp('2019-01-01') - pd.Timestamp('2015-01-01')).days

    def test_original_date_col_dropped(self):
        df = make_minimal_row()
        prep = Preprocessor()
        X = prep.fit_transform(df)
        assert 'host_since' not in X.columns
        assert 'first_review' not in X.columns
        assert 'last_review' not in X.columns

    def test_missing_date_imputed(self):
        df = make_minimal_row(first_review=np.nan, last_review=np.nan)
        prep = Preprocessor()
        X = prep.fit_transform(df)
        assert not np.isnan(X['first_review_days'].iloc[0])
        assert not np.isnan(X['last_review_days'].iloc[0])


class TestOrdinalEncoding:

    def test_within_an_hour_is_0(self):
        df = make_minimal_row(host_response_time='within an hour')
        prep = Preprocessor()
        X = prep.fit_transform(df)
        assert X['host_response_time'].iloc[0] == 0

    def test_few_days_is_3(self):
        df = make_minimal_row(host_response_time='a few days or more')
        prep = Preprocessor()
        X = prep.fit_transform(df)
        assert X['host_response_time'].iloc[0] == 3

    def test_missing_response_time_imputed(self):
        df = make_minimal_row(host_response_time=np.nan)
        prep = Preprocessor()
        X = prep.fit_transform(df)
        assert not np.isnan(X['host_response_time'].iloc[0])


class TestPropertyTypeGrouping:

    def test_rare_type_becomes_other(self):
        df = make_minimal_row(property_type='Yurt')
        prep = Preprocessor()
        X = prep.fit_transform(df)
        assert X['property_type_Other'].iloc[0] == 1.0

    def test_known_type_kept(self):
        df = make_minimal_row(property_type='Apartment')
        prep = Preprocessor()
        X = prep.fit_transform(df)
        assert X['property_type_Apartment'].iloc[0] == 1.0


class TestNoLeakage:

    def test_medians_from_train_applied_to_test(self):
        train = make_minimal_row(bathrooms=2.0)
        test = make_minimal_row(bathrooms=np.nan)
        test = test.drop(columns=['price'])
        prep = Preprocessor()
        prep.fit_transform(train)
        X_test = prep.transform(test)
        # Test NaN should be filled with train median (2.0), not test median
        assert X_test['bathrooms'].iloc[0] == 2.0

    def test_unseen_category_in_test_handled(self):
        train = make_minimal_row(room_type='Entire home/apt')
        test = make_minimal_row(room_type='Hotel room')  # unseen category
        test = test.drop(columns=['price'])
        prep = Preprocessor()
        prep.fit_transform(train)
        X_test = prep.transform(test)
        # handle_unknown='ignore' → all room_type OHE cols should be 0
        room_cols = [c for c in X_test.columns if c.startswith('room_type_')]
        assert X_test[room_cols].iloc[0].sum() == 0.0
