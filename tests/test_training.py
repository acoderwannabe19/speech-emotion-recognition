"""Tests for src/training.py: split, scale, and data preparation."""

import numpy as np
import pandas as pd
import pytest

from src.training import META_COLS, prepare_data, scale_features, split_data


@pytest.fixture
def sample_df():
    """Create a small synthetic dataset that mimics the real one."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame(
        {
            "labels": np.random.choice(
                ["male_angry", "female_happy", "male_sad", "female_neutral"],
                size=n,
            ),
            "source": np.random.choice(["RAVDESS", "TESS", "CREMA"], size=n),
            "path": [f"/fake/path/{i}.wav" for i in range(n)],
            "mfcc0_mean": np.random.randn(n),
            "mfcc0_std": np.random.rand(n),
            "mfcc1_mean": np.random.randn(n),
            "mfcc1_std": np.random.rand(n),
        }
    )
    return df


class TestSplitData:
    def test_returns_four_elements(self, sample_df):
        result = split_data(sample_df)
        assert len(result) == 4

    def test_no_meta_cols_in_X(self, sample_df):
        X_train, X_test, _, _ = split_data(sample_df)
        for col in META_COLS:
            assert col not in X_train.columns
            assert col not in X_test.columns

    def test_split_sizes(self, sample_df):
        X_train, X_test, y_train, y_test = split_data(sample_df, test_size=0.25)
        assert len(X_train) == 75
        assert len(X_test) == 25
        assert len(y_train) == 75
        assert len(y_test) == 25

    def test_y_are_labels(self, sample_df):
        _, _, y_train, y_test = split_data(sample_df)
        all_labels = set(y_train) | set(y_test)
        assert all_labels.issubset(set(sample_df["labels"]))

    def test_no_data_leakage_indices(self, sample_df):
        X_train, X_test, _, _ = split_data(sample_df)
        assert len(set(X_train.index) & set(X_test.index)) == 0


class TestScaleFeatures:
    def test_returns_three_elements(self, sample_df):
        X_train, X_test, _, _ = split_data(sample_df)
        result = scale_features(X_train, X_test, save_scaler=False)
        assert len(result) == 3

    def test_scaled_values_in_range(self, sample_df):
        X_train, X_test, _, _ = split_data(sample_df)
        X_train_s, _, _ = scale_features(X_train, X_test, save_scaler=False)
        # Train should be exactly [0, 1]
        assert X_train_s.min().min() >= -1e-10
        assert X_train_s.max().max() <= 1.0 + 1e-10

    def test_preserves_columns(self, sample_df):
        X_train, X_test, _, _ = split_data(sample_df)
        X_train_s, X_test_s, _ = scale_features(X_train, X_test, save_scaler=False)
        assert list(X_train_s.columns) == list(X_train.columns)
        assert list(X_test_s.columns) == list(X_test.columns)

    def test_preserves_index(self, sample_df):
        X_train, X_test, _, _ = split_data(sample_df)
        X_train_s, X_test_s, _ = scale_features(X_train, X_test, save_scaler=False)
        assert list(X_train_s.index) == list(X_train.index)
        assert list(X_test_s.index) == list(X_test.index)

    def test_scaler_fit_on_train_only(self, sample_df):
        """Verify the scaler statistics come from training data only (no leakage)."""
        X_train, X_test, _, _ = split_data(sample_df)
        _, _, scaler = scale_features(X_train, X_test, save_scaler=False)
        # Scaler's data_min_ should match X_train's min
        np.testing.assert_array_almost_equal(scaler.data_min_, X_train.min().values)
        np.testing.assert_array_almost_equal(scaler.data_max_, X_train.max().values)


class TestPrepareData:
    def test_returns_five_elements(self, sample_df):
        result = prepare_data(sample_df)
        assert len(result) == 5

    def test_x_train_is_dataframe(self, sample_df):
        X_train, _, _, _, _ = prepare_data(sample_df)
        assert isinstance(X_train, pd.DataFrame)

    def test_y_train_is_series(self, sample_df):
        _, _, y_train, _, _ = prepare_data(sample_df)
        assert isinstance(y_train, pd.Series)
