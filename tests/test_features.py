"""Tests for src/features.py — feature extraction and naming."""

import numpy as np
import pytest

from src.config import N_MFCC, RAVDESS_DIR
from src.features import extract_features, extract_mfccs, get_feature_names

# Use a real sample from the dataset for integration-style tests
SAMPLE_WAV = str(RAVDESS_DIR / "Actor_01" / "03-01-01-01-01-01-01.wav")


class TestGetFeatureNames:
    def test_default_returns_list(self):
        names = get_feature_names()
        assert isinstance(names, list)

    def test_default_length_matches_n_mfcc(self):
        # N x 6 (MFCC stats) + N x 2 (delta) + N x 2 (delta2) + 24 (chroma) + 10 (spectral)
        expected = N_MFCC * 6 + N_MFCC * 2 + N_MFCC * 2 + 24 + 10
        assert len(get_feature_names()) == expected

    def test_custom_n_mfcc(self):
        names = get_feature_names(n_mfcc=13)
        expected = 13 * 6 + 13 * 2 + 13 * 2 + 24 + 10
        assert len(names) == expected

    def test_names_are_strings(self):
        for name in get_feature_names():
            assert isinstance(name, str)

    def test_names_are_unique(self):
        names = get_feature_names()
        assert len(names) == len(set(names))

    def test_starts_with_mfcc(self):
        names = get_feature_names()
        assert names[0] == "mfcc0_mean"

    def test_ends_with_spectral_rolloff(self):
        names = get_feature_names()
        assert names[-1] == "spectral_rolloff_std"

    def test_contains_chroma_features(self):
        names = get_feature_names()
        chroma_names = [n for n in names if n.startswith("chroma")]
        assert len(chroma_names) == 24  # 12 bins x 2 stats

    def test_contains_delta_features(self):
        names = get_feature_names()
        delta_names = [n for n in names if n.startswith("delta_mfcc")]
        assert len(delta_names) == N_MFCC * 2


class TestExtractFeatures:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.features = extract_features(SAMPLE_WAV)

    def test_returns_numpy_array(self):
        assert isinstance(self.features, np.ndarray)

    def test_is_1d(self):
        assert self.features.ndim == 1

    def test_length_matches_feature_names(self):
        assert len(self.features) == len(get_feature_names())

    def test_no_nans(self):
        assert not np.isnan(self.features).any()

    def test_no_infs(self):
        assert not np.isinf(self.features).any()

    def test_features_are_finite(self):
        assert np.all(np.isfinite(self.features))

    def test_custom_n_mfcc(self):
        feats = extract_features(SAMPLE_WAV, n_mfcc=13)
        expected_len = 13 * 6 + 13 * 2 + 13 * 2 + 24 + 10
        assert len(feats) == expected_len


class TestBackwardAlias:
    def test_extract_mfccs_is_extract_features(self):
        assert extract_mfccs is extract_features
