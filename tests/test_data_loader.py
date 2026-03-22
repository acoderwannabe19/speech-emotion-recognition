"""Tests for src/data_loader.py — dataset loading and filtering."""

import pandas as pd
import pytest

from src.config import EXCLUDED_EMOTIONS
from src.data_loader import (
    load_all_datasets,
    load_crema,
    load_ravdess,
    load_savee,
    load_tess,
)

EXPECTED_COLUMNS = ["labels", "source", "path"]


class TestLoadRavdess:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.df = load_ravdess()

    def test_returns_dataframe(self):
        assert isinstance(self.df, pd.DataFrame)

    def test_has_required_columns(self):
        assert list(self.df.columns) == EXPECTED_COLUMNS

    def test_source_is_ravdess(self):
        assert (self.df["source"] == "RAVDESS").all()

    def test_labels_have_gender_emotion_format(self):
        for label in self.df["labels"]:
            parts = label.split("_")
            assert len(parts) == 2
            assert parts[0] in ("male", "female")

    def test_no_empty_paths(self):
        assert self.df["path"].str.len().min() > 0

    def test_all_paths_are_wav(self):
        assert self.df["path"].str.endswith(".wav").all()


class TestLoadSavee:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.df = load_savee()

    def test_returns_dataframe(self):
        assert isinstance(self.df, pd.DataFrame)

    def test_has_required_columns(self):
        assert list(self.df.columns) == EXPECTED_COLUMNS

    def test_source_is_savee(self):
        assert (self.df["source"] == "SAVEE").all()

    def test_all_male(self):
        assert self.df["labels"].str.startswith("male_").all()


class TestLoadTess:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.df = load_tess()

    def test_returns_dataframe(self):
        assert isinstance(self.df, pd.DataFrame)

    def test_has_required_columns(self):
        assert list(self.df.columns) == EXPECTED_COLUMNS

    def test_source_is_tess(self):
        assert (self.df["source"] == "TESS").all()

    def test_all_female(self):
        assert self.df["labels"].str.startswith("female_").all()


class TestLoadCrema:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.df = load_crema()

    def test_returns_dataframe(self):
        assert isinstance(self.df, pd.DataFrame)

    def test_has_required_columns(self):
        assert list(self.df.columns) == EXPECTED_COLUMNS

    def test_source_is_crema(self):
        assert (self.df["source"] == "CREMA").all()

    def test_has_both_genders(self):
        genders = self.df["labels"].str.split("_").str[0].unique()
        assert set(genders) == {"male", "female"}


class TestLoadAllDatasets:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.df = load_all_datasets()

    def test_returns_dataframe(self):
        assert isinstance(self.df, pd.DataFrame)

    def test_has_required_columns(self):
        assert list(self.df.columns) == EXPECTED_COLUMNS

    def test_contains_all_sources(self):
        sources = set(self.df["source"].unique())
        assert sources == {"RAVDESS", "SAVEE", "TESS", "CREMA"}

    def test_excluded_emotions_filtered_out(self):
        emotions = self.df["labels"].str.split("_").str[1].unique()
        for excluded in EXCLUDED_EMOTIONS:
            assert excluded not in emotions

    def test_no_surprise_in_labels(self):
        assert not self.df["labels"].str.contains("surprise").any()

    def test_index_is_contiguous(self):
        assert list(self.df.index) == list(range(len(self.df)))
