"""Tests for src/config.py: paths, mappings, and constants."""

from pathlib import Path

from src.config import (
    CREMA_DIR,
    CREMA_EMOTION_MAP,
    CREMA_FEMALE_IDS,
    DATA_PROCESSED,
    DATA_RAW,
    DURATION,
    EXCLUDED_EMOTIONS,
    MODELS_DIR,
    N_MFCC,
    OFFSET,
    PROJECT_ROOT,
    RAVDESS_DIR,
    RAVDESS_EMOTION_MAP,
    RAVDESS_GENDER_MAP,
    SAVEE_DIR,
    SAVEE_EMOTION_MAP,
    SR,
    TESS_DIR,
    TESS_EMOTION_MAP,
)


class TestPaths:
    def test_project_root_is_directory(self):
        assert PROJECT_ROOT.is_dir()

    def test_data_raw_under_project_root(self):
        assert DATA_RAW == PROJECT_ROOT / "data" / "raw"

    def test_data_processed_under_project_root(self):
        assert DATA_PROCESSED == PROJECT_ROOT / "data" / "processed"

    def test_models_dir_under_project_root(self):
        assert MODELS_DIR == PROJECT_ROOT / "models"

    def test_dataset_dirs_are_pathlib(self):
        for d in [RAVDESS_DIR, SAVEE_DIR, TESS_DIR, CREMA_DIR]:
            assert isinstance(d, Path)


class TestEmotionMappings:
    def test_ravdess_has_8_emotion_codes(self):
        assert set(RAVDESS_EMOTION_MAP.keys()) == {1, 2, 3, 4, 5, 6, 7, 8}

    def test_ravdess_calm_merged_to_neutral(self):
        assert RAVDESS_EMOTION_MAP[1] == "neutral"
        assert RAVDESS_EMOTION_MAP[2] == "neutral"

    def test_ravdess_gender_map(self):
        assert RAVDESS_GENDER_MAP[0] == "female"
        assert RAVDESS_GENDER_MAP[1] == "male"

    def test_savee_all_male(self):
        for label in SAVEE_EMOTION_MAP.values():
            assert label.startswith("male_")

    def test_tess_all_female(self):
        for label in TESS_EMOTION_MAP.values():
            assert label.startswith("female_")

    def test_crema_emotions_are_lowercase(self):
        for emotion in CREMA_EMOTION_MAP.values():
            assert emotion == emotion.lower()

    def test_crema_female_ids_is_set(self):
        assert isinstance(CREMA_FEMALE_IDS, set)
        assert len(CREMA_FEMALE_IDS) > 0


class TestExcludedEmotions:
    def test_surprise_excluded(self):
        assert "surprise" in EXCLUDED_EMOTIONS

    def test_excluded_is_set(self):
        assert isinstance(EXCLUDED_EMOTIONS, set)


class TestFeatureDefaults:
    def test_n_mfcc_positive(self):
        assert N_MFCC > 0

    def test_sr_standard(self):
        assert SR == 22050

    def test_duration_positive(self):
        assert DURATION > 0

    def test_offset_non_negative(self):
        assert OFFSET >= 0
