"""
Load and format each raw dataset into a unified DataFrame(labels, source, path).
"""

import os

import pandas as pd

from src.config import (
    CREMA_DIR,
    CREMA_EMOTION_MAP,
    CREMA_FEMALE_IDS,
    DATA_RAW,
    EXCLUDED_EMOTIONS,
    RAVDESS_DIR,
    RAVDESS_EMOTION_MAP,
    RAVDESS_GENDER_MAP,
    SAVEE_DIR,
    SAVEE_EMOTION_MAP,
    TESS_DIR,
    TESS_EMOTION_MAP,
)


def load_ravdess() -> pd.DataFrame:
    """Parse RAVDESS filenames into (labels, source, path)."""
    rows = []
    for subdir in sorted(os.listdir(RAVDESS_DIR)):
        subdir_path = RAVDESS_DIR / subdir
        if not subdir_path.is_dir():
            continue
        for f in os.listdir(subdir_path):
            if not f.endswith(".wav"):
                continue
            parts = f.split(".")[0].split("-")
            if len(parts) < 7:
                continue
            emotion = RAVDESS_EMOTION_MAP[int(parts[2])]
            gender = RAVDESS_GENDER_MAP[int(parts[6]) % 2]
            rows.append(
                {
                    "labels": f"{gender}_{emotion}",
                    "source": "RAVDESS",
                    "path": str(subdir_path / f),
                }
            )
    return pd.DataFrame(rows)


def load_savee() -> pd.DataFrame:
    """Parse SAVEE filenames into (labels, source, path)."""
    rows = []
    for f in os.listdir(SAVEE_DIR):
        if not f.endswith(".wav"):
            continue
        label = SAVEE_EMOTION_MAP.get(f[-8:-6], "male_error")
        rows.append(
            {
                "labels": label,
                "source": "SAVEE",
                "path": str(SAVEE_DIR / f),
            }
        )
    return pd.DataFrame(rows)


def load_tess() -> pd.DataFrame:
    """Parse TESS folder/filenames into (labels, source, path)."""
    rows = []
    for folder in sorted(os.listdir(TESS_DIR)):
        folder_path = TESS_DIR / folder
        if not folder_path.is_dir():
            continue
        label = TESS_EMOTION_MAP.get(folder, "Unknown")
        for f in os.listdir(folder_path):
            if f.endswith(".wav"):
                rows.append(
                    {
                        "labels": label,
                        "source": "TESS",
                        "path": str(folder_path / f),
                    }
                )
    return pd.DataFrame(rows)


def load_crema() -> pd.DataFrame:
    """Parse CREMA-D filenames into (labels, source, path)."""
    rows = []
    for f in sorted(os.listdir(CREMA_DIR)):
        if not f.endswith(".wav"):
            continue
        parts = f.split("_")
        speaker_id = int(parts[0])
        emotion = CREMA_EMOTION_MAP.get(parts[2], "Unknown")
        gender = "female" if speaker_id in CREMA_FEMALE_IDS else "male"
        rows.append(
            {
                "labels": f"{gender}_{emotion}",
                "source": "CREMA",
                "path": str(CREMA_DIR / f),
            }
        )
    return pd.DataFrame(rows)


def load_all_datasets() -> pd.DataFrame:
    """Concatenate all four datasets, drop excluded emotions, return a single DataFrame."""
    df = pd.concat(
        [load_savee(), load_ravdess(), load_tess(), load_crema()],
        axis=0,
        ignore_index=True,
    )
    # Drop rows whose emotion is in EXCLUDED_EMOTIONS
    if EXCLUDED_EMOTIONS:
        emotion_part = df["labels"].str.split("_").str[1]
        mask = ~emotion_part.isin(EXCLUDED_EMOTIONS)
        dropped = len(df) - mask.sum()
        df = df[mask].reset_index(drop=True)
        if dropped:
            print(f"Dropped {dropped} samples with excluded emotions: {EXCLUDED_EMOTIONS}")
    return df


def save_raw_dataset(df: pd.DataFrame) -> None:
    """Persist the combined raw dataset CSV."""
    df.to_csv(DATA_RAW / "dataset.csv", index=False)
