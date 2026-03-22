"""
Train / evaluate helpers: split, scale, save artefacts.

scaling is done AFTER the train/test split to prevent data leakage.
The scaler is fit on the training set only, then applied to the test set, separately.
"""

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from src.config import DATA_PROCESSED, MODELS_DIR

# columns that are NOT features
META_COLS = ["labels", "source", "path"]


def split_data(df: pd.DataFrame, test_size: float = 0.25, random_state: int = 42):
    """
    Split into train / test. Returns X_train, X_test, y_train, y_test
    (feature columns only for X, label column for y).
    """
    feature_cols = [c for c in df.columns if c not in META_COLS]

    X = df[feature_cols]
    y = df["labels"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        shuffle=True,
        random_state=random_state,
    )
    return X_train, X_test, y_train, y_test


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame, save_scaler: bool = True):
    """
    Fit a MinMaxScaler on X_train, transform both X_train and X_test.
    Optionally saves the fitted scaler to models/ for later inference.

    Returns
    -------
    X_train_scaled, X_test_scaled : pd.DataFrame
    scaler : MinMaxScaler (already fit)
    """
    scaler = MinMaxScaler()

    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )

    if save_scaler:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, MODELS_DIR / "scaler.joblib")

    return X_train_scaled, X_test_scaled, scaler


def prepare_data(df: pd.DataFrame, test_size: float = 0.25, random_state: int = 42):
    """
    Convenience wrapper: split → scale (no leakage).

    Returns
    -------
    X_train, X_test, y_train, y_test, scaler
    """
    X_train, X_test, y_train, y_test = split_data(
        df,
        test_size=test_size,
        random_state=random_state,
    )
    X_train, X_test, scaler = scale_features(X_train, X_test)
    return X_train, X_test, y_train, y_test, scaler


def save_processed(df: pd.DataFrame, filename: str = "dataset.csv") -> None:
    """Save any DataFrame to data/processed/."""
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PROCESSED / filename, index=False)
