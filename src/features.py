"""
Audio feature extraction utilities.

Extracts a rich set of features from each audio file:
  - N MFCCs  x 6 statistics (mean, std, min, max, skew, kurtosis)    = Nx6
  - N delta-MFCCs x 2 statistics (mean, std)                         = Nx2
  - N delta2-MFCCs x 2 statistics (mean, std)                        = Nx2
  - 12 Chroma bins x 2 statistics (mean, std)                        = 24
  - Spectral: ZCR, RMS, centroid, bandwidth, rolloff x 2 (mean, std) = 10

With N_MFCC=40 -> 240 + 80 + 80 + 24 + 10 = 434 features
With N_MFCC=20 -> 120 + 40 + 40 + 24 + 10 = 234 features
With N_MFCC=13 -> 78 + 26 + 26 + 24 + 10 = 164 features
"""

import librosa
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from src.config import N_MFCC, OFFSET, SR


def get_feature_names(n_mfcc: int = N_MFCC) -> list[str]:
    """Return an ordered list of feature column names."""
    names = []
    for i in range(n_mfcc):
        for stat in ("mean", "std", "min", "max", "skew", "kurt"):
            names.append(f"mfcc{i}_{stat}")
    for i in range(n_mfcc):
        for stat in ("mean", "std"):
            names.append(f"delta_mfcc{i}_{stat}")
    for i in range(n_mfcc):
        for stat in ("mean", "std"):
            names.append(f"delta2_mfcc{i}_{stat}")
    for i in range(12):
        for stat in ("mean", "std"):
            names.append(f"chroma{i}_{stat}")
    for feat in ("zcr", "rms", "spectral_centroid", "spectral_bandwidth", "spectral_rolloff"):
        for stat in ("mean", "std"):
            names.append(f"{feat}_{stat}")
    return names


def extract_features(
    file_path: str, sr: int = SR, offset: float = OFFSET, n_mfcc: int = N_MFCC
) -> np.ndarray:
    """
    Load one audio file and return a 1-D feature vector.

    With N_MFCC=40: 434 values.  With N_MFCC=20: 234 values.
    """
    y, sample_rate = librosa.load(
        file_path,
        res_type="kaiser_fast",
        sr=sr,
        offset=offset,
    )

    features = []

    # MFCCs: shape (n_mfcc, T)
    mfccs = librosa.feature.mfcc(y=y, sr=sample_rate, n_mfcc=n_mfcc)
    for i in range(n_mfcc):
        coeff = mfccs[i]
        features.extend(
            [
                np.mean(coeff),
                np.std(coeff),
                np.min(coeff),
                np.max(coeff),
                float(sp_stats.skew(coeff)),
                float(sp_stats.kurtosis(coeff)),
            ]
        )

    # Delta MFCCs (velocity)
    delta = librosa.feature.delta(mfccs)
    for i in range(n_mfcc):
        features.extend([np.mean(delta[i]), np.std(delta[i])])

    # Delta-delta MFCCs (acceleration)
    delta2 = librosa.feature.delta(mfccs, order=2)
    for i in range(n_mfcc):
        features.extend([np.mean(delta2[i]), np.std(delta2[i])])

    # Chroma: shape (12, T)
    chroma = librosa.feature.chroma_stft(y=y, sr=sample_rate)
    for i in range(12):
        features.extend([np.mean(chroma[i]), np.std(chroma[i])])

    # Spectral / energy descriptors
    zcr = librosa.feature.zero_crossing_rate(y)
    features.extend([np.mean(zcr), np.std(zcr)])

    rms = librosa.feature.rms(y=y)
    features.extend([np.mean(rms), np.std(rms)])

    centroid = librosa.feature.spectral_centroid(y=y, sr=sample_rate)
    features.extend([np.mean(centroid), np.std(centroid)])

    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sample_rate)
    features.extend([np.mean(bandwidth), np.std(bandwidth)])

    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sample_rate)
    features.extend([np.mean(rolloff), np.std(rolloff)])

    return np.array(features)


extract_mfccs = extract_features  # old name still works


def extract_features_for_dataset(df: pd.DataFrame, n_mfcc: int = N_MFCC) -> pd.DataFrame:
    """
    Given a DataFrame with a ``path`` column, extract features for every row.

    Returns
    -------
    DataFrame with columns: labels, source, path, mfcc0_mean, …, spectral_rolloff_std
    """
    col_names = get_feature_names(n_mfcc)
    feature_list = []
    errors = 0

    for idx, path in enumerate(df["path"]):
        try:
            feats = extract_features(path, n_mfcc=n_mfcc)
            feature_list.append(feats)
        except Exception:
            # On error (corrupt file, too short, etc.) fill with zeros
            feature_list.append(np.zeros(len(col_names)))
            errors += 1
        if (idx + 1) % 500 == 0 or idx == len(df) - 1:
            print(f"  Extracted {idx + 1}/{len(df)}  (errors: {errors})")

    features_df = pd.DataFrame(feature_list, columns=col_names)
    features_df.fillna(0, inplace=True)

    result = pd.concat(
        [df.reset_index(drop=True), features_df],
        axis=1,
    )
    return result
