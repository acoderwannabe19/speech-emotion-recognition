"""
Central configuration: paths, constants, and mappings for all datasets.
"""
from pathlib import Path

# ── Project root (one level up from src/) ──────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

print(f"Project root: {PROJECT_ROOT}")

# ── Raw data paths ─────────────────────────────────────────────────────────
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_INTERIM = PROJECT_ROOT / "data" / "interim"
MODELS_DIR = PROJECT_ROOT / "models"

RAVDESS_DIR = DATA_RAW / "ravdess-emotional-speech-audio" / "versions" / "1"
SAVEE_DIR = DATA_RAW / "surrey-audiovisual-expressed-emotion-savee" / "versions" / "1" / "ALL"
TESS_DIR = DATA_RAW / "toronto-emotional-speech-set-tess" / "versions" / "1" / "TESS Toronto emotional speech set data" / "TESS Toronto emotional speech set data"
CREMA_DIR = DATA_RAW / "cremad" / "versions" / "1" / "AudioWAV"

# ── Emotion mappings ──────────────────────────────────────────────────────
RAVDESS_EMOTION_MAP = {
    1: "neutral", 2: "neutral",   # merge calm → neutral
    3: "happy", 4: "sad",
    5: "angry", 6: "fear",
    7: "disgust", 8: "surprise",
}

RAVDESS_GENDER_MAP = {0: "female", 1: "male"}

SAVEE_EMOTION_MAP = {
    "_a": "male_angry",
    "_d": "male_disgust",
    "_f": "male_fear",
    "_h": "male_happy",
    "_n": "male_neutral",
    "sa": "male_sad",
    "su": "male_surprise",
}

TESS_EMOTION_MAP = {
    "OAF_angry": "female_angry",
    "YAF_angry": "female_angry",
    "OAF_disgust": "female_disgust",
    "YAF_disgust": "female_disgust",
    "OAF_Fear": "female_fear",
    "YAF_fear": "female_fear",
    "OAF_happy": "female_happy",
    "YAF_happy": "female_happy",
    "OAF_neutral": "female_neutral",
    "YAF_neutral": "female_neutral",
    "OAF_Pleasant_surprise": "female_surprise",
    "YAF_pleasant_surprised": "female_surprise",
    "OAF_Sad": "female_sad",
    "YAF_sad": "female_sad",
}

CREMA_EMOTION_MAP = {
    "SAD": "sad",
    "ANG": "angry",
    "DIS": "disgust",
    "FEA": "fear",
    "HAP": "happy",
    "NEU": "neutral",
}

CREMA_FEMALE_IDS = {
    1002, 1003, 1004, 1006, 1007, 1008, 1009, 1010, 1012, 1013,
    1018, 1020, 1021, 1024, 1025, 1028, 1029, 1030, 1037, 1043,
    1046, 1047, 1049, 1052, 1053, 1054, 1055, 1056, 1058, 1060,
    1061, 1063, 1072, 1073, 1074, 1075, 1076, 1078, 1079, 1082,
    1084, 1089, 1091,
}

# ── Emotions to exclude (too few samples / too imbalanced) ─────────────────
EXCLUDED_EMOTIONS = {"surprise"}

# ── Feature extraction defaults ───────────────────────────────────────────
N_MFCC = 40      
SR = 22050          # 22 050 Hz is librosa's default and sufficient for speech
DURATION = 3.0      
OFFSET = 0.5
