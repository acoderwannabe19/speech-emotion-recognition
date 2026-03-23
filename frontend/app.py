"""
Streamlit Speech Emotion Recognition frontend.

Run from the project root:
    streamlit run frontend/app.py
"""

import os
import sys
import tempfile
import warnings
from glob import glob

import joblib
import streamlit as st
from PIL import Image

warnings.filterwarnings("ignore")

# paths
FRONTEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(FRONTEND_DIR)
sys.path.insert(0, PROJECT_ROOT)

from src.features import extract_features  # noqa: E402

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
IMAGE_PATH = os.path.join(FRONTEND_DIR, "robot.png")


# sidebar: settings
st.sidebar.title("Settings")

# --- model selector ---
# auto-discover model files (ignore scaler files)
model_files = sorted(
    [
        os.path.basename(f)
        for f in glob(os.path.join(MODELS_DIR, "*.joblib"))
        if "scaler" not in os.path.basename(f).lower()
    ]
)
if not model_files:
    st.sidebar.error("No model files found in models/")
    st.stop()

selected_model = st.sidebar.selectbox("Model", model_files, index=0)

# --- scaler selector (auto-pair or manual) ---
scaler_files = sorted(
    [
        os.path.basename(f)
        for f in glob(os.path.join(MODELS_DIR, "*.joblib"))
        if "scaler" in os.path.basename(f).lower()
    ]
)
selected_scaler = st.sidebar.selectbox("Scaler", scaler_files, index=0) if scaler_files else None

# --- other settings ---
st.sidebar.markdown("---")
show_gender = st.sidebar.checkbox("Show detected gender", value=True)
show_quotes = st.sidebar.checkbox("Show inspirational quotes", value=True)


# ── load model + scaler ─────────────────────────────────────────────────────
@st.cache_resource
def load_model(model_name: str, scaler_name: str | None):
    model = joblib.load(os.path.join(MODELS_DIR, model_name))
    scaler = None
    if scaler_name:
        scaler = joblib.load(os.path.join(MODELS_DIR, scaler_name))
    return model, scaler


model, scaler = load_model(selected_model, selected_scaler)


# UI
st.title("Speech Emotion Recognition")

col1, col2 = st.columns([2, 2])

with col1:
    if os.path.exists(IMAGE_PATH):
        image = Image.open(IMAGE_PATH)
        st.image(image, caption="Your friendly emotion-detecting robot")

with col2:
    st.write("**Hi there! How are you feeling today?**")
    st.write("Upload an audio file or record your voice below")


# helpers
def predict_emotion(audio_path: str) -> str:
    """Extract features, scale, and predict."""
    features = extract_features(audio_path)
    if scaler is not None:
        features = scaler.transform(features.reshape(1, -1))
    else:
        features = features.reshape(1, -1)
    prediction = model.predict(features)[0]
    return prediction


def _parse_prediction(label: str):
    """Split 'female_angry' -> ('female', 'angry'). Plain label -> (None, label)."""
    for prefix in ("female", "male"):
        if label.startswith(prefix + "_"):
            return prefix, label[len(prefix) + 1 :]
    return None, label


EMOTION_QUOTES = {
    "angry": '*"Keep your temper — nobody else wants it."* — Evan Esar',
    "disgust": '*"Let it go, you deserve peace."*',
    "fear": '*"Courage is not the absence of fear, but the triumph over it."* — Nelson Mandela',
    "happy": '*"Happiness is not something ready-made. It comes from your own actions."* — Dalai Lama',
    "neutral": '*"Balance is not something you find, it\'s something you create."*',
    "surprise": '*"The only way to make sense out of change is to plunge into it."* — Alan Watts',
    "sad": '*"Tears are words that need to be written."* — Paulo Coelho',
}


def show_result(prediction: str):
    gender, emotion = _parse_prediction(prediction)
    with col2:
        st.write("Hmm, it seems you are…")
        if show_gender and gender:
            gender_tag = "(F)" if gender == "female" else "(M)"
            st.write(f"# {emotion.capitalize()} {gender_tag}")
            st.caption(f"Detected gender: **{gender.capitalize()}**")
        else:
            st.write(f"# {emotion.capitalize()}")
        if show_quotes:
            quote = EMOTION_QUOTES.get(emotion, "")
            if quote:
                st.write(quote)


def process_audio(audio_bytes: bytes):
    """Write bytes to a temp wav, predict, show result."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        prediction = predict_emotion(tmp_path)
        show_result(prediction)
    finally:
        os.unlink(tmp_path)


# audio input: upload then record
with col2:
    st.write("---")
    uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "ogg", "flac"])
    if uploaded_file is not None:
        st.audio(uploaded_file)
        process_audio(uploaded_file.read())

    st.write("---")
    st.write("**Or record directly:**")
    recorded_audio = st.audio_input("Record your voice")
    if recorded_audio is not None:
        process_audio(recorded_audio.read())
