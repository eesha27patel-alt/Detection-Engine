# ═══════════════════════════════════════════════════════════════
# Task 3 - Emotion Detection through Voice
# Author - Your Name
# Date - Today's Date
# ═══════════════════════════════════════════════════════════════

import streamlit as st
import librosa
import numpy as np
import pickle
import os

# ─────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────
st.set_page_config(
    page_title="Voice Emotion Detection",
    page_icon="🎤",
    layout="wide"
)

# ─────────────────────────────────────
# LOAD TRAINED MODEL
# ─────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_model():
    with open(os.path.join(BASE_DIR, "model", "emotion_model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(BASE_DIR, "model", "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(BASE_DIR, "model", "label_encoder.pkl"), "rb") as f:
        encoder = pickle.load(f)
    return model, scaler, encoder

model, scaler, encoder = load_model()

# ─────────────────────────────────────
# EMOTION MAPPINGS
# ─────────────────────────────────────
EMOTION_EMOJI = {
    "angry":   "😡 Angry",
    "disgust": "🤢 Disgust",
    "fear":    "😨 Fear",
    "happy":   "😊 Happy",
    "neutral": "😐 Neutral",
    "sad":     "😢 Sad",
    "surprise":"😲 Surprise"
}

EMOTION_COLOUR = {
    "angry":   "red",
    "disgust": "orange",
    "fear":    "purple",
    "happy":   "green",
    "neutral": "blue",
    "sad":     "gray",
    "surprise":"yellow"
}

# ─────────────────────────────────────
# GENDER DETECTION FUNCTION
# ─────────────────────────────────────
def detect_gender(audio, sample_rate):
    # Use pyin for reliable fundamental frequency estimation
    f0, voiced_flag, voiced_probs = librosa.pyin(
        audio, fmin=50, fmax=500, sr=sample_rate
    )
    # Keep only voiced frames with valid pitch
    valid_f0 = f0[~np.isnan(f0)]
    if len(valid_f0) == 0:
        return "unknown"
    median_pitch = np.median(valid_f0)
    # Male fundamental frequency: ~85-180 Hz
    # Female fundamental frequency: ~165-255 Hz
    if median_pitch > 180:
        return "female"
    else:
        return "male"

# ─────────────────────────────────────
# FEATURE EXTRACTION FUNCTION
# ─────────────────────────────────────
def extract_features(audio, sample_rate):
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
    chroma_mean = np.mean(chroma.T, axis=0)
    mel = librosa.feature.melspectrogram(y=audio, sr=sample_rate)
    mel_mean = np.mean(mel.T, axis=0)
    features = np.hstack([mfcc_mean, chroma_mean, mel_mean])
    return features

# ─────────────────────────────────────
# PREDICTION FUNCTION
# ─────────────────────────────────────
def predict_emotion(audio_path):
    audio, sample_rate = librosa.load(audio_path, res_type='kaiser_fast')
    gender = detect_gender(audio, sample_rate)
    if gender == "male":
        return None, "male"
    features = extract_features(audio, sample_rate)
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)
    emotion = encoder.inverse_transform(prediction)[0]
    return emotion, "female"

# ─────────────────────────────────────
# SHOW RESULT FUNCTION
# ─────────────────────────────────────
def show_result(emotion, gender):
    if gender == "male":
        st.error("⚠️ Male voice detected!")
        st.warning("This model works exclusively with female voices. Please upload or record a female voice.")
    else:
        emotion_display = EMOTION_EMOJI.get(emotion, emotion)
        colour = EMOTION_COLOUR.get(emotion, "white")
        st.success("✅ Female voice confirmed")
        st.divider()
        st.subheader("Detected Emotion:")
        st.markdown(
            f"<h1 style='text-align:center; color:{colour}'>{emotion_display}</h1>",
            unsafe_allow_html=True
        )

# ═══════════════════════════════════════════════════════════════
# STREAMLIT UI
# ═══════════════════════════════════════════════════════════════

st.title("🎤 Voice Emotion Detection System")
st.markdown("#### Detects emotions from female voice recordings")
st.divider()

# ─────────────────────────────────────
# MODE SELECTION
# ─────────────────────────────────────
mode = st.radio(
    "Select Mode:",
    ["📂 Upload Voice File", "🎙️ Live Recording"],
    horizontal=True
)

st.divider()

# ═══════════════════════════════════════════════════════════════
# UPLOAD MODE
# ═══════════════════════════════════════════════════════════════
if mode == "📂 Upload Voice File":
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📂 Upload Voice Note")
        uploaded_file = st.file_uploader(
            "Upload a voice file",
            type=["wav", "mp3", "ogg"]
        )
        if uploaded_file is not None:
            st.audio(uploaded_file, format="audio/wav")
            st.success("✅ Voice file uploaded!")

    with col2:
        st.subheader("🔍 Detection Result")
        if uploaded_file is None:
            st.info("Upload a voice file to detect emotion")

    st.divider()

    if st.button("🔍 Detect Emotion", use_container_width=True):
        if uploaded_file is None:
            st.warning("⚠️ Please upload a voice file first!")
        else:
            with st.spinner("🔄 Analysing voice..."):
                temp_path = os.path.join(BASE_DIR, "temp_audio.wav")
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                emotion, gender = predict_emotion(temp_path)
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            with col2:
                show_result(emotion, gender)

# ═══════════════════════════════════════════════════════════════
# LIVE RECORDING MODE
# ═══════════════════════════════════════════════════════════════
else:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🎙️ Record Your Voice")
        st.info("Click the microphone button below to record")

        # Built-in Streamlit audio recorder - no extra packages needed
        audio_bytes = st.audio_input("🎙️ Record your voice here")

        if audio_bytes is not None:
            st.audio(audio_bytes, format="audio/wav")
            st.success("✅ Recording captured!")

    with col2:
        st.subheader("🔍 Detection Result")
        if audio_bytes is None:
            st.info("Record your voice to detect emotion")

    st.divider()

    if audio_bytes is not None:
        if st.button("🔍 Detect Emotion from Recording", use_container_width=True):
            with st.spinner("🔄 Analysing voice..."):
                temp_path = os.path.join(BASE_DIR, "temp_recording.wav")
                with open(temp_path, "wb") as f:
                    f.write(audio_bytes.getvalue())
                emotion, gender = predict_emotion(temp_path)
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            with col2:
                show_result(emotion, gender)