"""
Test using EXACT same code as app.py to diagnose the fear-everything issue.
Simulates what happens when a user uploads a TESS file.
"""
import os
import pickle
import numpy as np
import librosa

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model EXACTLY like app.py does
with open(os.path.join(BASE_DIR, "model", "emotion_model.pkl"), "rb") as f:
    model = pickle.load(f)
with open(os.path.join(BASE_DIR, "model", "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)
with open(os.path.join(BASE_DIR, "model", "label_encoder.pkl"), "rb") as f:
    encoder = pickle.load(f)

print(f"Scaler expects: {scaler.n_features_in_} features")
print(f"Classes: {list(encoder.classes_)}")

# EXACT same extract_features as app.py
def extract_features(audio, sample_rate):
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
    chroma_mean = np.mean(chroma.T, axis=0)
    mel = librosa.feature.melspectrogram(y=audio, sr=sample_rate)
    mel_mean = np.mean(mel.T, axis=0)
    features = np.hstack([mfcc_mean, chroma_mean, mel_mean])
    return features

# EXACT same predict_emotion as app.py (minus gender check)
def predict_emotion(audio_path):
    audio, sample_rate = librosa.load(audio_path, res_type='kaiser_fast')
    print(f"  Audio shape: {audio.shape}, SR: {sample_rate}, Duration: {len(audio)/sample_rate:.2f}s")
    features = extract_features(audio, sample_rate)
    print(f"  Features shape: {features.shape}")
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)
    emotion = encoder.inverse_transform(prediction)[0]
    return emotion

# NOW simulate what happens when a file goes through upload
# (read bytes, write to temp, predict from temp)
def predict_via_upload_simulation(original_path):
    """Simulate the Streamlit upload path: read file bytes -> write to temp -> predict"""
    # Read file bytes (like uploaded_file.getvalue())
    with open(original_path, "rb") as f:
        file_bytes = f.read()
    print(f"  File bytes length: {len(file_bytes)}")
    
    # Write to temp (like the app does)
    temp_path = os.path.join(BASE_DIR, "test_temp_audio.wav")
    with open(temp_path, "wb") as f:
        f.write(file_bytes)
    
    # Predict from temp file
    result = predict_emotion(temp_path)
    os.remove(temp_path)
    return result

# Test files
test_cases = [
    ("dataset/TESS/OAF_angry/OAF_back_angry.wav", "angry"),
    ("dataset/TESS/OAF_Fear/OAF_back_fear.wav", "fear"),
    ("dataset/TESS/OAF_happy/OAF_back_happy.wav", "happy"),
    ("dataset/TESS/OAF_neutral/OAF_back_neutral.wav", "neutral"),
    ("dataset/TESS/OAF_Sad/OAF_back_sad.wav", "sad"),
]

print("\n=== TEST 1: Direct file load (like test_model.py) ===")
for rel_path, expected in test_cases:
    full_path = os.path.join(BASE_DIR, rel_path)
    predicted = predict_emotion(full_path)
    status = "OK" if predicted == expected else "WRONG"
    print(f"[{status}] {expected} -> {predicted}")

print("\n=== TEST 2: Via upload simulation (like app.py) ===")
for rel_path, expected in test_cases:
    full_path = os.path.join(BASE_DIR, rel_path)
    predicted = predict_via_upload_simulation(full_path)
    status = "OK" if predicted == expected else "WRONG"
    print(f"[{status}] {expected} -> {predicted}")
