"""Test the model on known dataset samples to verify predictions."""
import os
import pickle
import numpy as np
import librosa

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model
with open(os.path.join(BASE_DIR, "model", "emotion_model.pkl"), "rb") as f:
    model = pickle.load(f)
with open(os.path.join(BASE_DIR, "model", "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)
with open(os.path.join(BASE_DIR, "model", "label_encoder.pkl"), "rb") as f:
    encoder = pickle.load(f)

print(f"Scaler expects {scaler.n_features_in_} features")
print(f"Model classes: {list(encoder.classes_)}")

def extract_features(audio, sample_rate):
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta_mean = np.mean(mfcc_delta.T, axis=0)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    mfcc_delta2_mean = np.mean(mfcc_delta2.T, axis=0)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
    chroma_mean = np.mean(chroma.T, axis=0)
    mel = librosa.feature.melspectrogram(y=audio, sr=sample_rate)
    mel_mean = np.mean(mel.T, axis=0)
    contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)
    contrast_mean = np.mean(contrast.T, axis=0)
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(audio), sr=sample_rate)
    tonnetz_mean = np.mean(tonnetz.T, axis=0)
    zcr = librosa.feature.zero_crossing_rate(audio)
    zcr_mean = np.mean(zcr)
    rms = librosa.feature.rms(y=audio)
    rms_mean = np.mean(rms)
    bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate)
    bandwidth_mean = np.mean(bandwidth)
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)
    centroid_mean = np.mean(centroid)
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)
    rolloff_mean = np.mean(rolloff)
    features = np.hstack([
        mfcc_mean, mfcc_delta_mean, mfcc_delta2_mean,
        chroma_mean, mel_mean,
        contrast_mean, tonnetz_mean,
        zcr_mean, rms_mean,
        bandwidth_mean, centroid_mean, rolloff_mean
    ])
    return features

# Test on one file from each emotion folder
test_folders = {
    "OAF_angry": "angry",
    "OAF_Fear": "fear", 
    "OAF_happy": "happy",
    "OAF_neutral": "neutral",
    "OAF_Sad": "sad",
    "YAF_angry": "angry",
    "YAF_fear": "fear",
}

dataset_path = os.path.join(BASE_DIR, "dataset", "TESS")
print("\n--- Testing predictions on dataset samples ---")
correct = 0
total = 0
for folder, expected in test_folders.items():
    folder_path = os.path.join(dataset_path, folder)
    if not os.path.isdir(folder_path):
        print(f"SKIP: {folder} not found")
        continue
    files = [f for f in os.listdir(folder_path) if f.endswith('.wav')][:3]
    for fname in files:
        fpath = os.path.join(folder_path, fname)
        audio, sr = librosa.load(fpath, res_type='kaiser_fast')
        features = extract_features(audio, sr)
        print(f"  Feature count: {len(features)}")
        features_scaled = scaler.transform([features])
        pred = model.predict(features_scaled)
        predicted = encoder.inverse_transform(pred)[0]
        match = "OK" if predicted == expected else "WRONG"
        if predicted == expected:
            correct += 1
        total += 1
        print(f"  [{match}] {folder}/{fname}: expected={expected}, predicted={predicted}")

print(f"\nResult: {correct}/{total} correct")
