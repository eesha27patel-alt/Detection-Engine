# -*- coding: utf-8 -*-
# ================================================================
# Comprehensive Test Script for Task 3 - Emotion Detection
# Tests model accuracy, feature pipeline, gender detection,
# and simulated upload flow using TESS dataset samples
# ================================================================

import os
import sys
import pickle
import numpy as np
import librosa
from collections import defaultdict

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -----------------------------------------
# LOAD MODEL (exactly as app.py does)
# -----------------------------------------
print("=" * 60)
print("LOADING MODEL")
print("=" * 60)

with open(os.path.join(BASE_DIR, "model", "emotion_model.pkl"), "rb") as f:
    model = pickle.load(f)
with open(os.path.join(BASE_DIR, "model", "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)
with open(os.path.join(BASE_DIR, "model", "label_encoder.pkl"), "rb") as f:
    encoder = pickle.load(f)

print(f"  Model type       : {type(model).__name__}")
print(f"  Scaler expects   : {scaler.n_features_in_} features")
print(f"  Known classes    : {list(encoder.classes_)}")

# Check if model matches train_model.py expectation
if type(model).__name__ != "RandomForestClassifier":
    print(f"  [WARNING] train_model.py trains RandomForestClassifier,")
    print(f"            but saved model is {type(model).__name__}!")
    print(f"            The model may be from an older training run.")
print()

# -----------------------------------------
# FEATURE EXTRACTION (exact copy from app.py)
# -----------------------------------------
def extract_features(audio, sample_rate):
    """Exact same feature extraction as app.py"""
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
    chroma_mean = np.mean(chroma.T, axis=0)
    mel = librosa.feature.melspectrogram(y=audio, sr=sample_rate)
    mel_mean = np.mean(mel.T, axis=0)
    features = np.hstack([mfcc_mean, chroma_mean, mel_mean])
    return features

# -----------------------------------------
# GENDER DETECTION (exact copy from app.py)
# -----------------------------------------
def detect_gender(audio, sample_rate):
    """Exact same gender detection as app.py"""
    f0, voiced_flag, voiced_probs = librosa.pyin(
        audio, fmin=50, fmax=500, sr=sample_rate
    )
    valid_f0 = f0[~np.isnan(f0)]
    if len(valid_f0) == 0:
        return "unknown"
    median_pitch = np.median(valid_f0)
    if median_pitch > 180:
        return "female"
    else:
        return "male"

# -----------------------------------------
# PREDICTION (exact copy from app.py)
# -----------------------------------------
def predict_emotion(audio_path):
    """Exact same prediction pipeline as app.py"""
    audio, sample_rate = librosa.load(audio_path, res_type='kaiser_fast')
    gender = detect_gender(audio, sample_rate)
    if gender == "male":
        return None, "male"
    features = extract_features(audio, sample_rate)
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)
    emotion = encoder.inverse_transform(prediction)[0]
    return emotion, "female"

# -----------------------------------------
# UPLOAD SIMULATION (same as app.py flow)
# -----------------------------------------
def predict_via_upload_simulation(original_path):
    """Simulates the Streamlit upload: read bytes -> write temp -> predict"""
    with open(original_path, "rb") as f:
        file_bytes = f.read()
    temp_path = os.path.join(BASE_DIR, "_test_temp_audio.wav")
    with open(temp_path, "wb") as f:
        f.write(file_bytes)
    result = predict_emotion(temp_path)
    if os.path.exists(temp_path):
        os.remove(temp_path)
    return result

# -----------------------------------------
# EMOTION MAP (same as train_model.py)
# -----------------------------------------
EMOTION_MAP = {
    "angry": "angry",
    "disgust": "disgust",
    "fear": "fear",
    "happy": "happy",
    "neutral": "neutral",
    "pleasant_surprise": "surprise",
    "pleasant_surprised": "surprise",
    "sad": "sad"
}

# -----------------------------------------
# DISCOVER ALL TEST FILES
# -----------------------------------------
dataset_path = os.path.join(BASE_DIR, "dataset", "TESS")
test_files = []  # list of (file_path, expected_emotion, speaker, folder)

for folder_name in sorted(os.listdir(dataset_path)):
    folder_path = os.path.join(dataset_path, folder_name)
    if not os.path.isdir(folder_path):
        continue

    # Determine speaker (OAF = older female, YAF = younger female)
    folder_lower = folder_name.lower()
    if folder_lower.startswith("oaf"):
        speaker = "OAF"
    elif folder_lower.startswith("yaf"):
        speaker = "YAF"
    else:
        continue  # skip non-speaker folders

    # Determine emotion
    emotion = None
    for key in EMOTION_MAP:
        if key in folder_lower:
            emotion = EMOTION_MAP[key]
            break
    if emotion is None:
        continue

    # Get up to 5 sample files per folder for testing
    wav_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.wav')])
    for fname in wav_files[:5]:
        fpath = os.path.join(folder_path, fname)
        test_files.append((fpath, emotion, speaker, folder_name))

print(f"Found {len(test_files)} test files across {len(set(t[3] for t in test_files))} folders")
print()

# ================================================================
# TEST 1: Feature Dimension Check
# ================================================================
print("=" * 60)
print("TEST 1: Feature Dimension Consistency")
print("=" * 60)

sample_path = test_files[0][0]
audio, sr = librosa.load(sample_path, res_type='kaiser_fast')
features = extract_features(audio, sr)
expected_features = scaler.n_features_in_

print(f"  Extracted features : {len(features)}")
print(f"  Scaler expects     : {expected_features}")
if len(features) == expected_features:
    print(f"  [PASS] Feature dimensions match!")
else:
    print(f"  [FAIL] Dimension mismatch! Model will crash.")
print()

# ================================================================
# TEST 2: Gender Detection on TESS (all are female speakers)
# ================================================================
print("=" * 60)
print("TEST 2: Gender Detection (TESS = all female speakers)")
print("=" * 60)

gender_results = {"female": 0, "male": 0, "unknown": 0}
gender_wrong_files = []

for fpath, emotion, speaker, folder in test_files:
    audio, sr = librosa.load(fpath, res_type='kaiser_fast')
    gender = detect_gender(audio, sr)
    gender_results[gender] += 1
    if gender != "female":
        gender_wrong_files.append((os.path.basename(fpath), folder, gender))

total_gender = sum(gender_results.values())
print(f"  Female detected : {gender_results['female']}/{total_gender}")
print(f"  Male detected   : {gender_results['male']}/{total_gender}")
print(f"  Unknown         : {gender_results['unknown']}/{total_gender}")

if gender_results["male"] > 0 or gender_results["unknown"] > 0:
    wrong_count = gender_results['male'] + gender_results['unknown']
    pct = wrong_count / total_gender * 100
    print(f"  [WARNING] {wrong_count} TESS files ({pct:.1f}%) misclassified as non-female!")
    print(f"     These files would be REJECTED by the app (no emotion predicted):")
    for fname, folder, g in gender_wrong_files[:10]:
        print(f"       {folder}/{fname} -> detected as '{g}'")
    if len(gender_wrong_files) > 10:
        print(f"       ... and {len(gender_wrong_files) - 10} more")
else:
    print(f"  [PASS] All TESS files correctly detected as female")
print()

# ================================================================
# TEST 3: Emotion Prediction Accuracy (Direct Load)
# ================================================================
print("=" * 60)
print("TEST 3: Emotion Prediction Accuracy (Direct Load)")
print("=" * 60)

correct = 0
total = 0
per_emotion = defaultdict(lambda: {"correct": 0, "total": 0, "predictions": defaultdict(int)})
per_speaker = defaultdict(lambda: {"correct": 0, "total": 0})
wrong_predictions = []

for fpath, expected, speaker, folder in test_files:
    audio, sr = librosa.load(fpath, res_type='kaiser_fast')

    # Skip gender check - test emotion directly
    features = extract_features(audio, sr)
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)
    predicted = encoder.inverse_transform(prediction)[0]

    total += 1
    per_emotion[expected]["total"] += 1
    per_emotion[expected]["predictions"][predicted] += 1
    per_speaker[speaker]["total"] += 1

    if predicted == expected:
        correct += 1
        per_emotion[expected]["correct"] += 1
        per_speaker[speaker]["correct"] += 1
    else:
        wrong_predictions.append((os.path.basename(fpath), folder, expected, predicted))

accuracy = correct / total * 100 if total > 0 else 0
print(f"\n  Overall Accuracy: {correct}/{total} = {accuracy:.1f}%")
print()

# Per-emotion breakdown
print("  Per-Emotion Breakdown:")
print(f"  {'Emotion':<12} {'Correct':>8} {'Total':>6} {'Accuracy':>10}  Confusion")
print("  " + "-" * 70)
for emotion in sorted(per_emotion.keys()):
    data = per_emotion[emotion]
    acc = data["correct"] / data["total"] * 100 if data["total"] > 0 else 0
    # Show what wrong predictions were made
    wrong_preds = {k: v for k, v in data["predictions"].items() if k != emotion}
    confusion = ", ".join(f"{k}:{v}" for k, v in wrong_preds.items()) if wrong_preds else "-"
    print(f"  {emotion:<12} {data['correct']:>8} {data['total']:>6} {acc:>9.1f}%  {confusion}")

print()

# Per-speaker breakdown
print("  Per-Speaker Breakdown:")
for speaker in sorted(per_speaker.keys()):
    data = per_speaker[speaker]
    acc = data["correct"] / data["total"] * 100 if data["total"] > 0 else 0
    print(f"    {speaker}: {data['correct']}/{data['total']} = {acc:.1f}%")

print()

# Show some wrong predictions
if wrong_predictions:
    print(f"  Sample Wrong Predictions (showing up to 15):")
    for fname, folder, expected, predicted in wrong_predictions[:15]:
        print(f"    {folder}/{fname}: expected={expected}, got={predicted}")
else:
    print("  [PASS] All predictions correct!")
print()

# ================================================================
# TEST 4: Upload Simulation (end-to-end as app.py would do)
# ================================================================
print("=" * 60)
print("TEST 4: Full Upload Simulation (end-to-end like app.py)")
print("=" * 60)

# Test a subset (2 per emotion) through the full pipeline including gender detection
upload_test_files = []
seen_emotions = defaultdict(int)
for fpath, emotion, speaker, folder in test_files:
    if seen_emotions[emotion] < 2:
        upload_test_files.append((fpath, emotion, speaker, folder))
        seen_emotions[emotion] += 1

upload_correct = 0
upload_total = 0
upload_rejected = 0

for fpath, expected, speaker, folder in upload_test_files:
    predicted_emotion, detected_gender = predict_via_upload_simulation(fpath)

    if detected_gender == "male":
        upload_rejected += 1
        status = "REJECTED-male"
        print(f"  [WARNING] [{status}] {folder}/{os.path.basename(fpath)}: expected={expected}")
    else:
        upload_total += 1
        if predicted_emotion == expected:
            upload_correct += 1
            status = "OK"
        else:
            status = "WRONG"
        print(f"  [{status}] {folder}/{os.path.basename(fpath)}: expected={expected}, got={predicted_emotion}")

print()
if upload_total > 0:
    print(f"  Upload Accuracy: {upload_correct}/{upload_total} = {upload_correct/upload_total*100:.1f}%")
print(f"  Rejected as male: {upload_rejected}")
print()

# ================================================================
# TEST 5: Confidence Analysis (probability distribution)
# ================================================================
print("=" * 60)
print("TEST 5: Prediction Confidence Analysis")
print("=" * 60)

# Check if model supports predict_proba or decision_function
if hasattr(model, 'predict_proba'):
    low_confidence_count = 0
    confidence_by_emotion = defaultdict(list)

    for fpath, expected, speaker, folder in test_files[:35]:  # Test subset
        audio, sr = librosa.load(fpath, res_type='kaiser_fast')
        features = extract_features(audio, sr)
        features_scaled = scaler.transform([features])
        proba = model.predict_proba(features_scaled)[0]
        max_conf = np.max(proba)
        predicted_idx = np.argmax(proba)
        predicted = encoder.inverse_transform([predicted_idx])[0]

        confidence_by_emotion[expected].append(max_conf)

        if max_conf < 0.5:
            low_confidence_count += 1

    print(f"  Low confidence predictions (<50%): {low_confidence_count}/{min(len(test_files), 35)}")
    print()
    print(f"  {'Emotion':<12} {'Avg Conf':>10} {'Min':>8} {'Max':>8}")
    print("  " + "-" * 42)
    for emotion in sorted(confidence_by_emotion.keys()):
        confs = confidence_by_emotion[emotion]
        print(f"  {emotion:<12} {np.mean(confs)*100:>9.1f}% {np.min(confs)*100:>7.1f}% {np.max(confs)*100:>7.1f}%")
elif hasattr(model, 'decision_function'):
    print("  Model uses decision_function (SVC). Checking margin scores...")
    margin_by_emotion = defaultdict(list)

    for fpath, expected, speaker, folder in test_files[:35]:
        audio, sr = librosa.load(fpath, res_type='kaiser_fast')
        features = extract_features(audio, sr)
        features_scaled = scaler.transform([features])
        decision = model.decision_function(features_scaled)[0]
        max_margin = np.max(decision)
        margin_by_emotion[expected].append(max_margin)

    print(f"  {'Emotion':<12} {'Avg Margin':>12} {'Min':>8} {'Max':>8}")
    print("  " + "-" * 44)
    for emotion in sorted(margin_by_emotion.keys()):
        margins = margin_by_emotion[emotion]
        print(f"  {emotion:<12} {np.mean(margins):>11.3f} {np.min(margins):>7.3f} {np.max(margins):>7.3f}")
else:
    print("  Model does not support confidence analysis - skipping")

print()

# ================================================================
# SUMMARY
# ================================================================
print("=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
feat_ok = len(features) == expected_features
gender_ok = gender_results['male'] == 0 and gender_results['unknown'] == 0

print(f"  Feature dimensions    : {'PASS' if feat_ok else 'FAIL'}")
print(f"  Gender detection      : {'PASS' if gender_ok else 'ISSUES - some female voices detected as male'}")
print(f"  Overall accuracy      : {accuracy:.1f}%", end="")
if accuracy >= 80:
    print(" (GOOD)")
elif accuracy >= 60:
    print(" (MEDIOCRE)")
else:
    print(" (POOR)")
print(f"  Upload pipeline       : {'PASS' if upload_rejected == 0 else 'ISSUES - some files rejected as male'}")
print()

if accuracy < 70 or not gender_ok:
    print("  RECOMMENDATIONS:")
    if not gender_ok:
        print("    - Gender detection threshold (180 Hz) may be too aggressive")
        print("      Some female voices with lower pitch are being rejected")
    if accuracy < 70:
        print("    - Model accuracy is low, consider retraining")
        print("    - Current model is SVC but train_model.py uses RandomForest")
        print("    - Run: python train_model.py  to retrain with RandomForest + augmentation")
    print()
