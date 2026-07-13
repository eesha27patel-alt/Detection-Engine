# Task 3 - Emotion Detection through Voice
# Training Script with Data Augmentation for Real-World Robustness

import os
import numpy as np
import librosa
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

DATASET_PATH = "dataset/TESS"
MODEL_PATH = "model/emotion_model.pkl"
SCALER_PATH = "model/scaler.pkl"
ENCODER_PATH = "model/label_encoder.pkl"

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

def extract_features_from_audio(audio, sample_rate):
    """Extract features from audio array (shared between training and inference)."""
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
    chroma_mean = np.mean(chroma.T, axis=0)
    mel = librosa.feature.melspectrogram(y=audio, sr=sample_rate)
    mel_mean = np.mean(mel.T, axis=0)
    features = np.hstack([mfcc_mean, chroma_mean, mel_mean])
    return features

def augment_audio(audio, sample_rate):
    """Generate augmented versions of audio for robustness."""
    augmented = []

    # 1. Add white noise (simulates noisy mic/environment)
    for noise_level in [0.005, 0.01, 0.02]:
        noise = np.random.normal(0, noise_level, audio.shape)
        augmented.append(audio + noise)

    # 2. Pitch shift (simulates different speakers)
    for n_steps in [-2, -1, 1, 2]:
        shifted = librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=n_steps)
        augmented.append(shifted)

    # 3. Time stretch (simulates faster/slower speech)
    for rate in [0.85, 1.15]:
        stretched = librosa.effects.time_stretch(audio, rate=rate)
        augmented.append(stretched)

    # 4. Volume variation (simulates different recording levels)
    for gain in [0.5, 0.7, 1.3]:
        augmented.append(audio * gain)

    return augmented

def load_dataset():
    features_list = []
    labels_list = []
    print("Loading TESS dataset with data augmentation...")
    for folder_name in os.listdir(DATASET_PATH):
        folder_path = os.path.join(DATASET_PATH, folder_name)
        if not os.path.isdir(folder_path):
            continue
        folder_lower = folder_name.lower()
        emotion = None
        for key in EMOTION_MAP:
            if key in folder_lower:
                emotion = EMOTION_MAP[key]
                break
        if emotion is None:
            continue
        print(f"Processing folder: {folder_name} -> Emotion: {emotion}")
        for file_name in os.listdir(folder_path):
            if not file_name.endswith(".wav"):
                continue
            file_path = os.path.join(folder_path, file_name)
            try:
                audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')

                # Original features
                features = extract_features_from_audio(audio, sample_rate)
                if features is not None:
                    features_list.append(features)
                    labels_list.append(emotion)

                # Augmented features
                for aug_audio in augment_audio(audio, sample_rate):
                    aug_features = extract_features_from_audio(aug_audio, sample_rate)
                    if aug_features is not None:
                        features_list.append(aug_features)
                        labels_list.append(emotion)

            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    print(f"\nTotal samples loaded: {len(features_list)} (original + augmented)")
    return np.array(features_list), np.array(labels_list)

def train():
    X, y = load_dataset()
    if len(X) == 0:
        print("No data found! Check your TESS dataset path.")
        return
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    )

    print("\nTraining Random Forest model...")
    # Random Forest generalizes much better than SVM for varied audio
    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=encoder.classes_))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    os.makedirs("model", exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    with open(ENCODER_PATH, 'wb') as f:
        pickle.dump(encoder, f)
    print("\nModel saved successfully!")
    print(f"Emotions model can detect: {list(encoder.classes_)}")

if __name__ == "__main__":
    train()