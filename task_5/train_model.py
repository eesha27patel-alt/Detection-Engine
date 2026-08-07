# Task 5 - ASL Sign Language Detection
# Training Script (MediaPipe Hand Landmarks + Random Forest)

import os
import numpy as np
import cv2
import pickle
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

DATASET_PATH = os.path.join("dataset", "asl_alphabet_train", "asl_alphabet_train")
MODEL_PATH = "model/sign_model.pkl"
ENCODER_PATH = "model/label_encoder.pkl"
HAND_MODEL_PATH = "model/hand_landmarker.task"

# Initialize MediaPipe HandLandmarker
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=HAND_MODEL_PATH),
    running_mode=RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.3,
    min_hand_presence_confidence=0.3
)
detector = HandLandmarker.create_from_options(options)

def extract_landmarks(image):
    """Extract 21 hand landmarks (x, y, z) using MediaPipe.
    Returns 63 features (21 landmarks * 3 coords) or None if no hand detected."""
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)

    if result.hand_landmarks:
        hand = result.hand_landmarks[0]
        landmarks = []
        for lm in hand:
            landmarks.extend([lm.x, lm.y, lm.z])
        return np.array(landmarks)
    return None

def load_dataset(max_per_class=300):
    """Load ASL alphabet dataset and extract hand landmarks."""
    features_list = []
    labels_list = []
    print("Loading ASL Alphabet dataset with MediaPipe landmarks...")

    for folder_name in sorted(os.listdir(DATASET_PATH)):
        folder_path = os.path.join(DATASET_PATH, folder_name)
        if not os.path.isdir(folder_path):
            continue

        print(f"Processing: {folder_name}", end=" ")
        count = 0
        skip_count = 0
        files = os.listdir(folder_path)
        np.random.seed(42)
        np.random.shuffle(files)

        for file_name in files:
            if count >= max_per_class:
                break
            file_path = os.path.join(folder_path, file_name)
            try:
                img = cv2.imread(file_path)
                if img is None:
                    continue
                landmarks = extract_landmarks(img)
                if landmarks is not None:
                    features_list.append(landmarks)
                    labels_list.append(folder_name)
                    count += 1
                else:
                    skip_count += 1
            except Exception as e:
                skip_count += 1

        print(f"-> {count} samples ({skip_count} skipped)")

    print(f"\nTotal samples loaded: {len(features_list)}")
    if len(features_list) > 0:
        print(f"Feature vector size: {len(features_list[0])} (21 landmarks x 3 coords)")
    return np.array(features_list), np.array(labels_list)

def train():
    X, y = load_dataset()
    if len(X) == 0:
        print("No data found! Check your dataset path.")
        return

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    )

    print(f"\nTraining Random Forest model on hand landmarks...")
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"Number of classes: {len(encoder.classes_)}")

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=encoder.classes_))

    os.makedirs("model", exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    with open(ENCODER_PATH, 'wb') as f:
        pickle.dump(encoder, f)
    print("\nModel saved successfully!")
    print(f"Signs model can detect: {list(encoder.classes_)}")

    detector.close()

if __name__ == "__main__":
    train()
