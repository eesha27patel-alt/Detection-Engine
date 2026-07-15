# ═══════════════════════════════════════════════════════════════
# Task 5 - ASL Sign Language Detection
# Training Script (HOG features + OpenCV)
# ═══════════════════════════════════════════════════════════════

import os
import numpy as np
import cv2
import pickle
from skimage.feature import hog
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

DATASET_PATH = os.path.join("dataset", "asl_alphabet_train", "asl_alphabet_train")
MODEL_PATH = "model/sign_model.pkl"
ENCODER_PATH = "model/label_encoder.pkl"
IMG_SIZE = 64

def extract_features(image):
    """Extract HOG features from an image (BGR or grayscale).
    HOG captures shape/edge information and is robust to
    lighting changes and background variations."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))

    # HOG features - captures edge orientations (shape of hand)
    features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys'
    )
    return features

def load_dataset(max_per_class=500):
    """Load ASL alphabet dataset with a limit per class."""
    features_list = []
    labels_list = []
    print("Loading ASL Alphabet dataset with HOG features...")

    for folder_name in sorted(os.listdir(DATASET_PATH)):
        folder_path = os.path.join(DATASET_PATH, folder_name)
        if not os.path.isdir(folder_path):
            continue

        print(f"Processing: {folder_name}", end=" ")
        count = 0
        files = os.listdir(folder_path)
        np.random.shuffle(files)

        for file_name in files:
            if count >= max_per_class:
                break
            file_path = os.path.join(folder_path, file_name)
            try:
                img = cv2.imread(file_path)
                if img is None:
                    continue
                features = extract_features(img)
                features_list.append(features)
                labels_list.append(folder_name)
                count += 1
            except Exception as e:
                print(f"Error: {e}")

        print(f"-> {count} samples")

    print(f"\nTotal samples loaded: {len(features_list)}")
    print(f"Feature vector size: {len(features_list[0])}")
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

    print(f"\nTraining Random Forest model with HOG features...")
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"Number of classes: {len(encoder.classes_)}")

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=5,
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

if __name__ == "__main__":
    train()
