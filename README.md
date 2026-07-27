# 🤖 Machine Learning & Deep Learning Internship Projects

A unified collection of six machine learning and deep learning applications developed during the Python/Data Science internship. All projects are built using Python and deployed via Streamlit interactive web interfaces.

---

## 📋 Table of Contents
- [Overview](#overview)
- [Projects](#projects)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Results](#results)
- [Folder Structure](#folder-structure)

---

## Overview

This repository contains six end-to-end machine learning and deep learning projects covering:
- Computer Vision (Object Detection, Face Analysis)
- Audio Processing (Emotion Detection from Voice)
- Hand Gesture Recognition (Sign Language)
- Real-time Video Analysis (Drowsiness Detection)

Each project features a professional GUI built with Streamlit and combines pretrained deep learning models with custom trained classical ML models.

---

##  Projects

### Task 1 —  Car Colour Detection
Detects car colours in traffic images using YOLOv8 deep learning model.
- Blue cars marked with **red rectangles**
- Other cars marked with **blue rectangles**
- Counts people at traffic signal
- **Tech:** YOLOv8, OpenCV HSV, Streamlit

### Task 2 —  Animal Detection
Detects and classifies animals in images and videos.
- Carnivorous animals highlighted in **red**
- Non-carnivorous animals in **green**
- Pop-up alert with carnivore count
- **Tech:** YOLOv8 (COCO), Streamlit, OpenCV

### Task 3 —  Voice Emotion Detection
Detects emotions from female voice recordings.
- Trained SVM classifier on TESS dataset
- Detects 7 emotions: angry, disgust, fear, happy, neutral, sad, surprise
- Rejects male voices automatically
- **Accuracy: 97.50%**
- **Tech:** Librosa, SVM, scikit-learn, Streamlit

### Task 4 —  Nationality Detection
Predicts nationality, emotion, age and dress colour from face images.
- Indian → Age + Dress Colour + Emotion
- American → Age + Emotion
- African → Emotion + Dress Colour
- Other → Nationality + Emotion
- **Tech:** DeepFace, OpenCV HSV, Streamlit

### Task 5 —  Sign Language Detection
Recognises ASL alphabet letters from images and real-time video.
- Trained Random Forest on MediaPipe hand landmarks
- Operational only between **6PM to 10PM**
- **Accuracy: 90.79%**
- **Tech:** MediaPipe HandLandmarker, Random Forest, Streamlit

### Task 6 —  Drowsiness Detection
Detects sleeping and awake persons in images and videos.
- Sleeping persons marked with **red rectangles**
- Predicts age of sleeping persons
- Pop-up alert with sleeping count and ages
- **Tech:** YuNet (OpenCV), DeepFace, Streamlit

---

##  Tech Stack

| Category | Tools |
|---|---|
| **Deep Learning** | YOLOv8, DeepFace, MediaPipe, YuNet |
| **Classical ML** | SVM, Random Forest (scikit-learn) |
| **Computer Vision** | OpenCV, Pillow |
| **Audio Processing** | Librosa, soundfile, resampy |
| **GUI Framework** | Streamlit |
| **Language** | Python 3.8+ |

---

## Installation

**Step 1 — Clone the repository:**
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

**Step 2 — Create virtual environment:**
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

**Step 3 — Install dependencies:**
```bash
pip install streamlit ultralytics opencv-python Pillow numpy
pip install deepface mediapipe librosa soundfile resampy
pip install scikit-learn tf-keras
```

---

##  How to Run

### Task 1 — Car Colour Detection
```bash
cd task_1
streamlit run app.py
```

### Task 2 — Animal Detection
```bash
cd task_2
streamlit run app.py
```

### Task 3 — Voice Emotion Detection
```bash
# Train model first
cd task_3
python train_model.py

# Then run app
streamlit run app.py
```

### Task 4 — Nationality Detection
```bash
cd task_4
streamlit run app.py
```

### Task 5 — Sign Language Detection
```bash
# Train model first
cd task_5
python train_model.py

# Then run app
streamlit run app.py
```

### Task 6 — Drowsiness Detection
```bash
cd task_6
streamlit run app.py
```

>  Task 5 only works between **6PM and 10PM**

---

##  Results

| Task                    | Model                     | Accuracy |

| Car Colour Detection    | YOLOv8 pretrained         | —        |
| Animal Detection        | YOLOv8 pretrained         | —        |
| Voice Emotion Detection | SVM (TESS dataset)        | **97.50%** |
| Nationality Detection   | DeepFace pretrained       | — |
| Sign Language Detection | Random Forest + MediaPipe | **90.79%** |
| Drowsiness Detection    | YuNet + DeepFace          | — |

---

##  Folder Structure

```
 EMOTION/
│
├──  task_1/                  ← Car Colour Detection
│   └──  app.py
│
├──  task_2/                  ← Animal Detection
│   └──  app.py
│
├──  task_3/                  ← Voice Emotion Detection
│   ├──  app.py
│   ├──  train_model.py
│   └──  model/
│       ├── emotion_model.pkl
│       ├── scaler.pkl
│       └── label_encoder.pkl
│
├──  task_4/                  ← Nationality Detection
│   └──  app.py
│
├── task_5/                  ← Sign Language Detection
│   ├──  app.py
│   ├──  train_model.py
│   └──  model/
│       ├── sign_model.pkl
│       └── label_encoder.pkl
│
└── task_6/                  ← Drowsiness Detection
    └──  app.py
```

---

##  Notes

- Datasets (TESS, ASL Alphabet) are not included in this repository due to size
- Download TESS dataset from [Kaggle TESS](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess)
- Download ASL dataset from [Kaggle ASL](https://www.kaggle.com/datasets/ayuraj/asl-dataset)
- Place datasets in their respective `task_X/dataset/` folders before training

---

## 👤 Author

**[EESHA PATEL]**
Data Science Internship — 2026

