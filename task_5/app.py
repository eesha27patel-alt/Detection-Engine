# Task 5 - ASL Sign Language Detection
# Streamlit App (Upload / Camera Capture)

import streamlit as st
import numpy as np
import cv2
import pickle
import os
from datetime import datetime
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker, HandLandmarkerOptions, RunningMode
)

# PAGE CONFIGURATION
st.set_page_config(
    page_title="ASL Sign Language Detection",
    page_icon="",
    layout="wide"
)

# TIME RESTRICTION (6 PM - 10 PM only)
current_hour = datetime.now().hour
if current_hour < 18 or current_hour >= 22:
    st.title(" ASL Sign Language Detection")
    st.divider()
    st.error(" This app is only available between **6:00 PM and 10:00 PM**.")
    st.info(f"Current time: **{datetime.now().strftime('%I:%M %p')}**. Please come back during the allowed hours.")
    st.stop()

# LOAD MODEL & HAND DETECTOR
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_model():
    with open(os.path.join(BASE_DIR, "model", "sign_model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(BASE_DIR, "model", "label_encoder.pkl"), "rb") as f:
        encoder = pickle.load(f)
    return model, encoder

@st.cache_resource
def get_hand_detector():
    options = HandLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=os.path.join(BASE_DIR, "model", "hand_landmarker.task")
        ),
        running_mode=RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5
    )
    return HandLandmarker.create_from_options(options)

model, encoder = load_model()
detector = get_hand_detector()

# PREDICTION FUNCTION
def predict_sign(image):
    """Detect hand landmarks and predict sign."""
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)

    if result.hand_landmarks:
        hand = result.hand_landmarks[0]
        landmarks = []
        for lm in hand:
            landmarks.extend([lm.x, lm.y, lm.z])
        features = np.array(landmarks).reshape(1, -1)
        prediction = model.predict(features)
        sign = encoder.inverse_transform(prediction)[0]
        return sign
    return None

# SHOW RESULT
def show_result(sign):
    if sign:
        st.success(" Sign Detected!")
        st.divider()
        st.subheader("Detected Sign:")
        st.markdown(
            f"<h1 style='text-align:center; color:#4CAF50; font-size:80px;'>{sign}</h1>",
            unsafe_allow_html=True
        )
    else:
        st.error(" No hand detected in the image! Try again with your hand clearly visible.")

# UI

st.title(" ASL Sign Language Detection")
st.markdown("#### Detects American Sign Language alphabet signs")
st.divider()

mode = st.radio(
    "Select Mode:",
    [" Upload Image", " Camera Capture"],
    horizontal=True
)

st.divider()

# UPLOAD MODE
if mode == " Upload Image":
    col1, col2 = st.columns(2)

    with col1:
        st.subheader(" Upload Sign Image")
        uploaded_file = st.file_uploader(
            "Upload an image of an ASL sign",
            type=["jpg", "jpeg", "png", "bmp"]
        )
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.getvalue()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            st.image(image_rgb, caption="Uploaded Image")
            st.success(" Image uploaded!")

    with col2:
        st.subheader(" Detection Result")
        if uploaded_file is None:
            st.info("Upload an image to detect the sign")

    st.divider()

    if st.button(" Detect Sign", use_container_width=True):
        if uploaded_file is None:
            st.warning(" Please upload an image first!")
        else:
            with st.spinner(" Analysing sign..."):
                file_bytes = np.asarray(bytearray(uploaded_file.getvalue()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                sign = predict_sign(image)
            with col2:
                show_result(sign)

# CAMERA CAPTURE MODE
else:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader(" Capture Sign")
        st.info("Show your ASL sign and click the camera button to capture")
        camera_image = st.camera_input(" Capture your sign")

        if camera_image is not None:
            st.success(" Image captured!")

    with col2:
        st.subheader(" Detection Result")
        if camera_image is None:
            st.info("Capture an image to detect the sign")

    st.divider()

    if camera_image is not None:
        if st.button(" Detect Sign", use_container_width=True):
            with st.spinner(" Analysing sign..."):
                file_bytes = np.asarray(bytearray(camera_image.getvalue()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                sign = predict_sign(image)
            with col2:
                show_result(sign)
