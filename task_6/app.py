# Task 6 - Drowsiness Detection Model
# Uses OpenCV YuNet (face+eye landmarks) + DeepFace (age)

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
import urllib.request
from deepface import DeepFace

# PAGE CONFIGURATION
st.set_page_config(
    page_title="Drowsiness Detection",
    page_icon="",
    layout="wide"
)

# SETUP
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
YUNET_MODEL = os.path.join(BASE_DIR, "face_detection_yunet_2023mar.onnx")

if not os.path.exists(YUNET_MODEL):
    with st.spinner("⏬ Downloading face detection model..."):
        urllib.request.urlretrieve(
            "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
            YUNET_MODEL
        )

# GET AGE using DeepFace
def get_age(face_crop):
    try:
        if face_crop.size == 0 or face_crop.shape[0] < 20 or face_crop.shape[1] < 20:
            return "unknown"
        result = DeepFace.analyze(
            img_path=face_crop, actions=['age'],
            enforce_detection=False, silent=True
        )
        if isinstance(result, list):
            return result[0].get('age', 'unknown')
        return result.get('age', 'unknown')
    except Exception:
        return "unknown"

# Haar cascade for eye detection (reliable open/closed check)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye.xml'
)

# MAIN DETECTION FUNCTION
def detect_drowsiness(image):
    h, w = image.shape[:2]

    # YuNet face detector (0.7 confidence to reduce false positives)
    detector = cv2.FaceDetectorYN.create(YUNET_MODEL, "", (w, h), 0.7, 0.3, 5000)
    _, faces = detector.detect(image)

    total_people = 0
    sleeping_people = 0
    sleeping_ages = []

    if faces is not None:
        for face in faces:
            # Bounding box
            fx, fy, fw, fh = int(face[0]), int(face[1]), int(face[2]), int(face[3])

            # Skip tiny detections (less than 5% of image = likely false positive)
            min_face = int(min(w, h) * 0.05)
            if fw < min_face or fh < min_face:
                continue

            # Expand bounding box by 40% for better visualization
            expand_w = int(fw * 0.4)
            expand_h = int(fh * 0.4)
            x1 = max(0, fx - expand_w)
            y1 = max(0, fy - expand_h)
            x2 = min(w, fx + fw + expand_w)
            y2 = min(h, fy + fh + expand_h)

            if x2 <= x1 or y2 <= y1:
                continue

            total_people += 1
            face_color = image[y1:y2, x1:x2]
            face_gray = cv2.cvtColor(face_color, cv2.COLOR_BGR2GRAY)

            # Only search upper 50% of face for eyes (avoid detecting mouth/nose as eyes)
            face_h = face_gray.shape[0]
            upper_face = face_gray[0:int(face_h * 0.5), :]

            # Detect eyes using Haar cascade (strict: minNeighbors=5, require 2 eyes)
            eyes = eye_cascade.detectMultiScale(
                upper_face,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(20, 20)
            )

            # Draw eye landmark points from YuNet
            right_eye = (face[4], face[5])
            left_eye = (face[6], face[7])
            for (ex, ey) in [right_eye, left_eye]:
                cv2.circle(image, (int(ex), int(ey)), 3, (255, 255, 0), -1)

            #  SLEEPING (no eyes detected = eyes closed) 
            if len(eyes) == 0:
                sleeping_people += 1
                age = get_age(face_color.copy())
                sleeping_ages.append(age)

                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(
                    image, f"SLEEPING | Age: {age}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2
                )

            #  AWAKE (eyes detected = eyes open) 
            else:
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(
                    image, f"AWAKE ({len(eyes)} eyes)",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2
                )

    # Summary text
    cv2.putText(
        image,
        f"Total: {total_people} | Sleeping: {sleeping_people}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0, (0, 255, 255), 2
    )

    return image, total_people, sleeping_people, sleeping_ages

# STREAMLIT UI

st.title(" Drowsiness Detection System")
st.markdown("#### Detects sleeping and awake people in images and videos")
st.divider()

mode = st.radio(
    "Select Detection Mode:",
    [" Image Detection", " Video Detection"],
    horizontal=True
)

st.divider()

# IMAGE MODE
if mode == " Image Detection":

    col1, col2 = st.columns(2)

    with col1:
        st.subheader(" Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Original Image", use_container_width=True)
            st.success(" Image uploaded successfully!")
        else:
            st.info(" Upload an image to begin")

    with col2:
        st.subheader(" Detection Output")
        if uploaded_file is None:
            st.info("Detection result will appear here")

    st.divider()

    if st.button(" Detect Drowsiness", use_container_width=True):
        if uploaded_file is None:
            st.warning(" Please upload an image first!")
        else:
            with st.spinner(" Detecting drowsiness..."):
                file_bytes = np.asarray(
                    bytearray(uploaded_file.getvalue()),
                    dtype=np.uint8
                )
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                result_img, total, sleeping, ages = detect_drowsiness(img)
                result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

            with col2:
                st.image(
                    result_rgb,
                    caption="Detection Result",
                    use_container_width=True
                )

            #  SUMMARY 
            st.divider()
            st.subheader(" Detection Summary")

            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric(" Total People", total)
            with m2:
                st.metric(" Sleeping", sleeping)
            with m3:
                st.metric(" Awake", total - sleeping)

            #  POP-UP ALERT 
            if sleeping > 0:
                age_str = ", ".join([str(a) for a in ages])
                st.error(
                    f" ALERT: {sleeping} person(s) detected sleeping!\n"
                    f"Ages of sleeping persons: {age_str}"
                )
                st.warning(" Please wake up the sleeping person(s) immediately!")
            else:
                st.success(" No sleeping persons detected — everyone is awake!")

            #  LEGEND 
            st.divider()
            st.subheader(" Legend")
            l1, l2 = st.columns(2)
            with l1:
                st.error(" Red Rectangle = Sleeping Person")
            with l2:
                st.success("🟢 Green Rectangle = Awake Person")

# VIDEO MODE
else:
    st.subheader(" Upload Video")
    uploaded_video = st.file_uploader(
        "Choose a video file",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_video is not None:
        st.video(uploaded_video)
        st.success(" Video uploaded successfully!")

    st.divider()

    if st.button(" Detect Drowsiness in Video", use_container_width=True):
        if uploaded_video is None:
            st.warning(" Please upload a video first!")
        else:
            with st.spinner(" Processing video... please wait"):
                tfile = tempfile.NamedTemporaryFile(
                    delete=False, suffix=".mp4"
                )
                tfile.write(uploaded_video.getvalue())
                tfile.close()

                cap = cv2.VideoCapture(tfile.name)
                stframe = st.empty()
                max_sleeping = 0
                max_total = 0
                all_sleeping_ages = []
                frame_count = 0

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_count += 1
                    if frame_count % 5 != 0:
                        continue

                    result_frame, total, sleeping, ages = detect_drowsiness(frame)
                    result_rgb = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
                    stframe.image(result_rgb, caption=f"Frame {frame_count}",
                                  use_container_width=True)

                    max_sleeping = max(max_sleeping, sleeping)
                    max_total = max(max_total, total)
                    all_sleeping_ages.extend(ages)

                cap.release()
                os.unlink(tfile.name)

            st.divider()
            st.subheader(" Video Detection Summary")

            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric(" Max People Detected", max_total)
            with m2:
                st.metric(" Max Sleeping", max_sleeping)
            with m3:
                st.metric(" Frames Processed", frame_count // 5)

            if max_sleeping > 0:
                unique_ages = list(set([str(a) for a in all_sleeping_ages]))
                age_str = ", ".join(unique_ages)
                st.error(
                    f" ALERT: Up to {max_sleeping} person(s) detected sleeping!\n"
                    f"Detected ages: {age_str}"
                )
                st.warning(" Drowsiness detected — immediate attention required!")
            else:
                st.success(" No drowsiness detected in video!")