# ═══════════════════════════════════════════════════════════════
# Task 2 - Animal Detection System
# Author - Your Name
# Date - Today's Date
# Description - Detects animals in images/video, highlights carnivores
# ═══════════════════════════════════════════════════════════════

import streamlit as st
from PIL import Image
import cv2
import numpy as np
import tempfile
from ultralytics import YOLO

# ─────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────
st.set_page_config(
    page_title="Animal Detection",
    page_icon="🐾",
    layout="wide"
)

# ─────────────────────────────────────
# LOAD YOLO MODEL
# ─────────────────────────────────────
@st.cache_resource
def load_model():
    return YOLO('yolov8x.pt')

model = load_model()

# ─────────────────────────────────────
# DEFINE CARNIVOROUS ANIMALS
# COCO dataset classes that are animals
# ─────────────────────────────────────
ANIMAL_CLASSES = [
    "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe"
]

CARNIVOROUS_ANIMALS = ["cat", "dog", "bear"]
# Note: COCO dataset doesn't have lion/tiger natively,
# cat/dog/bear are the closest carnivore classes available

# ─────────────────────────────────────
# DETECTION FUNCTION (Works for both image and video frame)
# ─────────────────────────────────────
def detect_animals(image):
    results = model(image)[0]

    carnivore_count = 0
    total_animal_count = 0
    detected_animals = []

    for box in results.boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        confidence = float(box.conf[0])

        # Skip non-animal detections
        if class_name not in ANIMAL_CLASSES:
            continue

        # Use higher threshold for carnivores (avoids false positives)
        # and lower threshold for non-carnivores (catches distant animals)
        if class_name in CARNIVOROUS_ANIMALS and confidence < 0.75:
            continue
        if class_name not in CARNIVOROUS_ANIMALS and confidence < 0.5:
            continue

        total_animal_count += 1
        detected_animals.append(class_name)

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # ── CARNIVORE → RED BOX ──
        if class_name in CARNIVOROUS_ANIMALS:
            carnivore_count += 1
            box_colour = (0, 0, 255)  # Red
            label = f"{class_name.upper()} (Carnivore)"
        # ── NON-CARNIVORE → GREEN BOX ──
        else:
            box_colour = (0, 255, 0)  # Green
            label = f"{class_name.capitalize()}"

        cv2.rectangle(image, (x1, y1), (x2, y2), box_colour, 2)
        cv2.putText(
            image, label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, box_colour, 2
        )

    return image, total_animal_count, carnivore_count, detected_animals

# ═══════════════════════════════════════════════════════════════
# STREAMLIT UI
# ═══════════════════════════════════════════════════════════════

st.title("🐾 Animal Detection System")
st.markdown("#### Detects animals and highlights carnivorous species in RED")
st.divider()

# ─────────────────────────────────────
# MODE SELECTION - IMAGE OR VIDEO
# ─────────────────────────────────────
mode = st.radio(
    "Select Detection Mode:",
    ["📷 Image Detection", "🎥 Video Detection"],
    horizontal=True
)

st.divider()

# ═══════════════════════════════════════════════════════════════
# IMAGE MODE
# ═══════════════════════════════════════════════════════════════
if mode == "📷 Image Detection":

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📂 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an animal image",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Original Image", use_container_width=True)

    with col2:
        st.subheader("🔍 Detection Output")
        if uploaded_file is None:
            st.info("Upload an image to see detection results")

    st.divider()

    if st.button("🔍 Detect Animals", use_container_width=True):
        if uploaded_file is None:
            st.warning("⚠️ Please upload an image first!")
        else:
            with st.spinner("🔄 Detecting animals..."):
                # Read image
                uploaded_file.seek(0)
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                # Run detection
                result_img, total_count, carnivore_count, animals = detect_animals(img)
                result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

            with col2:
                st.image(result_rgb, caption="Detection Result", use_container_width=True)

            # ── SUMMARY ──
            st.divider()
            st.subheader("📊 Detection Summary")

            m1, m2 = st.columns(2)
            with m1:
                st.metric("🐾 Total Animals Detected", total_count)
            with m2:
                st.metric("🔴 Carnivores Detected", carnivore_count)

            # ── POP-UP STYLE ALERT ──
            if carnivore_count > 0:
                st.error(f"⚠️ ALERT: {carnivore_count} carnivorous animal(s) detected!")
            else:
                st.success("✅ No carnivorous animals detected")

            # ── ANIMAL LIST ──
            if animals:
                st.subheader("🗂️ Detected Species")
                for animal in set(animals):
                    count = animals.count(animal)
                    tag = "🔴 Carnivore" if animal in CARNIVOROUS_ANIMALS else "🟢 Non-Carnivore"
                    st.write(f"- **{animal.capitalize()}** : {count} — {tag}")

# ═══════════════════════════════════════════════════════════════
# VIDEO MODE
# ═══════════════════════════════════════════════════════════════
else:
    st.subheader("📂 Upload Video")
    uploaded_video = st.file_uploader(
        "Choose an animal video",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_video is not None:
        st.video(uploaded_video)

        if st.button("🔍 Detect Animals in Video", use_container_width=True):
            with st.spinner("🔄 Processing video... this may take a moment"):

                # Save uploaded video temporarily
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tfile.write(uploaded_video.read())

                cap = cv2.VideoCapture(tfile.name)

                stframe = st.empty()
                max_carnivore_count = 0
                all_animals = []

                frame_count = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_count += 1
                    # Process every 5th frame for speed
                    if frame_count % 5 != 0:
                        continue

                    result_frame, total_count, carnivore_count, animals = detect_animals(frame)
                    result_rgb = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)

                    stframe.image(result_rgb, caption="Live Detection", use_container_width=True)

                    max_carnivore_count = max(max_carnivore_count, carnivore_count)
                    all_animals.extend(animals)

                cap.release()

            # ── FINAL SUMMARY AFTER VIDEO ──
            st.divider()
            st.subheader("📊 Video Detection Summary")

            if max_carnivore_count > 0:
                st.error(f"⚠️ ALERT: Up to {max_carnivore_count} carnivorous animal(s) detected in video!")
            else:
                st.success("✅ No carnivorous animals detected in video")

            if all_animals:
                st.subheader("🗂️ All Detected Species")
                for animal in set(all_animals):
                    count = all_animals.count(animal)
                    tag = "🔴 Carnivore" if animal in CARNIVOROUS_ANIMALS else "🟢 Non-Carnivore"
                    st.write(f"- **{animal.capitalize()}** : detected {count} times — {tag}")