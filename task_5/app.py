# ═══════════════════════════════════════════════════════════════
# Task 5 - ASL Sign Language Detection
# Streamlit App with Live Camera Detection (HOG + OpenCV)
# ═══════════════════════════════════════════════════════════════

import streamlit as st
import numpy as np
import cv2
import pickle
import os
import time
from skimage.feature import hog

# ─────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────
st.set_page_config(
    page_title="ASL Sign Language Detection",
    page_icon="🤟",
    layout="wide"
)

# ─────────────────────────────────────
# LOAD TRAINED MODEL
# ─────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_SIZE = 64

@st.cache_resource
def load_model():
    with open(os.path.join(BASE_DIR, "model", "sign_model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(BASE_DIR, "model", "label_encoder.pkl"), "rb") as f:
        encoder = pickle.load(f)
    return model, encoder

model, encoder = load_model()

# ─────────────────────────────────────
# FEATURE EXTRACTION (HOG)
# ─────────────────────────────────────
def extract_features(image):
    """Extract HOG features - captures shape/edges, robust to background."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys'
    )
    return features

# ─────────────────────────────────────
# PREDICTION FUNCTION
# ─────────────────────────────────────
def predict_sign_from_frame(frame):
    """Predict ASL sign from an OpenCV frame."""
    features = extract_features(frame)
    prediction = model.predict([features])
    sign = encoder.inverse_transform(prediction)[0]
    return sign

# ─────────────────────────────────────
# SHOW RESULT
# ─────────────────────────────────────
def show_result(sign):
    st.success(f"✅ Sign Detected!")
    st.divider()
    st.subheader("Detected Sign:")
    st.markdown(
        f"<h1 style='text-align:center; color:#4CAF50; font-size:80px;'>{sign}</h1>",
        unsafe_allow_html=True
    )

# ═══════════════════════════════════════════════════════════════
# STREAMLIT UI
# ═══════════════════════════════════════════════════════════════

st.title("🤟 ASL Sign Language Detection")
st.markdown("#### Detects American Sign Language alphabet signs")
st.divider()

# ─────────────────────────────────────
# MODE SELECTION
# ─────────────────────────────────────
mode = st.radio(
    "Select Mode:",
    ["📂 Upload Image", "📷 Live Camera Detection"],
    horizontal=True
)

st.divider()

# ═══════════════════════════════════════════════════════════════
# UPLOAD MODE
# ═══════════════════════════════════════════════════════════════
if mode == "📂 Upload Image":
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📂 Upload Sign Image")
        uploaded_file = st.file_uploader(
            "Upload an image of an ASL sign",
            type=["jpg", "jpeg", "png", "bmp"]
        )
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.getvalue()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            st.image(image_rgb, caption="Uploaded Image", width="stretch")
            st.success("✅ Image uploaded!")

    with col2:
        st.subheader("🔍 Detection Result")
        if uploaded_file is None:
            st.info("Upload an image to detect the sign")

    st.divider()

    if st.button("🔍 Detect Sign", use_container_width=True):
        if uploaded_file is None:
            st.warning("⚠️ Please upload an image first!")
        else:
            with st.spinner("🔄 Analysing sign..."):
                file_bytes = np.asarray(bytearray(uploaded_file.getvalue()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                sign = predict_sign_from_frame(image)
            with col2:
                show_result(sign)

# ═══════════════════════════════════════════════════════════════
# LIVE CAMERA MODE
# ═══════════════════════════════════════════════════════════════
else:
    st.subheader("📷 Live Camera Detection")
    st.markdown(
        "Show your ASL hand sign to the camera. "
        "The model will **continuously detect** the sign in real-time."
    )
    st.info("💡 **Tip:** Use a plain/light background behind your hand for best results.")

    col1, col2 = st.columns([2, 1])

    with col2:
        st.subheader("🔍 Detected Sign")
        sign_placeholder = st.empty()
        sign_placeholder.info("Start camera to detect signs")
        st.divider()
        history_header = st.empty()
        history_area = st.empty()

    with col1:
        run_camera = st.toggle("🎥 Start Camera", value=False)
        frame_placeholder = st.empty()

    if run_camera:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("❌ Cannot access camera! Please check your camera connection.")
        else:
            sign_history = []
            last_sign = ""
            frame_count = 0

            history_header.markdown("**📝 Sign History:**")

            while run_camera:
                ret, frame = cap.read()
                if not ret:
                    st.warning("⚠️ Failed to read from camera")
                    break

                frame_count += 1

                # Draw ROI box
                h, w = frame.shape[:2]
                roi_size = min(h, w) - 40
                roi_x = (w - roi_size) // 2
                roi_y = (h - roi_size) // 2
                cv2.rectangle(frame, (roi_x, roi_y),
                              (roi_x + roi_size, roi_y + roi_size),
                              (0, 255, 0), 3)
                cv2.putText(frame, "Show sign here", (roi_x, roi_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                # Predict every 5 frames
                if frame_count % 5 == 0:
                    roi = frame[roi_y:roi_y + roi_size, roi_x:roi_x + roi_size]
                    sign = predict_sign_from_frame(roi)

                    if sign != "nothing":
                        sign_placeholder.markdown(
                            f"<h1 style='text-align:center; color:#4CAF50; "
                            f"font-size:100px;'>{sign}</h1>",
                            unsafe_allow_html=True
                        )
                        if sign != last_sign:
                            sign_history.append(sign)
                            last_sign = sign
                            if len(sign_history) > 20:
                                sign_history = sign_history[-20:]
                            history_area.markdown(
                                f"` {'  '.join(sign_history)} `"
                            )
                    else:
                        sign_placeholder.info("👋 Show a sign to the camera")

                    cv2.putText(frame, f"Sign: {sign}", (10, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame_rgb, channels="RGB",
                                        width="stretch")
                time.sleep(0.03)

            cap.release()
    else:
        frame_placeholder.info("👆 Toggle the switch above to start the camera")
