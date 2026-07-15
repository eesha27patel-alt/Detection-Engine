# ═══════════════════════════════════════════════════════════════
# Task 4 - Nationality Detection Model
# Author - Your Name
# Date - Today's Date
# Description - Detects nationality, emotion, age and dress colour
# ═══════════════════════════════════════════════════════════════

import streamlit as st
from deepface import DeepFace
from PIL import Image
import cv2
import numpy as np
import os

# ─────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────
st.set_page_config(
    page_title="Nationality Detection",
    page_icon="🌍",
    layout="wide"
)

# ─────────────────────────────────────
# DRESS COLOUR DETECTION FUNCTION
# Detects dominant colour below face region
# ─────────────────────────────────────
def get_dress_colour(image, face_region):
    h, w = image.shape[:2]

    # Get face coordinates
    fx = face_region.get('x', 0)
    fy = face_region.get('y', 0)
    fw = face_region.get('w', w//3)
    fh = face_region.get('h', h//3)

    # Crop body region below face — use wider area (3x face width) and taller (2x face height)
    body_y_start = min(fy + fh, h)
    body_y_end = min(fy + fh * 3, h)
    face_center_x = fx + fw // 2
    body_x_start = max(face_center_x - fw * 2, 0)
    body_x_end = min(face_center_x + fw * 2, w)

    body_crop = image[body_y_start:body_y_end, body_x_start:body_x_end]

    if body_crop.size == 0:
        return "unknown"

    # Convert to HSV for colour detection
    hsv = cv2.cvtColor(body_crop, cv2.COLOR_BGR2HSV)

    # Mask out skin-tone pixels (HSV: H=0-25, S=40-170, V=80-255)
    skin_lower = np.array([0, 40, 80])
    skin_upper = np.array([25, 170, 255])
    skin_mask = cv2.inRange(hsv, skin_lower, skin_upper)
    non_skin_mask = cv2.bitwise_not(skin_mask)

    colour_ranges = {
        "red":    ([0, 70, 50],    [10, 255, 255]),
        "blue":   ([100, 50, 50],  [130, 255, 255]),
        "light blue": ([85, 50, 100], [105, 255, 255]),
        "green":  ([40, 40, 50],   [80, 255, 255]),
        "white":  ([0, 0, 170],    [180, 50, 255]),
        "black":  ([0, 0, 0],      [180, 255, 50]),
        "yellow": ([20, 50, 50],   [35, 255, 255]),
        "orange": ([10, 50, 50],   [20, 255, 255]),
        "purple": ([130, 30, 50],  [160, 255, 255]),
        "pink":   ([160, 30, 50],  [180, 255, 255]),
        "grey":   ([0, 0, 50],     [180, 30, 170]),
    }

    max_pixels = 0
    detected_colour = "unknown"
    colour_pixel_counts = {}

    for colour_name, (lower, upper) in colour_ranges.items():
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        # Only count non-skin pixels
        mask = cv2.bitwise_and(mask, non_skin_mask)
        pixel_count = cv2.countNonZero(mask)
        colour_pixel_counts[colour_name] = pixel_count
        if pixel_count > max_pixels:
            max_pixels = pixel_count
            detected_colour = colour_name

    # Priority: if light blue wins but white is close (within 50%), it's likely white
    if detected_colour == "light blue" and "white" in colour_pixel_counts:
        white_count = colour_pixel_counts["white"]
        light_blue_count = colour_pixel_counts["light blue"]
        if white_count > light_blue_count * 0.5:
            detected_colour = "white"

    return detected_colour

# ─────────────────────────────────────
# NATIONALITY MAPPING FUNCTION
# Maps DeepFace race output to our categories
# ─────────────────────────────────────
def map_nationality(race):
    race = race.lower()
    if race == "indian":
        return "Indian"
    elif race in ["white", "caucasian"]:
        return "American"
    elif race in ["black", "african"]:
        return "African"
    else:
        return "Other"

# ─────────────────────────────────────
# MAIN ANALYSIS FUNCTION
# ─────────────────────────────────────
def analyse_image(image_path):
    # Read image
    image = cv2.imread(image_path)

    # Run DeepFace analysis with retinaface for better face detection
    results = DeepFace.analyze(
        img_path=image_path,
        actions=['age', 'emotion', 'race'],
        enforce_detection=False,
        detector_backend='retinaface',
        align=True
    )

    # DeepFace returns list — get first face
    if isinstance(results, list):
        result = results[0]
    else:
        result = results

    # Extract predictions
    dominant_race = result.get('dominant_race', 'unknown')
    dominant_emotion = result.get('dominant_emotion', 'unknown')
    emotion_scores = result.get('emotion', {})
    age = result.get('age', 'unknown')
    face_region = result.get('region', {})

    # ── Post-processing: fix common emotion misclassifications ──
    # DeepFace's FER2013 model has known confusion between similar emotions
    if emotion_scores:
        # Debug: print scores to terminal
        print(f"\n[DEBUG] Raw emotion scores: {emotion_scores}")
        print(f"[DEBUG] Dominant emotion (raw): {dominant_emotion}")

        fear_score = emotion_scores.get('fear', 0)
        angry_score = emotion_scores.get('angry', 0)
        disgust_score = emotion_scores.get('disgust', 0)
        sad_score = emotion_scores.get('sad', 0)
        neutral_score = emotion_scores.get('neutral', 0)
        surprise_score = emotion_scores.get('surprise', 0)
        happy_score = emotion_scores.get('happy', 0)

        # Sort all scores to find top 2
        sorted_scores = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)
        top_emotion = sorted_scores[0][0]
        second_emotion = sorted_scores[1][0] if len(sorted_scores) > 1 else None

        # Fix 1: Fear vs Angry
        if dominant_emotion == 'fear':
            if (angry_score + disgust_score) > fear_score:
                dominant_emotion = 'angry'
            elif angry_score > 0 and (fear_score - angry_score) < 15:
                dominant_emotion = 'angry'

        # Fix 2: Neutral → Sad (aggressive)
        # If neutral is dominant but sad is anywhere in top 3 and > 5%, correct to sad
        if dominant_emotion == 'neutral':
            top3_emotions = [e[0] for e in sorted_scores[:3]]
            if 'sad' in top3_emotions and sad_score > 5:
                dominant_emotion = 'sad'

        # Fix 3: If happy is not clearly dominant and sad signals exist
        if dominant_emotion == 'neutral' and sad_score > happy_score:
            dominant_emotion = 'sad'

        print(f"[DEBUG] Corrected emotion: {dominant_emotion}")

    # Map race to nationality
    nationality = map_nationality(dominant_race)

    # Get dress colour
    dress_colour = get_dress_colour(image, face_region)

    return nationality, dominant_emotion, age, dress_colour, face_region, emotion_scores

# ─────────────────────────────────────
# DRAW FACE RECTANGLE FUNCTION
# ─────────────────────────────────────
def draw_face_box(image_path, face_region, label):
    image = cv2.imread(image_path)
    x = face_region.get('x', 0)
    y = face_region.get('y', 0)
    w = face_region.get('w', 0)
    h = face_region.get('h', 0)

    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(
        image, label,
        (x, y-10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7, (0, 255, 0), 2
    )

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# ═══════════════════════════════════════════════════════════════
# STREAMLIT UI
# ═══════════════════════════════════════════════════════════════

st.title("🌍 Nationality Detection System")
st.markdown("#### Detects nationality, emotion, age and dress colour from face images")
st.divider()

# Initialize session state for results
if "results" not in st.session_state:
    st.session_state.results = None
if "result_image" not in st.session_state:
    st.session_state.result_image = None

# ─────────────────────────────────────
# UPLOAD SECTION
# ─────────────────────────────────────
st.subheader("📂 Upload Image")
uploaded_file = st.file_uploader(
    "Upload a face image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=400)
    st.success("✅ Image uploaded successfully!")
else:
    st.info("👆 Upload a face image to begin")
    # Clear old results when no file is uploaded
    st.session_state.results = None
    st.session_state.result_image = None

# ─────────────────────────────────────
# DETECT BUTTON
# ─────────────────────────────────────
st.divider()

if st.button("🔍 Analyse Nationality", use_container_width=True):
    if uploaded_file is None:
        st.warning("⚠️ Please upload an image first!")
    else:
        with st.spinner("🔄 Analysing face... please wait"):
            # Save image temporarily using getvalue() which always works
            temp_path = os.path.join(os.path.dirname(__file__), "temp_face.jpg")
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getvalue())

            # Run analysis
            nationality, emotion, age, dress_colour, face_region, emotion_scores = analyse_image(temp_path)

            # Draw face box
            label = f"{nationality}"
            result_image = draw_face_box(temp_path, face_region, label)

            # Clean up temp file
            os.remove(temp_path)

            # Store results in session state so they persist across re-runs
            st.session_state.results = {
                "nationality": nationality,
                "emotion": emotion,
                "age": age,
                "dress_colour": dress_colour,
                "emotion_scores": emotion_scores,
            }
            st.session_state.result_image = result_image

# ─────────────────────────────────────
# DISPLAY RESULTS (from session state)
# ─────────────────────────────────────
if st.session_state.results is not None:
    res = st.session_state.results
    nationality = res["nationality"]
    emotion = res["emotion"]
    age = res["age"]
    dress_colour = res["dress_colour"]
    emotion_scores = res.get("emotion_scores", {})

    # Show result image
    if st.session_state.result_image is not None:
        st.image(st.session_state.result_image, caption="Detection Result", width=400)

    # ─────────────────────────────────────
    # RESULTS SECTION
    # ─────────────────────────────────────
    st.divider()
    st.subheader("📊 Analysis Results")

    # ── INDIAN ──
    if nationality == "Indian":
        st.info("🇮🇳 Nationality detected: **Indian**")
        st.markdown("Showing: Nationality + Age + Dress Colour + Emotion")
        st.divider()
        r1, r2, r3, r4 = st.columns(4)
        with r1:
            st.metric("🌍 Nationality", "Indian")
        with r2:
            st.metric("🎭 Emotion", emotion.capitalize())
        with r3:
            st.metric("🎂 Age", f"{age} years")
        with r4:
            st.metric("👗 Dress Colour", dress_colour.capitalize())

    # ── AMERICAN ──
    elif nationality == "American":
        st.info("🇺🇸 Nationality detected: **American**")
        st.markdown("Showing: Nationality + Age + Emotion")
        st.divider()
        r1, r2, r3 = st.columns(3)
        with r1:
            st.metric("🌍 Nationality", "American")
        with r2:
            st.metric("🎭 Emotion", emotion.capitalize())
        with r3:
            st.metric("🎂 Age", f"{age} years")

    # ── AFRICAN ──
    elif nationality == "African":
        st.info("🌍 Nationality detected: **African**")
        st.markdown("Showing: Nationality + Emotion + Dress Colour")
        st.divider()
        r1, r2, r3 = st.columns(3)
        with r1:
            st.metric("🌍 Nationality", "African")
        with r2:
            st.metric("🎭 Emotion", emotion.capitalize())
        with r3:
            st.metric("👗 Dress Colour", dress_colour.capitalize())

    # ── OTHER ──
    else:
        st.info(f"🌐 Nationality detected: **Other**")
        st.markdown("Showing: Nationality + Emotion")
        st.divider()
        r1, r2 = st.columns(2)
        with r1:
            st.metric("🌍 Nationality", nationality)
        with r2:
            st.metric("🎭 Emotion", emotion.capitalize())

    # ─────────────────────────────────────
    # OUTPUT SUMMARY BOX
    # ─────────────────────────────────────
    st.divider()
    st.subheader("📋 Complete Output Summary")
    st.markdown(f"""
    | Attribute | Result |
    |---|---|
    | 🌍 Detected Nationality | **{nationality}** |
    | 🎭 Detected Emotion | **{emotion.capitalize()}** |
    | 🎂 Estimated Age | **{age} years** |
    | 👗 Dress Colour | **{dress_colour.capitalize()}** |
    """)