# ═══════════════════════════════════════════════════════════════
# Task 1 - Car Colour Detection System
# Author - Your Name
# Date - Today's Date
# Description - Detects car colours and counts people at traffic signal
# ═══════════════════════════════════════════════════════════════

import streamlit as st
from PIL import Image
import cv2
import numpy as np
from ultralytics import YOLO

# ─────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────
st.set_page_config(
    page_title="Car Colour Detection",
    page_icon="🚗",
    layout="wide"
)

# ─────────────────────────────────────
# CUSTOM CSS FOR BETTER LOOKS
# ─────────────────────────────────────
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stButton>button {
        background-color: #4e9af1;
        color: white;
        font-size: 18px;
        border-radius: 10px;
        padding: 10px;
    }
    .stButton>button:hover {
        background-color: #1f6feb;
    }
    .result-box {
        background-color: #1e1e2e;
        padding: 20px;
        border-radius: 10px;
        margin-top: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────
# LOAD YOLO MODEL
# ─────────────────────────────────────
@st.cache_resource
def load_model():
    return YOLO('yolov8s.pt')  # 'small' model for better accuracy

model = load_model()

# ─────────────────────────────────────
# COLOUR DETECTION FUNCTION
# ─────────────────────────────────────
def get_car_colour(image_crop):
    # ── Crop to central 60% to avoid background, road, and window edges ──
    h, w = image_crop.shape[:2]
    margin_x = int(w * 0.2)
    margin_y = int(h * 0.2)
    # Ensure we still have a valid region
    if margin_x * 2 < w and margin_y * 2 < h:
        center_crop = image_crop[margin_y:h - margin_y, margin_x:w - margin_x]
    else:
        center_crop = image_crop

    # Convert cropped car image to HSV
    hsv = cv2.cvtColor(center_crop, cv2.COLOR_BGR2HSV)

    # Define colour ranges in HSV format
    # Red wraps around H=0/180 so we need two ranges for it
    colour_ranges = {
        "red_low":  ([0, 70, 50],     [10, 255, 255]),
        "red_high": ([170, 70, 50],   [180, 255, 255]),
        "orange":   ([11, 100, 100],  [20, 255, 255]),
        "yellow":   ([21, 70, 100],   [35, 255, 255]),
        "green":    ([36, 50, 50],    [85, 255, 255]),
        "blue":     ([100, 70, 50],   [130, 255, 255]),
        "white":    ([0, 0, 200],     [180, 25, 255]),
        "silver":   ([0, 0, 120],     [180, 30, 199]),
        "grey":     ([0, 0, 51],      [180, 40, 119]),
        "black":    ([0, 0, 0],       [180, 50, 50]),
    }

    # Find which colour has the most matching pixels
    total_pixels = hsv.shape[0] * hsv.shape[1]
    colour_counts = {}

    for colour_name, (lower, upper) in colour_ranges.items():
        lower = np.array(lower)
        upper = np.array(upper)

        # Create a mask for this colour range
        mask = cv2.inRange(hsv, lower, upper)

        # Count matching pixels
        pixel_count = cv2.countNonZero(mask)
        colour_counts[colour_name] = pixel_count

    # Merge the two red ranges
    colour_counts["red"] = colour_counts.pop("red_low", 0) + colour_counts.pop("red_high", 0)

    # Find the dominant colour
    detected_colour = "unknown"
    max_pixels = 0

    for colour_name, pixel_count in colour_counts.items():
        if pixel_count > max_pixels:
            max_pixels = pixel_count
            detected_colour = colour_name

    # Require at least 5% of pixels to match, otherwise label unknown
    if max_pixels < total_pixels * 0.05:
        detected_colour = "unknown"

    return detected_colour

# ─────────────────────────────────────
# MAIN DETECTION FUNCTION
# ─────────────────────────────────────
def detect_cars_and_people(uploaded_file):
    # Read uploaded image as bytes
    file_bytes = np.asarray(
        bytearray(uploaded_file.read()),
        dtype=np.uint8
    )
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Keep a clean copy for colour detection (before any rectangles are drawn)
    clean_image = image.copy()

    # ═══════════════════════════════════════════════════════════
    # TWO-PASS DETECTION: Separate passes for people and vehicles
    # so that NMS for people can NEVER suppress car detections
    # ═══════════════════════════════════════════════════════════

    # COCO class IDs: person=0, car=2, motorcycle=3, bus=5, truck=7
    PERSON_CLASSES = [0]
    VEHICLE_CLASSES = [2, 3, 5, 7]

    # Pass 1: Detect PEOPLE only
    people_results = model(image, conf=0.25, classes=PERSON_CLASSES, max_det=100)[0]

    # Pass 2: Detect VEHICLES only
    vehicle_results = model(image, conf=0.10, classes=VEHICLE_CLASSES, max_det=100)[0]

    # Counters
    people_count = 0
    car_count = 0
    colour_summary = []
    all_detected_classes = []

    # ── PROCESS PEOPLE ──
    for box in people_results.boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        confidence = float(box.conf[0])
        all_detected_classes.append(f"{class_name} ({confidence:.0%})")

        people_count += 1
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Draw green rectangle for people
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image,
            f"Person {confidence:.0%}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (0, 255, 0), 2
        )

    # ── PROCESS VEHICLES (YOLO) ──
    for box in vehicle_results.boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        confidence = float(box.conf[0])
        all_detected_classes.append(f"{class_name} ({confidence:.0%})")

        car_count += 1
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Crop from the CLEAN image (no rectangles drawn yet on this copy)
        car_crop = clean_image[y1:y2, x1:x2]

        # Skip if crop is empty
        if car_crop.size == 0:
            continue

        # Detect colour of this car
        colour = get_car_colour(car_crop)
        colour_summary.append(colour)

        # Blue car → RED rectangle
        if colour == "blue":
            rect_colour = (0, 0, 255)
            label = f"Blue Car - RED BOX"
        # Other cars → BLUE rectangle
        else:
            rect_colour = (255, 0, 0)
            label = f"{colour.capitalize()} Car"

        # Draw rectangle around car
        cv2.rectangle(image, (x1, y1), (x2, y2), rect_colour, 2)

        # Draw label above rectangle
        cv2.putText(
            image,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, rect_colour, 2
        )

    # ═══════════════════════════════════════════════════════════
    # FALLBACK: Contour-based car detection for cartoon/illustration images
    # If YOLO found 0 vehicles, use colour segmentation + scoring
    # ═══════════════════════════════════════════════════════════
    if car_count == 0:
        img_h, img_w = clean_image.shape[:2]
        img_area = img_h * img_w

        # Build a mask of all person bounding boxes (to exclude them)
        person_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        for box in people_results.boxes:
            px1, py1, px2, py2 = map(int, box.xyxy[0])
            expand = 15
            person_mask[max(0, py1 - expand):min(img_h, py2 + expand),
                        max(0, px1 - expand):min(img_w, px2 + expand)] = 255

        # Mask out the top 15% (sky) and bottom 10% (ground)
        sky_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        sky_mask[:int(img_h * 0.15), :] = 255
        sky_mask[int(img_h * 0.90):, :] = 255

        # Convert clean image to HSV
        hsv_full = cv2.cvtColor(clean_image, cv2.COLOR_BGR2HSV)

        # Focus on saturated vehicle colours only
        car_colour_ranges = {
            "red":    [([0, 70, 70], [10, 255, 255]), ([165, 70, 70], [180, 255, 255])],
            "blue":   [([95, 70, 50], [130, 255, 255])],
            "yellow": [([21, 80, 100], [35, 255, 255])],
            "orange": [([11, 80, 80], [20, 255, 255])],
            "pink":   [([140, 50, 50], [165, 255, 255])],
        }

        # Collect ALL candidate regions with scores
        candidates = []

        for colour_name, ranges in car_colour_ranges.items():
            combined_mask = np.zeros((img_h, img_w), dtype=np.uint8)
            for (lower, upper) in ranges:
                mask = cv2.inRange(hsv_full, np.array(lower), np.array(upper))
                combined_mask = cv2.bitwise_or(combined_mask, mask)

            # Remove sky/ground and person regions
            combined_mask = cv2.bitwise_and(combined_mask, cv2.bitwise_not(sky_mask))
            combined_mask = cv2.bitwise_and(combined_mask, cv2.bitwise_not(person_mask))

            # Morphological cleanup
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)
                area_pct = area / img_area

                # Skip very tiny regions (less than 0.5% of image)
                if area_pct < 0.005:
                    continue

                cx, cy, cw, ch = cv2.boundingRect(contour)
                bbox_area = cw * ch
                aspect_ratio = cw / ch if ch > 0 else 0
                fill_ratio = area / bbox_area if bbox_area > 0 else 0

                # Skip if wider than 60% of image or taller than 40%
                if cw > img_w * 0.60 or ch > img_h * 0.40:
                    continue

                # Skip if aspect ratio is extremely vertical (< 0.8)
                if aspect_ratio < 0.8:
                    continue

                # Skip if touching 3+ edges (definitely background)
                edges = 0
                if cx <= 3: edges += 1
                if cy <= 3: edges += 1
                if cx + cw >= img_w - 3: edges += 1
                if cy + ch >= img_h - 3: edges += 1
                if edges >= 3:
                    continue

                # Check person overlap
                car_region = person_mask[cy:cy + ch, cx:cx + cw]
                person_overlap = cv2.countNonZero(car_region) / bbox_area if bbox_area > 0 else 1
                if person_overlap > 0.5:
                    continue

                # ── SCORING: rate how "car-like" this region is ──
                score = 0.0

                # Size score: prefer regions that are 2-10% of image
                if area_pct >= 0.02:
                    score += 30
                elif area_pct >= 0.01:
                    score += 20
                elif area_pct >= 0.005:
                    score += 10

                # Aspect ratio score: prefer wider-than-tall shapes
                if 1.5 <= aspect_ratio <= 3.5:
                    score += 25
                elif 1.2 <= aspect_ratio <= 4.5:
                    score += 15
                elif 0.8 <= aspect_ratio <= 1.2:
                    score += 5

                # Fill ratio: well-filled shapes are more likely solid objects
                if fill_ratio > 0.5:
                    score += 15
                elif fill_ratio > 0.35:
                    score += 8

                # Position: prefer middle of image vertically
                center_y = (cy + ch / 2) / img_h
                if 0.30 <= center_y <= 0.70:
                    score += 15
                elif 0.20 <= center_y <= 0.80:
                    score += 8

                # Penalize edge-touching and person overlap
                score -= edges * 10
                score -= person_overlap * 20

                candidates.append({
                    'x': cx, 'y': cy, 'w': cw, 'h': ch,
                    'colour': colour_name, 'score': score,
                })

        # Sort candidates by score (best first)
        candidates.sort(key=lambda c: c['score'], reverse=True)

        # Remove overlapping candidates (keep higher-scoring ones)
        final_cars = []
        for cand in candidates:
            if cand['score'] < 30:
                continue

            # Check overlap with already accepted cars
            is_overlap = False
            for accepted in final_cars:
                ox1 = max(cand['x'], accepted['x'])
                oy1 = max(cand['y'], accepted['y'])
                ox2 = min(cand['x'] + cand['w'], accepted['x'] + accepted['w'])
                oy2 = min(cand['y'] + cand['h'], accepted['y'] + accepted['h'])
                if ox1 < ox2 and oy1 < oy2:
                    overlap = (ox2 - ox1) * (oy2 - oy1)
                    smaller_area = min(cand['w'] * cand['h'],
                                       accepted['w'] * accepted['h'])
                    if overlap / smaller_area > 0.3:
                        is_overlap = True
                        break
            if is_overlap:
                continue

            final_cars.append(cand)

        # Draw the accepted cars
        for car in final_cars:
            cx, cy, cw, ch = car['x'], car['y'], car['w'], car['h']
            colour_name = car['colour']

            car_count += 1
            colour_summary.append(colour_name)
            all_detected_classes.append(f"car-fallback ({colour_name})")

            if colour_name == "blue":
                rect_colour = (0, 0, 255)
                label = f"Blue Car - RED BOX"
            else:
                rect_colour = (255, 0, 0)
                label = f"{colour_name.capitalize()} Car"

            cv2.rectangle(image, (cx, cy), (cx + cw, cy + ch), rect_colour, 3)
            cv2.putText(image, label, (cx, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, rect_colour, 2)

    # Show people count on top of image
    if people_count > 0:
        cv2.putText(
            image,
            f"People at Signal: {people_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2, (0, 255, 255), 3
        )

    return image, car_count, people_count, colour_summary, all_detected_classes

# ═══════════════════════════════════════════════════════════════
# STREAMLIT UI
# ═══════════════════════════════════════════════════════════════

# ─────────────────────────────────────
# HEADER
# ─────────────────────────────────────
st.title("🚗 Car Colour Detection System")
st.markdown("#### Detects car colours and counts people at traffic signals")
st.divider()

# ─────────────────────────────────────
# TWO COLUMN LAYOUT
# ─────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("📂 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose a traffic image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Original Image", use_column_width=True)
        st.success("✅ Image uploaded successfully!")
    else:
        st.info("👆 Please upload a traffic image to begin")

with col2:
    st.subheader("🔍 Detection Output")
    if uploaded_file is None:
        st.info("Detection result will appear here after you click Detect")

# ─────────────────────────────────────
# DETECT BUTTON
# ─────────────────────────────────────
st.divider()

if st.button("🔍 Detect Cars and People", use_container_width=True):
    if uploaded_file is None:
        st.warning("⚠️ Please upload an image first!")
    else:
        with st.spinner("🔄 Detecting... Please wait"):
            # Reset file pointer before reading again
            uploaded_file.seek(0)

            # Run detection
            result_image, car_count, people_count, colours, all_detected_classes = detect_cars_and_people(uploaded_file)

            # Convert BGR to RGB for Streamlit display
            result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)

        # Show what was detected (debug info)
        with st.expander("🔬 Debug: All Detections", expanded=False):
            if all_detected_classes:
                st.write(", ".join(all_detected_classes))
            else:
                st.write("No objects detected")

        # Show result image on right column
        with col2:
            st.image(
                result_image_rgb,
                caption="Detection Result",
                use_column_width=True
            )

        # ─────────────────────────────────────
        # RESULTS SUMMARY
        # ─────────────────────────────────────
        st.divider()
        st.subheader("📊 Detection Summary")

        # Metric cards
        m1, m2, m3, m4 = st.columns(4)

        with m1:
            st.metric("🚗 Total Cars", car_count)
        with m2:
            st.metric("👥 People at Signal", people_count)
        with m3:
            blue_cars = colours.count("blue")
            st.metric("🔵 Blue Cars Detected", blue_cars)
        with m4:
            other_cars = car_count - blue_cars
            st.metric("🚘 Other Cars", other_cars)

        # ─────────────────────────────────────
        # COLOUR BREAKDOWN
        # ─────────────────────────────────────
        if colours:
            st.divider()
            st.subheader("🎨 Car Colour Breakdown")

            breakdown_cols = st.columns(len(set(colours)))
            for i, colour in enumerate(set(colours)):
                count = colours.count(colour)
                with breakdown_cols[i]:
                    st.metric(
                        f"{colour.capitalize()} Cars",
                        count
                    )

        # ─────────────────────────────────────
        # LEGEND
        # ─────────────────────────────────────
        st.divider()
        st.subheader("🗺️ Detection Legend")
        l1, l2, l3 = st.columns(3)
        with l1:
            st.error("🔴 Red Rectangle = Blue Car")
        with l2:
            st.info("🔵 Blue Rectangle = Other Colour Car")
        with l3:
            st.success("🟢 Green Rectangle = Person")

        # ─────────────────────────────────────
        # ALERTS
        # ─────────────────────────────────────
        st.divider()
        if people_count > 0:
            st.warning(f"⚠️ {people_count} people detected at the traffic signal!")
        else:
            st.info("✅ No people detected at the signal")

        if blue_cars > 0:
            st.error(f"🔵 {blue_cars} blue car(s) detected — marked with RED rectangle")