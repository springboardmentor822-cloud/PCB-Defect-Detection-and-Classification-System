import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
import os
from tensorflow.keras.models import load_model

# ---------------- Page Configuration ----------------
st.set_page_config(page_title="PCB Defect Detection", layout="centered")

st.title("🔍 PCB Defect Detection")
st.write("CNN-based Defect Detection with False Positive Suppression")

# ---------------- Sidebar Controls ----------------
st.sidebar.header("Detection Settings")

conf_threshold = st.sidebar.slider(
    "Confidence Threshold (%)",
    min_value=40,
    max_value=95,
    value=70,
    step=5
) / 100.0

st.sidebar.write(f"🔧 Threshold: {conf_threshold:.2f}")

# 🔑 Key parameter to suppress false positives
MIN_DEFECT_PATCHES = 3

# ---------------- Load Model ----------------
@st.cache_resource
def load_cnn_model():
    model_path = os.path.join("..", "models", "pcb_defect_model_v3.h5")
    return load_model(model_path)

model = load_cnn_model()

# ---------------- Image Upload ----------------
uploaded_file = st.file_uploader(
    "Upload PCB Image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:
    start_time = time.time()

    # ---------------- Read Image ----------------
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    image_np = cv2.resize(image_np, (256, 256))

    st.subheader("Original PCB Image")
    st.image(image_np, use_column_width=True)

    annotated = image_np.copy()
    raw_detections = []

    # ---------------- Parameters ----------------
    PATCH_SIZE = 64
    STRIDE = 32
    MERGE_DISTANCE = 50
    BOX_SIZE = 28

    # ---------------- Patch-based Detection ----------------
    for y in range(0, 256 - PATCH_SIZE + 1, STRIDE):
        for x in range(0, 256 - PATCH_SIZE + 1, STRIDE):

            patch = image_np[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            patch = cv2.resize(patch, (224, 224))
            patch = patch / 255.0
            patch = np.expand_dims(patch, axis=0)

            preds = model.predict(patch, verbose=0)[0]

            # Binary or multi-class safe
            if preds.shape[0] == 1:
                conf = float(preds[0])
            else:
                conf = float(np.max(preds))

            if conf > conf_threshold:
                cx = x + PATCH_SIZE // 2
                cy = y + PATCH_SIZE // 2
                raw_detections.append((cx, cy, conf))

    # ---------------- Merge Nearby Detections ----------------
    final_detections = []

    for det in raw_detections:
        cx, cy, conf = det
        merged = False

        for i, (fx, fy, fconf) in enumerate(final_detections):
            dist = np.sqrt((cx - fx)**2 + (cy - fy)**2)
            if dist < MERGE_DISTANCE:
                if conf > fconf:
                    final_detections[i] = (cx, cy, conf)
                merged = True
                break

        if not merged:
            final_detections.append(det)

    # ---------------- FALSE POSITIVE SUPPRESSION ----------------
    if len(final_detections) < MIN_DEFECT_PATCHES:
        final_detections = []   # Treat image as NON-DEFECTIVE

    # ---------------- Draw SMALL RED BOXES + CONFIDENCE ----------------
    for (cx, cy, conf) in final_detections:
        x1 = max(cx - BOX_SIZE // 2, 0)
        y1 = max(cy - BOX_SIZE // 2, 0)
        x2 = min(cx + BOX_SIZE // 2, 255)
        y2 = min(cy + BOX_SIZE // 2, 255)

        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            2
        )

        cv2.putText(
            annotated,
            f"{conf*100:.1f}%",
            (x1, y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 255),
            1,
            cv2.LINE_AA
        )

    detection_time = time.time() - start_time

    # ---------------- Display ----------------
    st.subheader("Final Detection Output")
    st.image(annotated, use_column_width=True)

    st.info(f"⏱ Detection Time: {detection_time:.3f} seconds")

    if final_detections:
        st.success(f"⚠️ Defect Detected ({len(final_detections)} regions)")
    else:
        st.success("✅ PCB is Non-Defective")