import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image
import time
from collections import Counter

st.set_page_config(page_title="PCB Defect Detection", layout="wide")


model = tf.keras.models.load_model("pcb_defect_model.h5")

class_names = [
    'Missing_hole',
    'Mouse_bite',
    'Open_circuit',
    'Short',
    'Spur',
    'Spurious_copper'
]

IMG_SIZE = 128


st.markdown("""
<style>
.big-title {
    font-size:40px !important;
    font-weight:700;
    text-align:center;
}
.metric-box {
    background-color:#111827;
    padding:15px;
    border-radius:10px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-title">🔍 PCB Defect Detection & Classification</p>', unsafe_allow_html=True)

st.write("Upload a clean PCB template and a test PCB image to detect and classify defects.")

col1, col2 = st.columns(2)

with col1:
    template_file = st.file_uploader("📤 Upload Template (Clean PCB)", type=["jpg","png","jpeg"])

with col2:
    test_file = st.file_uploader("📤 Upload Test (Defective PCB)", type=["jpg","png","jpeg"])


if template_file and test_file:

    start_time = time.time()

    template = np.array(Image.open(template_file))
    test = np.array(Image.open(test_file))

    test = cv2.resize(test, (template.shape[1], template.shape[0]))

    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    test_gray = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)

    template_gray = cv2.GaussianBlur(template_gray, (7,7), 0)
    test_gray = cv2.GaussianBlur(test_gray, (7,7), 0)


    diff = cv2.absdiff(template_gray, test_gray)

    
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5,5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.dilate(thresh, kernel, iterations=2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    output_image = test.copy()
    detected_defects = []
    confidences = []

    for cnt in contours:

        area = cv2.contourArea(cnt)
        if area < 700:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        mean_intensity = np.mean(diff[y:y+h, x:x+w])
        if mean_intensity < 15:
            continue

        roi = test[y:y+h, x:x+w]
        if roi.size == 0:
            continue

        roi_resized = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
        roi_array = np.expand_dims(roi_resized, axis=0)
        roi_array = preprocess_input(roi_array)

        prediction = model.predict(roi_array, verbose=0)
        class_index = np.argmax(prediction)
        label = class_names[class_index]
        confidence = np.max(prediction)

        if confidence < 0.5:
            continue

        detected_defects.append(label)
        confidences.append(confidence)

    
        cv2.rectangle(output_image, (x, y), (x+w, y+h),
                      (0, 0, 255), 3)

        cv2.putText(output_image,
                    f"{label} ({confidence:.2f})",
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2)

    end_time = time.time()
    processing_time = round(end_time - start_time, 2)

    st.markdown("## 📌 Detection Result")

    if detected_defects:
        st.success("Defects Detected")
    else:
        st.success("PCB is defect-free")


    st.markdown("## 🖼 Image Comparison")

    colA, colB, colC = st.columns(3)

    with colA:
        st.image(template, caption="Template Image")

    with colB:
        st.image(test, caption="Test Image")

    with colC:
        st.image(output_image, caption="Detected Output")

    st.markdown("## 📊 Inspection Statistics")

    defect_count = Counter(detected_defects)

    colS1, colS2, colS3 = st.columns(3)

    with colS1:
        st.metric("Total Defects", len(detected_defects))

    with colS2:
        if detected_defects:
            st.metric("Highest Confidence", f"{max(confidences):.2f}")
        else:
            st.metric("Highest Confidence", "0")

    with colS3:
        st.metric("Processing Time (sec)", processing_time)

    if defect_count:
        st.write("### Defect Type Count")
        for defect, count in defect_count.items():
            st.write(f"• {defect}: {count}")

    _, buffer = cv2.imencode('.jpg', output_image)

    st.download_button(
        label="⬇ Download Result Image",
        data=buffer.tobytes(),
        file_name="pcb_detection_result.jpg",
        mime="image/jpeg"
    )
