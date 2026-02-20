import streamlit as st
from PIL import Image
import os
import sys

# -------------------------
# FIX IMPORT PATH
# -------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from roi_extraction.backend_detect import detect_and_annotate

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="PCB Defect Detection System",
    page_icon="🟢",
    layout="wide"
)

# -------------------------
# CSS
# -------------------------
st.markdown("""
<style>
.main { background-color: #f4f6f9; }
h1 { color: #0b6623; font-weight: 800; }
.card {
    background-color: white;
    padding: 20px;
    border-radius: 14px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
}
.footer {
    text-align: center;
    color: gray;
    margin-top: 40px;
}
.stButton>button {
    background-color: #0b6623;
    color: white;
    font-size: 18px;
    padding: 12px 28px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# HEADER
# -------------------------
st.markdown("<h1>🟢 PCB Defect Detection & Classification</h1>", unsafe_allow_html=True)
st.markdown(
    "An end-to-end system demonstrating **preprocessing, localization, and classification** of PCB defects."
)

st.divider()

# -------------------------
# IMAGE UPLOAD
# -------------------------
st.markdown("### 📤 Upload PCB Images")

col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    template_file = st.file_uploader(
        "🧩 Template PCB Image (Defect-Free)",
        type=["jpg", "png", "jpeg"]
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    test_file = st.file_uploader(
        "🔍 Test PCB Image (Possibly Defective)",
        type=["jpg", "png", "jpeg"]
    )
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# MAIN PIPELINE
# -------------------------
if template_file and test_file:

    template_img = Image.open(template_file)
    test_img = Image.open(test_file)

    st.divider()
    st.markdown("### 🖼️ Uploaded Images")

    c1, c2 = st.columns(2)
    c1.image(template_img, caption="Template PCB", use_container_width=True)
    c2.image(test_img, caption="Test PCB", use_container_width=True)

    st.divider()

    defect_label = st.selectbox(
        "🏷️ Ground Truth Defect Type",
        [
            "Missing_hole",
            "Mouse_bite",
            "Open_circuit",
            "Short",
            "Spur",
            "Spurious_copper"
        ]
    )

    if st.button("🚀 Run Complete Pipeline", use_container_width=True):

        with st.spinner("Processing PCB image..."):
            os.makedirs("temp", exist_ok=True)

            template_path = "temp/template.png"
            test_path = "temp/test.png"
            output_dir = "temp/results"

            template_img.save(template_path)
            test_img.save(test_path)

            result = detect_and_annotate(
                template_path,
                test_path,
                output_dir,
                defect_label
            )

        st.success("✅ Processing completed successfully!")

        # -------------------------
        # PROCESS-BASED TABS
        # -------------------------
        tab1, tab2, tab3, tab4 = st.tabs([
            "🧪 Defect Mask (Preprocessing)",
            "📦 ROI Localization",
            "🧠 Model Information",
            "✅ Final Annotated Output"
        ])

        # -------------------------
        # TAB 1: MASK
        # -------------------------
        with tab1:
            st.subheader("Binary Defect Mask")
            st.image(result["mask_path"], use_container_width=True)
            st.metric(
                "Preprocessing Time",
                f"{result['timings']['Preprocessing']:.3f} sec"
            )

        # -------------------------
        # TAB 2: ROI ONLY
        # -------------------------
        with tab2:
            st.subheader("Localized Defect Regions (Bounding Boxes Only)")
            st.image(result["roi_path"], use_container_width=True)
            st.metric("ROIs Detected", result["roi_count"])
            st.metric(
                "Localization Time",
                f"{result['timings']['Localization']:.3f} sec"
            )

        # -------------------------
        # TAB 3: MODEL DETAILS
        # -------------------------
        with tab3:
            st.subheader("Classification Model Details")
            st.markdown("""
- **Model Used:** Custom CNN (PyTorch)  
- **Input Size:** 128 × 128  
- **Optimizer:** Adam  
- **Loss Function:** Cross-Entropy  
- **Achieved Accuracy:** **84%**
            """)

        # -------------------------
        # TAB 4: FINAL OUTPUT
        # -------------------------
        with tab4:
            st.subheader("Final Annotated PCB Output")
            st.image(result["final_path"], use_container_width=True)
            st.metric(
                "Total Processing Time",
                f"{result['timings']['Total']:.3f} sec"
            )

else:
    st.info("⬆️ Upload both template and test images to begin.")

# -------------------------
# FOOTER
# -------------------------
st.markdown(
    "<div class='footer'>PCB Defect Detection System • Internship Project</div>",
    unsafe_allow_html=True
)
