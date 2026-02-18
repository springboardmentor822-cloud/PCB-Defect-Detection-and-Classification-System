import streamlit as st
from backend import run_inference
from PIL import Image
from pathlib import Path
import tempfile
import pandas as pd
from src.module6_yolo.yolo_inference import run_yolo_detection


st.set_page_config(
    page_title="PCB Defect Detection",
    layout="wide"
)

st.title("PCB Defect Detection & Classification")
st.caption(
    "An end-to-end system demonstrating preprocessing, "
    "ROI localization, and classification of PCB defects."
)

st.subheader("Upload PCB Images")

col1, col2 = st.columns(2)

with col1:
    template_img = st.file_uploader(
        "Template PCB Image (Optional)",
        type=["jpg", "png", "jpeg"]
    )

with col2:
    test_img = st.file_uploader(
        "Test PCB Image (Required)",
        type=["jpg", "png", "jpeg"]
    )


if st.button("Run Defect Detection"):

    if test_img is None:
        st.error("Please upload a defect image.")
    else:
        with st.spinner("Processing images..."):

            
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir = Path(tmpdir)

                
                test_path = tmpdir / test_img.name
                Image.open(test_img).save(test_path)

                
                if template_img is not None:
                    template_path = tmpdir / template_img.name
                    Image.open(template_img).save(template_path)
                else:
                    template_path = None  # Auto-detect inside pipeline

                
                result = run_inference(
                    template_path,
                    test_path
                )

        st.success("Processing completed successfully!")


        tabs = st.tabs([
            "ROI Localization",
            "Model Information",
            "Final Annotated Output"
        ])

        with tabs[0]:
            st.subheader("ROI Localization")
            st.image(result["roi_image"], use_container_width=True)

        with tabs[1]:
            st.subheader("Classification Model Details")
            st.write("**Model Used:**", result["model_info"]["model"])
            st.write("**Input Size:**", result["model_info"]["input_size"])
            st.write("**Optimizer:**", result["model_info"]["optimizer"])
            st.write("**Loss Function:**", result["model_info"]["loss"])
            st.write("**Achieved Accuracy:**", result["model_info"]["accuracy"])

        with tabs[2]:
            st.subheader("Final Annotated PCB Image")
            st.image(result["annotated_image"], use_container_width=True)

            with open(result["annotated_image"], "rb") as f:
                st.download_button(
                    label="Download Annotated Image",
                    data=f,
                    file_name="pcb_annotated.png",
                    mime="image/png"
                )


st.markdown("---")
st.header("Unseen PCB Defect Detection (YOLO)")

yolo_image = st.file_uploader(
    "Upload PCB Image (No Template Required)",
    type=["jpg", "png"],
    key="yolo"
)

if yolo_image:

    if st.button("Run YOLO Detection"):

        with st.spinner("Running YOLO detection..."):

            
            import tempfile
            from pathlib import Path

            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir = Path(tmpdir)
                image_path = tmpdir / yolo_image.name

                with open(image_path, "wb") as f:
                    f.write(yolo_image.getbuffer())

                output_image = run_yolo_detection(str(image_path))

        st.success("YOLO Detection Completed!")

        st.image(output_image, use_container_width=True)

        with open(output_image, "rb") as f:
            st.download_button(
                label="Download YOLO Result",
                data=f,
                file_name="yolo_result.png",
                mime="image/png"
            )
