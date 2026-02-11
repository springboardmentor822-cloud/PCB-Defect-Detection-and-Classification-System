"""
PCB Defect Detection Web Application
Built with Streamlit for easy deployment and beautiful UI
Enhanced with template comparison and CSV export
"""

import streamlit as st
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import io
import sys
from datetime import datetime

# Add backend to path
backend_dir = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_dir))

from utils.inference import run_detection_pipeline, run_template_comparison_pipeline, ModelLoader
from utils.visualization import display_detection_summary, display_defect_legend, display_model_info
from utils.export import generate_csv_log, prepare_image_download


# Page configuration
st.set_page_config(
    page_title="PCB Defect Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #6366f1;
        --secondary-color: #8b5cf6;
        --success-color: #10b981;
        --warning-color: #f59e0b;
        --danger-color: #ef4444;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.9;
    }
    
    /* Upload section */
    .upload-section {
        background: #f8fafc;
        padding: 2rem;
        border-radius: 10px;
        border: 2px dashed #cbd5e1;
        text-align: center;
        margin: 1rem 0;
    }
    
    /* Results section */
    .results-header {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        padding: 1rem 2rem;
        border-radius: 8px;
        color: white;
        margin: 1rem 0;
    }
    
    /* Metric cards */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        border: 1px solid #cbd5e1;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(99, 102, 241, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(99, 102, 241, 0.4);
    }
    
    /* Download button */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(16, 185, 129, 0.3);
    }
    
    .stDownloadButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(16, 185, 129, 0.4);
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: #f8fafc;
    }
    
    /* Info boxes */
    .stAlert {
        border-radius: 8px;
        border-left: 4px solid #6366f1;
    }
    
    /* Image containers */
    .stImage {
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Dataframe */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)


def main():
    """Main application function."""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🔍 PCB Defect Detection System</h1>
        <p>Advanced AI-powered detection and classification of PCB defects</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/circuit.png", width=80)
        st.title("⚙️ Settings")
        
        # Detection mode selection
        st.subheader("Detection Mode")
        detection_mode = st.radio(
            "Choose detection mode:",
            ["Auto Detection", "Template Comparison"],
            help="Auto: Detect defects automatically. Template: Compare test image with template."
        )
        
        st.divider()
        
        # Model paths
        st.subheader("Model Configuration")
        
        base_dir = Path(__file__).parent.parent / "backend"
        detector_path = base_dir / "models" / "best_pcb.pt"
        classifier_path = base_dir / "models" / "classifier_best.pt"
        
        # Check if models exist
        detector_exists = detector_path.exists()
        classifier_exists = classifier_path.exists()
        
        if detector_exists:
            st.success("✅ YOLOv8 Detector: Loaded")
        else:
            st.error(f"❌ YOLOv8 Detector: Not found")
            st.caption(f"Expected at: {detector_path}")
        
        if classifier_exists:
            st.success("✅ EfficientNet Classifier: Loaded")
        else:
            st.error(f"❌ Classifier: Not found")
            st.caption(f"Expected at: {classifier_path}")
        
        st.divider()
        
        # Display model info and legend
        display_model_info()
        display_defect_legend()
        
        st.divider()
        
        # Device info
        device = ModelLoader.get_device()
        st.info(f"🖥️ Computing on: **{device}**")
    
    # Main content
    if not (detector_exists and classifier_exists):
        st.error("⚠️ Models not found! Please ensure the models are in the correct location.")
        st.info("""
        **Expected model locations:**
        - YOLOv8 Detector: `backend/models/best_pcb.pt`
        - EfficientNet Classifier: `backend/models/classifier_best.pt`
        """)
        return
    
    # Different UI based on detection mode
    if detection_mode == "Auto Detection":
        render_auto_detection_mode(detector_path, classifier_path)
    else:
        render_template_comparison_mode(detector_path, classifier_path)


def render_auto_detection_mode(detector_path, classifier_path):
    """Render the auto detection mode UI."""
    
    st.subheader("📤 Upload PCB Image")
    uploaded_file = st.file_uploader(
        "Choose a PCB image (JPG, PNG, JPEG)",
        type=['jpg', 'png', 'jpeg'],
        help="Upload a PCB image to detect and classify defects",
        key="auto_upload"
    )
    
    if uploaded_file is not None:
        # Clear old results if new file uploaded
        current_file = uploaded_file.name
        if 'last_uploaded_file' not in st.session_state or st.session_state['last_uploaded_file'] != current_file:
            # New file uploaded, clear previous results
            for key in ['annotated_image', 'detections', 'original_image', 'upload_filename']:
                if key in st.session_state:
                    del st.session_state[key]
            st.session_state['last_uploaded_file'] = current_file
        
        # Read and display original image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        original_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # Create two columns for before/after
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📷 Original Image")
            original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            st.image(original_rgb, width="stretch")
        
        # Run detection button
        if st.button("🚀 Run Detection & Classification", use_container_width=True, key="auto_detect"):
            with st.spinner("🔄 Running AI models... This may take a moment..."):
                try:
                    # Run the pipeline
                    annotated_image, detections = run_detection_pipeline(
                        original_image,
                        str(detector_path),
                        str(classifier_path)
                    )
                    
                    # Store results in session state
                    st.session_state['annotated_image'] = annotated_image
                    st.session_state['detections'] = detections
                    st.session_state['original_image'] = original_image
                    st.session_state['upload_filename'] = uploaded_file.name
                    
                    st.success("✅ Detection complete!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error during detection: {str(e)}")
                    st.exception(e)
        
        # Display results if available
        if 'annotated_image' in st.session_state:
            with col2:
                st.subheader("🎯 Detection Results")
                annotated_rgb = cv2.cvtColor(st.session_state['annotated_image'], cv2.COLOR_BGR2RGB)
                st.image(annotated_rgb, width="stretch")
            
            # Show detection count
            st.info(f"🔍 Found {len(st.session_state['detections'])} defect(s) in the image")
            
            # Display detection summary below
            st.divider()
            display_detection_summary(st.session_state['detections'])
            
            # Download section
            st.divider()
            st.subheader("⬇️ Download Results")
            
            col_dl1, col_dl2 = st.columns(2)
            
            with col_dl1:
                # Download annotated image
                img_bytes = prepare_image_download(st.session_state['annotated_image'])
                st.download_button(
                    label="📥 Download Annotated Image",
                    data=img_bytes,
                    file_name=f"pcb_defects_{st.session_state.get('upload_filename', 'result.png')}",
                    mime="image/png",
                    use_container_width=True
                )
            
            with col_dl2:
                # Download CSV report
                csv_bytes = generate_csv_log(st.session_state['detections'])
                st.download_button(
                    label="📊 Download CSV Report",
                    data=csv_bytes,
                    file_name=f"defect_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            # Clear results button
            if st.button("🔄 Clear Results", use_container_width=True, key="auto_clear"):
                for key in ['annotated_image', 'detections', 'original_image', 'upload_filename']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
    
    else:
        # Show instructions when no file is uploaded
        st.info("""
        👆 **Get Started:**
        1. Upload a PCB image using the file uploader above
        2. Click "Run Detection & Classification" to analyze the image
        3. View the results with bounding boxes and defect classifications
        4. Download the annotated image and CSV report
        """)
        
        show_defect_types_info()


def render_template_comparison_mode(detector_path, classifier_path):
    """Render the template comparison mode UI."""
    
    st.subheader("📤 Upload Images for Comparison")
    
    col_up1, col_up2 = st.columns(2)
    
    with col_up1:
        st.markdown("**Template Image (Golden PCB)**")
        template_file = st.file_uploader(
            "Upload template image",
            type=['jpg', 'png', 'jpeg'],
            help="Upload a defect-free template PCB image",
            key="template_upload"
        )
    
    with col_up2:
        st.markdown("**Test Image (PCB to Inspect)**")
        test_file = st.file_uploader(
            "Upload test image",
            type=['jpg', 'png', 'jpeg'],
            help="Upload the PCB image to inspect for defects",
            key="test_upload"
        )
    
    if template_file is not None and test_file is not None:
        # Clear old results if new files uploaded
        current_files = f"{template_file.name}_{test_file.name}"
        if 'last_template_files' not in st.session_state or st.session_state['last_template_files'] != current_files:
            # New files uploaded, clear previous results
            for key in ['template_annotated', 'template_detections', 'template_image', 'test_image', 'test_filename']:
                if key in st.session_state:
                    del st.session_state[key]
            st.session_state['last_template_files'] = current_files
        
        # Read images
        template_bytes = np.asarray(bytearray(template_file.read()), dtype=np.uint8)
        template_image = cv2.imdecode(template_bytes, cv2.IMREAD_COLOR)
        
        test_bytes = np.asarray(bytearray(test_file.read()), dtype=np.uint8)
        test_image = cv2.imdecode(test_bytes, cv2.IMREAD_COLOR)
        
        # Display uploaded images
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📋 Template Image")
            template_rgb = cv2.cvtColor(template_image, cv2.COLOR_BGR2RGB)
            st.image(template_rgb, width="stretch")
        
        with col2:
            st.subheader("🔬 Test Image")
            test_rgb = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
            st.image(test_rgb, width="stretch")
        
        # Run comparison button
        if st.button("🚀 Run Template Comparison", use_container_width=True, key="template_detect"):
            with st.spinner("🔄 Comparing images and detecting defects..."):
                try:
                    # Run template comparison pipeline
                    annotated_image, detections = run_template_comparison_pipeline(
                        template_image,
                        test_image,
                        str(detector_path),
                        str(classifier_path),
                        use_subtraction=True
                    )
                    
                    # Store results in session state
                    st.session_state['template_annotated'] = annotated_image
                    st.session_state['template_detections'] = detections
                    st.session_state['template_image'] = template_image
                    st.session_state['test_image'] = test_image
                    st.session_state['test_filename'] = test_file.name
                    
                    st.success("✅ Comparison complete!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error during comparison: {str(e)}")
                    st.exception(e)
        
        # Display results if available
        if 'template_annotated' in st.session_state:
            st.divider()
            st.subheader("🎯 Detection Results")
            
            # Display result image below
            result_rgb = cv2.cvtColor(st.session_state['template_annotated'], cv2.COLOR_BGR2RGB)
            st.image(result_rgb, width="stretch", caption="Detected Defects on Test Image")
            
            # Show detection count
            st.info(f"🔍 Found {len(st.session_state['template_detections'])} defect(s) in the comparison")
            
            # Display detection summary
            st.divider()
            display_detection_summary(st.session_state['template_detections'])
            
            # Download section
            st.divider()
            st.subheader("⬇️ Download Results")
            
            col_dl1, col_dl2 = st.columns(2)
            
            with col_dl1:
                # Download annotated image
                img_bytes = prepare_image_download(st.session_state['template_annotated'])
                st.download_button(
                    label="📥 Download Annotated Image",
                    data=img_bytes,
                    file_name=f"comparison_result_{st.session_state.get('test_filename', 'result.png')}",
                    mime="image/png",
                    use_container_width=True
                )
            
            with col_dl2:
                # Download CSV report
                csv_bytes = generate_csv_log(st.session_state['template_detections'])
                st.download_button(
                    label="📊 Download CSV Report",
                    data=csv_bytes,
                    file_name=f"comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            # Clear results button
            if st.button("🔄 Clear Results", use_container_width=True, key="template_clear"):
                for key in ['template_annotated', 'template_detections', 'template_image', 'test_image', 'test_filename']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
    
    else:
        # Show instructions
        st.info("""
        👆 **Get Started with Template Comparison:**
        1. Upload a **template image** (defect-free golden PCB)
        2. Upload a **test image** (PCB to inspect)
        3. Click "Run Template Comparison" to find differences
        4. View detected defects with color-coded bounding boxes
        5. Download the annotated image and CSV report
        
        💡 **Tip:** Template comparison uses image subtraction to highlight differences between the template and test images, then applies AI classification.
        """)
        
        show_defect_types_info()


def show_defect_types_info():
    """Display information about supported defect types."""
    st.subheader("📋 Supported Defect Types")
    
    defect_info = {
        "Missing Hole": "Holes that should be present but are missing",
        "Mouse Bite": "Small indentations on the edge of the PCB",
        "Open Circuit": "Breaks in the conductive traces",
        "Short": "Unwanted connections between traces",
        "Spur": "Unwanted protrusions from traces",
        "Spurious Copper": "Excess copper in unwanted areas"
    }
    
    cols = st.columns(2)
    for idx, (defect, description) in enumerate(defect_info.items()):
        with cols[idx % 2]:
            st.markdown(f"**{defect}**")
            st.caption(description)


if __name__ == "__main__":
    main()
