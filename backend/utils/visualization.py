"""
Visualization utilities for displaying detection results.
"""

import streamlit as st
from typing import List, Dict
import pandas as pd


def display_detection_summary(detections: List[Dict]):
    """
    Display a summary of detections in a nice format.
    
    Args:
        detections: List of detection dictionaries
    """
    if not detections:
        st.info("✅ No defects detected! PCB appears to be clean.")
        return
    
    st.subheader(f"🔍 Detected {len(detections)} Defect(s)")
    
    # Create summary statistics
    defect_counts = {}
    for det in detections:
        label = det['label']
        defect_counts[label] = defect_counts.get(label, 0) + 1
    
    # Display as metrics
    cols = st.columns(min(len(defect_counts), 3))
    for idx, (defect_type, count) in enumerate(defect_counts.items()):
        with cols[idx % 3]:
            st.metric(
                label=defect_type.replace('_', ' ').title(),
                value=f"{count} detected"
            )
    
    # Display detailed table
    st.subheader("📊 Detailed Results")
    
    df_data = []
    for idx, det in enumerate(detections, 1):
        df_data.append({
            'ID': idx,
            'Defect Type': det['label'].replace('_', ' ').title(),
            'Confidence': f"{det['confidence']:.2%}",
            'Location (x1, y1, x2, y2)': str(det['bbox'])
        })
    
    df = pd.DataFrame(df_data)
    st.dataframe(df, width="stretch", hide_index=True)


def display_defect_legend():
    """Display a legend of defect types and their colors."""
    st.sidebar.markdown("### 🎨 Defect Color Legend")
    
    defect_colors = {
        'Missing Hole': '🟡',
        'Mouse Bite': '🟣',
        'Open Circuit': '🔴',
        'Short': '🟠',
        'Spur': '🔵',
        'Spurious Copper': '🟢'
    }
    
    for defect, emoji in defect_colors.items():
        st.sidebar.markdown(f"{emoji} **{defect}**")


def display_model_info():
    """Display information about the models being used."""
    with st.sidebar.expander("ℹ️ Model Information"):
        st.markdown("""
        **Detection Model:** YOLOv8  
        Detects regions of interest (ROIs) on PCB images
        
        **Classification Model:** EfficientNet-B0  
        Classifies detected defects into 6 categories:
        - Missing Hole
        - Mouse Bite
        - Open Circuit
        - Short
        - Spur
        - Spurious Copper
        """)
