"""
Export utilities for PCB defect detection results.
Handles CSV generation and image download preparation.
"""

import pandas as pd
import cv2
import numpy as np
from datetime import datetime
from typing import List, Dict
import io


def generate_csv_log(detections: List[Dict], filename: str = "defect_report") -> bytes:
    """
    Generate a CSV log of all detected defects.
    
    Args:
        detections: List of detection dictionaries with bbox, label, confidence
        filename: Base filename for the CSV
    
    Returns:
        CSV data as bytes for download
    """
    if not detections:
        # Return empty CSV with headers
        df = pd.DataFrame(columns=['ID', 'Defect_Type', 'Confidence_%', 'X1', 'Y1', 'X2', 'Y2', 'Timestamp'])
    else:
        csv_data = []
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        for idx, det in enumerate(detections, 1):
            x1, y1, x2, y2 = det['bbox']
            csv_data.append({
                'ID': idx,
                'Defect_Type': det['label'],
                'Confidence_%': f"{det['confidence'] * 100:.2f}",
                'X1': x1,
                'Y1': y1,
                'X2': x2,
                'Y2': y2,
                'Timestamp': timestamp
            })
        
        df = pd.DataFrame(csv_data)
    
    # Convert to CSV bytes
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_bytes = csv_buffer.getvalue().encode('utf-8')
    
    return csv_bytes


def prepare_image_download(image: np.ndarray, format: str = 'PNG') -> bytes:
    """
    Prepare an image for download by encoding it to bytes.
    
    Args:
        image: Image in BGR format (OpenCV)
        format: Image format (PNG, JPG)
    
    Returns:
        Image data as bytes
    """
    if format.upper() == 'PNG':
        _, buffer = cv2.imencode('.png', image)
    elif format.upper() in ['JPG', 'JPEG']:
        _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    return buffer.tobytes()


def create_detection_summary(detections: List[Dict]) -> Dict:
    """
    Create a summary of detections for display.
    
    Args:
        detections: List of detection dictionaries
    
    Returns:
        Dictionary with summary statistics
    """
    if not detections:
        return {
            'total_defects': 0,
            'defect_counts': {},
            'avg_confidence': 0.0,
            'defect_types': []
        }
    
    defect_counts = {}
    total_confidence = 0.0
    
    for det in detections:
        label = det['label']
        defect_counts[label] = defect_counts.get(label, 0) + 1
        total_confidence += det['confidence']
    
    return {
        'total_defects': len(detections),
        'defect_counts': defect_counts,
        'avg_confidence': total_confidence / len(detections),
        'defect_types': list(defect_counts.keys())
    }
