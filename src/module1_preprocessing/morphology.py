import cv2
import numpy as np

def clean_mask(mask):
    """
    Remove tiny noise while keeping real defect regions
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)

    return closed
