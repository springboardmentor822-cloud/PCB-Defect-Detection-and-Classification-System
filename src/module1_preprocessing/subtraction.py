import cv2
import numpy as np


def align_images(template, defect):
    """
    Align defect image to template using ECC alignment
    """
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    defect_gray = cv2.cvtColor(defect, cv2.COLOR_BGR2GRAY)

    warp_matrix = np.eye(2, 3, dtype=np.float32)

    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        100,
        1e-6
    )

    try:
        cv2.findTransformECC(
            template_gray,
            defect_gray,
            warp_matrix,
            cv2.MOTION_EUCLIDEAN,
            criteria
        )

        aligned = cv2.warpAffine(
            defect,
            warp_matrix,
            (template.shape[1], template.shape[0]),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
        )

        return aligned

    except cv2.error:
        return defect


def subtract_and_threshold(template, aligned):
    """
    Subtract template from aligned defect image
    and generate raw defect mask
    """

    # Absolute difference
    diff = cv2.absdiff(template, aligned)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (11, 11), 0)
    blur = cv2.medianBlur(blur, 5)

    _, mask = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return mask
