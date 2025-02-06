import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple

# ---------- Data Classes ----------

hsv_t = Tuple[int, int, int]
bgr_t = Tuple[int, int, int]


@dataclass
class Color:
    """Stores color name, HSV ranges, and BGR values for drawing."""
    name: str
    hsv_ranges: List[Tuple[hsv_t, hsv_t]]
    bgr: bgr_t

# ---------- Global Color Definitions ----------

RED = Color(
    name='RED',
    hsv_ranges=[
        ((0, 80, 100), (10, 200, 255)),
        ((160, 80, 100), (180, 200, 255))
    ],
    bgr=(0, 0, 255)
)

BLUE = Color(
    name='BLUE',
    hsv_ranges=[
        ((105, 50, 100), (120, 200, 255))
    ],
    bgr=(255, 0, 0)
)

YELLOW = Color(
    name='YELLOW',
    hsv_ranges=[
        ((20, 100, 100), (30, 255, 255))
    ],
    bgr=(0, 255, 255)
)

COLOR_DEFINITIONS = [RED, BLUE, YELLOW]


# ---------- Color Block Detector ----------

def compute_hue_std_flip(h_array: np.ndarray, flip_threshold: float = 90.0) -> float:
    # Ensure float
    h_float = h_array.astype(np.float32)

    # 1) Direct std
    std1 = np.std(h_float)

    # 2) Flip
    shifted = h_float.copy()
    mask = (shifted < flip_threshold)
    shifted[mask] += 180.0
    std2 = np.std(shifted)

    return float(min(std1, std2))
