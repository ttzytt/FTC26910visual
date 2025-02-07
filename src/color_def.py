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


RED_R9000P = Color(
    name="RED_R9000P",
    hsv_ranges=[
        ((0, 70, 50), (3, 160, 225)),
        ((165, 70, 50), (180, 160, 225)),
    ],
    bgr=(0, 0, 255)
)

BLUE_R9000P = Color(
    name="BLUE_R9000P",
    hsv_ranges=[
        ((110, 80, 70), (125, 180, 230)),
    ],
    bgr=(255, 0, 0)
)

YELLOW_R9000P = Color(
    name="YELLOW_R9000P",
    hsv_ranges=[
        ((17, 60, 140), (32, 125, 255)),
    ],
    bgr=(0, 255, 255)
)

COLOR_DEF_R9000P = [RED_R9000P, BLUE_R9000P, YELLOW_R9000P]

RED_LL = Color(
    name="RED_LL",
    hsv_ranges=[
        ((0, 190, 90), (5, 255, 250)),
        ((160, 190, 90), (180, 255, 250)),
    ],
    bgr=(0, 0, 255)
)

YELLOW_LL = Color(
    name="YELLOW_LL",
    hsv_ranges=[
        ((15, 130, 160), (35, 255, 250)),
    ],
    bgr=(0, 255, 255)
)

BLUE_LL = Color(
    name="BLUE_LL",
    hsv_ranges=[
        ((100, 220, 70), (125, 255, 230)),
    ],
    bgr=(255, 0, 0)
)