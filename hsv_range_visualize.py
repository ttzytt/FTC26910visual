import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

# ---------- Data Types ----------

hsv_t = Tuple[int, int, int]
bgr_t = Tuple[int, int, int]

@dataclass
class Color:
    """Stores color name, HSV ranges, and BGR values for drawing."""
    name: str
    hsv_ranges: List[Tuple[hsv_t, hsv_t]]
    bgr: bgr_t

# ---------- Define Global Color Constants ----------

RED = Color(
    name='RED',
    hsv_ranges=[
        ((0, 100, 100), (10, 255, 255)),
        ((160, 100, 100), (180, 255, 255))
    ],
    bgr=(0, 0, 255)
)

BLUE = Color(
    name='BLUE',
    hsv_ranges=[
        ((215 // 2, 40, 50), (230 // 2, 120, 255))
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

# Color margin
color_margin = {
    'H': 0,
    'S': 0,
    'V': 0
}

# ---------- Helper Functions ----------

def check_in_range(h: int, s: int, v: int, color_def: Color) -> bool:
    """
    Check if (h, s, v) is within any HSV range of the given color_def,
    considering global color_margin.
    """
    for (lower, upper) in color_def.hsv_ranges:
        lh = max(0, lower[0] - color_margin['H'])
        ls = max(0, lower[1] - color_margin['S'])
        lv = max(0, lower[2] - color_margin['V'])

        uh = min(180, upper[0] + color_margin['H'])
        us = min(255, upper[1] + color_margin['S'])
        uv = min(255, upper[2] + color_margin['V'])

        if lh <= h <= uh and ls <= s <= us and lv <= v <= uv:
            return True
    return False

def generate_color_map(color_def: Color, fixed_v: int = 255) -> np.ndarray:
    """
    Generate a 2D image (height=256, width=181) where:
      - x-axis = H (0..180)
      - y-axis = S (0..255)
    We fix V = fixed_v.
    If (H, S, fixed_v) is in color range, show the corresponding BGR color; else black.
    """
    img = np.zeros((256, 181, 3), dtype=np.uint8)

    for h in range(181):      # H from 0 to 180
        for s in range(256):  # S from 0 to 255
            if check_in_range(h, s, fixed_v, color_def):
                hsv_pixel = np.uint8([[[h, s, fixed_v]]])  # shape (1,1,3)
                bgr_pixel = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)
                img[s, h] = bgr_pixel[0, 0]
            else:
                # remains black
                pass

    return img

# ---------- Main ----------

def main():
    for color_def in COLOR_DEFINITIONS:
        color_map = generate_color_map(color_def, fixed_v=255)
        filename = f"{color_def.name.lower()}_map.png"
        cv2.imwrite(filename, color_map)
        print(f"Saved color map for {color_def.name} as '{filename}'")

if __name__ == "__main__":
    main()
