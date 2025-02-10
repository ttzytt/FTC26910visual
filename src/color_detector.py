import cv2
import numpy as np
from src.color_defs import *
from src.block import Block
from typing import List, TypedDict
from src.detector import *
from src.type_defs import *
from enum import Enum

class DebugType(Enum):
    SINGLE_COLOR_MASK = "single_color_mask"
    COMBINED_MASK = "combined_mask"

class ColorDetector(Detector):
    """
    Detects color blocks by:
      1) Preprocessing (brightness, blur)
      2) Creating color masks
      3) Finding contours
      4) Computing mean & std(H, S, V) inside each contour
    """

    def __init__(self, detecting_colors: List[Color], preproc_cfg: PreprocCfg = PreprocCfg(), debug_option : List[DebugType] | bool = []):
        # Basic image processing parameters
        super().__init__(detecting_colors, preproc_cfg, debug_option, DebugType)
        self.min_contour_area = 1000
        # Thresholds for std(H, S, V)
        self.std_threshold_hsv = (3, 50, 50)

    def process_frame(self, frame: np.ndarray) -> List[Block]:
        """Main entry: preprocess and detect blocks, while saving debug images."""
        # 1) Preprocessing
        preprocessed = self._preprocess(frame)

        # 2) Convert to HSV
        hsv = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2HSV)

        # 3) Detect blocks
        blocks = self._detect_blocks(hsv)

        return blocks

    def _detect_blocks(self, hsv: np.ndarray) -> List[Block]:
        blocks = []
        combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

        for color_def in self.detecting_colors:
            mask = self.create_color_mask(hsv, color_def)
            if DebugType.SINGLE_COLOR_MASK in self.debug_option:
                self.debug_images[f'{color_def.name}_mask'] = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            combined_mask = cv2.bitwise_or(combined_mask, mask)

            contours = self._find_contours(mask)
            color_blocks = self._process_contours(contours, color_def, hsv)
            blocks.extend(color_blocks)

        combined_bgr = cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2BGR)
        if DebugType.COMBINED_MASK in self.debug_option:
            self.debug_images['combined_mask'] = combined_bgr

        return blocks

    def _find_contours(self, mask: np.ndarray) -> List[np.ndarray]:
        """Find external contours with area > min_contour_area."""
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [c for c in contours if cv2.contourArea(c) > self.min_contour_area]

    def _process_contours(self,
                          contours: List[np.ndarray],
                          color_def: Color,
                          hsv: np.ndarray) -> List[Block]:
        """Compute mean & std(H, S, V) inside each contour and filter by thresholds."""
        blocks = []
        for cnt in contours:
            rect = cv2.minAreaRect(cnt)
            (cx, cy), (w, h), angle = rect

            # Normalize orientation
            if w < h:
                w, h = h, w
                angle += 90

            # Get bounding rect (for ROI)
            x_min, y_min, w_int, h_int = cv2.boundingRect(cnt)
            if w_int == 0 or h_int == 0:
                continue

            # Create local mask (contour inside bounding box)
            contour_mask = np.zeros((h_int, w_int), dtype=np.uint8)
            shifted_cnt = cnt - [x_min, y_min]
            cv2.drawContours(contour_mask, [shifted_cnt], 0, (255,), -1)

            # Extract HSV ROI
            hsv_roi = hsv[y_min:y_min + h_int, x_min:x_min + w_int]
            hsv_masked = cv2.bitwise_and(hsv_roi, hsv_roi, mask=contour_mask)

            # Split channels and extract valid pixels
            h_ch, s_ch, v_ch = cv2.split(hsv_masked)
            h_valid = h_ch[contour_mask == 255].astype(np.float32)
            s_valid = s_ch[contour_mask == 255].astype(np.float32)
            v_valid = v_ch[contour_mask == 255].astype(np.float32)

            if len(h_valid) == 0:
                continue

            # Compute mean & std for H, S, V
            mean_h = float(np.mean(h_valid))
            mean_s = float(np.mean(s_valid))
            mean_v = float(np.mean(v_valid))

            std_h = compute_hue_std_flip(h_valid, flip_threshold=90.0)
            std_s = float(np.std(s_valid))
            std_v = float(np.std(v_valid))

            # Create a new Block
            if std_h <= self.std_threshold_hsv[0] and \
               std_s <= self.std_threshold_hsv[1] and \
               std_v <= self.std_threshold_hsv[2]:

                block = Block(
                    center=(cx, cy),
                    size=(w, h),
                    angle=angle,
                    color=color_def,
                    color_std=(std_h, std_s, std_v),
                    mean_hsv=(mean_h, mean_s, mean_v),
                    # store the original contour (absolute coordinates)
                    contour=cnt
                )
                blocks.append(block)
        return blocks
