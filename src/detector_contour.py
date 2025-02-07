import cv2
import numpy as np
from src.color_def import *
from src.block import Block
from typing import List
from src.detectors import *


class ContourVizResults(TypedDict, total=False):
    """Defines the structure of the dictionary returned by visualize()."""
    final_detection: np.ndarray
    avg_HSV: np.ndarray
    original: np.ndarray
    preprocessed: np.ndarray
    hsv_space: np.ndarray
    combined_mask: np.ndarray

class ColorBlockDetectorContour(Detector):
    """
    Detects color blocks by:
      1) Preprocessing (brightness, blur)
      2) Creating color masks
      3) Finding contours
      4) Computing mean & std(H, S, V) inside each contour
    """

    def __init__(self, detecting_colors: List[Color]):
        # Basic image processing parameters
        self.blur_size = 35
        self.brightness = 0
        self.erode_iter = 7
        self.dilate_iter = 6
        self.min_contour_area = 1000
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        # Thresholds for std(H, S, V)
        self.std_threshold_hsv = (3, 50, 50)

        # Storage for debug images (intermediate steps)
        self._debug_images : ContourVizResults = {}
        self.detecting_colors = detecting_colors

    def process_frame(self, frame: np.ndarray) -> List[Block]:
        """Main entry: preprocess and detect blocks, while saving debug images."""
        self._debug_images = {}

        # Save original frame
        self._debug_images['original'] = frame.copy()

        # 1) Preprocessing
        preprocessed = self._preprocess(frame)
        self._debug_images['preprocessed'] = preprocessed.copy()

        # 2) Convert to HSV
        hsv = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2HSV)
        hsv_bgr_like = cv2.cvtColor(
            hsv, cv2.COLOR_HSV2BGR)  # just for visualization
        self._debug_images['hsv_space'] = hsv_bgr_like

        # 3) Detect blocks
        blocks = self._detect_blocks(hsv)

        return blocks

    def get_debug_images(self) -> ContourVizResults:
        """Returns debug images for visualization."""
        return self._debug_images

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Adjust brightness and apply Gaussian blur."""
        frame = cv2.convertScaleAbs(frame, alpha=1, beta=self.brightness)
        if self.blur_size > 0:
            kernel_size = self.blur_size | 1
            frame = cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)
        return frame

    def _detect_blocks(self, hsv: np.ndarray) -> List[Block]:
        blocks = []
        combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

        for color_def in self.detecting_colors:
            mask = self._create_color_mask(hsv, color_def)
            combined_mask = cv2.bitwise_or(combined_mask, mask)

            contours = self._find_contours(mask)
            color_blocks = self._process_contours(contours, color_def, hsv)
            blocks.extend(color_blocks)

        combined_bgr = cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2BGR)
        self._debug_images['combined_mask'] = combined_bgr

        return blocks

    def _create_color_mask(self, hsv: np.ndarray, color_def: Color) -> np.ndarray:
        """
        Create a mask for each color definition, applying morphological operations.
        The recognized areas retain their original colors (converted from HSV to BGR).
        
        Args:
            hsv (np.ndarray): The HSV-converted image.
            color_def (Color): The color definition (with HSV ranges and BGR info).
        
        Returns:
            np.ndarray: A binary mask after morphological ops.
        """
        # Step 1: Initialize an empty mask
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

        # Step 2: Apply color thresholds for each HSV range
        for (lower, upper) in color_def.hsv_ranges:

            # Apply threshold to get binary mask
            tmp_mask = cv2.inRange(hsv, np.array(list(lower)), np.array(list(upper)))
            mask = cv2.bitwise_or(mask, tmp_mask)

        # Step 3: For debug: raw mask in color
        hsv_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        raw_mask_colored = cv2.bitwise_and(hsv_bgr, hsv_bgr, mask=mask)
        debug_raw = f"{color_def.name.lower()}_mask_raw"
        self._debug_images[debug_raw] = raw_mask_colored

        # Step 4: Morphological operations (erode & dilate)
        mask_morph = cv2.erode(mask, self.kernel, iterations=self.erode_iter)
        mask_morph = cv2.dilate(mask_morph, self.kernel,
                                iterations=self.dilate_iter)

        # Step 5: For debug: morph mask in color
        morph_mask_colored = cv2.bitwise_and(hsv_bgr, hsv_bgr, mask=mask_morph)
        debug_morph = f"{color_def.name.lower()}_mask_morph"
        self._debug_images[debug_morph] = morph_mask_colored

        return mask_morph

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
