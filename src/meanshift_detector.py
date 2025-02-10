import cv2
import numpy as np
from typing import List, TypedDict
from src.detector import Detector
from src.color_defs import Color, compute_hue_std_flip
from src.block import Block


class MeanShiftVizResults(TypedDict, total=False):
    """Typed dictionary structure for debug images."""
    original: np.ndarray
    preprocessed: np.ndarray
    mean_shift_filtered: np.ndarray
    mask_after_threshold: np.ndarray
    final_detection: np.ndarray


class MeanshiftDetector(Detector):
    """
    Detect color blocks by:
      1) Preprocessing (brightness, blur)
      2) Mean Shift filtering to smooth colors
      3) For each color, threshold => morphological cleaning => findContours
      4) Build Block from each contour
    """

    def __init__(self, detecting_colors: List[Color]):
        # Basic image processing parameters
        self.blur_size = 0
        self.brightness = 0
        self.mask_erode_iter = 0
        self.mask_dilate_iter = 0
        self.min_area = 1000  # Minimum region area
        self.spatial_radius = 30  # For pyrMeanShiftFiltering
        self.color_radius = 50
        self.max_level = 1  # for iterative level in meanShift
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        # Optional HSV std check
        self.std_threshold_hsv = (3, 50, 50)
        self.detecting_colors = detecting_colors

        # Storage for debug images
        self._debug_images: MeanShiftVizResults = {}

    def process_frame(self, frame: np.ndarray) -> List[Block]:
        """
        Main entry: preprocess, do mean shift filtering, color threshold, build Blocks
        """
        # Clear old debug images
        self._debug_images = {}

        # Save original
        self._debug_images['original'] = frame.copy()

        # 1) Preprocess (e.g. brightness, blur)
        preprocessed = self._preprocess(frame)
        self._debug_images['preprocessed'] = preprocessed.copy()

        st_time = cv2.getTickCount()
        # 2) Mean Shift filtering
        mean_shift_bgr = cv2.pyrMeanShiftFiltering(
            preprocessed,
            sp=self.spatial_radius,
            sr=self.color_radius,
            maxLevel=self.max_level
        )
        ed_time = cv2.getTickCount()
        print("Mean Shift Time:", (ed_time - st_time) / cv2.getTickFrequency())
        self._debug_images['mean_shift_filtered'] = mean_shift_bgr.copy()

        # Convert to HSV for color thresholding
        mean_shift_hsv = cv2.cvtColor(mean_shift_bgr, cv2.COLOR_BGR2HSV)

        # 3) For each color, threshold => morphological => findContours => create Blocks
        blocks: List[Block] = []
        for color_def in self.detecting_colors:
            color_blocks = self._detect_blocks_for_color(
                mean_shift_hsv, mean_shift_bgr, color_def)
            blocks.extend(color_blocks)

        return blocks

    def get_debug_images(self) -> MeanShiftVizResults:
        return self._debug_images

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Adjust brightness and optionally blur."""
        # Adjust brightness
        frame = cv2.convertScaleAbs(frame, alpha=1, beta=self.brightness)
        # Gaussian blur if needed
        if self.blur_size > 0:
            kernel_size = self.blur_size | 1  # ensure odd
            frame = cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)
        return frame

    def _detect_blocks_for_color(self, hsv_img: np.ndarray, bgr_img: np.ndarray, color_def: Color) -> List[Block]:
        """
        For a single color definition, threshold => morphological cleaning => findContours => build Blocks
        """
        blocks: List[Block] = []

        # A) Combine all HSV ranges for this color
        mask = np.zeros(hsv_img.shape[:2], dtype=np.uint8)
        for (lower, upper) in color_def.hsv_ranges:
            lower_np = np.array(lower, dtype=np.uint8)
            upper_np = np.array(upper, dtype=np.uint8)
            tmp_mask = cv2.inRange(hsv_img, lower_np, upper_np)
            mask = cv2.bitwise_or(mask, tmp_mask)

        # B) Morphological cleaning
        if self.mask_erode_iter > 0:
            mask = cv2.erode(mask, self.kernel, iterations=self.mask_erode_iter)
        if self.mask_dilate_iter > 0:
            mask = cv2.dilate(mask, self.kernel, iterations=self.mask_dilate_iter)

        # Optional: store debug for each color
        mask_debug = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        self._debug_images[f"{color_def.name}_mask_after_threshold"] = mask_debug

        # C) Find contours in the mask
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        h, w = mask.shape[:2]
        for cnt in contours:
            region_area = cv2.contourArea(cnt)
            if region_area < self.min_area:
                continue

            # minAreaRect
            rect = cv2.minAreaRect(cnt)
            (rx, ry), (rw, rh), angle = rect
            if rw < rh:
                rw, rh = rh, rw
                angle += 90

            # Build a mask for this contour to compute stats
            contour_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(contour_mask, [cnt], 0, 255, -1)

            # Extract HSV region
            hsv_region = cv2.bitwise_and(hsv_img, hsv_img, mask=contour_mask)

            # Mean HSV
            mean_hsv = cv2.mean(hsv_region, mask=contour_mask)[:3]

            # Std HSV
            h_ch, s_ch, v_ch = cv2.split(hsv_region)
            h_valid = h_ch[contour_mask == 255].astype(np.float32)
            s_valid = s_ch[contour_mask == 255].astype(np.float32)
            v_valid = v_ch[contour_mask == 255].astype(np.float32)

            std_h = compute_hue_std_flip(h_valid)
            std_s = float(np.std(s_valid))
            std_v = float(np.std(v_valid))

            # Create Block
            block = Block(
                center=(rx, ry),
                size=(rw, rh),
                angle=angle,
                color=color_def,
                mean_hsv=(mean_hsv[0], mean_hsv[1], mean_hsv[2]),
                color_std=(std_h, std_s, std_v),
                contour=cnt
            )
            blocks.append(block)

        return blocks
