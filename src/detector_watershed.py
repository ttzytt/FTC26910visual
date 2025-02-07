import cv2
import numpy as np
from typing import List, TypedDict
from src.detectors import Detector
from src.color_def import Color, compute_hue_std_flip
from src.block import Block


class WatershedVizResults(TypedDict, total=False):
    """Defines the structure of the dictionary returned by visualize()."""
    original: np.ndarray
    preprocessed: np.ndarray
    combined_mask: np.ndarray
    sure_bg: np.ndarray
    sure_fg: np.ndarray
    unknown: np.ndarray
    final_detection: np.ndarray

class ColorBlockDetectorWatershed(Detector):
    """
    Detect color blocks by:
      1) Preprocessing (brightness, blur)
      2) Combining all color ranges to create a single mask
      3) Using distance transform + watershed to segment
      4) Extract each labeled region, build Block
    """

    def __init__(self, detecting_colors):
        # Basic image processing parameters
        self.blur_size = 50
        self.brightness = 0
        self.mask_erode_iter = 2
        self.mask_dilate_iter = 2
        self.min_area = 1000  # minimum region area
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.sure_fg_min_dis_ratio = 0.7
        self.detecting_colors = detecting_colors
        # For optional HSV std check:
        self.std_threshold_hsv = (3, 50, 50)

        # Storage for debug images (intermediate steps)
        self._debug_images: WatershedVizResults = {}

    def process_frame(self, frame: np.ndarray) -> List[Block]:
        """
        Main entry: preprocess, do watershed, return Blocks
        """
        # Clear old debug images
        self._debug_images = {}

        # Save original
        self._debug_images['original'] = frame.copy()

        # 1) Preprocess
        preprocessed = self._preprocess(frame)
        self._debug_images['preprocessed'] = preprocessed.copy()

        # 2) Convert to HSV
        hsv = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2HSV)

        # 3) Detect blocks via watershed
        blocks = self._detect_blocks(hsv, frame)

        return blocks

    def get_debug_images(self) -> WatershedVizResults:
        return self._debug_images

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Adjust brightness and apply Gaussian blur."""
        # Adjust brightness
        frame = cv2.convertScaleAbs(frame, alpha=1, beta=self.brightness)
        # Apply blur
        if self.blur_size > 0:
            kernel_size = self.blur_size | 1  # ensure odd
            frame = cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)
        return frame

    def _detect_blocks(self, hsv: np.ndarray, frame_bgr: np.ndarray) -> List[Block]:
        """
        Perform multiple watershed passes: one per color in self.detecting_colors.

        Workflow for each color:
        1) Create a color-specific mask.
        2) Morphological cleaning.
        3) _watershed_segment => obtains markers.
        4) For each labeled region:
            - Compute shape (minAreaRect)
            - (Optionally) compute HSV stats
            - Assign the color (since this pass is specific to one color)
            - Construct a Block
        Accumulate all blocks from each color and return.
        """

        all_blocks: List[Block] = []
        # For debugging, we can store intermediate images per color
        # e.g. "blue_mask", "blue_sure_bg", etc.
        self._debug_images = {}

        # Save original just once
        self._debug_images['original'] = frame_bgr.copy()

        # 1) Preprocess (optional brightness, blur)
        preprocessed = self._preprocess(frame_bgr)
        self._debug_images['preprocessed'] = preprocessed.copy()

        # 2) Convert to HSV
        hsv_img = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2HSV)

        # For each color, run its own Watershed pass
        for color_def in self.detecting_colors:
            # A) Create color-specific mask
            color_mask = self._create_color_mask(hsv_img, color_def)
            # Optional morphological ops if not already inside _create_color_mask
            color_mask = cv2.morphologyEx(
                color_mask, cv2.MORPH_OPEN, self.kernel, iterations=2)
            color_mask = cv2.morphologyEx(
                color_mask, cv2.MORPH_CLOSE, self.kernel, iterations=2)

            # Save debug mask
            mask_bgr = cv2.cvtColor(color_mask, cv2.COLOR_GRAY2BGR)
            self._debug_images[f"{color_def.name}_mask"] = mask_bgr

            # B) Watershed on this color mask
            markers, num_labels, sure_bg, sure_fg, unknown = self._watershed_segment(
                color_mask, frame_bgr)

            # Debug images for sure_bg, sure_fg, unknown
            self._debug_images[f"{color_def.name}_sure_bg"] = cv2.cvtColor(
                sure_bg, cv2.COLOR_GRAY2BGR)
            self._debug_images[f"{color_def.name}_sure_fg"] = cv2.cvtColor(
                sure_fg, cv2.COLOR_GRAY2BGR)

            unknown_bgr = np.zeros_like(frame_bgr)
            unknown_bgr[unknown == 255] = (128, 128, 128)
            self._debug_images[f"{color_def.name}_unknown"] = unknown_bgr

            # C) Build blocks from each label
            h, w = color_mask.shape[:2]
            for lbl in range(2, num_labels + 1):
                region_mask = (markers == lbl)
                region_area = np.count_nonzero(region_mask)
                if region_area < self.min_area:
                    continue

                # extract coordinates
                region_pts = np.argwhere(region_mask)
                if len(region_pts) == 0:
                    continue

                # minAreaRect
                coords_xy = np.fliplr(region_pts).astype(np.float32)  # (x, y)
                coords_xy_list = coords_xy[:, np.newaxis, :]
                rect = cv2.minAreaRect(coords_xy_list)
                (rx, ry), (rw, rh), angle = rect
                if rw < rh:
                    rw, rh = rh, rw
                    angle += 90

                # optional: compute HSV stats in that region
                label_mask = np.zeros((h, w), dtype=np.uint8)
                label_mask[region_mask] = 255

                hsv_region = cv2.bitwise_and(hsv_img, hsv_img, mask=label_mask)

                # For mean
                mean_hsv = cv2.mean(hsv_region, mask=label_mask)[:3]

                # For std
                h_ch, s_ch, v_ch = cv2.split(hsv_region)
                h_valid = h_ch[label_mask == 255].astype(np.float32)
                s_valid = s_ch[label_mask == 255].astype(np.float32)
                v_valid = v_ch[label_mask == 255].astype(np.float32)

                std_h = compute_hue_std_flip(h_valid)
                std_s = float(np.std(s_valid))
                std_v = float(np.std(v_valid))

                # We can retrieve the contour for accurate shape
                contours, _ = cv2.findContours(
                    label_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours:
                    continue
                largest_contour = max(contours, key=cv2.contourArea)

                # create Block
                block = Block(
                    center=(rx, ry),
                    size=(rw, rh),
                    angle=angle,
                    color=color_def,  # since this pass is specifically for color_def
                    mean_hsv=(mean_hsv[0], mean_hsv[1], mean_hsv[2]),
                    color_std=(std_h, std_s, std_v),
                    contour=largest_contour
                )
                all_blocks.append(block)

        return all_blocks



    def _watershed_segment(self, mask: np.ndarray, frame_bgr: np.ndarray):
        """
        Given a binary mask (255=FG, 0=BG), apply morphological + distance transform,
        compute sure_fg, sure_bg, unknown, then run cv2.watershed on frame_bgr.
        
        Returns:
          markers, num_labels, sure_bg, sure_fg, unknown
        """
        # Step 1: sure_bg by dilate
        sure_bg = cv2.dilate(mask, self.kernel, iterations=7)

        # Step 2: distance transform => sure_fg
        dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        max_val = dist_transform.max()
        _, sure_fg = cv2.threshold(dist_transform, self.sure_fg_min_dis_ratio * max_val, 255, 0)
        # set to 255 if larger than threshold, else 0
        sure_fg = sure_fg.astype(np.uint8)

        # Step 3: unknown = sure_bg - sure_fg
        unknown = cv2.subtract(sure_bg, sure_fg)

        # Step 4: connectedComponents => markers
        num_labels, markers = cv2.connectedComponents(sure_fg)
        # in connected component, 0 is for background
        # but we want background to be 1, so add 1 to all labels
        markers += 1 
        markers[unknown == 255] = 0

        # Step 5: watershed
        # 1: background 
        # 0: unknown
        # other: different regions
        cv2.watershed(frame_bgr, markers)

        return markers, num_labels, sure_bg, sure_fg, unknown

    def _create_color_mask(self, hsv: np.ndarray, color_def: Color) -> np.ndarray:
        """
        Create a mask for a single color definition, with morphological ops.
        """
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

        for (lower, upper) in color_def.hsv_ranges:
            lower_np = np.array(lower, dtype=np.uint8)
            upper_np = np.array(upper, dtype=np.uint8)
            tmp_mask = cv2.inRange(hsv, lower_np, upper_np)
            mask = cv2.bitwise_or(mask, tmp_mask)

        # Morphology
        mask_morph = cv2.erode(
            mask, self.kernel, iterations=self.mask_erode_iter)
        mask_morph = cv2.dilate(mask_morph, self.kernel,
                                iterations=self.mask_dilate_iter)
        return mask_morph
