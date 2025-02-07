import cv2
import numpy as np
from typing import List, TypedDict
from .detectors import Detector
from .color_def import Color, compute_hue_std_flip
from .block import Block


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
        self.blur_size = 35
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
        1) Combine all color masks into one total mask.
        2) Perform morphological cleaning.
        3) Call _watershed_segment => obtain markers.
        4) For each segmented label:
        - Compute color coverage.
        - Assign the best matching color.
        - Construct a Block instance.
        """

        blocks: List[Block] = []

        # ========== A. Combine all color masks into a single "combined_mask" ==========

        combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

        # Dictionary to store individual color masks for color assignment later
        color_masks = {}

        for color_def in self.detecting_colors:
            mask_color = self._create_color_mask(hsv, color_def)
            combined_mask = cv2.bitwise_or(combined_mask, mask_color)
            color_masks[color_def.name] = mask_color  # Store for coverage analysis

        # Debugging: Store the combined mask visualization
        self._debug_images['combined_mask'] = cv2.cvtColor(
            combined_mask, cv2.COLOR_GRAY2BGR)

        # ========== B. Morphological operations (Opening & Closing) for noise reduction ==========

        mask_cleaned = cv2.morphologyEx(
            combined_mask, cv2.MORPH_OPEN, self.kernel, iterations=2)
        mask_cleaned = cv2.morphologyEx(
            mask_cleaned, cv2.MORPH_CLOSE, self.kernel, iterations=2)

        # ========== C. Apply Watershed Segmentation ==========

        markers, num_labels, sure_bg, sure_fg, unknown = self._watershed_segment(
            mask_cleaned, frame_bgr)

        # Debugging: Save foreground, background, and unknown region visualizations
        self._debug_images['sure_bg'] = cv2.cvtColor(sure_bg, cv2.COLOR_GRAY2BGR)
        self._debug_images['sure_fg'] = cv2.cvtColor(sure_fg, cv2.COLOR_GRAY2BGR)

        unknown_bgr = np.zeros_like(frame_bgr)
        unknown_bgr[unknown == 255] = (128, 128, 128)  # Gray for unknown region
        self._debug_images['unknown'] = unknown_bgr


        # ========== D. Iterate over detected labels, compute shape & assign color ==========

        h, w = mask_cleaned.shape[:2]
        for lbl in range(2, num_labels + 1):
            # Extract the region corresponding to the current label
            mask_region = (markers == lbl)
            region_area = np.count_nonzero(mask_region)

            # Skip if the region is too small
            if region_area < self.min_area:
                continue

            # Extract pixel coordinates belonging to this region
            region_pts = np.argwhere(mask_region)  # (y, x) format

            if len(region_pts) == 0:
                continue

            # Convert to (x, y) format
            coords_xy = np.fliplr(region_pts).astype(np.float32)  # Swap x and y

            # Prepare for minimum area rectangle computation
            coords_xy_list = coords_xy[:, np.newaxis, :]
            rect = cv2.minAreaRect(coords_xy_list)  # Compute minimum bounding box
            (rx, ry), (rw, rh), angle = rect

            # Ensure width is the larger dimension
            if rw < rh:
                rw, rh = rh, rw
                angle += 90

            # --- E. Compute color coverage to assign the best matching color ---

            # Create a binary mask for the detected region (same size as the original image)
            label_mask = np.zeros((h, w), dtype=np.uint8)
            label_mask[mask_region] = 255

            best_color: Color = self.detecting_colors[0]  # Default color
            best_coverage = 0.0

            for cdef in self.detecting_colors:
                # Compute overlap between label_mask and each color mask
                overlap_mask = cv2.bitwise_and(label_mask, color_masks[cdef.name])
                overlap_area = np.count_nonzero(overlap_mask)

                # Compute coverage ratio (how much of the region belongs to this color)
                coverage_ratio = overlap_area / float(region_area)

                # Select the color with the highest coverage
                if coverage_ratio > best_coverage:
                    best_coverage = coverage_ratio
                    best_color = cdef

            # --- F. Create Block object and append to results ---

            hsv_region = cv2.bitwise_and(hsv, hsv, mask=label_mask)

            mean_hsv = cv2.mean(hsv_region, mask=label_mask)[:3]
            h_channel, s_channel, v_channel = cv2.split(hsv_region)

            h_valid = h_channel[label_mask == 255].astype(np.float32)
            s_valid = s_channel[label_mask == 255].astype(np.float32)
            v_valid = v_channel[label_mask == 255].astype(np.float32)
            
            std_h = compute_hue_std_flip(h_valid, flip_threshold=90.0)
            std_s = float(np.std(s_valid))
            std_v = float(np.std(v_valid))

            contours, _ = cv2.findContours(
                label_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            largest_contour = max(contours, key=cv2.contourArea)

            block = Block(
                center=(rx, ry),
                size=(rw, rh),
                angle=angle,
                color=best_color, 
                mean_hsv=(mean_hsv[0], mean_hsv[1], mean_hsv[2]),
                color_std=(std_h, std_s, std_v),
                contour=largest_contour
            )
            blocks.append(block)

        return blocks


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
