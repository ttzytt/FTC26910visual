import cv2
import numpy as np
from enum import Enum
from typing import List, Sequence, Type 
from src.detector import *  # Assume this defines a base Detector class
from src.type_defs import VizResults, img_t
from src.color_defs import Color, compute_hue_std_flip
from src.block import Block
# or wherever your PreprocCfg is defined
from src.preprocessor import *


class WatershedDetector(Detector):
    """
    Detect color blocks by:
      1) Using parent-class preprocessing (brightness, blur, etc.)
      2) Watershed-based segmentation for each color
      3) Building Block objects from labeled regions
    """

    class DebugType(Enum):
        SINGLE_COLOR_MASK = "single_color_mask"
        COMBINED_MASK = "combined_mask"
        WATERSHED_BG = "watershed_bg"
        COMBINED_WATERSHED_BG = "combined_watershed_bg"
        WATERSHED_FG = "watershed_fg"
        COMBINED_WATERSHED_FG = "combined_watershed_fg"
        WATERSHED_DIST_TRANSFORM = "watershed_dist_transform"
        COMBINED_WATERSHED_DIST_TRANSFORM = "combined_watershed_dist_transform"
        WATERSHED_UNKNOWN = "watershed_unknown"
        COMBINED_WATERSHED_UNKNOWN = "combined_watershed_unknown"
        WATERSHED_SEG = "watershed_seg"
        COMBINED_WATERSHED_SEG = "combined_watershed_seg"
    

    def __init__(
        self,
        detecting_colors: List[Color],
        preproc_cfg: PreprocCfg = PreprocCfg(),
        debug_option: List[DebugType] | bool = []
    ) -> None:
        """
        :param detecting_colors: List of Color objects to detect
        :param preproc_cfg: Preprocessing configuration (inherited from parent)
        :param debug_option: Which debug images to store
        """
        super().__init__(detecting_colors, preproc_cfg, debug_option, self.DebugType)

        # Additional WatershedDetector-specific parameters
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.mask_erode_iter = 2
        self.mask_dilate_iter = 2
        self.min_area = 1000   # minimum region area
        self.sure_fg_min_dis_ratio = 0.7
        self.std_threshold_hsv = (3, 50, 50)  # H, S, V thresholds
        self.debug_option : List[WatershedDetector.DebugType]

    def process_frame(self, frame: img_t) -> List[Block]:
        """
        Main entry point. Uses parent-class preprocessing,
        then applies watershed-based segmentation per color.
        """
        # if selected watershed and it is combined, then in debug_option have to
        # select the single-color version 

        combined_watershed_dbg_imgs = [dbg for dbg in self.debug_option if dbg.value.startswith("combined_watershed")]
        for dbg in combined_watershed_dbg_imgs:
            non_combined = self.DebugType[dbg.value.replace("combined_", "").upper()]
            if non_combined not in self.debug_option:
                print(f"Warning: {dbg.value} is selected but {non_combined.value} is not. ")

        preprocessed = self._preprocess(frame)
        hsv = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2HSV)
        blocks = self._detect_blocks(hsv, preprocessed)

        watershed_dbg_imgs_to_combine = [step for step in self.DebugType if 'watershed' in step.value]
        watershed_dbg_imgs_to_combine = [step for step in watershed_dbg_imgs_to_combine if f"combined_{step.value}" 
                                         in [dbg_type.value for dbg_type in self.debug_option]]

        for step in watershed_dbg_imgs_to_combine:
            self.debug_images[f"combined_{step.value}"] = self._merge_debug_imgs(
                step.value)

        return blocks

    def _detect_blocks(self, hsv: img_t, frame_bgr: img_t) -> List[Block]:
        """
        Perform watershed segmentation for each color in self.detecting_colors.
        """
        all_blocks: List[Block] = []

        # (Optional) create a combined mask for debugging
        combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

        for color_def in self.detecting_colors:
            # A) Create color-specific mask
            color_mask = self.create_color_mask(hsv, color_def)

            # Debug: store single color mask
            if self.DebugType.SINGLE_COLOR_MASK in self.debug_option:
                self.debug_images[f"{color_def.name}_mask"] = cv2.cvtColor(
                    color_mask, cv2.COLOR_GRAY2BGR)

            # Combine to the overall mask (optional, for debugging)
            combined_mask = cv2.bitwise_or(combined_mask, color_mask)

            # B) Watershed on this color mask
            markers, num_labels, sure_bg, sure_fg, unknown = self._watershed_segment(
                color_mask, color_def, frame_bgr)

            # Debug images for each color
            # Convert to BGR for consistent display
            unknown_bgr = np.zeros_like(frame_bgr)
            unknown_bgr[unknown == 255] = (128, 128, 128)

            # C) Build blocks from each labeled region
            h, w = hsv.shape[:2]
            for lbl in range(2, num_labels + 1):
                region_mask = (markers == lbl)
                region_area = np.count_nonzero(region_mask)
                if region_area < self.min_area:
                    continue

                # Extract region coordinates
                region_pts = np.argwhere(region_mask)
                if len(region_pts) == 0:
                    continue

                # minAreaRect
                coords_xy = np.fliplr(region_pts).astype(np.float32)  # (x, y)
                coords_xy_list = coords_xy[:, np.newaxis, :]
                rect = cv2.minAreaRect(coords_xy_list)
                (rx, ry), (rw, rh), angle = rect
                # Normalize orientation
                if rw < rh:
                    rw, rh = rh, rw
                    angle += 90

                # Create label_mask for HSV stats
                label_mask = np.zeros((h, w), dtype=np.uint8)
                label_mask[region_mask] = 255

                hsv_region = cv2.bitwise_and(hsv, hsv, mask=label_mask)
                # Split channels
                h_ch, s_ch, v_ch = cv2.split(hsv_region)
                h_valid = h_ch[label_mask == 255].astype(np.float32)
                s_valid = s_ch[label_mask == 255].astype(np.float32)
                v_valid = v_ch[label_mask == 255].astype(np.float32)

                # Compute mean, std
                mean_hsv = cv2.mean(hsv_region, mask=label_mask)[:3]
                std_h = float(compute_hue_std_flip(h_valid))
                std_s = float(np.std(s_valid))
                std_v = float(np.std(v_valid))

                # Filter by thresholds
                if std_h <= self.std_threshold_hsv[0] and \
                   std_s <= self.std_threshold_hsv[1] and \
                   std_v <= self.std_threshold_hsv[2]:
                    # retrieve contour if needed
                    contours, _ = cv2.findContours(
                        label_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if not contours:
                        continue
                    largest_contour = max(contours, key=cv2.contourArea)

                    block = Block(
                        center=(rx, ry),
                        size=(rw, rh),
                        angle=angle,
                        color=color_def,
                        mean_hsv=(mean_hsv[0], mean_hsv[1], mean_hsv[2]),
                        color_std=(std_h, std_s, std_v),
                        contour=largest_contour
                    )
                    all_blocks.append(block)

        # Optionally store the combined mask in debug
        if self.DebugType.COMBINED_MASK in self.debug_option:
            combined_bgr = cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2BGR)
            self.debug_images['combined_mask'] = combined_bgr

        return all_blocks

    def _watershed_segment(
        self,
        mask: np.ndarray,
        color_def: Color,
        frame_bgr: np.ndarray
    ) -> tuple[np.ndarray, int, np.ndarray, np.ndarray, np.ndarray]:
        """
        Performs distance transform and watershed segmentation.
        Returns (markers, num_labels, sure_bg, sure_fg, unknown).
        Stores debug images if enabled in debug_option.
        """

        color_name = color_def.name

        # ------------------- Step 1: Obtain sure background (BG) -------------------
        sure_bg = cv2.dilate(mask, self.kernel, iterations=5)

        if self.DebugType.WATERSHED_BG in self.debug_option:
            # Convert to BGR format and store
            sure_bg_bgr = cv2.cvtColor(sure_bg, cv2.COLOR_GRAY2BGR)
            sure_bg_bgr[sure_bg == 255] = color_def.bgr
            self.debug_images[f"watershed_bg_{color_name}"] = sure_bg_bgr

        # ------------------- Step 2: Distance transform to obtain sure foreground (FG) -------------------
        dist_transform = cv2.distanceTransform(
            mask, cv2.DIST_L2, cv2.DIST_MASK_5)

        # Normalize and apply color map for visualization
        if self.DebugType.WATERSHED_DIST_TRANSFORM in self.debug_option:
            dist_vis = cv2.normalize(dist_transform, dist_transform.copy(),
                                     0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            dist_color = cv2.applyColorMap(dist_vis, cv2.COLORMAP_JET)
            dist_color[mask == 0] = 0  # Set BG to black
            self.debug_images[f"watershed_dist_transform_{color_name}"] = dist_color

        max_val = dist_transform.max()
        _, sure_fg = cv2.threshold(dist_transform,
                                   self.sure_fg_min_dis_ratio * max_val,
                                   255, 0)
        sure_fg = sure_fg.astype(np.uint8)

        if self.DebugType.WATERSHED_FG in self.debug_option:
            # Convert FG to BGR format and store
            sure_fg_bgr = cv2.cvtColor(sure_fg, cv2.COLOR_GRAY2BGR)
            sure_fg_bgr[sure_fg == 255] = color_def.bgr
            self.debug_images[f"watershed_fg_{color_name}"] = sure_fg_bgr

        # ------------------- Step 3: Compute unknown region -------------------
        unknown = cv2.subtract(sure_bg, sure_fg)

        if self.DebugType.WATERSHED_UNKNOWN in self.debug_option:
            # Mark unknown region in gray
            unknown_bgr = np.zeros_like(frame_bgr)
            # Gray color for unknown area
            unknown_bgr[unknown == 255] = (128, 128, 128)
            self.debug_images[f"watershed_unknown_{color_name}"] = unknown_bgr

        # ------------------- Step 4: Label foreground using connected components -------------------
        num_labels, markers = cv2.connectedComponents(sure_fg)
        markers += 1  # Ensure background has label 1
        markers[unknown == 255] = 0  # Mark unknown region with 0

        # ------------------- Step 5: Apply watershed algorithm -------------------
        cv2.watershed(frame_bgr, markers)

        # ------------------- Step 6: Visualize segmentation results -------------------
        if self.DebugType.WATERSHED_SEG in self.debug_option:
            seg_vis = np.zeros_like(frame_bgr)

            rng = np.random.default_rng(42)  # fixed seed for reproducibility
            label_colors = {}

            # Generate a unique random BGR color for each label
            for lbl in range(2, num_labels + 2):
                color = rng.integers(0, 256, size=3, dtype=np.uint8)
                label_colors[lbl] = color

            # Overlay color for each labeled region using array mask
            for lbl in range(2, num_labels + 1):
                region_mask = (markers == lbl)  # boolean mask of shape (h, w)
                if not np.any(region_mask):
                    continue
                seg_vis[region_mask] = label_colors[lbl]

            self.debug_images[f"watershed_seg_{color_name}"] = seg_vis

        return markers, num_labels, sure_bg, sure_fg, unknown
