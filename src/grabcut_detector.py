import cv2
import numpy as np
from enum import Enum
from typing import List, Sequence
from src.detector import Detector
from src.color_defs import Color, compute_hue_std_flip
from src.block import Block
from src.type_defs import VizResults, img_t
from src.preprocessor import PreprocCfg


class GrabCutDetector(Detector):
    """
    Detect color blocks by GrabCut segmentation.
    """

    class DebugType(Enum):
        GRABCUT_MASK = "grabcut_mask"
        COMBINED_GRABCUT_MASK = "combined_grabcut_mask"

    def __init__(
        self,
        detecting_colors: List[Color],
        preproc_cfg: PreprocCfg = PreprocCfg(),
        debug_option: List[DebugType] | bool = []
    ):
        super().__init__(detecting_colors, preproc_cfg, debug_option, self.DebugType)
        self.iter_count = 5  # Number of GrabCut iterations

    def process_frame(self, frame: img_t) -> List[Block]:
        preprocessed = self._preprocess(frame)
        hsv = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2HSV)

        blocks: List[Block] = []
        combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

        for color_def in self.detecting_colors:
            mask_bgr = self._grabcut_single_color(preprocessed, color_def)
            mask_gray = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2GRAY)
            if self.DebugType.GRABCUT_MASK in self.debug_option:
                self.debug_images[f'grabcut_{color_def.name}'] = mask_bgr

            # Find contours and build blocks
            contours, _ = cv2.findContours(mask_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if cv2.contourArea(cnt) < 1000:
                    continue
                rect = cv2.minAreaRect(cnt)
                (cx, cy), (w, h), angle = rect
                # Orient
                if w < h:
                    w, h = h, w
                    angle += 90
                # Compute mean & std
                block = self._create_block(cnt, hsv, color_def, (cx, cy), (w, h), angle)
                if block:
                    blocks.append(block)

            combined_mask = cv2.bitwise_or(combined_mask, mask_gray)

        if self.DebugType.COMBINED_GRABCUT_MASK in self.debug_option:
            # Convert combined mask to BGR
            mask_bgr = cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2BGR)
            self.debug_images['combined_grabcut_mask'] = mask_bgr

        return blocks

    def _grabcut_single_color(self, image: np.ndarray, color_def: Color) -> np.ndarray:
        """
        Run GrabCut for a single color range.
        """
        # Create an approximate mask from color range
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        color_mask = self.create_color_mask(hsv, color_def)

        # Prepare GrabCut buffers
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)

        # Initial rectangle
        x, y, w, h = cv2.boundingRect(color_mask)
        if w == 0 or h == 0:
            return np.zeros_like(image)

        # Convert mask to mode
        gc_mask = np.where(color_mask > 0, cv2.GC_PR_FGD, cv2.GC_BGD).astype('uint8')
        cv2.grabCut(image, gc_mask, (x, y, w, h), bgdModel, fgdModel, self.iter_count, cv2.GC_INIT_WITH_MASK)

        # GC_PR_FGD or GC_FGD => definitely foreground
        final_mask = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype('uint8')
        output = cv2.bitwise_and(image, image, mask=final_mask)
        return output

    def _create_block(
        self,
        cnt: np.ndarray,
        hsv: np.ndarray,
        color_def: Color,
        center: tuple[float, float],
        size: tuple[float, float],
        angle: float
    ) -> Block | None:
        """
        Compute mean & std(H, S, V) inside contour, filter, and create a Block.
        """
        x_min, y_min, w_int, h_int = cv2.boundingRect(cnt)
        if w_int == 0 or h_int == 0:
            return None

        local_mask = np.zeros((h_int, w_int), dtype=np.uint8)
        shifted_cnt = cnt - [x_min, y_min]
        cv2.drawContours(local_mask, [shifted_cnt], 0, (255,), -1)

        hsv_roi = hsv[y_min:y_min + h_int, x_min:x_min + w_int]
        hsv_masked = cv2.bitwise_and(hsv_roi, hsv_roi, mask=local_mask)

        h_ch, s_ch, v_ch = cv2.split(hsv_masked)
        valid_idx = (local_mask == 255)
        if not np.any(valid_idx):
            return None

        h_valid = h_ch[valid_idx].astype(float)
        s_valid = s_ch[valid_idx].astype(float)
        v_valid = v_ch[valid_idx].astype(float)

        mean_h = float(np.mean(h_valid))
        mean_s = float(np.mean(s_valid))
        mean_v = float(np.mean(v_valid))

        std_h = compute_hue_std_flip(h_valid, flip_threshold=90.0)
        std_s = float(np.std(s_valid))
        std_v = float(np.std(v_valid))

        # Simple filtering example
        if std_h > 15 or std_s > 50 or std_v > 50:
            return None

        block = Block(
            center=center,
            size=size,
            angle=angle,
            color=color_def,
            color_std=(std_h, std_s, std_v),
            mean_hsv=(mean_h, mean_s, mean_v),
            contour=cnt
        )
        return block