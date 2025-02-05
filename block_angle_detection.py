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

@dataclass
class Block:
    """Represents a detected color block with position, size, angle, color info, and HSV stats."""
    center: Tuple[float, float]
    size: Tuple[float, float]
    angle: float
    color: Color
    color_std: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    mean_hsv: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    contour: np.ndarray = field(default=None)  # store the absolute contour for visualization


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
    
    return min(std1, std2)



class ColorBlockDetector:
    """
    Detects color blocks by:
      1) Preprocessing (brightness, blur)
      2) Creating color masks
      3) Finding contours
      4) Computing mean & std(H, S, V) inside each contour
    """
    def __init__(self):
        # Basic image processing parameters
        self.blur_size = 35
        self.brightness = 0
        self.erode_iter = 7
        self.dilate_iter = 6
        self.min_contour_area = 1000
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        
        # Tolerance margins for HSV
        self.color_margin = {
            'H': 3,
            'S': 20,
            'V': 150
        }

        # Thresholds for std(H, S, V)
        self.std_threshold_hsv = (3, 50, 50)

        # Storage for debug images (intermediate steps)
        self._debug_images = {}

    def process_frame(self, frame: np.ndarray) -> List[Block]:
        """Main entry: preprocess and detect blocks, while saving debug images."""
        self._debug_images.clear()  # clear from previous frame

        # Save original frame
        self._debug_images['original'] = frame.copy()

        # 1) Preprocessing
        preprocessed = self._preprocess(frame)
        self._debug_images['preprocessed'] = preprocessed.copy()

        # 2) Convert to HSV
        hsv = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2HSV)
        hsv_bgr_like = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)  # just for visualization
        self._debug_images['hsv_space'] = hsv_bgr_like

        # 3) Detect blocks
        blocks = self._detect_blocks(hsv)

        return blocks

    def get_debug_images(self) -> dict:
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

        for color_def in COLOR_DEFINITIONS:
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
            lower_margin = np.array([
                max(0, lower[0] - self.color_margin['H']),
                max(0, lower[1] - self.color_margin['S']),
                max(0, lower[2] - self.color_margin['V'])
            ], dtype=np.uint8)
            
            upper_margin = np.array([
                min(180, upper[0] + self.color_margin['H']),
                min(255, upper[1] + self.color_margin['S']),
                min(255, upper[2] + self.color_margin['V'])
            ], dtype=np.uint8)
            
            # Apply threshold to get binary mask
            tmp_mask = cv2.inRange(hsv, lower_margin, upper_margin)
            mask = cv2.bitwise_or(mask, tmp_mask)

        # Step 3: For debug: raw mask in color
        hsv_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        raw_mask_colored = cv2.bitwise_and(hsv_bgr, hsv_bgr, mask=mask)
        debug_raw = f"{color_def.name.lower()}_mask_raw"
        self._debug_images[debug_raw] = raw_mask_colored

        # Step 4: Morphological operations (erode & dilate)
        mask_morph = cv2.erode(mask, self.kernel, iterations=self.erode_iter)
        mask_morph = cv2.dilate(mask_morph, self.kernel, iterations=self.dilate_iter)

        # Step 5: For debug: morph mask in color
        morph_mask_colored = cv2.bitwise_and(hsv_bgr, hsv_bgr, mask=mask_morph)
        debug_morph = f"{color_def.name.lower()}_mask_morph"
        self._debug_images[debug_morph] = morph_mask_colored

        return mask_morph

    def _find_contours(self, mask: np.ndarray) -> List[np.ndarray]:
        """Find external contours with area > min_contour_area."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
            cv2.drawContours(contour_mask, [shifted_cnt], 0, 255, -1)

            # Extract HSV ROI
            hsv_roi = hsv[y_min:y_min + h_int, x_min:x_min + w_int]
            hsv_masked = cv2.bitwise_and(hsv_roi, hsv_roi, mask=contour_mask)

            # Split channels and extract valid pixels
            h_ch, s_ch, v_ch = cv2.split(hsv_masked)
            h_valid = h_ch[contour_mask == 255]
            s_valid = s_ch[contour_mask == 255]
            v_valid = v_ch[contour_mask == 255]

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
                    contour=cnt  # store the original contour (absolute coordinates)
                )
                blocks.append(block)
        return blocks


# ---------- Block Visualizer ----------

class BlockVisualizer:
    """
    Manages two modes:
      - Mode 0: Final detection only (single window)
      - Mode 1: Debug images (multiple windows, intermediate steps) + one additional window
                showing each block region filled with its average HSV color.
    """
    def __init__(self):
        self.mode = 0
        self.prev_mode = -1  # force initialization
        self.main_window = "Block Detection"

    def toggle_mode(self):
        # only two modes for now: 0 -> 1 -> 0 -> 1 ...
        self.mode = (self.mode + 1) % 2

    def visualize(self, frame: np.ndarray, blocks: List[Block], debug_images: dict):
        """
        Decide which visualization to show based on mode.
        Only destroy/recreate windows if the mode changed.
        """
        if self.mode != self.prev_mode:
            cv2.destroyAllWindows()
            self.prev_mode = self.mode

        if self.mode == 0:
            # Show final detection
            self.show_final_result(frame, blocks)
        else:
            # Show debug images + a window for average HSV fill
            self.show_debug_images(debug_images)
            self.show_avg_hsv_fill(frame, blocks)

    def show_final_result(self, frame: np.ndarray, blocks: List[Block]):
        """Draw bounding boxes and put text for each block."""
        output = frame.copy()
        for block in blocks:
            box = cv2.boxPoints((block.center, block.size, block.angle))
            box = np.int0(box)
            cv2.drawContours(output, [box], 0, block.color.bgr, 2)

            # Text lines with extra info: avgH, avgS, avgV
            lines = [
                f"{block.color.name}: {block.angle:.1f} deg",
                f"stdHSV=({block.color_std[0]:.1f}, {block.color_std[1]:.1f}, {block.color_std[2]:.1f})",
                f"avgHSV=({block.mean_hsv[0]:.1f}, {block.mean_hsv[1]:.1f},{block.mean_hsv[2]:.1f})"
            ]
            x0, y0 = int(block.center[0]), int(block.center[1])
            for i, line in enumerate(lines):
                offset_y = i * 15
                cv2.putText(
                    output,
                    line,
                    (x0, y0 + offset_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1
                )

        cv2.putText(
            output,
            "Final Detection",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2
        )
        cv2.imshow(self.main_window, output)

    def show_debug_images(self, debug_images: dict):
        """Display intermediate debug images, each in its own window."""
        for name, img in debug_images.items():
            if img is None:
                continue
            display = img.copy()
            cv2.putText(display, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 255, 0), 2)
            cv2.imshow(name, display)

    def show_avg_hsv_fill(self, frame: np.ndarray, blocks: List[Block]):
        """
        Create a black canvas the same size as 'frame', then fill each block's contour
        with the block's average HSV color (converted to BGR). Show this in a new window.
        """
        canvas = np.zeros_like(frame)  # black canvas
        for block in blocks:
            # Convert mean_hsv -> BGR
            hsv_pixel = np.uint8([[[block.mean_hsv[0], block.mean_hsv[1], block.mean_hsv[2]]]])
            bgr_pixel = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)
            avg_color = (int(bgr_pixel[0,0,0]), int(bgr_pixel[0,0,1]), int(bgr_pixel[0,0,2]))

            # Fill the contour with this color
            cv2.drawContours(canvas, [block.contour], 0, avg_color, -1)

        cv2.putText(
            canvas,
            "Blocks filled w/ average HSV",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2
        )
        cv2.imshow("Avg HSV Debug", canvas)


# ---------- Main Loop ----------

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    detector = ColorBlockDetector()
    visualizer = BlockVisualizer()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        blocks = detector.process_frame(frame)
        debug_imgs = detector.get_debug_images()

        # Visualize based on the current mode
        visualizer.visualize(frame, blocks, debug_imgs)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('m'):
            # Toggle modes (0 -> 1 -> 0 -> 1)
            visualizer.toggle_mode()

    cap.release()
    cv2.destroyAllWindows()
