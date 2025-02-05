import cv2
import numpy as np
import enum
from typing import List, Tuple

class Color(enum.Enum):
    RED = {'hsv_ranges': [( (0, 100, 100), (10, 255, 255) ), ( (160, 100, 100), (180, 255, 255) )], 'bgr': (0, 0, 255)}
    # GREEN = {'hsv_ranges': [( (35, 100, 100), (85, 255, 255) )], 'bgr': (0, 255, 0)}
    BLUE = {'hsv_ranges': [( (100, 100, 100), (130, 255, 255) )], 'bgr': (255, 0, 0)}

class Block:
    def __init__(self, center: Tuple[float, float], size: Tuple[float, float], angle: float, color: Color):
        self.center = center  # (x, y)
        self.size = size      # (width, height)
        self.angle = angle    # degree
        self.color = color

class ColorBlockDetector:
    def __init__(self):
        # Image processing parameters
        self.blur_size = 35
        self.brightness = -10
        self.erode_iter = 1
        self.dilate_iter = 1
        self.min_contour_area = 100
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        
        # Color detection parameters
        self.color_margin = {
            'H': 5,
            'S': 5,
            'V': 50
        }

    def process_frame(self, frame: np.ndarray) -> List[Block]:
        processed = self._preprocess(frame)
        hsv = cv2.cvtColor(processed, cv2.COLOR_BGR2HSV)
        return self._detect_blocks(hsv)

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        frame = cv2.convertScaleAbs(frame, alpha=1, beta=self.brightness)
        if self.blur_size > 0:
            kernel_size = self.blur_size | 1  # Ensure odd
            return cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)
        return frame

    def _detect_blocks(self, hsv: np.ndarray) -> List[Block]:
        blocks = []
        for color in Color:
            mask = self._create_color_mask(hsv, color)
            contours = self._find_contours(mask)
            blocks += self._process_contours(contours, color)
        return blocks

    def _create_color_mask(self, hsv: np.ndarray, color: Color) -> np.ndarray:
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in color.value['hsv_ranges']:
            lower = np.array([
                max(0, lower[0] - self.color_margin['H']),
                max(0, lower[1] - self.color_margin['S']),
                max(0, lower[2] - self.color_margin['V'])
            ])
            upper = np.array([
                min(180, upper[0] + self.color_margin['H']),
                min(255, upper[1] + self.color_margin['S']),
                min(255, upper[2] + self.color_margin['V'])
            ])
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))
        
        mask = cv2.erode(mask, self.kernel, iterations=self.erode_iter)
        mask = cv2.dilate(mask, self.kernel, iterations=self.dilate_iter)
        return mask

    def _find_contours(self, mask: np.ndarray):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [c for c in contours if cv2.contourArea(c) > self.min_contour_area]

    def _process_contours(self, contours: List[np.ndarray], color: Color) -> List[Block]:
        blocks = []
        for cnt in contours:
            rect = cv2.minAreaRect(cnt)
            (x, y), (w, h), angle = rect
            if w < h:  # Ensure width is always the longer side
                w, h = h, w
                angle += 90
            blocks.append(Block((x, y), (w, h), angle, color))
        return blocks

class BlockVisualizer:
    @staticmethod
    def draw_blocks(frame: np.ndarray, blocks: List[Block]) -> np.ndarray:
        for block in blocks:
            box = cv2.boxPoints( (block.center, block.size, block.angle) )
            box = np.int0(box)
            cv2.drawContours(frame, [box], 0, block.color.value['bgr'], 2)
            
            text = f"{block.angle:.1f} deg"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(frame,
                        (int(block.center[0]-text_size[0]/2), int(block.center[1]-text_size[1]/2)),
                        (int(block.center[0]+text_size[0]/2), int(block.center[1]+text_size[1]/2)),
                        block.color.value['bgr'], -1)
            cv2.putText(frame, text,
                      (int(block.center[0]-text_size[0]/2), int(block.center[1]+text_size[1]/2)),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
        return frame


def runPipeline(image, llrobot):
    detector = ColorBlockDetector()
    visualizer = BlockVisualizer()

    blocks = detector.process_frame(image)
    image = visualizer.draw_blocks(image, blocks)

    return [], image, []

