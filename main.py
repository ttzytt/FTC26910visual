import cv2
import numpy as np
from src.color_def import *
from src.detector_contour import ColorBlockDetectorContour
from src.detector_watershed import ColorBlockDetectorWatershed
from src.detector_meanshift import ColorBlockDetectorMeanShift
from src.visualizer import BlockVisualizer
from src.utils.serializer import *
# ---------- Global Color Definitions ----------

RED = Color(
    name='RED',
    hsv_ranges=[
        ((0, 50, 100), (10, 200, 255)),
        ((160, 50, 100), (180, 200, 255))
    ],
    bgr=(0, 0, 255)
)

BLUE = Color(
    name='BLUE',
    hsv_ranges=[
        ((100, 50, 50), (120, 255, 255))
    ],
    bgr=(255, 0, 0)
)

YELLOW = Color(
    name='YELLOW',
    hsv_ranges=[
        ((20, 50, 100), (30, 255, 255))
    ],
    bgr=(0, 255, 255)
)

COLOR_DEFINITIONS = [BLUE]

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    detector = ColorBlockDetectorWatershed(COLOR_DEFINITIONS)
    visualizer = BlockVisualizer()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        blocks = detector.process_frame(frame)
        debug_imgs = detector.get_debug_images()

        # Visualize based on the current mode
        visualizer.visualize(frame, blocks, debug_imgs) # type: ignore

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('m'):
            # Toggle modes (0 -> 1 -> 0 -> 1)
            visualizer.toggle_mode()

    cap.release()
    cv2.destroyAllWindows()