import cv2
import numpy as np
from src.color_defs import *
from src.color_detector import ColorDetector
from src.watershed_detector import WatershedDetector
from src.meanshift_detector import MeanshiftDetector
from src.visualizer import BlockVisualizer
from src.utils.serializer import *
from src.preprocessor import *
# ---------- Global Color Definitions ----------

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    preproc_cfg = PreprocCfg(debug_steps=True)
    detector = WatershedDetector(COLOR_DEF_R9000P, preproc_cfg, True)
    visualizer = BlockVisualizer()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        blocks = detector.process_frame(frame)
        debug_imgs = detector.debug_images

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