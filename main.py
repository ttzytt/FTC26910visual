import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple
from src.color_def import *
from src.detector import ColorBlockDetector
from src.visualizer import BlockVisualizer

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