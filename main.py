import cv2
import numpy as np
from src.color_def import *
from src.detector import ColorBlockDetector
from src.visualizer import BlockVisualizer
from src.utils.serializer import *

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
        blocks_json = json.dumps([block.__dict__ for block in blocks], cls=NumpyEncoder)
        encoded = string_to_doubles(blocks_json)
        decoded = doubles_to_string(encoded)
        print (decoded)

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