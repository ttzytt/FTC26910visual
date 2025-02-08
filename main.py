import cv2
import numpy as np
from src.color_def import *
from src.detector_contour import ColorBlockDetectorContour
from src.detector_watershed import ColorBlockDetectorWatershed
from src.detector_meanshift import ColorBlockDetectorMeanShift
from src.visualizer import BlockVisualizer
from src.utils.serializer import *
# ---------- Global Color Definitions ----------

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    detector = ColorBlockDetectorWatershed(COLOR_DEF_R9000P)
    visualizer = BlockVisualizer()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        blocks = detector.process_frame(frame)
        debug_imgs = detector.get_debug_images()

        # Visualize based on the current mode
        visualizer.visualize(frame, blocks, debug_imgs) # type: ignore

        serialized_blocks = serialize_to_floats(blocks)
        print(serialized_blocks)
        deserailized_blocks = deserialize_from_floats(serialized_blocks)
        for block in deserailized_blocks:
            print(block)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('m'):
            # Toggle modes (0 -> 1 -> 0 -> 1)
            visualizer.toggle_mode()

    cap.release()
    cv2.destroyAllWindows()