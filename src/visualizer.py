# -------------------- visualizer.py --------------------
import cv2
import numpy as np
from typing import List, Dict
from src.color_defs import *
from src.block import Block
from src.type_defs import *
from src.detector import Detector
from src.preprocessor import PreprocType
from src.debug_controls import DebugControlWidget

try:
    from PySide6.QtWidgets import QApplication
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False

NOT_HEADLESS = hasattr(cv2, 'imshow')


class BlockVisualizer:
    """
    Manages visualization modes with PySide6 debug controls
    """

    def __init__(self, detector: Detector, show: bool = True):
        self.mode = 0
        self.prev_mode = 0
        self.main_window = "Block Detection"
        self.show = show and NOT_HEADLESS
        self.detector = detector

        if PYQT_AVAILABLE:
            # Initialize GUI components
            self.qt_app = QApplication.instance() or QApplication([])
            self.debug_controls = DebugControlWidget(detector.DebugType)
            self.debug_controls.options_changed.connect(
                self._update_debug_options)

            # Initialize states
            self._init_states()
            self.active_windows = set()

            self.overlay_dbg_img_on_frame = True
            self.overlay_alpha = 0.5
        else:
            self.qt_app = None
            self.debug_controls = None
            self.active_windows = set()
            self.overlay_dbg_img_on_frame = False
            self.overlay_alpha = 0.5

    def _init_states(self):
        if not PYQT_AVAILABLE:
            return

        # Detector states
        detector_states = {
            opt: opt in self.detector.debug_option
            for opt in self.detector.DebugType
        }

        # Preprocessor states
        assert isinstance(self.detector.preproc.cfg.debug_steps, set)
        preproc_states = {
            step: step in self.detector.preproc.cfg.debug_steps
            for step in PreprocType
        }

        assert self.debug_controls is not None
        self.debug_controls.set_states(detector_states, preproc_states)

    def toggle_mode(self):
        """Switch between mode 0 and mode 1."""
        if not PYQT_AVAILABLE:
            return
        assert self.debug_controls is not None
        self.mode = (self.mode + 1) % 2
        if self.mode == 1:
            self.debug_controls.show()
        else:
            self.debug_controls.hide()

        cv2.destroyAllWindows()
        self.active_windows.clear()

    def visualize(self, frame: np.ndarray, blocks: List[Block]) -> VizResults:
        """Main visualization processing"""
        if PYQT_AVAILABLE:
            assert self.qt_app is not None
            self.qt_app.processEvents()

        debug_images = self.detector.debug_images

        results: VizResults = {}

        # Generate final result
        final_result = self.gen_final_result(frame, blocks)
        if self._valid_image(final_result):
            results['final detection'] = final_result

        if PYQT_AVAILABLE:
            # Handle debug mode
            if self.mode == 1:
                if self._valid_image(frame):
                    results['original'] = frame.copy()

                debug_outputs = self.gen_debug_imgs(debug_images, frame)
                results.update(debug_outputs)

            # Update windows
            if self.show:
                current_windows = set(results.keys())
                if self.mode == 1:
                    current_windows.add("Debug Controls")

                # Close unused windows
                for win in self.active_windows - current_windows:
                    try:
                        cv2.destroyWindow(win)
                    except cv2.error:
                        pass

                # Update OpenCV windows
                for name, img in results.items():
                    if self._valid_image(img):
                        cv2.imshow(name, img)

                self.active_windows = current_windows

        return results

    def _update_debug_options(self):
        """Update detector with current GUI states"""
        if not PYQT_AVAILABLE:
            return
        assert self.debug_controls is not None
        detector_states, preproc_states = self.debug_controls.get_states()

        # Update detector
        self.detector.debug_option = [ # type: ignore
            opt for opt, enabled in detector_states.items() if enabled
        ]

        # Update preprocessor
        self.detector.preproc.cfg.debug_steps = {
            step for step, enabled in preproc_states.items() if enabled
        }

        print(f"Detector: {self.detector.debug_option}")
        print(f"Preprocessor: {self.detector.preproc.cfg.debug_steps}")

    def _valid_image(self, img: np.ndarray) -> bool:
        """Validate image dimensions"""
        return (
            isinstance(img, np.ndarray) and
            img.size > 0 and
            img.shape[0] > 0 and
            img.shape[1] > 0
        )

    @staticmethod
    def gen_final_result(frame: np.ndarray, blocks: List[Block]) -> np.ndarray:
        """Draw bounding boxes and put text for each block."""
        output = frame.copy()
        for block in blocks:
            box = cv2.boxPoints(
                (block.center, block.size, block.angle))  # type: ignore
            box = np.intp(box)
            cv2.drawContours(output, [box], 0,
                             block.color.bgr, 2)  # type: ignore

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

        return output

    def gen_debug_imgs(self, debug_images: VizResults, frame : img_t) -> Dict[str, np.ndarray]:
        """
        Only display debug images for which the corresponding checkbox is selected.
        For each debug image, if its name does not start with any of the selected keys
        (and if it has a color suffix, the suffix is separated by an underscore), 
        then the corresponding window is destroyed.
        """
        results = {}
        selected_keys = {opt.value for opt in self.detector.debug_option}
        assert isinstance(self.detector.preproc.cfg.debug_steps, set)
        selected_keys.update({step.value for step in self.detector.preproc.cfg.debug_steps})
        for name, img in debug_images.items():
            selected = False
            for key in selected_keys:
                # Check if the debug image name is exactly the key or starts with "key_"
                if name == key or name.startswith(f"{key}_"):
                    selected = True
                    break
    
            if not selected:
                continue
    
            if self._valid_image(img):
                display = img.copy()
                cv2.putText(display, name, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                results[name] = display
    
        if self.overlay_dbg_img_on_frame:
            for name, img in results.items():
                if name == 'final detection':
                    continue
                results[name] = cv2.addWeighted(frame, self.overlay_alpha, img, 1 - self.overlay_alpha, 0)

        return results
