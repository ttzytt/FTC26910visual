import cv2
import numpy as np
from .color_def import *
from .block import Block

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
            box = cv2.boxPoints(
                (block.center, block.size, block.angle))  # type: ignore
            box = np.intp(box)
            cv2.drawContours(output, [box], 0, block.color.bgr, 2)  # type: ignore

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
            hsv_pixel = np.uint8(
                [[[block.mean_hsv[0], block.mean_hsv[1], block.mean_hsv[2]]]])  # type: ignore
            bgr_pixel = cv2.cvtColor(
                hsv_pixel, cv2.COLOR_HSV2BGR)  # type: ignore
            avg_color = (int(bgr_pixel[0, 0, 0]), int(
                bgr_pixel[0, 0, 1]), int(bgr_pixel[0, 0, 2]))

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
