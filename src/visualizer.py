import cv2
import numpy as np
from typing import List, Dict, Optional, TypedDict
from src.color_defs import *
from src.block import Block
from src.type_defs import *

NOT_HEADLESS = hasattr(cv2, 'imshow')

class BlockVisualizer:
    """
    Manages two modes:
      - Mode 0: Final detection only (single window)
      - Mode 1: Debug images (multiple windows, intermediate steps) + one additional window
                showing each block region filled with its average HSV color.

    Parameters:
        show (bool): Whether to display images using OpenCV's cv2.imshow(). If False, functions return images.
    """

    def __init__(self, show: bool = True):
        self.mode = 0
        self.prev_mode = 0  # Force initialization
        self.main_window = "Block Detection"
        self.show = show and NOT_HEADLESS

    def toggle_mode(self):
        """Switch between mode 0 and mode 1."""
        self.mode = (self.mode + 1) % 2

    def visualize(self, frame: np.ndarray, blocks: List[Block], debug_images: VizResults) -> VizResults:
        """
        Decide which visualization to show based on mode.
        Only destroy/recreate windows if the mode changed.

        Parameters:
            frame (np.ndarray): The original frame.
            blocks (List[Block]): List of detected blocks.
            debug_images (Dict[str, np.ndarray]): Dictionary of debug images.

        Returns:
            Optional[Dict[str, np.ndarray]]: If `self.show=False`, returns a dictionary of images.
        """
        if self.mode != self.prev_mode:
            cv2.destroyAllWindows()
            self.prev_mode = self.mode

        results : VizResults= {}
        final_result = self.gen_final_result(frame, blocks)
        results['final etection'] = final_result
        if self.mode == 1:
            # debug mode
            results['original'] = frame.copy()
            debug_outputs = self.gen_debug_imgs(debug_images)
            avg_hsv_image = self.gen_avg_hsv_fill(frame, blocks)

            for name, img in debug_outputs.items(): results[name] = img
            results['avg HSV'] = avg_hsv_image

        if self.show:
            for name, img in results.items():
                cv2.imshow(name, img)
        return results 

    def gen_final_result(self, frame: np.ndarray, blocks: List[Block]) -> np.ndarray:
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

        return output

    def gen_debug_imgs(self, debug_images: VizResults) -> Dict[str, np.ndarray]:
        """Display intermediate debug images, or return them if `self.show` is False."""
        results = {}
        for name, img in debug_images.items():
            if not isinstance(img, np.ndarray):
                continue
            display = img.copy()
            cv2.putText(display, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 255, 0), 2)
            
            results[name] = display

        return results

    def gen_avg_hsv_fill(self, frame: np.ndarray, blocks: List[Block]) -> np.ndarray:
        """
        Create a black canvas the same size as 'frame', then fill each block's contour
        with the block's average HSV color (converted to BGR). Show this in a new window
        or return the processed image.
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

        return canvas
