import cv2
import numpy as np
from typing import List, Dict
from dataclasses import dataclass, field
from typing import Tuple

# ------------------------------------------------------------------
# (1) Color + Block definitions
# ------------------------------------------------------------------


@dataclass
class Color:
    """Stores color name, HSV ranges, and BGR values for drawing."""
    name: str
    hsv_ranges: List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]
    bgr: Tuple[int, int, int]


@dataclass
class Block:
    """
    Represents a detected color block with position, size, angle, color info,
    and optional HSV stats or contour.
    """
    center: Tuple[float, float]
    size: Tuple[float, float]
    angle: float
    color: Color
    color_std: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    mean_hsv: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    contour: np.ndarray = field(default_factory=lambda: np.array([]))


# Example color definitions
RED = Color(
    name='RED',
    hsv_ranges=[
        ((0, 80, 50), (20, 200, 255)),
        ((160, 80, 50), (180, 200, 255))
    ],
    bgr=(0, 0, 255)
)

BLUE = Color(
    name='BLUE',
    hsv_ranges=[
        ((80, 50, 50), (140, 255, 255))
    ],
    bgr=(255, 0, 0)
)

YELLOW = Color(
    name='YELLOW',
    hsv_ranges=[
        ((20, 20, 30), (60, 255, 255))
    ],
    bgr=(0, 255, 255)
)

COLOR_DEFINITIONS = [YELLOW]

# ------------------------------------------------------------------
# (2) Detector Class with Debug
# ------------------------------------------------------------------


class BlobFloodFillDetector:
    """
    Detect color blocks by:
      1) Creating color mask in HSV
      2) Blob detection to find keypoints
      3) Flood fill from each keypoint to get the full region
      4) minAreaRect to form a Block

    Also stores intermediate debug images for each step.
    """

    def __init__(self, debug: bool = True):
        # Basic morphological parameters
        self.erode_iter = 2
        self.dilate_iter = 2
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        # For blob detection
        self.min_area = 500  # minimum area for a blob
        self.debug = debug
        # We'll store debug images in a dictionary
        self.debug_images: Dict[str, np.ndarray] = {}

    def detect_blocks(self, frame_bgr: np.ndarray) -> List[Block]:
        """
        Main entry point. Returns the list of Blocks and
        populates self.debug_images with intermediate steps.
        """
        # Clear debug images each time
        self.debug_images.clear()

        # Convert to HSV
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

        blocks: List[Block] = []

        for color_def in COLOR_DEFINITIONS:
            # 1) Create the color mask
            mask = self._create_color_mask(hsv, color_def)

            # 2) Detect blobs on this mask
            keypoints = self._detect_blobs(mask)

            # Save a debug image showing keypoints
            if self.debug:
                debug_kp = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                print("keypoints count: ", len(keypoints))
                for kp in keypoints:
                    x, y = int(kp.pt[0]), int(kp.pt[1])
                    cv2.circle(debug_kp, (x, y), 5, (0, 255, 0), -1)
                self.debug_images[f"{color_def.name}_blob_keypoints"] = debug_kp

            # 3) For each keypoint => flood fill => find contour => build Block
            idx_blob = 0
            for kp in keypoints:
                sub_blocks = self._floodfill_contour_and_build(
                    mask, kp.pt, color_def, idx_blob)
                blocks.extend(sub_blocks)
                idx_blob += 1

        return blocks

    def _create_color_mask(self, hsv: np.ndarray, color_def: Color) -> np.ndarray:
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for (lower, upper) in color_def.hsv_ranges:
            lower_np = np.array(lower, dtype=np.uint8)
            upper_np = np.array(upper, dtype=np.uint8)
            tmp_mask = cv2.inRange(hsv, lower_np, upper_np)
            mask = cv2.bitwise_or(mask, tmp_mask)

        # morphological cleanup
        mask = cv2.erode(mask, self.kernel, iterations=self.erode_iter)
        mask = cv2.dilate(mask, self.kernel, iterations=self.dilate_iter)

        # Save mask as debug image if needed
        if self.debug:
            c_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            self.debug_images[f"{color_def.name}_mask"] = c_bgr

        return mask

    def _detect_blobs(self, mask: np.ndarray) -> List[cv2.KeyPoint]:
        params = cv2.SimpleBlobDetector_Params()

        # 1) Basic area filter
        params.filterByArea = True
        params.minArea = self.min_area
        # Optionally define a maxArea if needed:
        # params.maxArea = 10000

        # 2) Circularity filter
        # For a perfect circle, circularity=1.0
        # For a square, circularity ~ 0.785
        # Non-square rectangles -> typically <0.785
        params.filterByCircularity = True
        # If you want to exclude near-perfect circles:
        # e.g. let’s allow shapes with circularity up to ~0.85
        params.maxCircularity = 0.85
        # If you also want to exclude extremely elongated shapes,
        # set a minCircularity > 0.1 or 0.2
        params.minCircularity = 0.3

        # 3) Convexity filter
        # A rectangle is convex, so its convexity ~1.0.
        # This can help exclude highly concave shapes.
        params.filterByConvexity = False
        params.minConvexity = 0.9  # only keep blobs that are fairly convex
        # e.g. if you want to exclude shapes with notches or large indentations

        # 4) Inertia filter (related to elongation)
        # The inertia ratio for perfect circles ~1.0.
        # For squares or rectangles, it might vary (0.3..0.9),
        # depending on how “thin” or “squashed” the shape is.
        # You can tune these to exclude extremely elongated lines or weird shapes.
        params.filterByInertia = False
        params.minInertiaRatio = 0.2
        # Possibly define maxInertiaRatio = e.g. 0.8

        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(mask)
        return keypoints


    def _floodfill_contour_and_build(
        self,
        mask: np.ndarray,
        pt_xy: Tuple[float, float],
        color_def: Color,
        idx_blob: int
    ) -> List[Block]:
        blocks: List[Block] = []

        cx, cy = int(pt_xy[0]), int(pt_xy[1])
        if (cx < 0 or cy < 0 or cx >= mask.shape[1] or cy >= mask.shape[0]):
            return blocks

        # Copy the mask so we don't modify the original
        temp_mask = mask.copy()

        # floodFill requires a mask bigger by 2 in each dimension
        flood_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)

        # We can use zero tolerance => fill exact same color, or set loDiff, upDiff
        flags = 8  # 8-neighborhood
        cv2.floodFill(temp_mask, flood_mask, (cx, cy),
                      255, (0,), (0,), flags=flags)

        # isolate the region that became 255
        blob_region = cv2.inRange(temp_mask, 255, 255)
        sub_contours, _ = cv2.findContours(
            blob_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not sub_contours:
            return blocks

        # pick the largest contour
        largest_contour = max(sub_contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        if area < self.min_area:
            return blocks

        # OPTIONAL: save a debug image of the floodfilled area
        if self.debug:
            flood_vis = cv2.cvtColor(blob_region, cv2.COLOR_GRAY2BGR)
            cv2.circle(flood_vis, (cx, cy), 5, (0, 0, 255), -1)
            self.debug_images[f"{color_def.name}_flood_blob_{idx_blob}"] = flood_vis

        rect = cv2.minAreaRect(largest_contour)
        (rx, ry), (rw, rh), angle = rect
        if rw < rh:
            rw, rh = rh, rw
            angle += 90

        block = Block(
            center=(rx, ry),
            size=(rw, rh),
            angle=angle,
            color=color_def,
            contour=largest_contour
        )
        blocks.append(block)
        return blocks

# ------------------------------------------------------------------
# (3) Example usage with debug
# ------------------------------------------------------------------


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    detector = BlobFloodFillDetector(debug=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        blocks = detector.detect_blocks(frame)

        # Show final detection
        out = frame.copy()
        for b in blocks:
            box = cv2.boxPoints((b.center, b.size, b.angle))
            box = np.intp(box)
            cv2.drawContours(out, [box], 0, b.color.bgr, 2)

        cv2.imshow("Final Detection", out)

        # Display debug images
        for name, dbg_img in detector.debug_images.items():
            cv2.imshow(name, dbg_img)

        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()
