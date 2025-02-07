import cv2
import numpy as np
from typing import Tuple


def watershed_color_demo(frame_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Demonstration of color-based segmentation using Watershed.
    Returns:
      - annotated image showing the watershed result,
      - a visualization image of (foreground, background, unknown).
    """

    # 1) Convert to HSV and threshold for a certain color range (e.g. “blue”).
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    # Example: detect “blue” color
    lower_blue = np.array([100, 30, 30], dtype=np.uint8)
    upper_blue = np.array([130, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # 2) Morphological cleanup (remove noise)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 'mask' is now a clean binary image: 255=likely blue, 0=background

    # 3) Mark sure background and sure foreground
    # Sure background: we can dilate the mask a bit
    sure_bg = cv2.dilate(mask, kernel, iterations=3)

    # Sure foreground: distance transform -> threshold
    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    max_val = dist_transform.max()
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * max_val, 255, 0)
    sure_fg = np.uint8(sure_fg)  # must be uint8

    # 4) Define unknown area = sure_bg - sure_fg
    unknown = cv2.subtract(sure_bg, sure_fg)

    # 5) Marker labeling
    num_labels, markers = cv2.connectedComponents(sure_fg)
    # Add 1 to all labels so that sure background is 1 instead of 0
    markers = markers + 1
    # Mark the unknown region as 0
    markers[unknown == 255] = 0

    # 6) Apply watershed on the original image (BGR).
    cv2.watershed(frame_bgr, markers)  # modifies 'markers' in-place

    # After watershed, each separate region has an ID > 1, boundary is marked as -1
    # We can color each region or highlight boundaries.

    # 7) Create an output image to visualize the watershed result
    result = frame_bgr.copy()
    boundary = (markers == -1)
    result[boundary] = [0, 0, 255]  # red boundary

    # Optionally color each region differently, ignoring the background = 1
    for i in range(2, num_labels + 1):
        mask_region = (markers == i)
        color = np.random.randint(0, 255, size=3)
        result[mask_region] = color
    result[boundary] = [0, 0, 255]  # keep boundary on top

    # 8) Create a separate visualization for foreground / background / unknown
    #    with different colors
    fg_bg_unknown = create_fg_bg_unknown_viz(
        frame_bgr.shape, sure_fg, sure_bg, unknown)

    return result, fg_bg_unknown


def create_fg_bg_unknown_viz(shape: tuple, sure_fg: np.ndarray, sure_bg: np.ndarray, unknown: np.ndarray) -> np.ndarray:
    """
    Create a visualization image where each pixel shows whether it's
    foreground, background, or unknown region.

    We'll use:
      - White for foreground
      - Green for background
      - Gray for unknown
      - Black for outside all these
    """
    # We assume 'shape' is e.g. (height, width, 3).
    # sure_fg, sure_bg, unknown are single-channel.
    height, width = shape[:2]

    viz = np.zeros((height, width, 3), dtype=np.uint8)

    # 1) Unknown region => gray
    #    (because unknown has 255 for that region)
    viz[unknown == 255] = (128, 128, 128)  # gray

    # 2) sure_bg => green
    #    but careful not to overwrite unknown => we do it only where unknown==0
    #    sure_bg also uses 255 => white for that region
    bg_mask = (sure_bg == 255) & (unknown == 0)
    viz[bg_mask] = (0, 255, 0)  # green

    # 3) sure_fg => white
    #    but only set it where unknown==0
    fg_mask = (sure_fg == 255) & (unknown == 0)
    viz[fg_mask] = (255, 255, 255)  # white

    return viz


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Apply the watershed color-based segmentation
        segmented, fg_bg_unknown_viz = watershed_color_demo(frame)

        cv2.imshow("Original", frame)
        cv2.imshow("Watershed Result", segmented)
        cv2.imshow("FG/BG/Unknown", fg_bg_unknown_viz)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()
