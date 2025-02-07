import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Optional
import os
import sys
import matplotlib.pyplot as plt  # Added for histogram plotting

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from color_def import hsv_t, Color


def compute_hsv_ranges(h_values, s_values, v_values, coverage: float) -> List[Tuple[hsv_t, hsv_t]]:
    """Handle circular Hue values for colors like red"""
    h_sorted = np.sort(h_values)

    # Calculate percentiles accounting for circular Hue space
    lower_p = (1 - coverage) / 2 * 100
    upper_p = 100 - lower_p

    h_low = np.percentile(h_sorted, lower_p)
    h_high = np.percentile(h_sorted, upper_p)

    s_low = np.percentile(s_values, lower_p)
    s_high = np.percentile(s_values, upper_p)
    v_low = np.percentile(v_values, lower_p)
    v_high = np.percentile(v_values, upper_p)

    ranges = []
    if h_high - h_low > 90:  # Threshold for splitting red ranges
        # Split into two ranges for red
        ranges.append(((0, int(s_low), int(v_low)),
                       (10, int(s_high), int(v_high))))
        ranges.append(((170, int(s_low), int(v_low)),
                       (180, int(s_high), int(v_high))))
    else:
        # Single range for other colors
        ranges.append(((int(h_low), int(s_low), int(v_low)),
                       (int(h_high), int(s_high), int(v_high))))
    return ranges


class RotatedROISelector:
    """Interactive rotated ROI selection"""

    def __init__(self, image):
        self.image = image
        self.center = (0, 0)
        self.size = (100, 50)
        self.angle = 0
        self.dragging = False
        self.current_corner = -1
        self.corners = np.zeros((4, 2), dtype=np.float32)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self.current_corner = self.find_nearest_corner(x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self.corners[self.current_corner] = (x, y)
            self.update_rotated_rect()

    def find_nearest_corner(self, x, y):
        distances = [np.linalg.norm(c - (x, y)) for c in self.corners]
        return np.argmin(distances)

    def update_rotated_rect(self):
        rect = cv2.minAreaRect(self.corners.astype(int))
        self.center, self.size, self.angle = rect

    def select(self):
        cv2.namedWindow("Rotated ROI")
        cv2.setMouseCallback("Rotated ROI", self.mouse_callback)

        # Initialize corners with default rectangle
        h, w = self.image.shape[:2]
        self.corners = np.array([[w//3, h//3],
                                 [2*w//3, h//3],
                                 [2*w//3, 2*h//3],
                                 [w//3, 2*h//3]], dtype=np.float32)

        while True:
            img = self.image.copy()

            # Draw current rotated rectangle.
            # Use box.astype(np.int32) to avoid deprecation warnings.
            box = cv2.boxPoints((self.center, self.size, self.angle))
            cv2.drawContours(img, [box.astype(np.int32)], 0, (0, 255, 0), 2)

            # Draw corners
            for x, y in self.corners:
                cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)

            cv2.imshow("Rotated ROI", img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        cv2.destroyAllWindows()
        return self.center, self.size, self.angle


def extract_rotated_roi(image, center, size, angle):
    """
    Extract the rotated region as an upright rectangle.
    Convert size to integer values and ensure center is a tuple of floats.
    """
    patch_size = (int(size[0]), int(size[1]))
    center_pt = (float(center[0]), float(center[1]))

    # Rotate the entire image about the center by angle.
    M = cv2.getRotationMatrix2D(center_pt, angle, 1)
    rotated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    # Extract the rotated ROI.
    roi = cv2.getRectSubPix(rotated, patch_size, center_pt)
    return roi


def main():
    image_name = "limelight.png"
    color_name = input(
        "Enter the color name of the block being measured: ").strip()
    image = cv2.imread(f"../../blk_imgs/{image_name}")
    if image is None:
        print("Could not load image")
        return

    # 1. Select rotated ROI interactively.
    selector = RotatedROISelector(image)
    center, size, angle = selector.select()

    # 2. Extract the ROI and convert to HSV.
    roi = extract_rotated_roi(image, center, size, angle)
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # 3. Calculate HSV distributions.
    H = hsv_roi[:, :, 0].flatten()
    S = hsv_roi[:, :, 1].flatten()
    V = hsv_roi[:, :, 2].flatten()

    # Set lower bound, upper bound and interval for the coverage percentages.
    # For example, lower=0.5, upper=1, interval=0.1 will generate: 0.5, 0.6, 0.7, 0.8, 0.9, 1.0.
    coverage_lower = 0.5   # 50% coverage
    coverage_upper = 1.0   # 100% coverage
    coverage_interval = 0.1  # 10% step

    # Create an array of coverage values (e.g. 0.5, 0.6, 0.7, ..., 1.0)
    coverage_levels = np.arange(
        coverage_lower, coverage_upper + coverage_interval/2, coverage_interval)

    # For each coverage level, compute the HSV ranges.
    computed_hsv_ranges = []   # Will store tuples: (coverage, list_of_ranges)
    for cov in coverage_levels:
        ranges_for_cov = compute_hsv_ranges(H, S, V, cov)
        computed_hsv_ranges.append((cov, ranges_for_cov))

    # Print the computed ranges to the terminal.
    print(f"Detected HSV ranges for {color_name}:")
    for cov, ranges in computed_hsv_ranges:
        # Compute the corresponding lower and upper percentiles for this coverage.
        lower_percentile = (1 - cov) / 2 * 100
        upper_percentile = 100 - lower_percentile
        percentile_interval = upper_percentile - lower_percentile
        print(f"Coverage: {cov*100:.0f}% "
              f"(Percentiles: {lower_percentile:.1f}%-{upper_percentile:.1f}%, Interval: {percentile_interval:.1f}%)")
        for idx, (lower, upper) in enumerate(ranges):
            print(f"  Range {idx+1}:")
            print(f"    Lower: {lower}")
            print(f"    Upper: {upper}")

    # Save the computed ranges and percentile settings to a CSV file.
    import csv
    img_no_format = image_name.split('.')[0]
    csv_output_path = f"./output/{img_no_format}_{color_name}_hsv_ranges.csv"
    with open(csv_output_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        # Write header row.
        writer.writerow(["Coverage", "LowerPercentile", "UpperPercentile", "PercentileInterval",
                         "RangeNumber", "Hue Lower", "Hue Upper", "Sat Lower", "Sat Upper", "Val Lower", "Val Upper"])
        for cov, ranges in computed_hsv_ranges:
            lower_percentile = (1 - cov) / 2 * 100
            upper_percentile = 100 - lower_percentile
            percentile_interval = upper_percentile - lower_percentile
            for idx, (lower, upper) in enumerate(ranges):
                writer.writerow([f"{cov*100:.0f}%", lower_percentile, upper_percentile, percentile_interval,
                                 idx+1, lower[0], upper[0], lower[1], upper[1], lower[2], upper[2]])
    print(f"HSV ranges saved to CSV file: {csv_output_path}")

    # Plot histograms of the HSV distributions.
    fig, axs = plt.subplots(3, 1, figsize=(15, 12), dpi=300)

    axs[0].hist(H, bins=180, color='red', alpha=0.7)
    axs[0].set_title("Hue Distribution")
    axs[0].set_xlabel("Hue")
    axs[0].set_ylabel("Frequency")
    axs[0].set_xticks(range(0, 181, 5))

    axs[1].hist(S, bins=256, color='green', alpha=0.7)
    axs[1].set_title("Saturation Distribution")
    axs[1].set_xlabel("Saturation")
    axs[1].set_ylabel("Frequency")
    axs[1].set_xticks(range(0, 256, 10))

    axs[2].hist(V, bins=256, color='blue', alpha=0.7)
    axs[2].set_title("Value Distribution")
    axs[2].set_xlabel("Value")
    axs[2].set_ylabel("Frequency")
    axs[2].set_xticks(range(0, 256, 10))

    plt.tight_layout()
    img_no_format = image_name.split('.')[0]
    plt.savefig(f"./output/{img_no_format}_{color_name}_histogram.png")


if __name__ == "__main__":
    main()
