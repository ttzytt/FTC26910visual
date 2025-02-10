from typing import Tuple, List, Optional, Set, TypeVar, Annotated, Literal, TypedDict
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass, field
import cv2
from enum import Enum
from src.type_defs import *


class PreprocType(Enum):
    """Enumeration of available preprocessing steps."""
    AUTO_WB = "auto_wb"                # Auto White Balance
    BRIGHTNESS = "brightness"          # Brightness Adjustment
    HIST_EQUALIZE = "hist_equalize"    # Histogram Equalization
    CLAHE = "clahe"                    # CLAHE
    GAUSSIAN = "gaussian"              # Gaussian Blur
    MEDIAN = "median"                  # Median Blur
    BILATERAL = "bilateral"            # Bilateral Filter
    GUIDED = "guided"                  # Guided Filter
    MORPH_OPEN = "morph_open"          # Morphological Opening
    MORPH_CLOSE = "morph_close"        # Morphological Closing
# TODO: sharpening

DEF_STEPS = [
    PreprocType.MORPH_OPEN, 
    # PreprocType.MORPH_CLOSE,
    PreprocType.BILATERAL,
]

# ------------------- Preprocess Configuration -------------------

@dataclass
class PreprocCfg:
    """Configuration for image preprocessing steps."""
    preprocess_steps: List[PreprocType] = field(default_factory=lambda: DEF_STEPS)
    # Steps for which debug output is desired
    debug_steps: Set[PreprocType] | bool = field(default_factory=set)

    brightness: int = 0

    CLAHE_clip_limit: int = 2
    CLAHE_grid_size: int = 8

    gaussian_kernel_size: int = 5
    gaussian_sigma: int = 0

    median_kernel_size: int = 5

    bilateral_d: int = 11
    bilateral_sigma_color: int = 100
    bilateral_sigma_space: int = 20

    guided_radius: int = 5
    guided_eps: float = 0.1

    morph_open_kernel_size: int = 5
    morph_open_iter: int = 1

    morph_close_kernel_size: int = 5
    morph_close_iter: int = 1

# ------------------- Preprocessor Class -------------------


class Preproc:
    """Applies a sequence of preprocessing steps to an image based on the given configuration.
    Also outputs debug images for steps specified in cfg.debug_steps.
    """

    def __init__(self, cfg: PreprocCfg):
        self.cfg = cfg
        self.debug_images: VizResults = {}  # Store intermediate debug results

        # Check if any debug step is specified but not present in the preprocess_steps
        if isinstance(self.cfg.debug_steps, bool):
            if self.cfg.debug_steps:
                self.cfg.debug_steps = {step for step in PreprocType}
            else: self.cfg.debug_steps = set()
            # if true then add all steps to debug_steps
            return
    
        if len(self.cfg.debug_steps) > 0:
            missing_debug = {
                step for step in self.cfg.debug_steps if not self.cfg.preprocess_steps or step not in self.cfg.preprocess_steps}
            if missing_debug:
                for step in missing_debug:
                    print(
                        f"Warning: Debug step '{step.value}' is specified but not included in preprocess_steps.")

    def process(self, image: img_t) -> img_t:
        """Processes the image based on the configured preprocessing steps.
        Stores intermediate debug outputs for steps specified in debug_steps.
        """
        if not self.cfg.preprocess_steps:
            return image

        for step in self.cfg.preprocess_steps:
            image = self._apply_step(image, step)
            # If the current step is in the debug_steps, store a copy of the intermediate result.
            if isinstance(self.cfg.debug_steps, set) and step in self.cfg.debug_steps:
                self.debug_images[step.value] = image.copy()
        return image

    def _apply_step(self, image: img_t, step: PreprocType) -> img_t:
        """Applies a specific preprocessing step to the image."""
        processors = {
            PreprocType.AUTO_WB: self._auto_white_balance,
            PreprocType.BRIGHTNESS: self._adjust_brightness,
            PreprocType.HIST_EQUALIZE: self._hist_equalize,
            PreprocType.CLAHE: self._clahe,
            PreprocType.GAUSSIAN: self._gaussian,
            PreprocType.MEDIAN: self._median,
            PreprocType.BILATERAL: self._bilateral,
            PreprocType.GUIDED: self._guided_filter,
            PreprocType.MORPH_OPEN: self._morph_open,
            PreprocType.MORPH_CLOSE: self._morph_close,
        }
        return processors[step](image)

    def _auto_white_balance(self, image: img_bgr_t) -> img_bgr_t:
        """Applies automatic white balance using OpenCV's built-in method if available."""
        if hasattr(cv2, "xphoto") and hasattr(cv2.xphoto, "createSimpleWB"):
            wb = cv2.xphoto.createSimpleWB()
            return wb.balanceWhite(image)
        else:
            # Fallback method: Normalize LAB color space (less accurate)
            result = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            avg_a = np.mean(result[:, :, 1])
            avg_b = np.mean(result[:, :, 2])
            result[:, :, 1] = result[:, :, 1] - \
                ((avg_a - 128) * (result[:, :, 0] / 255.0))
            result[:, :, 2] = result[:, :, 2] - \
                ((avg_b - 128) * (result[:, :, 0] / 255.0))
            return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)

    def _adjust_brightness(self, image: img_t) -> img_t:
        """Adjusts the brightness of the image."""
        return cv2.convertScaleAbs(image, alpha=1, beta=self.cfg.brightness)

    def _hist_equalize(self, image: img_t) -> img_t:
        """Applies histogram equalization to enhance contrast."""
        if len(image.shape) == 2:  # Grayscale
            return cv2.equalizeHist(image)
        else:  # Color image
            ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
            return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    def _clahe(self, image: img_t) -> img_t:
        """Applies CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
        clahe = cv2.createCLAHE(clipLimit=self.cfg.CLAHE_clip_limit,
                                tileGridSize=(self.cfg.CLAHE_grid_size, self.cfg.CLAHE_grid_size))
        if len(image.shape) == 2:
            return clahe.apply(image)
        else:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def _gaussian(self, image: img_t) -> img_t:
        """Applies Gaussian blur for noise reduction."""
        return cv2.GaussianBlur(image,
                                (self.cfg.gaussian_kernel_size,
                                 self.cfg.gaussian_kernel_size),
                                self.cfg.gaussian_sigma)

    def _median(self, image: img_t) -> img_t:
        """Applies median filtering for salt-and-pepper noise removal."""
        return cv2.medianBlur(image, self.cfg.median_kernel_size)

    def _bilateral(self, image: img_t) -> img_t:
        """Applies bilateral filtering for noise reduction while preserving edges."""
        return cv2.bilateralFilter(image,
                                   self.cfg.bilateral_d,
                                   self.cfg.bilateral_sigma_color,
                                   self.cfg.bilateral_sigma_space)

    def _guided_filter(self, image: img_t) -> img_t:
        """Applies guided filtering using OpenCV's ximgproc module if available.
        Falls back to bilateral filtering otherwise.
        """
        if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "guidedFilter"):
            # Apply guided filtering using OpenCV's ximgproc module
            return cv2.ximgproc.guidedFilter(image, image, self.cfg.guided_radius, self.cfg.guided_eps)
        else:
            # Fallback method: Apply bilateral filter as an alternative (since guided filtering is edge-preserving)
            print("Warning: Guided filtering is skipped as OpenCV's ximgproc module is unavailable. Using Bilateral filter instead.")
            return cv2.bilateralFilter(image,
                                       self.cfg.bilateral_d,
                                       self.cfg.bilateral_sigma_color,
                                       self.cfg.bilateral_sigma_space)

    def _morph_open(self, image: img_t) -> img_t:
        """Applies morphological opening (erosion followed by dilation) to remove noise."""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (self.cfg.morph_open_kernel_size, self.cfg.morph_open_kernel_size))
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=self.cfg.morph_open_iter)

    def _morph_close(self, image: img_t) -> img_t:
        """Applies morphological closing (dilation followed by erosion) to close small holes."""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (self.cfg.morph_close_kernel_size, self.cfg.morph_close_kernel_size))
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=self.cfg.morph_close_iter)
