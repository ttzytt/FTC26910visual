from abc import ABC, abstractmethod
from src.block import Block
from typing import List, Sequence, Type
import numpy as np
from dataclasses import dataclass
from src.type_defs import *
from src.color_defs import *
from src.preprocessor import *

EnumSub = TypeVar('EnumSub', bound=Enum)

class Detector(ABC):


    def __init__(self, detecting_colors: List[Color], 
                preproc_cfg: PreprocCfg, 
                debug_option: Sequence[EnumSub] | bool, 
                debug_type: Type[EnumSub]):
        self.preproc_cfg = preproc_cfg
        self.preproc = Preproc(self.preproc_cfg)
        self.detecting_colors = detecting_colors
        self.debug_images: VizResults = {}
        if isinstance(debug_option, bool):
            debug_option = list(debug_type) if debug_option else []
        self.debug_option = debug_option

    @property
    @abstractmethod
    def DebugType(self) -> Type[EnumSub]:
        pass

    @abstractmethod
    def process_frame(self, frame: img_bgr_t) -> List[Block]:
        pass

    def _preprocess(self, frame: img_bgr_t) -> img_bgr_t:
        ret = self.preproc.process(frame)
        self.debug_images.update(self.preproc.debug_images)
        return ret

    @staticmethod
    def create_color_mask(frame_hsv: img_hsv_t, colors: List[Color] | Color) -> img_gray_t:
        mask = np.zeros(frame_hsv.shape[:2], dtype=np.uint8)
        if isinstance(colors, Color):
            colors = [colors]
        for color in colors:
            for (lower, upper) in color.hsv_ranges:
                temp_mask = cv2.inRange(
                    frame_hsv, np.array(lower), np.array(upper))
                mask = cv2.bitwise_or(mask, temp_mask)
        return mask

    def _merge_debug_imgs(self, prefix: str) -> img_bgr_t:
        """
        Merge debug images with the given prefix according to order of `self.detecting_colors`
        Later images have higher priority
        """
        ret = None
        for color in self.detecting_colors:
            img_name = f"{prefix}_{color.name}"
            if img_name not in self.debug_images:
                continue
            img = self.debug_images[img_name]
            if ret is None:
                ret = np.zeros_like(img)
            # Create a mask where the current image is not black (i.e., has color)
            mask = np.any(img > 0, axis=-1)
            ret[mask > 0] = img[mask > 0]
        assert ret is not None, f"No debug images found with prefix {prefix}"
        return ret
