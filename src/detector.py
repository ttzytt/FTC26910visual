from abc import ABC, abstractmethod
from src.block import Block
from typing import List, Sequence, Type
import numpy as np
from dataclasses import dataclass
from src.type_defs import *
from src.color_defs import *
from src.preprocessor import *


class Detector(ABC):

    EnumSub = TypeVar('EnumSub', bound=Enum)

    def __init__(self, detecting_colors: List[Color], 
                preproc_cfg: PreprocCfg, 
                debug_option: Sequence[EnumSub] | bool, 
                debug_type: Type[EnumSub]):
        self.preproc_cfg = preproc_cfg
        self.preproc = Preproc(self.preproc_cfg)
        self.detecting_colors = detecting_colors
        self.debug_images: VizResults = {}
        print(debug_option)
        if isinstance(debug_option, bool):
            debug_option = list(debug_type) if debug_option else []
        print(debug_option)
        self.debug_option = debug_option

    @abstractmethod
    def process_frame(self, frame: img_bgr_t) -> List[Block]:
        pass

    def _preprocess(self, frame: img_bgr_t) -> img_bgr_t:
        ret = self.preproc.process(frame)
        for name, img in self.preproc.debug_images.items():
            self.debug_images[name] = np.array(img, dtype=np.uint8)
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
