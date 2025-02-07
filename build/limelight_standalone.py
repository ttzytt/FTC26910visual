#!/usr/bin/env python
import contextlib as __stickytape_contextlib

@__stickytape_contextlib.contextmanager
def __stickytape_temporary_dir():
    import tempfile
    import shutil
    dir_path = tempfile.mkdtemp()
    try:
        yield dir_path
    finally:
        shutil.rmtree(dir_path)

with __stickytape_temporary_dir() as __stickytape_working_dir:
    def __stickytape_write_module(path, contents):
        import os, os.path

        def make_package(path):
            parts = path.split("/")
            partial_path = __stickytape_working_dir
            for part in parts:
                partial_path = os.path.join(partial_path, part)
                if not os.path.exists(partial_path):
                    os.mkdir(partial_path)
                    with open(os.path.join(partial_path, "__init__.py"), "wb") as f:
                        f.write(b"\n")

        make_package(os.path.dirname(path))

        full_path = os.path.join(__stickytape_working_dir, path)
        with open(full_path, "wb") as module_file:
            module_file.write(contents)

    import sys as __stickytape_sys
    __stickytape_sys.path.insert(0, __stickytape_working_dir)

    __stickytape_write_module('detector_contour.py', b'import cv2\r\nimport numpy as np\r\nfrom src.color_def import *\r\nfrom src.block import Block\r\nfrom typing import List\r\nfrom src.detectors import *\r\n\r\n\r\nclass ContourVizResults(TypedDict, total=False):\r\n    """Defines the structure of the dictionary returned by visualize()."""\r\n    final_detection: np.ndarray\r\n    avg_HSV: np.ndarray\r\n    original: np.ndarray\r\n    preprocessed: np.ndarray\r\n    hsv_space: np.ndarray\r\n    combined_mask: np.ndarray\r\n\r\nclass ColorBlockDetectorContour(Detector):\r\n    """\r\n    Detects color blocks by:\r\n      1) Preprocessing (brightness, blur)\r\n      2) Creating color masks\r\n      3) Finding contours\r\n      4) Computing mean & std(H, S, V) inside each contour\r\n    """\r\n\r\n    def __init__(self, detecting_colors: List[Color]):\r\n        # Basic image processing parameters\r\n        self.blur_size = 35\r\n        self.brightness = 0\r\n        self.erode_iter = 7\r\n        self.dilate_iter = 6\r\n        self.min_contour_area = 1000\r\n        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))\r\n\r\n        # Thresholds for std(H, S, V)\r\n        self.std_threshold_hsv = (3, 50, 50)\r\n\r\n        # Storage for debug images (intermediate steps)\r\n        self._debug_images : ContourVizResults = {}\r\n        self.detecting_colors = detecting_colors\r\n\r\n    def process_frame(self, frame: np.ndarray) -> List[Block]:\r\n        """Main entry: preprocess and detect blocks, while saving debug images."""\r\n        self._debug_images = {}\r\n\r\n        # Save original frame\r\n        self._debug_images[\'original\'] = frame.copy()\r\n\r\n        # 1) Preprocessing\r\n        preprocessed = self._preprocess(frame)\r\n        self._debug_images[\'preprocessed\'] = preprocessed.copy()\r\n\r\n        # 2) Convert to HSV\r\n        hsv = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2HSV)\r\n        hsv_bgr_like = cv2.cvtColor(\r\n            hsv, cv2.COLOR_HSV2BGR)  # just for visualization\r\n        self._debug_images[\'hsv_space\'] = hsv_bgr_like\r\n\r\n        # 3) Detect blocks\r\n        blocks = self._detect_blocks(hsv)\r\n\r\n        return blocks\r\n\r\n    def get_debug_images(self) -> ContourVizResults:\r\n        """Returns debug images for visualization."""\r\n        return self._debug_images\r\n\r\n    def _preprocess(self, frame: np.ndarray) -> np.ndarray:\r\n        """Adjust brightness and apply Gaussian blur."""\r\n        frame = cv2.convertScaleAbs(frame, alpha=1, beta=self.brightness)\r\n        if self.blur_size > 0:\r\n            kernel_size = self.blur_size | 1\r\n            frame = cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)\r\n        return frame\r\n\r\n    def _detect_blocks(self, hsv: np.ndarray) -> List[Block]:\r\n        blocks = []\r\n        combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)\r\n\r\n        for color_def in self.detecting_colors:\r\n            mask = self._create_color_mask(hsv, color_def)\r\n            combined_mask = cv2.bitwise_or(combined_mask, mask)\r\n\r\n            contours = self._find_contours(mask)\r\n            color_blocks = self._process_contours(contours, color_def, hsv)\r\n            blocks.extend(color_blocks)\r\n\r\n        combined_bgr = cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2BGR)\r\n        self._debug_images[\'combined_mask\'] = combined_bgr\r\n\r\n        return blocks\r\n\r\n    def _create_color_mask(self, hsv: np.ndarray, color_def: Color) -> np.ndarray:\r\n        """\r\n        Create a mask for each color definition, applying morphological operations.\r\n        The recognized areas retain their original colors (converted from HSV to BGR).\r\n        \r\n        Args:\r\n            hsv (np.ndarray): The HSV-converted image.\r\n            color_def (Color): The color definition (with HSV ranges and BGR info).\r\n        \r\n        Returns:\r\n            np.ndarray: A binary mask after morphological ops.\r\n        """\r\n        # Step 1: Initialize an empty mask\r\n        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)\r\n\r\n        # Step 2: Apply color thresholds for each HSV range\r\n        for (lower, upper) in color_def.hsv_ranges:\r\n\r\n            # Apply threshold to get binary mask\r\n            tmp_mask = cv2.inRange(hsv, np.array(list(lower)), np.array(list(upper)))\r\n            mask = cv2.bitwise_or(mask, tmp_mask)\r\n\r\n        # Step 3: For debug: raw mask in color\r\n        hsv_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)\r\n        raw_mask_colored = cv2.bitwise_and(hsv_bgr, hsv_bgr, mask=mask)\r\n        debug_raw = f"{color_def.name.lower()}_mask_raw"\r\n        self._debug_images[debug_raw] = raw_mask_colored\r\n\r\n        # Step 4: Morphological operations (erode & dilate)\r\n        mask_morph = cv2.erode(mask, self.kernel, iterations=self.erode_iter)\r\n        mask_morph = cv2.dilate(mask_morph, self.kernel,\r\n                                iterations=self.dilate_iter)\r\n\r\n        # Step 5: For debug: morph mask in color\r\n        morph_mask_colored = cv2.bitwise_and(hsv_bgr, hsv_bgr, mask=mask_morph)\r\n        debug_morph = f"{color_def.name.lower()}_mask_morph"\r\n        self._debug_images[debug_morph] = morph_mask_colored\r\n\r\n        return mask_morph\r\n\r\n    def _find_contours(self, mask: np.ndarray) -> List[np.ndarray]:\r\n        """Find external contours with area > min_contour_area."""\r\n        contours, _ = cv2.findContours(\r\n            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)\r\n        return [c for c in contours if cv2.contourArea(c) > self.min_contour_area]\r\n\r\n    def _process_contours(self,\r\n                          contours: List[np.ndarray],\r\n                          color_def: Color,\r\n                          hsv: np.ndarray) -> List[Block]:\r\n        """Compute mean & std(H, S, V) inside each contour and filter by thresholds."""\r\n        blocks = []\r\n        for cnt in contours:\r\n            rect = cv2.minAreaRect(cnt)\r\n            (cx, cy), (w, h), angle = rect\r\n\r\n            # Normalize orientation\r\n            if w < h:\r\n                w, h = h, w\r\n                angle += 90\r\n\r\n            # Get bounding rect (for ROI)\r\n            x_min, y_min, w_int, h_int = cv2.boundingRect(cnt)\r\n            if w_int == 0 or h_int == 0:\r\n                continue\r\n\r\n            # Create local mask (contour inside bounding box)\r\n            contour_mask = np.zeros((h_int, w_int), dtype=np.uint8)\r\n            shifted_cnt = cnt - [x_min, y_min]\r\n            cv2.drawContours(contour_mask, [shifted_cnt], 0, (255,), -1)\r\n\r\n            # Extract HSV ROI\r\n            hsv_roi = hsv[y_min:y_min + h_int, x_min:x_min + w_int]\r\n            hsv_masked = cv2.bitwise_and(hsv_roi, hsv_roi, mask=contour_mask)\r\n\r\n            # Split channels and extract valid pixels\r\n            h_ch, s_ch, v_ch = cv2.split(hsv_masked)\r\n            h_valid = h_ch[contour_mask == 255].astype(np.float32)\r\n            s_valid = s_ch[contour_mask == 255].astype(np.float32)\r\n            v_valid = v_ch[contour_mask == 255].astype(np.float32)\r\n\r\n            if len(h_valid) == 0:\r\n                continue\r\n\r\n            # Compute mean & std for H, S, V\r\n            mean_h = float(np.mean(h_valid))\r\n            mean_s = float(np.mean(s_valid))\r\n            mean_v = float(np.mean(v_valid))\r\n\r\n            std_h = compute_hue_std_flip(h_valid, flip_threshold=90.0)\r\n            std_s = float(np.std(s_valid))\r\n            std_v = float(np.std(v_valid))\r\n\r\n            # Create a new Block\r\n            if std_h <= self.std_threshold_hsv[0] and \\\r\n               std_s <= self.std_threshold_hsv[1] and \\\r\n               std_v <= self.std_threshold_hsv[2]:\r\n\r\n                block = Block(\r\n                    center=(cx, cy),\r\n                    size=(w, h),\r\n                    angle=angle,\r\n                    color=color_def,\r\n                    color_std=(std_h, std_s, std_v),\r\n                    mean_hsv=(mean_h, mean_s, mean_v),\r\n                    # store the original contour (absolute coordinates)\r\n                    contour=cnt\r\n                )\r\n                blocks.append(block)\r\n        return blocks\r\n')
    __stickytape_write_module('src/color_def.py', b'import cv2\r\nimport numpy as np\r\nfrom dataclasses import dataclass, field\r\nfrom typing import List, Tuple\r\n\r\n# ---------- Data Classes ----------\r\n\r\nhsv_t = Tuple[int, int, int]\r\nbgr_t = Tuple[int, int, int]\r\n\r\n\r\n@dataclass\r\nclass Color:\r\n    """Stores color name, HSV ranges, and BGR values for drawing."""\r\n    name: str\r\n    hsv_ranges: List[Tuple[hsv_t, hsv_t]]\r\n    bgr: bgr_t\r\n\r\n\r\n# ---------- Color Block Detector ----------\r\n\r\ndef compute_hue_std_flip(h_array: np.ndarray, flip_threshold: float = 90.0) -> float:\r\n    # Ensure float\r\n    h_float = h_array.astype(np.float32)\r\n\r\n    # 1) Direct std\r\n    std1 = np.std(h_float)\r\n\r\n    # 2) Flip\r\n    shifted = h_float.copy()\r\n    mask = (shifted < flip_threshold)\r\n    shifted[mask] += 180.0\r\n    std2 = np.std(shifted)\r\n\r\n    return float(min(std1, std2))\r\n\r\n\r\nRED_R9000P = Color(\r\n    name="RED_R9000P",\r\n    hsv_ranges=[\r\n        ((0, 70, 50), (3, 160, 225)),\r\n        ((165, 70, 50), (180, 160, 225)),\r\n    ],\r\n    bgr=(0, 0, 255)\r\n)\r\n\r\nBLUE_R9000P = Color(\r\n    name="BLUE_R9000P",\r\n    hsv_ranges=[\r\n        ((110, 80, 70), (125, 180, 230)),\r\n    ],\r\n    bgr=(255, 0, 0)\r\n)\r\n\r\nYELLOW_R9000P = Color(\r\n    name="YELLOW_R9000P",\r\n    hsv_ranges=[\r\n        ((17, 60, 140), (32, 125, 255)),\r\n    ],\r\n    bgr=(0, 255, 255)\r\n)\r\n\r\nCOLOR_DEF_R9000P = [RED_R9000P, BLUE_R9000P, YELLOW_R9000P]\r\n\r\nRED_LL = Color(\r\n    name="RED_LL",\r\n    hsv_ranges=[\r\n        ((0, 190, 90), (5, 255, 250)),\r\n        ((160, 190, 90), (180, 255, 250)),\r\n    ],\r\n    bgr=(0, 0, 255)\r\n)\r\n\r\nYELLOW_LL = Color(\r\n    name="YELLOW_LL",\r\n    hsv_ranges=[\r\n        ((15, 130, 160), (35, 255, 250)),\r\n    ],\r\n    bgr=(0, 255, 255)\r\n)\r\n\r\nBLUE_LL = Color(\r\n    name="BLUE_LL",\r\n    hsv_ranges=[\r\n        ((100, 220, 70), (125, 255, 230)),\r\n    ],\r\n    bgr=(255, 0, 0)\r\n)\r\n\r\nCOLOR_DEF_LL = [RED_LL, YELLOW_LL, BLUE_LL]')
    __stickytape_write_module('src/block.py', b'import numpy as np\r\nfrom dataclasses import dataclass, field\r\nfrom .color_def import Color\r\nfrom typing import List, Tuple\r\n\r\n@dataclass\r\nclass Block:\r\n    """Represents a detected color block with position, size, angle, color info, and HSV stats."""\r\n    center: Tuple[float, float]\r\n    size: Tuple[float, float]\r\n    angle: float\r\n    color: Color\r\n    color_std: Tuple[float, float, float] = (0.0, 0.0, 0.0)\r\n    mean_hsv: Tuple[float, float, float] = (0.0, 0.0, 0.0)\r\n    # store the absolute contour for visualization\r\n    contour: np.ndarray = field(default_factory=lambda: np.array([]))\r\n')
    __stickytape_write_module('src/detectors.py', b'from abc import ABC, abstractmethod\r\nfrom src.block import Block\r\nfrom typing import List\r\nimport numpy as np\r\nfrom typing import TypedDict, Dict \r\n\r\nVizResults = Dict[str, np.ndarray]\r\n\r\nclass Detector(ABC):\r\n    @abstractmethod\r\n    def process_frame(self, frame: np.ndarray) -> List[Block]:\r\n        pass\r\n    @abstractmethod\r\n    def get_debug_images(self) -> Dict[str, np.ndarray]: \r\n        pass')
    __stickytape_write_module('src/visualizer.py', b'import cv2\r\nimport numpy as np\r\nfrom typing import List, Dict, Optional, TypedDict\r\nfrom src.color_def import *\r\nfrom src.block import Block\r\nfrom src.detectors import VizResults\r\n\r\nclass BlockVisualizer:\r\n    """\r\n    Manages two modes:\r\n      - Mode 0: Final detection only (single window)\r\n      - Mode 1: Debug images (multiple windows, intermediate steps) + one additional window\r\n                showing each block region filled with its average HSV color.\r\n\r\n    Parameters:\r\n        show (bool): Whether to display images using OpenCV\'s cv2.imshow(). If False, functions return images.\r\n    """\r\n\r\n    def __init__(self, show: bool = True):\r\n        self.mode = 0\r\n        self.prev_mode = 0  # Force initialization\r\n        self.main_window = "Block Detection"\r\n        self.show = show  # Whether to display images or return them\r\n\r\n    def toggle_mode(self):\r\n        """Switch between mode 0 and mode 1."""\r\n        self.mode = (self.mode + 1) % 2\r\n\r\n    def visualize(self, frame: np.ndarray, blocks: List[Block], debug_images: VizResults) -> Optional[VizResults]:\r\n        """\r\n        Decide which visualization to show based on mode.\r\n        Only destroy/recreate windows if the mode changed.\r\n\r\n        Parameters:\r\n            frame (np.ndarray): The original frame.\r\n            blocks (List[Block]): List of detected blocks.\r\n            debug_images (Dict[str, np.ndarray]): Dictionary of debug images.\r\n\r\n        Returns:\r\n            Optional[Dict[str, np.ndarray]]: If `self.show=False`, returns a dictionary of images.\r\n        """\r\n        if self.mode != self.prev_mode:\r\n            cv2.destroyAllWindows()\r\n            self.prev_mode = self.mode\r\n\r\n        results : VizResults= {}\r\n\r\n        if self.mode == 0:\r\n            final_result = self.show_final_result(frame, blocks)\r\n            if not self.show:\r\n                results[\'final_detection\'] = final_result\r\n        else:\r\n            debug_outputs = self.show_debug_images(debug_images)\r\n            avg_hsv_image = self.show_avg_hsv_fill(frame, blocks)\r\n\r\n            if not self.show:\r\n                for name, img in debug_outputs.items(): results[name] = img\r\n                results[\'avg_HSV\'] = avg_hsv_image\r\n\r\n        return results if not self.show else None\r\n\r\n    def show_final_result(self, frame: np.ndarray, blocks: List[Block]) -> np.ndarray:\r\n        """Draw bounding boxes and put text for each block."""\r\n        output = frame.copy()\r\n        for block in blocks:\r\n            box = cv2.boxPoints(\r\n                (block.center, block.size, block.angle))  # type: ignore\r\n            box = np.intp(box)\r\n            cv2.drawContours(output, [box], 0, block.color.bgr, 2)  # type: ignore\r\n\r\n            # Text lines with extra info: avgH, avgS, avgV\r\n            lines = [\r\n                f"{block.color.name}: {block.angle:.1f} deg",\r\n                f"stdHSV=({block.color_std[0]:.1f}, {block.color_std[1]:.1f}, {block.color_std[2]:.1f})",\r\n                f"avgHSV=({block.mean_hsv[0]:.1f}, {block.mean_hsv[1]:.1f},{block.mean_hsv[2]:.1f})"\r\n            ]\r\n            x0, y0 = int(block.center[0]), int(block.center[1])\r\n            for i, line in enumerate(lines):\r\n                offset_y = i * 15\r\n                cv2.putText(\r\n                    output,\r\n                    line,\r\n                    (x0, y0 + offset_y),\r\n                    cv2.FONT_HERSHEY_SIMPLEX,\r\n                    0.5,\r\n                    (255, 255, 255),\r\n                    1\r\n                )\r\n\r\n        cv2.putText(\r\n            output,\r\n            "Final Detection",\r\n            (10, 30),\r\n            cv2.FONT_HERSHEY_SIMPLEX,\r\n            1.0,\r\n            (0, 255, 0),\r\n            2\r\n        )\r\n\r\n        if self.show:\r\n            cv2.imshow(self.main_window, output)\r\n        return output\r\n\r\n    def show_debug_images(self, debug_images: VizResults) -> Dict[str, np.ndarray]:\r\n        """Display intermediate debug images, or return them if `self.show` is False."""\r\n        results = {}\r\n        for name, img in debug_images.items():\r\n            if not isinstance(img, np.ndarray):\r\n                continue\r\n            display = img.copy()\r\n            cv2.putText(display, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,\r\n                        1.0, (0, 255, 0), 2)\r\n\r\n            if self.show:\r\n                cv2.imshow(name, display)\r\n            else:\r\n                results[name] = display\r\n\r\n        return results\r\n\r\n    def show_avg_hsv_fill(self, frame: np.ndarray, blocks: List[Block]) -> np.ndarray:\r\n        """\r\n        Create a black canvas the same size as \'frame\', then fill each block\'s contour\r\n        with the block\'s average HSV color (converted to BGR). Show this in a new window\r\n        or return the processed image.\r\n        """\r\n        canvas = np.zeros_like(frame)  # black canvas\r\n        for block in blocks:\r\n            # Convert mean_hsv -> BGR\r\n            hsv_pixel = np.uint8(\r\n                [[[block.mean_hsv[0], block.mean_hsv[1], block.mean_hsv[2]]]])  # type: ignore\r\n            bgr_pixel = cv2.cvtColor(\r\n                hsv_pixel, cv2.COLOR_HSV2BGR)  # type: ignore\r\n            avg_color = (int(bgr_pixel[0, 0, 0]), int(\r\n                bgr_pixel[0, 0, 1]), int(bgr_pixel[0, 0, 2]))\r\n\r\n            # Fill the contour with this color\r\n            cv2.drawContours(canvas, [block.contour], 0, avg_color, -1)\r\n\r\n        cv2.putText(\r\n            canvas,\r\n            "Blocks filled w/ average HSV",\r\n            (10, 30),\r\n            cv2.FONT_HERSHEY_SIMPLEX,\r\n            1.0,\r\n            (255, 255, 255),\r\n            2\r\n        )\r\n\r\n        if self.show:\r\n            cv2.imshow("Avg HSV Debug", canvas)\r\n        return canvas\r\n')
    __stickytape_write_module('utils/serializer.py', b'from src.block import Block\r\nfrom ctypes import Structure, c_int16, c_int32, c_float\r\nimport struct\r\nfrom typing import Tuple, Optional, List\r\nfrom enum import Enum\r\n\r\nclass SerializedColor(Enum):\r\n    YELLOW = 1\r\n    RED = 2\r\n    BLUE = 4\r\n\r\n\r\ndef get_serialized_color(color_name: str) -> Optional[SerializedColor]:\r\n    """Maps detected color names to their corresponding SerializedColor enum based on keyword matching."""\r\n    color_name_upper = color_name.upper() \r\n\r\n    for enum_member in SerializedColor:\r\n        if enum_member.name in color_name_upper:\r\n            return enum_member\r\n\r\n    return None  # Return None if no match is found\r\n\r\nclass SerializedBlock(Structure):\r\n    _pack_ = 1 # no padding\r\n    _fields_ = [\r\n        ("center_x", c_int16),\r\n        ("center_y", c_int16),\r\n        ("width", c_int16),\r\n        ("height", c_int16),\r\n        ("angle", c_float), \r\n        ("color", c_int32)\r\n    ]\r\n\r\n    def pack_to_two_doubles(self) -> Tuple[float, float]:\r\n        raw_bytes = bytes(self)\r\n        assert len(raw_bytes) == 16, f"Expected 16 bytes, got {len(raw_bytes)}"\r\n        dbl1, dbl2 = struct.unpack(\'<2d\', raw_bytes)\r\n        return dbl1, dbl2\r\n\r\n    @staticmethod\r\n    def from_block(block: \'Block\') -> \'SerializedBlock\':\r\n        """Creates a SerializedBlock from a Block object."""\r\n        color_enum = get_serialized_color(block.color.name)\r\n        assert color_enum is not None, f"Unknown color: {block.color}"\r\n        return SerializedBlock(\r\n            center_x=int(block.center[0]),  # Ensure int conversion\r\n            center_y=int(block.center[1]),\r\n            width=int(block.size[0]),\r\n            height=int(block.size[1]),\r\n            angle=float(block.angle),  # Ensure float conversion\r\n            color=color_enum.value  # Store enum as int\r\n        )\r\n        \r\n    @classmethod\r\n    def from_bytes(cls, raw_bytes: bytes) -> \'SerializedBlock\':\r\n        """Creates a SerializedBlock from raw bytes."""\r\n        assert len(raw_bytes) == 16, f"Expected 16 bytes, got {len(raw_bytes)}"\r\n        return cls.from_buffer_copy(raw_bytes)\r\n    \r\n    @classmethod\r\n    def from_doubles(cls, dbl1: float, dbl2: float) -> \'SerializedBlock\':\r\n        """Creates a SerializedBlock from two doubles."""\r\n        raw_bytes = struct.pack(\'<2d\', dbl1, dbl2)\r\n        return cls.from_bytes(raw_bytes)\r\n    \r\ndef serialize_to_doubles(blocks: List[\'Block\']) -> List[float]:\r\n    """Serializes a list of Blocks into a list of doubles."""\r\n    serialized_blocks = [SerializedBlock.from_block(block) for block in blocks]\r\n    return [dbl for serialized_block in serialized_blocks for dbl in serialized_block.pack_to_two_doubles()]')
    __stickytape_write_module('src/utils/serializer.py', b'from src.block import Block\r\nfrom ctypes import Structure, c_int16, c_int32, c_float\r\nimport struct\r\nfrom typing import Tuple, Optional, List\r\nfrom enum import Enum\r\n\r\nclass SerializedColor(Enum):\r\n    YELLOW = 1\r\n    RED = 2\r\n    BLUE = 4\r\n\r\n\r\ndef get_serialized_color(color_name: str) -> Optional[SerializedColor]:\r\n    """Maps detected color names to their corresponding SerializedColor enum based on keyword matching."""\r\n    color_name_upper = color_name.upper() \r\n\r\n    for enum_member in SerializedColor:\r\n        if enum_member.name in color_name_upper:\r\n            return enum_member\r\n\r\n    return None  # Return None if no match is found\r\n\r\nclass SerializedBlock(Structure):\r\n    _pack_ = 1 # no padding\r\n    _fields_ = [\r\n        ("center_x", c_int16),\r\n        ("center_y", c_int16),\r\n        ("width", c_int16),\r\n        ("height", c_int16),\r\n        ("angle", c_float), \r\n        ("color", c_int32)\r\n    ]\r\n\r\n    def pack_to_two_doubles(self) -> Tuple[float, float]:\r\n        raw_bytes = bytes(self)\r\n        assert len(raw_bytes) == 16, f"Expected 16 bytes, got {len(raw_bytes)}"\r\n        dbl1, dbl2 = struct.unpack(\'<2d\', raw_bytes)\r\n        return dbl1, dbl2\r\n\r\n    @staticmethod\r\n    def from_block(block: \'Block\') -> \'SerializedBlock\':\r\n        """Creates a SerializedBlock from a Block object."""\r\n        color_enum = get_serialized_color(block.color.name)\r\n        assert color_enum is not None, f"Unknown color: {block.color}"\r\n        return SerializedBlock(\r\n            center_x=int(block.center[0]),  # Ensure int conversion\r\n            center_y=int(block.center[1]),\r\n            width=int(block.size[0]),\r\n            height=int(block.size[1]),\r\n            angle=float(block.angle),  # Ensure float conversion\r\n            color=color_enum.value  # Store enum as int\r\n        )\r\n        \r\n    @classmethod\r\n    def from_bytes(cls, raw_bytes: bytes) -> \'SerializedBlock\':\r\n        """Creates a SerializedBlock from raw bytes."""\r\n        assert len(raw_bytes) == 16, f"Expected 16 bytes, got {len(raw_bytes)}"\r\n        return cls.from_buffer_copy(raw_bytes)\r\n    \r\n    @classmethod\r\n    def from_doubles(cls, dbl1: float, dbl2: float) -> \'SerializedBlock\':\r\n        """Creates a SerializedBlock from two doubles."""\r\n        raw_bytes = struct.pack(\'<2d\', dbl1, dbl2)\r\n        return cls.from_bytes(raw_bytes)\r\n    \r\ndef serialize_to_doubles(blocks: List[\'Block\']) -> List[float]:\r\n    """Serializes a list of Blocks into a list of doubles."""\r\n    serialized_blocks = [SerializedBlock.from_block(block) for block in blocks]\r\n    return [dbl for serialized_block in serialized_blocks for dbl in serialized_block.pack_to_two_doubles()]')
    from detector_contour import ColorBlockDetectorContour
    from src.visualizer import BlockVisualizer
    from utils.serializer import *
    from src.color_def import COLOR_DEF_LL
    from src.utils.serializer import *
    from ctypes import sizeof
    import cv2
    
    RET_DBL_ARR_SIZE = 32
    MAX_RET_BLK_CNT = RET_DBL_ARR_SIZE * 8 / sizeof(SerializedBlock)
    
    def runPipeline(image, llrobot):
        detector = ColorBlockDetectorContour(COLOR_DEF_LL)
        visualizer = BlockVisualizer(show=False)
    
        blocks = detector.process_frame(image)
        image = visualizer.show_final_result(image, blocks)
    
        # sort the block by area, which can be calculated by contour
        blocks = sorted(blocks, key=lambda x: cv2.contourArea(x.contour), reverse=True)
        
        #select the first MAX_RET_BLK_CNT blocks if exceed
    
        if len(blocks) > MAX_RET_BLK_CNT:
            blocks = blocks[:MAX_RET_BLK_CNT]
    
        serialized_blocks = serialize_to_doubles(blocks)
    
        return [], image['Final_Detection'], serialized_blocks  # type: ignore