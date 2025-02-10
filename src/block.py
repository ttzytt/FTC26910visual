import numpy as np
from dataclasses import dataclass, field
from src.color_defs import Color
from typing import List, Tuple

@dataclass
class Block:
    """Represents a detected color block with position, size, angle, color info, and HSV stats."""
    center: Tuple[float, float]
    size: Tuple[float, float]
    angle: float
    color: Color
    color_std: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    mean_hsv: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    # store the absolute contour for visualization
    contour: np.ndarray = field(default_factory=lambda: np.array([]))
