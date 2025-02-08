from abc import ABC, abstractmethod
from src.block import Block
from typing import List
import numpy as np
from typing import TypedDict, Dict 

VizResults = Dict[str, np.ndarray]

class Detector(ABC):
    def __init__(self):
        pass
    @abstractmethod
    def process_frame(self, frame: np.ndarray) -> List[Block]:
        pass
    
    @abstractmethod
    def get_debug_images(self) -> Dict[str, np.ndarray]: 
        pass
    
    # def _preprocess(self, frame: np.ndarray) -> np.ndarray: 
    #     # Example preprocessing: convert frame to grayscale
    #     pass