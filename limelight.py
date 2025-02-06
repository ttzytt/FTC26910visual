from src.detector import ColorBlockDetector
from src.visualizer import BlockVisualizer
from src.utils.double_encoder import *
import json

def runPipeline(image, llrobot):
    detector = ColorBlockDetector()
    visualizer = BlockVisualizer(show=False)

    blocks = detector.process_frame(image)
    image = visualizer.visualize(image, blocks, detector.get_debug_images())

    return [], image['Final_Detection'], [] # type: ignore