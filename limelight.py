from src.detector import ColorBlockDetector
from src.visualizer import BlockVisualizer
from utils.serializer import *
import json

def runPipeline(image, llrobot):
    detector = ColorBlockDetector()
    visualizer = BlockVisualizer(show=False)

    blocks = detector.process_frame(image)
    image = visualizer.visualize(image, blocks, detector.get_debug_images())
    blocks_json = json.dumps([block.__dict__ for block in blocks], cls=NumpyEncoder)
    encoded = string_to_doubles(blocks_json)
    return [], image['Final_Detection'], encoded # type: ignore