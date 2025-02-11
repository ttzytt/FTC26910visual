from color_detector import ColorDetector
from watershed_detector import WatershedDetector
from meanshift_detector import MeanshiftDetector
from src.visualizer import BlockVisualizer
from src.utils.serializer import *
from color_defs import COLOR_DEF_LL
from src.utils.serializer import *
from ctypes import sizeof
import cv2

MAX_RET_BLK_CNT = 5

def runPipeline(image, llrobot):
    # can change the algorithm with 3 options: contour, watershed, meanshift
    detector = ColorDetector(COLOR_DEF_LL)
    visualizer = BlockVisualizer(show=False)

    blocks = detector.process_frame(image)
    image = visualizer.gen_final_result(image, blocks)

    # sort the block by area, which can be calculated by contour
    blocks = sorted(blocks, key=lambda x: cv2.contourArea(x.contour), reverse=True)
    
    #select the first MAX_RET_BLK_CNT blocks if exceed

    if len(blocks) > MAX_RET_BLK_CNT:
        blocks = blocks[:MAX_RET_BLK_CNT]

    serialized_blocks = serialize_to_floats(blocks)

    return [], image, serialized_blocks 