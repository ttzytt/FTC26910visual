from src.detector_contour import ColorBlockDetectorContour
from src.detector_watershed import ColorBlockDetectorWatershed
from src.detector_meanshift import ColorBlockDetectorMeanShift
from src.visualizer import BlockVisualizer
from src.utils.serializer import *
from src.color_def import COLOR_DEF_LL
from src.utils.serializer import *
from ctypes import sizeof
import cv2

MAX_RET_BLK_CNT = 5

def runPipeline(image, llrobot):
    # can change the algorithm with 3 options: contour, watershed, meanshift
    detector = ColorBlockDetectorContour(COLOR_DEF_LL)
    visualizer = BlockVisualizer(show=False)

    blocks = detector.process_frame(image)
    image = visualizer.show_final_result(image, blocks)

    # sort the block by area, which can be calculated by contour
    blocks = sorted(blocks, key=lambda x: cv2.contourArea(x.contour), reverse=True)
    
    #select the first MAX_RET_BLK_CNT blocks if exceed

    if len(blocks) > MAX_RET_BLK_CNT:
        blocks = blocks[:MAX_RET_BLK_CNT]

    serialized_blocks = serialize_to_floats(blocks)

    return [], image, serialized_blocks 