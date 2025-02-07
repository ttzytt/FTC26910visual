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