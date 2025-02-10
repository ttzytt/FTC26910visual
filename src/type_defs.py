from typing import Tuple
import numpy as np 
import numpy.typing as npt
from typing import TypeVar, Annotated, Literal, Dict

hsv_t = Tuple[int, int, int]
bgr_t = Tuple[int, int, int]


Dtype = TypeVar('Dtype', bound=np.generic)


array_NxNx3_t = Annotated[npt.NDArray[Dtype], Literal['N', 'N', 3]]
array_NxNx1_t = Annotated[npt.NDArray[Dtype], Literal['N', 'N', 1]]

img_t = array_NxNx3_t

img_hsv_t = array_NxNx3_t

img_bgr_t = array_NxNx3_t

img_gray_t = array_NxNx1_t

VizResults = Dict[str, img_t | img_gray_t]