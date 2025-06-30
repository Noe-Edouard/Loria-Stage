import numpy as np
from skimage.feature import hessian_matrix
from typing import Literal, Callable, TypedDict

from utils.logger import setup_logger
from utils.decorator import log_call, log_time

from utils.config import SegmentationParams


class Segmenter:
    def __init__(self, method: Literal['thresholding']):
        self.method = method
        self.selector = {
            'thresholding': self.thresholding
        }
    @log_call
    def segment_data(self, data: np.ndarray, segmentation_params: SegmentationParams) -> Callable[..., list[np.ndarray]]:
        
        segmentation_params = segmentation_params or SegmentationParams()
        
        if self.method not in self.selector:
            raise ValueError(f"Unknown segmentation method: {self.method}. Valid methods: {[key for key, value in self.selector.items()]}")
        return self.selector[self.method](data, **segmentation_params.to_dict())
    
    @log_call 
    def thresholding(self, array: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (array > threshold).astype(np.uint8)


    