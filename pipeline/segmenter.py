import numpy as np
from typing import Literal, Callable, Optional
from sklearn.metrics import precision_recall_curve

from core.config import SegmentationConfig
from core.logger import Logger, setup_logger
from utils.helpers import normalize_data
from utils.decorator import log_call

class Segmenter:
    
    def __init__(self, logger : Logger = setup_logger()):
        self.logger = logger
        self.selector = {
            'thresholding': self.thresholding
        }
        
    @log_call()
    def segment_data(self, data: np.ndarray, method: Literal['thresholding'], segmentation_params: SegmentationConfig, ground_truth: Optional[np.ndarray] = None) -> Callable[..., list[np.ndarray]]:
        
        segmentation_params = segmentation_params or SegmentationConfig()
        
        if method not in self.selector:
            raise ValueError(f"Unknown segmentation method: {method}. Valid methods: {[key for key, value in self.selector.items()]}")
        return self.selector[method](data, ground_truth=ground_truth, **segmentation_params.to_dict())
    
    @log_call ()
    def thresholding(self, array: np.ndarray, threshold: float = 0.5, ground_truth: Optional[np.ndarray] = None) -> np.ndarray:
        # https://sirineamrane.medium.com/from-auc-roc-to-optimal-treshold-selection-a-guide-for-binary-classification-679bae8ea1bf
        arr_normalized = normalize_data(array)
        gt_normalized = normalize_data(ground_truth)
        if ground_truth is not None:
            precision, recall, thresholds = precision_recall_curve(gt_normalized.ravel(), arr_normalized.ravel())
            f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
            threshold = thresholds[np.argmax(f1_scores)]

        return (arr_normalized > threshold).astype(np.uint8), threshold



    