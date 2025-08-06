import numpy as np
from numpy import ndarray
from typing import Literal, Callable, Optional, Tuple
from sklearn.metrics import precision_recall_curve

from core.config.experiment import SegmentationConfig
from core.io.logger import Logger, setup_logger
from core.utils.helpers import normalize_data

class Segmenter:
    
    def __init__(self, logger : Logger = setup_logger()):
        self.logger = logger
        self.selector = {
            'thresholding': self.thresholding
        }
        
    def segment_data(
        self, 
        data: ndarray, 
        method: Literal['thresholding'], 
        segmentation_params: SegmentationConfig, 
        ground_truth: Optional[ndarray] = None
    ) -> Callable[..., Tuple[ndarray, float]]:
        
        segmentation_params = segmentation_params or SegmentationConfig()
        
        if method not in self.selector:
            raise ValueError(f"Unknown segmentation method: {method}. Valid methods: {[key for key, value in self.selector.items()]}")
        return self.selector[method](data=data, ground_truth=ground_truth, **segmentation_params.to_dict())

    def thresholding(
        self, 
        data: ndarray, 
        threshold: float = 0.5, 
        ground_truth: Optional[ndarray] = None
    ) -> Tuple[ndarray, float]:
        # https://sirineamrane.medium.com/from-auc-roc-to-optimal-treshold-selection-a-guide-for-binary-classification-679bae8ea1bf
        data_normalized = normalize_data(data)
        if ground_truth is not None:
            gt_normalized = normalize_data(ground_truth)
            precision, recall, thresholds = precision_recall_curve(gt_normalized.ravel(), data_normalized.ravel())
            f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
            threshold = thresholds[np.argmax(f1_scores)]
        elif threshold is None:
            threshold = 0.5
        return (data_normalized > threshold).astype(np.uint8), threshold



    