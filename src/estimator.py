import numpy as np
from skimage.feature import hessian_matrix

from typing import Literal, Callable, TypedDict
from utils.logger import setup_logger
from utils.decorator import log_call, log_time
from utils.config import HessianParams

class Estimator:
    def __init__(self, method: Literal['default', 'farid']):
        self.method = method
        self.selector = {
            'default': self.default,
            'farid': self.farid
        }

    def compute_hessian(self) -> Callable[..., list[np.ndarray]]:
       
        if self.method not in self.selector:
            raise ValueError(f"Unknown differentiation method: {self.method}. Valid methods: {[key for key, value in self.selector.items()]}")
        return self.selector[self.method]
        
    def default(self, data, scale, **hessian_params) -> list[np.ndarray]:
        return hessian_matrix(image = data, sigma=scale, **hessian_params)
    
    def farid(self, hessian_params) -> list[np.ndarray]:
        pass
