import numpy as np
from skimage.feature import hessian_matrix
from typing import Literal, Callable

from utils.decorator import log_call, log_time
from utils.config import HessianParams
from utils.logger import Logger, setup_logger


class Derivator:
    
    def __init__(self, method: Literal['default', 'farid'], logger: Logger = setup_logger()):
        self.logger = logger
        self.method = method
        self.selector = {
            'default': self.default,
            'farid': self.farid
        }

    def hessian_function(self) -> Callable[..., list[np.ndarray]]:
       
        if self.method not in self.selector:
            raise ValueError(f"Unknown differentiation method: {self.method}. Valid methods: {[key for key, value in self.selector.items()]}")
        return self.selector[self.method]
        
    def default(self, data, sigma, **hessian_params) -> list[np.ndarray]:
        return hessian_matrix(image = data, sigma=sigma, **hessian_params)
    
    def farid(self, hessian_params) -> list[np.ndarray]:
        pass
