import numpy as np
from numpy import ndarray
import gc
import dask.array as da
from dask.diagnostics import ProgressBar
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from skimage.filters import frangi as frangi_skimage
from typing import Callable, Optional, Sequence, Literal, Tuple

from core.utils.helpers import normalize_data
from core.config.experiment import HessianConfig, EnhancementConfig, ProcessingConfig
from core.io.logger import Logger, setup_logger
from core.utils.helpers import check_gpu_available


class Enhancer:
    
    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu and check_gpu_available()
        if use_gpu and check_gpu_available():
            import cupy as cp
            self.xp = cp
        else:
            import numpy as np
            self.xp = np
        
        self.selector = {
            'frangi': self.frangi,
        }

    def select_enhancement_function(self, method: Literal['frangi']) -> Callable[..., ndarray]:
        if method not in self.selector:
            raise ValueError(f"Unknown enhancement method: {method}. Valid methods : {list(self.selector.keys())}")
        
        return self.selector[method]
    
    def frangi(
        self,
        image: ndarray,
        hessian_function: Callable[..., list[ndarray]] = hessian_matrix,
        hessian_params: dict = {'mode': 'reflect', 'cval': 0},
        scales: Optional[Sequence[int]] = range(0, 10, 2),
        alpha: float = 0.5,
        beta: float = 0.5,
        gamma: Optional[float] = None,
        black_ridges: Optional[bool] = True,
        skimage = False,
    ) -> ndarray:
       
        xp = self.xp
        hessian_function.__self__.use_gpu = self.use_gpu
        
        if skimage:
            # use_gaussian_derivative=True by default (can't be changed)
            return frangi_skimage(image, sigmas=scales, alpha=alpha, beta=beta, gamma=gamma, black_ridges = black_ridges)
        
        else:
            
            image = image.astype(xp.float32, copy=False)
            if not black_ridges:
                image = -image
            
            filtered_image = xp.zeros_like(image)
            for scale in scales:
                hessian = hessian_function(image, sigma=scale, **hessian_params)
                
                eigvals = hessian_matrix_eigvals(hessian)
                eigvals = xp.take_along_axis(eigvals, xp.abs(eigvals).argsort(0), axis=0)
                
                if image.ndim == 2:
                    lambda1, lambda2 = xp.maximum(eigvals, 1e-10)
                    r_a = xp.inf
                    r_b = lambda1 / lambda2
                    
                else:  # ndim == 3
                    lambda1, lambda2, lambda3 = xp.maximum(eigvals, 1e-10)
                    r_a = lambda2 / lambda3
                    r_b = lambda1 / xp.sqrt(lambda2 * lambda3)
                
                s = xp.sqrt((eigvals**2).sum(axis=0))

                # Compute gamma
                if gamma is None:
                    gamma = s.max() / 2 if s.max() != 0 else 1
                
                vesselness = 1.0 - xp.exp(-(r_a**2) / (2 * alpha**2))  # Plateness
                vesselness *= xp.exp(-(r_b**2) / (2 * beta**2))        # Blobness
                vesselness *= (1.0 - xp.exp(-(s**2) / (2 * gamma**2))) # Brightness
                
                filtered_image = xp.maximum(filtered_image, vesselness)
                
                # Free memory
                del hessian, eigvals, vesselness, s
                gc.collect()
                
            return filtered_image
        
   