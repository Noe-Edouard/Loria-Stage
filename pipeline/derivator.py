import numpy as np
from typing import Literal, Callable
from skimage.feature import hessian_matrix
from scipy.ndimage import gaussian_filter
from scipy.ndimage import convolve1d

from core.logger import Logger, setup_logger


class Derivator:
    
    def __init__(self, logger: Logger = setup_logger()):
        self.logger = logger
        self.selector = {
            'default': self.default,
            'gaussian': self.gaussian,
            'farid': self.farid,
            'cubic': self.cubic,
            'trigonometric': self.trigonometric,
            'catmull': self.catmull,
            'bspline': self.bspline,
            'bezier': self.bezier,
        }

    def select_hessian_function(self, method: Literal['default', 'gaussian', 'farid', 'cubic', 'trigonometric', 'catmull', 'bspline', 'bezier']) -> Callable[..., list[np.ndarray]]:

        if method not in self.selector:
            raise ValueError(f"Unknown differentiation method: {method}. Valid methods: {[key for key, value in self.selector.items()]}")
        
        return self.selector[method]
        
    
    def compute_hessian(self, data: np.ndarray, filters: list[list], sigma: float = 1.0, mode:str = 'reflect') -> list[np.ndarray]:
            
            p  = np.array(filters[0])
            d1 = np.array(filters[1])  
            d2 = np.array(filters[2])
            
            data = gaussian_filter(data, sigma=sigma)
                
            if data.ndim == 2:
                Hxx = convolve1d(convolve1d(data, d2, axis=1, mode=mode), p, axis=0, mode=mode)  # Dxx = f * d2_x * p_y
                Hyy = convolve1d(convolve1d(data, p, axis=1, mode=mode), d2, axis=0, mode=mode)  # Dxx = f * d2_y * p_x
                Hxy = convolve1d(convolve1d(data, d1, axis=1, mode=mode), d1, axis=0, mode=mode) # Dxy = f * d1_x * d1_y
                
                return Hxx, Hxy, Hyy
            

            elif data.ndim ==3:
                Hxx = convolve1d(convolve1d(convolve1d(data, d2, axis=0, mode=mode),  p, axis=1, mode=mode),  p, axis=2, mode=mode)   # Dxx = f * d2_x * p_y  * p_z
                Hyy = convolve1d(convolve1d(convolve1d(data, p,  axis=0, mode=mode), d2, axis=1, mode=mode),  p, axis=2, mode=mode)   # Dxx = f * p_x  * d2_y * p_z
                Hzz = convolve1d(convolve1d(convolve1d(data, p, axis=0, mode=mode),   p, axis=1, mode=mode), d2, axis=2, mode=mode)   # Dxx = f * p_x  * p_y  * d2_z

                Hxy = convolve1d(convolve1d(convolve1d(data, d1, axis=0, mode=mode), d1, axis=1, mode=mode),  p, axis=2, mode=mode)   # Dxx = f * d1_x * d1_y * p_z
                Hyz = convolve1d(convolve1d(convolve1d(data, p,  axis=0, mode=mode), d1, axis=1, mode=mode), d1, axis=2, mode=mode)   # Dxx = f * p_x  * d1_y * d1_z
                Hzx = convolve1d(convolve1d(convolve1d(data, d1, axis=0, mode=mode),  p, axis=1, mode=mode), d1, axis=2, mode=mode)   # Dxx = f * d1_x * p_y  * d1_z
                
            return  Hxx, Hxy, Hzx, Hyy, Hyz, Hzz
    
    
    def default(self, data: np.ndarray, sigma: int, mode: str = 'reflect', cval: float = 0.0, order: str = 'rc', **unused) -> list[np.ndarray]:
        return hessian_matrix(
            image = data, 
            sigma=sigma, 
            mode=mode, 
            cval=cval, 
            order=order, 
            use_gaussian_derivatives=False
        )
        
    def gaussian(self, data: np.ndarray, sigma: int, mode: str = 'reflect', cval: float = 0.0, order: str = 'rc', **unused) -> list[np.ndarray]:
        return hessian_matrix(
            image = data, 
            sigma=sigma, 
            mode=mode, 
            cval=cval, 
            order=order, 
            use_gaussian_derivatives=True
        )
    
    
    def farid(self, data: np.ndarray, sigma: float = 1.0, mode: str = 'reflect', order:int = 7, **unused):
        if order == 5:
            p  = [0.030320, 0.249724, 0.439911, 0.249724, 0.030320]
            d1 = [-0.104550, -0.292315, 0.0, 0.292315, 0.104550]
            d2 = [0.232905, 0.002668, -0.471147, 0.002668, 0.232905]
        elif order == 7:
            p  = [0.0047, 0.0693, 0.2454, 0.3611, 0.2454, 0.0693, 0.0047]
            d1 = [0.0187, 0.1253, 0.1930, 0.0, -0.1930, -0.1253, -0.0187]
            d2 = [0.0553, 0.1378, -0.0566, -0.2731, -0.0566, 0.1378, 0.0553]
        
        return self.compute_hessian(data=data, filters=[p, d1, d2], sigma=sigma, mode=mode) 
    
    def cubic(self, data: np.ndarray, sigma: float = 1.0, mode: str = 'reflect', **unused):
        p  = [0.0039, -0.0703, 0.2461, 0.6406, 0.2461, -0.0703, 0.0039]
        d1 = [-0.0026, 0.0938, -0.6797, 0.0, 0.6797, -0.0938, 0.0026]
        d2 = [-0.0312, 0.3125, 0.0312, -0.6250, 0.0312, 0.3125, -0.0312]
        
        return self.compute_hessian(data=data, filters=[p, d1, d2], sigma=sigma, mode=mode) 


    def trigonometric(self, data: np.ndarray, sigma: float = 1.0, mode: str = 'reflect', **unused):
        p  = [0.0043, -0.0745, 0.2457, 0.6490, 0.2457, -0.0745, 0.0043]
        d1 = [-0.0061, 0.1966, -1.3282, 0.0, 1.3282, -0.1966, 0.0061]
        d2 = [-0.1272, 1.2203, 0.1272, -2.4405, 0.1272, 1.2203, -0.1272]
        
        return self.compute_hessian(data=data, filters=[p, d1, d2], sigma=sigma, mode=mode) 


    def catmull(self, data: np.ndarray, sigma: float = 1.0, mode: str = 'reflect', **unused):
        p  = [0.0039, -0.0703, 0.2461, 0.6406, 0.2461, -0.0703, 0.0039]
        d1 = [-0.0078, 0.1562, -0.7891, 0.0, 0.7891, -0.1562, 0.0078]
        d2 = [-0.0312, 0.3125, 0.0312, -0.6250, 0.0312, 0.3125, -0.0312]
        
        return self.compute_hessian(data=data, filters=[p, d1, d2], sigma=sigma, mode=mode) 


    def bspline(self, data: np.ndarray, sigma: float = 1.0, mode: str = 'reflect', **unused):
        p  = [0.0004, 0.0200, 0.2496, 0.4601, 0.2496, 0.0200, 0.0004]
        d1 = [-0.0026, -0.0729, -0.3464, 0.0, 0.3464, 0.0729, 0.0026]
        d2 = [0.0104, 0.2292, -0.0104, -0.4583, -0.0104, 0.2292, 0.0104]
        
        return self.compute_hessian(data=data, filters=[p, d1, d2], sigma=sigma, mode=mode) 


    def bezier(self, data: np.ndarray, sigma: float = 1.0, mode: str = 'reflect', **unused):
        p  = [0.0156, 0.0938, 0.2344, 0.3125, 0.2344, 0.0938, 0.0156]
        d1 = [0.0938, 0.3750, 0.4688, 0.0, -0.4688, -0.3750, -0.0938]
        d2 = [0.3750, 0.7500, -0.3750, -1.5000, -0.3750, 0.7500, 0.3750]
        
        return self.compute_hessian(data=data, filters=[p, d1, d2], sigma=sigma, mode=mode) 




