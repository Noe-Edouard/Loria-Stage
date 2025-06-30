import numpy as np
import gc
import dask.array as da
from dask.diagnostics import ProgressBar
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from skimage.filters import frangi as frangi_skimage
from typing import Callable, Optional, Sequence, Literal, Tuple

from utils.decorator import log_call
from utils.helpers import normalize_data
from utils.config import HessianParams, EnhancementParams, ProcessingParams
from utils.logger import Logger, setup_logger


class Enhancer:
    
    def __init__(self, method: Literal['frangi'], logger : Logger = setup_logger()):
        self.logger = logger
        self.method = method
        self.selector = {
            'frangi': self.frangi,
        }

    @log_call()
    def enhance_data(
        self, 
        data: np.ndarray, 
        processing_params: ProcessingParams,
        enhancement_params: EnhancementParams, 
        hessian_params: HessianParams,
    ) -> np.ndarray:
        
        # Prepare params
        filter_params, enhancement_function, normalize, parallelize, chunk_size, overlap_size = self._prepare_enhancement(
            processing_params=processing_params,
            enhancement_params=enhancement_params,
            hessian_params=hessian_params,
        )

        # Processing (seq/para)
        processed_data = self._process_data(
            data=data,
            normalize=normalize,
            parallelize=parallelize,
            chunk_size=chunk_size,
            overlap_size=overlap_size,
            enhancement_function=enhancement_function,
            filter_params=filter_params,
        )
        
        return processed_data
    
    

    def _prepare_enhancement(
        self,
        processing_params: ProcessingParams,
        enhancement_params: EnhancementParams,
        hessian_params: HessianParams,
    ) -> tuple[dict, Callable, bool, bool, tuple[int, ...], int]:

        processing_params = processing_params.to_dict()
        enhancement_params = enhancement_params.to_dict()
        hessian_params = hessian_params.to_dict()
        
        if self.method not in self.selector:
            raise ValueError(f"Unknown enhancement method: {self.method}. Valid methods : {list(self.selector.keys())}")
        
        # Get enhancement function
        enhancement_function = self.selector[self.method]
        
        filter_params = enhancement_params.copy()
        parallelize = processing_params.get('parallelize', False)
        normalize = processing_params.get('normalize', False)

        
        # Nest hessian params
        if hessian_params:
            filter_params['hessian_params'] = hessian_params
            
        # Compute overlap size
        if processing_params.get('overlap_size', None) is None:
            scales_range = enhancement_params.get('scales_range', None)
            if scales_range:
                overlap_size = scales_range[1]
            else:
                scales = enhancement_params.get('scales', None)
                if scales:
                    overlap_size = int(np.max(scales))
                else:
                    overlap_size = 10  # default value
        else:
            overlap_size = processing_params['overlap_size']
    
        
        # Set Chunk size
        chunk_size = processing_params.get('chunk_size', None)
        
        return filter_params, enhancement_function, normalize, parallelize, chunk_size, overlap_size


    def _process_data(
        self,
        data: np.ndarray,
        normalize: bool,
        parallelize: bool,
        chunk_size: Tuple[int, ...],
        overlap_size: int,
        enhancement_function: Callable,
        filter_params: dict,
    ) -> np.ndarray:

        # Parallel processing
        if parallelize and data.ndim == 3:
            if chunk_size is None:
                chunk_size = tuple(min(64, s) for s in data.shape)
                
            dask_data = da.from_array(data, chunks=chunk_size)

            processed_chunks = da.map_overlap(
                dask_data,
                enhancement_function,
                depth=overlap_size,
                boundary='reflect',
                dtype=np.float32,
                **filter_params
            )

            print("Parallel processing in progress...")
            with ProgressBar():
                result = processed_chunks.compute()
        
        # Sequential processing (or 2D)
        else:
            print("Sequential processing in progress...")
            result = enhancement_function(data, **filter_params)
        
        if normalize:
            result = normalize_data(result)
        
        return result
    
    
    @log_call()
    def frangi(
        self,
        image: np.ndarray,
        hessian_function: Callable[..., list[np.ndarray]] = hessian_matrix,
        scales: Optional[Sequence[int]] = range(0, 10, 2),
        scales_number: Optional[int] = None,
        scales_range: Optional[Tuple[int, int]] = None,
        alpha: float = 0.5,
        beta: float = 0.5,
        gamma: Optional[float] = None,
        black_ridges: Optional[bool] = False,
        hessian_params: dict = {'mode': 'reflect', 'cval': 0, 'use_gaussian_derivatives': True},
        skimage = False,
    ) -> np.ndarray:
                
        
        if skimage:
            self.logger.info('The enhancement is done with frangi function from skimage.')
            # use_gaussian_derivative=True by default (can't be changed)
            return frangi_skimage(image, sigmas=scales, alpha=alpha, beta=beta, gamma=gamma, black_ridges = black_ridges)
        
        else:
            
            if scales_number and scales_range:
                scales = np.linspace(scales_range[0], scales_range[1], scales_number, dtype=int)
        
            image = image.astype(np.float32, copy=False)
            if not black_ridges:
                image = -image
            
            filtered_image = np.zeros_like(image)
            for scale in scales:
                hessian = hessian_function(image, sigma=scale, **hessian_params)
                
                eigvals = hessian_matrix_eigvals(hessian)
                eigvals = np.take_along_axis(eigvals, np.abs(eigvals).argsort(0), axis=0)
                # eigvals = np.sort(np.abs(eigvals), axis=0)
                
                # eigvals[eigvals <= 0] = 1e-10
                if image.ndim == 2:
                    lambda1, lambda2 = np.maximum(eigvals, 1e-10)
                    r_a = np.inf
                    r_b = lambda1 / lambda2
                    
                else:  # ndim == 3
                    lambda1, lambda2, lambda3 = np.maximum(eigvals, 1e-10)
                    r_a = lambda2 / lambda3
                    r_b = lambda1 / np.sqrt(lambda2 * lambda3)
                
                s = np.sqrt((eigvals**2).sum(axis=0))

                # Compute gamma
                if gamma is None:
                    gamma = s.max() / 2 if s.max() != 0 else 1
                self.logger.info(f'gamma = {gamma}')
                vesselness = 1.0 - np.exp(-(r_a**2) / (2 * alpha**2))  # Plateness
                vesselness *= np.exp(-(r_b**2) / (2 * beta**2))        # Blobness
                vesselness *= (1.0 - np.exp(-(s**2) / (2 * gamma**2))) # Brightness
                
                filtered_image = np.maximum(filtered_image, vesselness)
                
                # Free memory
                del hessian, eigvals, vesselness, s
                gc.collect()
                
            return filtered_image
        
   