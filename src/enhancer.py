import numpy as np
import gc
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from typing import Callable, Optional, Sequence, Literal, Tuple, Dict, Any

import dask.array as da
from dask.diagnostics import ProgressBar

from utils.decorator import log_call
from utils.helpers import normalize_data
from utils.config import HessianParams, EnhancementParams, ProcessingParams


class Enhancer:
    
    def __init__(self, method: Literal['frangi']):
        self.method = method
        self.selector = {
            'frangi': self.frangi
        }

    @log_call
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


    @log_call
    def frangi(
        self,
        image: np.ndarray,
        compute_hessian: Callable[..., list[np.ndarray]] = hessian_matrix,
        scales: Optional[Sequence[int]] = range(1, 10, 2),
        scales_number: Optional[int] = None,
        scales_range: Optional[Tuple[int, int]] = None,
        alpha: float = 0.5,
        beta: float = 0.5,
        gamma: Optional[float] = None,
        hessian_params: dict = {'mode': 'reflect', 'cval': 0, 'use_gaussian_derivatives': False},
    ) -> np.ndarray:
        
        if scales_number and scales_range:
            scales = np.linspace(scales_range[0], scales_range[1], scales_number, dtype=int)
        
        image = image.astype(np.float32, copy=False)
        filtered_image = np.zeros_like(image)
        
        for scale in scales:
            hessian = compute_hessian(image, scale=scale, **hessian_params)
            
            eigvals = hessian_matrix_eigvals(hessian)
            # eigvals = np.take_along_axis(eigvals, np.abs(eigvals).argsort(0), axis=0)
            eigvals = np.sort(np.abs(eigvals), axis=0)
            
            
            # !div0
            eigvals[eigvals == 0] = 1e-10

            if image.ndim == 2:
                lambda1, lambda2 = eigvals
                r_b = np.abs(lambda1) / np.abs(lambda2)
                vesselness = 1.0
            else:  # ndim == 3
                lambda1, lambda2, lambda3 = eigvals
                r_a = np.abs(lambda2) / np.abs(lambda3)
                r_b = np.abs(lambda1) / np.sqrt(np.abs(lambda2 * lambda3))
                vesselness = 1.0 - np.exp(-(r_a**2) / (2 * alpha**2))    # Plateness
            
            s = np.sqrt((eigvals**2).sum(axis=0))

            # Compute gamma
            if gamma is None:
                gamma = s.max() / 2 if s.max() != 0 else 1

            vesselness *= np.exp(-(r_b**2) / (2 * beta**2))              # Blobness
            vesselness *= (1.0 - np.exp(-(s**2) / (2 * gamma**2)))       # Brightness
            
            filtered_image = np.maximum(filtered_image, vesselness)
            
            # Free memory
            del hessian, eigvals, vesselness, s
            gc.collect()
            
        return filtered_image
    
    