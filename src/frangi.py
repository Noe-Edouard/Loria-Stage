from skimage.feature import hessian_matrix, hessian_matrix_eigvals
import numpy as np
import gc

def frangi_filter(image, scales=range(1, 10, 2), alpha=0.5, beta=0.5, gamma=None, mode='reflect', cval=0):
    
    image = image.astype(np.float32, copy=False)
    filtered_image = np.zeros_like(image)
    
    for scale in scales:
        hess = hessian_matrix(image, scale, mode=mode, cval=cval, use_gaussian_derivatives=True)
        hess = [h.astype(np.float32) for h in hess]
        
        eigvals = hessian_matrix_eigvals(hess)
        eigvals = np.sort(np.abs(eigvals), axis=0)
        
        if image.ndim == 2:
            lambda1, lambda2 = np.maximum(eigvals, 1e-10)
            r_b = abs(lambda1) / abs(lambda2) 
            vesselness = 1.0
            
        else:  # ndim == 3
            lambda1, lambda2, lambda3 = np.maximum(eigvals, 1e-10)
            r_a = abs(lambda2) / abs(lambda3) 
            r_b = abs(lambda1) / np.sqrt(abs(lambda2 * lambda3))  
            vesselness = 1.0 - np.exp(-(r_a**2) / (2 * alpha**2))  # plateness
            
        s = np.sqrt((eigvals**2).sum(0))
        if gamma is None:
            gamma = s.max() / 2 if s.max() != 0 else 1
            # print(f'gamma = {gamma}')

        vesselness *= np.exp(-(r_b**2) / (2 * beta**2))            # blobness
        vesselness *= 1.0 - np.exp(-(s**2) / (2 * gamma**2))       # brightness
        filtered_image = np.maximum(filtered_image, vesselness)
        
        # Free memory
        del hess, eigvals, vesselness, s, r_b
        gc.collect()
        
    return filtered_image
