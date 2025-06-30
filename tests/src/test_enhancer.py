import numpy as np
import pytest
from src.enhancer import Enhancer
from skimage.feature import hessian_matrix


@pytest.fixture
def enhancer():
    return Enhancer(method='frangi')
    

def test_frangi(enhancer):
    # 2D
    image_2d = np.random.rand(64, 64).astype(np.float32)
    result_2d = enhancer.frangi(image_2d, scales_range=(1, 10), scales_number = 2)
    assert result_2d.shape == image_2d.shape
    assert result_2d.dtype == np.float32
    
    # 3D
    image_3d = np.random.rand(32, 32, 32).astype(np.float32)
    result_3d = enhancer.frangi(image_3d)
    assert result_3d.shape == image_3d.shape
    assert result_3d.dtype == np.float32

def test_enhancement(enhancer):
    # 2D
    image_2d = np.random.rand(64, 64).astype(np.float32)
    result_2d = enhancer.apply_enhancement(image_2d)
    assert result_2d.shape == image_2d.shape
    
    # 3D SEQ
    image_3d = np.random.rand(32, 32, 32).astype(np.float32)
    result_3d = enhancer.apply_enhancement(image_3d, parallelize=False)
    assert result_3d.shape == image_3d.shape
    
    # 3D PARA
    image_para = np.random.rand(64, 64, 64).astype(np.float32)
    result_para = enhancer.apply_enhancement(
        image_para,
        parallelize=True,
        chunk_size=(32, 32, 32)
    )
    assert result_para.shape == image_para.shape
