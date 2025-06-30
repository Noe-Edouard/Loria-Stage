import numpy as np

from src.enhancer import Enhancer
from utils.loader import Loader
from utils.viewer import Viewer
from utils.saver import Saver
from utils.logger import setup_logger
from utils.helpers import normalize_data


def test_frangi():
    logger = setup_logger(name='tests', debug_mode=True)
    loader = Loader(input_dir='data/test', logger=logger)
    enhancer = Enhancer(method='frangi', logger=logger)
    viewer = Viewer()
    saver = Saver(output_dir='outputs/test', logger=logger)
    
    # 2D
    image_2d = loader.load_data('test.jpg')
    
    result_2d_own = normalize_data(enhancer.frangi(image_2d, skimage=False))
    result_2d_skimage = normalize_data(enhancer.frangi(image_2d, skimage=True))
    
    figure_2d = viewer.display_images([image_2d, result_2d_own, result_2d_skimage], ['Original', 'Enhance (frangi own)', 'Enhance (frangi skimage)'])
    histogram_2d = viewer.display_histogram([result_2d_own - result_2d_skimage], ['Histogram of (result_frangi_own - result_frangi_skimage) for 2D test'])
    
    saver.save_plot(figure_2d, 'test_enhancer_2d_comparison')
    saver.save_plot(histogram_2d, 'test_enhance_2dr_histogram')
    
    assert np.mean(np.abs(result_2d_skimage - result_2d_own)) < 1e-2
        
    # 3D
    image_3d = loader.load_data('test.nii', crop=True)
    
    result_3d_own = enhancer.frangi(image_3d, skimage=False, gamma=15)
    result_3d_skimage = enhancer.frangi(image_3d, skimage=True, gamma=15)
    
    figure_3d = viewer.display_slices([image_3d, result_3d_own, result_3d_skimage], ['Original', 'Enhance (frangi own)', 'Enhance (frangi skimage)'])
    histogram_3d = viewer.display_histogram([result_3d_skimage - result_3d_own], ['Histogram of (result_frangi_own - result_frangi_skimage) for 3D test'])
    
    saver.save_animation(figure_3d, 'test_enhancer_3d_comparison')
    saver.save_plot(histogram_3d, 'test_enhancer_3d_histogram')
    
    assert np.max(np.abs(result_2d_own - result_2d_skimage)) < 1e-2
