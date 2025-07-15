import numpy as np

from pipeline.enhancer import Enhancer
from pipeline.derivator import Derivator
from core.loader import Loader
from view.viewer import Viewer
from core.saver import Saver
from core.logger import setup_logger
from utils.helpers import normalize_data

def test_farid():
    logger = setup_logger(name='tests', debug_mode=True)
    loader = Loader(input_dir='data/test', logger=logger)
    enhancer = Enhancer(method='frangi', logger=logger)
    hessian_default = Derivator('default', logger).select_hessian_function()
    hessian_farid = Derivator('farid', logger).select_hessian_function()
    
    viewer = Viewer()
    saver = Saver(experiment_name="test_farid", output_dir='test', logger=logger)
    
    # 2D
    # image_2d = loader.load_data('test.jpg')
    
    # result_2d_default = normalize_data(enhancer.frangi(image=image_2d, hessian_function=hessian_farid))
    # result_2d_farid = normalize_data(enhancer.frangi(image=image_2d, hessian_function=hessian_farid))
    
    # figure_2d = viewer.display_images([image_2d, result_2d_default, result_2d_farid], ['Original', 'Hessian (default)', 'Hessian (farid)'])
    # histogram_2d = viewer.display_histogram([result_2d_default - result_2d_farid], ['Histogram of (result_hessian_default) - result_hessian_farid) for 2D test'])
    
    # saver.save_plot(figure_2d, 'test_farid_2d_comparison')
    # saver.save_plot(histogram_2d, 'test_farid_2d_histogram')
    
    # assert np.mean(np.abs(result_2d_default - result_2d_farid)) < 1e-3
        
    # 3D
    image_3d = loader.load_data('test.nii', crop=True)
    
    # result_3d_default = normalize_data(enhancer.frangi(image=image_3d, hessian_function=hessian_default, gamma=15))
    result_3d_farid_true = normalize_data(enhancer.frangi(image=image_3d, hessian_function=hessian_farid, gamma=15, hessian_params={'mode': 'reflect', 'cval': 0, 'use_gaussian_derivatives': True}))
    result_3d_farid_false = normalize_data(enhancer.frangi(image=image_3d, hessian_function=hessian_farid, gamma=15, hessian_params={'mode': 'reflect', 'cval': 0, 'use_gaussian_derivatives': False}))
    result_3d_farid_none = normalize_data(enhancer.frangi(image=image_3d, hessian_function=hessian_farid, gamma=15, hessian_params={'mode': 'reflect', 'cval': 0, 'use_gaussian_derivatives': None}))
    
    figure_3d = viewer.display_slices([result_3d_farid_true, result_3d_farid_false, result_3d_farid_none], ['True', 'False', 'None'])
    
    saver.save_animation(figure_3d, 'test_enhancer_3d_comparison')
    

    
    # figure_3d = viewer.display_slices([image_3d, result_3d_default, result_3d_farid], ['Original', 'Hessian default', 'Hessian farid'])
    # histogram_3d = viewer.display_histogram([result_3d_default - result_3d_farid], ['Histogram of (result_hessian_default - result_hessian_farid) for 3D test'])
    
    # saver.save_animation(figure_3d, 'test_enhancer_3d_comparison')
    # saver.save_plot(histogram_3d, 'test_enhancer_3d_histogram')
    
    # assert np.max(np.abs(result_3d_farid - result_3d_default)) < 1e-3
