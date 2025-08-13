import numpy as np
import pytest
from core.configs.config import SetupConfig, ExperimentConfig

class DummyEnhancer:
    def enhance_data(self, data, method, processing_params, enhancement_params, hessian_params):
        return data

class DummySegmenter:
    def segment_data(self, data, method, segmentation_params, ground_truth):
        return data, 0.5

class DummyEnhancementConfig:
    alpha = 0.1
    beta = 0.1
    gamma = 0.1
    

class DummyProcessingConfig:
    parallelize = False
    chunk_size = [1, 1, 1]
    overlap_size = 0
    

class DummyHessianConfig:
    mode = 'reflect'

class DummySegmentationConfig:
    threshold = 0.5

class DummyMethodsConfig:
    hessian = 'default'
    enhancer = 'frangi'
    segmenter = 'thresholding'
    

class DummyLoadConfig:
    normalize = True
    crop = False
    input_file = 'dummy_input'
    output_file = 'dummy_output'

def dummy_config():
    setup = SetupConfig(
        name='test', input_dir='.', output_dir='.', log_dir='.', log_file='test.log',
        debug_mode=True, display_mode=False, save_mode=False
    )
    experiment = ExperimentConfig(
        loading=DummyLoadConfig(),
        methods=DummyMethodsConfig(),
        processing=DummyProcessingConfig(),
        hessian=DummyHessianConfig(),
        enhancement=DummyEnhancementConfig(),
        segmentation=DummySegmentationConfig()
    )
    return setup, experiment
