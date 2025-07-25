from pathlib import Path
from numpy import ndarray
from typing import Optional 

from core.logger import setup_logger
from core.config import SetupConfig, ExperimentConfig, ExperimentData
from core.loader import Loader
from core.saver import Saver
from view.viewer import Viewer
from utils.decorator import log_section, log_time
from pipeline.derivator import Derivator
from pipeline.enhancer import Enhancer 
from pipeline.segmenter import Segmenter


class Pipeline:
    
    def __init__(self, setup: SetupConfig):
        
        # Setup
        self.setup = setup

        # Logger
        self.logger = setup_logger(log_file=self.setup.log_file, debug_mode=self.setup.debug_mode)

        # Loader
        self.loader = Loader(input_dir=self.setup.input_dir, logger=self.logger)
        
        # Derivator
        self.derivator = Derivator(logger=self.logger)
        
        # Enhancer
        self.enhancer = Enhancer(logger=self.logger)
        
        # Segmenter
        self.segmenter = Segmenter(logger=self.logger)

        # Viewer
        self.viewer = Viewer(display_mode=self.setup.display_mode)
        
        # Saver
        self.saver = Saver(experiment_name=self.setup.name, output_dir=self.setup.output_dir, logger=self.logger)
        
        self.logger.info(f'[INIT] Pipeline initialized - Experiment {self.setup.name}')
    
    def display_analytics(self, data_raw: ndarray, data_enhanced: ndarray, data_segmented: ndarray, config: ExperimentConfig):
        if data_raw.ndim == 2:
            figure = self.viewer.display_images([data_raw, data_enhanced, data_segmented], ["RAW", "ENHANCED", "SEGMENTED"])
            if self.setup.save_mode:
                self.saver.save_plot(figure, filename='results')
        else:
            histogram = self.viewer.display_histograms([data_raw, data_enhanced, data_segmented], ['RAW', 'ENHANCED', 'SEGMENTED'])
            slices = self.viewer.display_slices([data_raw, data_enhanced, data_segmented], ['RAW', 'ENHANCED', 'SEGMENTED'])
            volume = self.viewer.display_volume(volume=data_enhanced, threshold=config.segmentation.threshold)
            if self.setup.save_mode:
                self.saver.save_plot(histogram, 'histogram')
                self.saver.save_animation(slices, 'slices')
                self.saver.save_plot(volume, 'volume')
        
    @log_section("Pipeline execution")
    @log_time()
    def run(self, config: ExperimentConfig, data_raw: Optional[ndarray] = None, data_gt: Optional[ndarray] = None) -> ExperimentData:
        # Load Data
        if data_raw is None:
            data_raw = self.loader.load_data(
                filename=config.load.input_file,
                normalize=config.load.normalize,
                crop=config.load.crop,
                target_shape=config.load.target_shape,
            )
        if data_gt is None:
            if config.load.gt_file is not None:
                data_gt = self.loader.load_data(
                    filename=config.load.gt_file,
                    normalize=config.load.normalize,
                    crop=config.load.crop,
                    target_shape=config.load.target_shape,
                )
            else :
                data_gt = None
            
        # Select Derivator
        config.enhancement.hessian_function = (
            self.derivator.select_hessian_function(config.methods.derivator)
        )
        
        # Enhance Data
        data_enhanced = self.enhancer.enhance_data(
            data=data_raw,
            method=config.methods.enhancer,
            processing_params=config.processing,
            enhancement_params=config.enhancement,
            hessian_params=config.hessian
        )
        
        # Segment Data
        data_segmented, threshold = self.segmenter.segment_data(
            data=data_enhanced,
            ground_truth=data_gt,
            method=config.methods.segmenter,
            segmentation_params=config.segmentation
        )
        config.segmentation.threshold = threshold
        
        # Save data
        if self.setup.save_mode:
            self.saver.save_data(data_enhanced, f'data_enhanced', '.npz')
            self.saver.save_data(data_segmented, f'data_segmented', '.npz')

        # Display analytics
        self.display_analytics(data_raw, data_enhanced, data_segmented, config)
               
        # Return results
        experiment_data = ExperimentData(
            enhanced=data_enhanced,
            segmented=data_segmented,
            config=config,
        )
        return experiment_data