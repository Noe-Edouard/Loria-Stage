from pathlib import Path
from numpy import ndarray
from typing import Optional 

from core.logger import setup_logger
from core.config import SetupConfig, ExperimentConfig, ExperimentData
from utils.decorator import log_call, log_time
from core.loader import Loader
from view.viewer import Viewer
from core.saver import Saver
from utils.decorator import log_section
from pipeline.derivator import Derivator
from pipeline.enhancer import Enhancer 
from pipeline.segmenter import Segmenter


class Pipeline:
    
    @log_call()
    def __init__(self, setup: SetupConfig):
        self.setup = setup
        
        # Paths
        self.log_dir = Path(self.setup.log_dir)
        self.input_dir = Path(self.setup.input_dir)
        self.output_dir = Path(self.setup.output_dir)

        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        
        
        # Logger
        self.log_file = self.setup.log_file
        self.debug_mode = self.setup.debug_mode
        self.logger = setup_logger(self.log_file, log_dir=self.log_dir, debug_mode=self.debug_mode)

        # Loader
        self.loader = Loader(self.input_dir, self.logger)
        
        # Derivator
        self.derivator = Derivator(self.logger)
        
        # Enhancer
        self.enhancer = Enhancer(self.logger)
        
        # Segmenter
        self.segmenter = Segmenter(self.logger)
        
        # Viewer
        self.display_mode = self.setup.display_mode
        if self.display_mode:
            self.viewer = Viewer()
        
        # Saver
        self.save_mode = self.setup.save_mode
        if self.save_mode:
            self.saver = Saver(experiment_name=self.setup.name, output_dir=self.output_dir, logger=self.logger)
        
        self.logger.info(f'[INIT] Pipeline initialized - Experiment {self.setup.name}')
        
        
    @log_section("Pipeline execution")
    @log_time()
    def run(self, config: ExperimentConfig, data_raw: Optional[ndarray] = None, ground_truth: Optional[ndarray] = None) -> ExperimentData:
        # Load data
        if data_raw is None:
            data_raw = self.loader.load_data(
                filename=config.load.input_file,
                normalize=config.load.normalize,
                crop=config.load.crop,
                target_shape=config.load.target_shape,
            )
        # Select Derivator
        config.enhancement.hessian_function = self.derivator.select_hessian_function(config.methods.derivator)
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
            ground_truth=ground_truth,
            method=config.methods.segmenter,
            segmentation_params=config.segmentation
        )
        config.segmentation.threshold = threshold
        # Save data
        if self.save_mode:
            self.saver.save_data(data_enhanced, f'{config.load.output_file}_enhanced', '.npz')
            self.saver.save_data(data_segmented, f'{config.load.output_file}_segmented', '.npz')
        # Display analytics
        if self.display_mode:
            if data_raw.ndim == 2:
                figure = self.viewer.display_images([data_raw, data_enhanced, data_segmented],["RAW", "ENHANCED", "SEGMENTED"])
                if self.save_mode:
                    self.saver.save_plot(figure, filename=config.load.output_file)
            else:
                histogram = self.viewer.display_histograms([data_raw, data_enhanced, data_segmented], ['RAW', 'ENHANCED', 'SEGMENTED'])
                slices = self.viewer.display_slices([data_raw, data_enhanced, data_segmented], ['RAW', 'ENHANCED', 'SEGMENTED'])
                volume = self.viewer.display_volume(volume=data_enhanced, threshold=config.segmentation.threshold)
                if self.save_mode:
                    self.saver.save_plot(histogram, config.load.output_file)
                    self.saver.save_animation(slices, config.load.output_file)
                    self.saver.save_plot(volume, config.load.output_file)
        # Return results
        experiment_data = ExperimentData(
            enhanced=data_enhanced,
            segmented=data_segmented,
            config=config,
        )
        return experiment_data