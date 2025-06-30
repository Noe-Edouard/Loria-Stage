from pathlib import Path
from datetime import datetime 

from utils.logger import setup_logger
from utils.config import Config
from utils.decorator import log_call, log_time
from utils.loader import Loader
from utils.viewer import Viewer
from utils.saver import Saver

from src.derivator import Derivator
from src.enhancer import Enhancer 
from src.segmenter import Segmenter


class Pipeline:
    
    @log_call()
    def __init__(self, config: Config):
        self.config = config.config
        
        # Paths
        self.run_name = self.config.experiment.name
        self.input_file = self.config.experiment.input_file
        self.log_file = self.config.experiment.log_file
        
        self.input_dir = Path(self.config.experiment.input_dir)
        self.output_dir = Path(self.config.experiment.output_dir)
        self.log_dir = Path(self.config.experiment.log_dir)
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Logger
        self.debug_mode = self.config.experiment.debug_mode
        self.logger = setup_logger(self.log_file, log_dir=self.log_dir, debug_mode=self.debug_mode)
        self.logger.info(f'[INIT] Pipeline initialized - Experiment {self.config.experiment.name}')

        # Loader
        self.loader = Loader(self.input_dir, self.logger)
        
        # Derivator
        self.derivator = Derivator(self.config.methods.derivator, self.logger)
        
        # Enhancer
        self.enhancer = Enhancer(self.config.methods.enhancer, self.logger)
        
        # Segmenter
        self.segmenter = Segmenter(self.config.methods.segmenter, self.logger)
        
        # Viewer
        self.viewer = Viewer()
        
        # Saver
        self.saver = Saver(output_dir=self.output_dir, logger=self.logger)
        
        
    @log_time()
    @log_call()
    def run(self):
        self.logger.info("[START] Pipeline execution start.")
        data = self.loader.load_data(
            filename=self.input_file,
            normalize=self.config.experiment.normalize,
            crop=self.config.experiment.crop,
            target_shape=self.config.experiment.target_shape,
        )
        self.config.enhancement.hessian_function = self.derivator.hessian_function()
        enhanced_data = self.enhancer.enhance_data(
            data=data,
            processing_params=self.config.processing,
            enhancement_params=self.config.enhancement,
            hessian_params=self.config.hessian
        )
        
        segmented_data = self.segmenter.segment_data(
            data=enhanced_data,
            segmentation_params=self.config.segmentation
        )
        
        # Save data

        self.saver.save_data(enhanced_data, f'{self.run_name}_enhanced', '.npz')
        self.saver.save_data(segmented_data, f'{self.run_name}_segmented', '.npz')
        
        # Display analytics
        if data.ndim == 2:
            figure = self.viewer.display_images([data, enhanced_data, segmented_data],['RAW', 'ENHANCED', 'SEGMENTED'])
            self.saver.save_plot(figure, filename=self.run_name)
        else:
            histogram = self.viewer.display_histogram([data, enhanced_data, segmented_data], ['RAW', 'ENHANCED', 'SEGMENTED'])
            self.saver.save_plot(histogram, self.run_name)
            
            slices = self.viewer.display_slices([data, enhanced_data, segmented_data], ['RAW', 'ENHANCED', 'SEGMENTED'])
            self.saver.save_animation(slices, self.run_name)
            
            volume = self.viewer.display_volume(volume=enhanced_data, threshold=self.config.segmentation.threshold)
            self.saver.save_plot(volume, self.run_name)
            
        self.logger.info("[END] Pipelien execution end.")

        return data, enhanced_data, segmented_data