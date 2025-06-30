from pathlib import Path
from utils.logger import setup_logger
from utils.config import Config, ConfigParams
from utils.decorator import log_call, log_time
from datetime import datetime 
from utils.io import DataIO
from src.estimator import Estimator
from src.enhancer import Enhancer 
from src.segmenter import Segmenter
from utils.viewer import Viewer

class Pipeline:
    @log_call
    def __init__(self, config: Config):
        self.config = config.config
        
        # Paths
        self.run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_") + self.config.experiment.name
        
        self.input_file = self.config.experiment.input_file
        self.output_file = self.config.experiment.output_file
        
        self.input_dir = Path(self.config.experiment.input_dir)
        self.output_dir = Path(self.config.experiment.output_dir)
        self.log_dir = Path(self.config.experiment.log_dir)
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Logger
        self.debug_mode = self.config.experiment.debug_mode
        self.logger = setup_logger("pipeline", log_dir=self.log_dir, debug_mode=self.debug_mode)
        self. logger.info(f'[INIT] Pipeline initialized - Experiment {self.config.experiment.name}')

        # Loader
        self.io = DataIO(self.input_dir, self.output_dir, self.logger)
        
        # Estimator
        self.estimator = Estimator(self.config.methods.estimator)
        
        # Enhancer
        self.enhancer = Enhancer(self.config.methods.enhancer)
        
        # Segmenter
        self.segmenter = Segmenter(self.config.methods.segmenter)
        
        # Viewer
        self.viewer = Viewer()
        
    @log_time
    @log_call
    def run(self):
        self.logger.info("[START] Pipeline execution start.")
        data = self.io.load_data(
            filename=self.input_file,
            normalize=self.config.experiment.normalize,
            crop=self.config.experiment.crop,
            target_shape=self.config.experiment.target_shape,
        )
        self.config.enhancement.compute_hessian = self.estimator.compute_hessian()
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
        self.path_enhancement = f'{self.run_name}_enhanced_{self.output_file}'
        self.path_segmented = f'{self.run_name}_segmented_{self.output_file}'
        self.io.save_data(enhanced_data, self.path_enhancement)
        self.io.save_data(segmented_data, self.path_segmented)
        
        
        self.viewer.display_histogram([data, enhanced_data, segmented_data], ['RAW', 'ENHANCED', 'SEGMENTED'])
        self.viewer.display_slices([data, enhanced_data, segmented_data], ['RAW', 'ENHANCED', 'SEGMENTED'])
        self.viewer.display_volume(volume=enhanced_data, threshold=self.config.segmentation.threshold)
        
        self.logger.info("[END] Pipelien execution end.")

        return data, enhanced_data, segmented_data