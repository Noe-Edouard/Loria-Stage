import numpy as np
from numpy import ndarray
from copy import deepcopy
from typing import Any

from core.pipeline.pipeline import Pipeline
from benchmark.benchmarks.base import BenchmarkBase
from benchmark.analytics.enhancement import AnalyticsEnhancement
from core.io.saver import Saver
from core.io.loader import Loader
from core.io.logger import Logger
from core.config.figure import FigureData
from core.config.benchmark import BenchmarkData
from core.config.experiment import ExperimentConfig, Experiment

class BenchmarkEnhancement(BenchmarkBase):
    
    def __init__(self, 
            save_mode: bool, 
            display_mode: bool, 
            pipeline: Pipeline,
            logger: Logger, 
            loader: Loader, 
            saver: Saver,
        ):
        super().__init__(save_mode, display_mode, pipeline, logger, loader, saver)
        self.analytics = AnalyticsEnhancement(self.logger)
    
    def _update_config(self, config: ExperimentConfig, param: str, value: Any) -> ExperimentConfig:
        new_config = deepcopy(config)
        if param == 'scales_min':
            scales = np.arange(value, 20, 1)
            setattr(new_config.enhancement, 'scales', scales)
        elif param == 'scales_max':
            scales = np.arange(1, value, 1)
            setattr(new_config.enhancement, 'scales', scales)
        else:    
            setattr(new_config.enhancement, param, value)
            
        return new_config
    
    
    def _run_experiment(self,
            data_raw: ndarray,
            data_gt: ndarray,
            experiment_config: ExperimentConfig,
        ) -> Experiment:
        
        # Run Pipeline
        experiment = self.pipeline.run(
            config=experiment_config,
            data_raw=data_raw,
            data_gt=data_gt,
        )
        
        return experiment
    
    
    def _create_figures(self, benchmark_data: BenchmarkData) -> list[FigureData]:
        
        figures = []
        
        # Alpha/Beta/Gamma
        figures.append(self.analytics.get_params_curves(
            benchmark_results=benchmark_data.results
        ))
        
        # Scales
        figures.append(self.analytics.get_scales_curves(
            benchmark_results=benchmark_data.results
        ))
        
        return figures
    
   
