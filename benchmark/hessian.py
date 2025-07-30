from numpy import ndarray
from copy import deepcopy
from typing import Any


from core.pipeline.pipeline import Pipeline
from benchmark.grid_search import GridSearcher
from benchmark.base import BenchmarkBase
from benchmark.analytics.hessian import AnalyticsHessian
from core.config.experiment import ExperimentConfig, Experiment
from core.config.benchmark import BenchmarkData
from core.config.figure import FigureData
from core.saver import Saver
from core.loader import Loader
from core.logger import Logger
from core.saver import Saver



class BenchmarkHessian(BenchmarkBase):
    
    def __init__(self, 
            save_mode: bool, 
            display_mode: bool, 
            pipeline: Pipeline,
            logger: Logger, 
            loader: Loader, 
            saver: Saver,
            grid_searcher: GridSearcher,
        ):
        
        super().__init__(save_mode, display_mode, pipeline, logger, loader, saver)

        self.grid_searcher = grid_searcher
        self.analytics = AnalyticsHessian(self.logger)
     
    
    def _update_config(self, config: ExperimentConfig, param: str, value: Any) -> ExperimentConfig:
        new_config = deepcopy(config)
        setattr(new_config.methods, param, value)
        
        return new_config
    
    
    def _run_experiment(self,
            data_raw: ndarray,
            data_gt: ndarray,
            experiment_config: ExperimentConfig,
        ) -> Experiment:
        
        # Grid Search
        best_params, best_score, best_result = self.grid_searcher.search(
            data_raw=data_raw,
            data_gt=data_gt,
            config=experiment_config,
        )
        
        # Update Config
        for key, value in best_params.items():
            setattr(best_result.config.enhancement, key, value)
            
        return best_result
    
    
    def _create_figures(self, benchmark_data: BenchmarkData) ->list[FigureData]:
        
        # Parse experiments
        experiments = []
        data_raw = benchmark_data.data_raw
        data_gt = benchmark_data.data_gt
        for params, values in benchmark_data.results.items():
            for value, experiment in values.items():
                experiments.append(experiment)
        
        figures: list[FigureData] = []
        
        # Histogram
        figures.append(self.analytics.get_histograms(
            experiments=experiments,
            data_raw=data_raw,
            data_gt=data_gt,
        ))
        
        # Config
        figures.append(self.analytics.get_configs(
            experiments=experiments,
        ))
                
        # Metrics
        figures.append(self.analytics.get_metrics(
            experiments=experiments,
        ))
        
        # Curves
        figures.append(self.analytics.get_curves(
            experiments=experiments,
            ground_truth=data_gt,
        ))
        
        # Views
        figures.extend(self.analytics.get_views(
            experiments=experiments,
            data_gt=data_gt,
            data_raw=data_raw,
        ))
        
        return figures
    
