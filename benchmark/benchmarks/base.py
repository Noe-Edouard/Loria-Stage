import matplotlib.pyplot as plt
from numpy import ndarray
from typing import Any, Optional
from abc import ABC, abstractmethod
from pathlib import Path


from core.pipeline.pipeline import Pipeline
from core.io.saver import Saver
from core.io.loader import Loader
from core.io.logger import Logger
from core.io.saver import Saver
from core.config.figure import FigureData
from core.config.experiment import ExperimentConfig, Experiment
from core.config.benchmark import BenchmarkConfig, BenchmarkResults, BenchmarkData
from core.config.metrics import Metrics
from benchmark.analytics.base import AnalyticsBase
from benchmark.metrics import dice, mcc, roc, pr
from core.utils.decorator import log_section

class BenchmarkBase(ABC):

    def __init__(self, 
            save_mode: bool, 
            display_mode: bool, 
            pipeline: Pipeline,
            logger: Logger, 
            loader: Loader, 
            saver: Saver,
            analytics: Optional[AnalyticsBase] = None,
        ):
        
        self.save_mode = save_mode
        self.display_mode = display_mode
        self.pipeline = pipeline
        self.logger = logger
        self.loader = loader
        self.saver = saver
        self.analytics = analytics
        

    @abstractmethod
    def _update_config(self, config: ExperimentConfig, param: str, value: Any) -> ExperimentConfig:
        """Met à jour dans la config de l'expérience le paramètre étudié dans le benchmark"""
        pass
    
    
    @abstractmethod
    def _run_experiment(self, data_raw: ndarray,
            data_gt: ndarray,
            experiment_config: ExperimentConfig,
        ) -> Experiment:
        """Lance l'expérience avec les paramètre de la config"""
        pass
    
    
    @abstractmethod
    def _create_figures(self, benchmark_data: BenchmarkData) -> list[FigureData]:
        """Construit les figures utiles pour l'analyse du benchmark."""
        pass
    

    def _save_figures(self, figures: list[FigureData], dirname: str = None):
        for i, figure in enumerate(figures):
            if figure.name is None:
                figure.name = f"figure_{i}"
                
        for figure in figures:
            if self.display_mode:
                if figure.mode == 'text':
                    self.logger.info(f'{figure.figure}')
                else:
                    plt.show()
            if self.save_mode:
                self.saver.save_figure(figure, dirname)
    

    def _compute_metrics(self, data_segmented: ndarray, data_gt: ndarray) -> Metrics:
        
        metrics = Metrics(
            dice=dice(data_gt, data_segmented),
            mcc=mcc(data_gt, data_segmented),
            roc=roc(data_gt, data_segmented),
            pr=pr(data_gt, data_segmented),
        )
        
        return metrics
    
    @log_section('Benchmark execution')
    def run(self, 
            file_raw: str, 
            file_gt: str,
            benchmark_config: BenchmarkConfig, 
            experiment_config: ExperimentConfig, 
        ) -> BenchmarkResults:
        
        # Load data
        data_raw = self.loader.load_data(
            filename=file_raw,
            normalize=experiment_config.loading.normalize,
            crop=experiment_config.loading.crop,
            target_shape=experiment_config.loading.target_shape,
        )
        data_gt = self.loader.load_data(
            filename=file_gt,
            normalize=experiment_config.loading.normalize,
            crop=experiment_config.loading.crop,
            target_shape=experiment_config.loading.target_shape,
        )

        
        # Update config
        experiment_config.loading.raw_file = file_raw
        experiment_config.loading.gt_file = file_gt
        
        results: BenchmarkResults = {param: {value: None for value in values} for param, values in benchmark_config.params.items()}
        i = 0
        for param, values in benchmark_config.params.items():
            for value in values:
                
                # Update config
                experiment_config = self._update_config(
                    config=experiment_config, 
                    param=param, 
                    value=value
                )
                
                # Run experiment 
                experiment = self._run_experiment(
                    data_raw=data_raw, 
                    data_gt=data_gt,
                    experiment_config=experiment_config,
                )
                
                # Compute metrics
                metrics = self._compute_metrics(
                    data_segmented=experiment.data_segmented,
                    data_gt=data_gt,
                )
                
                # Store experiment
                experiment.metrics = metrics 
                experiment.id = f"{Path(file_raw).name}{i}"
                results[param][value] = experiment
                
                # Udpate id
                i+=1

                
        benchmark_data = BenchmarkData(
            data_raw=data_raw,
            data_gt=data_gt,
            results=results,
        )
        
        figures = self._create_figures(benchmark_data)
        self.saver.save_results(results, 'benchmark', Path(file_raw).stem)
        self._save_figures(figures, Path(file_raw).stem)
        
        return results
    

