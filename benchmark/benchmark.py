from numpy import ndarray
from pathlib import Path
from itertools import product
from copy import deepcopy
from typing import Tuple
import numpy as np
from pipeline.pipeline import Pipeline
from view.analytics import Analytics
from core.config import ExperimentConfig, BenchmarkConfig, BenchmarkData, ExperimentData, ConfigBuilder, MainConfig, MetricsConfig
from core.saver import Saver
from core.loader import Loader
from core.config import SetupConfig, ConfigBuilder
from core.logger import setup_logger
from benchmark.metrics import mcc
from benchmark.grid_search import GridSearch
from utils.decorator import log_time
from utils.decorator import log_section

from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from core.logger import Logger
from utils.parallelizer import Parallelizer
from pipeline.pipeline import Pipeline
from core.saver import Saver
from pipeline.enhancer import Enhancer

class BenchmarkBase(ABC):

    def __init__(self, save_mode: bool, display_mode: bool, logger: Logger, parallelizer: Parallelizer, saver: Saver, pipeline: Pipeline = None, enhancer: Enhancer = None):
        self.save_mode = save_mode
        self.display_mode = display_mode
        self.enhancer = enhancer
        self.pipeline = pipeline
        self.parallelizer = parallelizer
        self.logger = logger
        self.saver = saver
        
    def _display_figure(self, figure: plt.Figure, filename: str):
        if self.display_mode:
            plt.show()
        if self.save_mode:
            self.saver.save_plot(figure, filename)

    
    @abstractmethod
    def process_image(self, *args, **kargs):
        pass
    
    @abstractmethod
    def compute_scores(self, *args, **kargs):
        pass
    
    @abstractmethod
    def plot_scores(self, *args, **kargs):
        pass
    
    @abstractmethod
    def run(self, *args, **kargs):
        pass
    
    
class BenchmarkDerivator(BenchmarkBase):
    
    
    def _compute_metrics(self, results: list[ExperimentData], data_gt: ndarray) -> list[MetricsConfig]:
        return self.analytics.get_metrics(results, data_gt)

    def _display_results(self, benchmark_data: BenchmarkData):
        histograms = self.analytics.display_histograms(
            experiments=benchmark_data.experiments,
            data_raw=benchmark_data.data_raw,
            data_gt=benchmark_data.data_gt,
        )
        configs = self.analytics.display_configs(
            experiments=benchmark_data.experiments,
        )
        metrics = self.analytics.display_metrics(
            experiments=benchmark_data.experiments,
            data_gt=benchmark_data.data_gt,
        )
        curves = self.analytics.display_curves(
            experiments=benchmark_data.experiments,
            data_gt=benchmark_data.data_gt,
        )
        views, titles, modes = self.analytics.display_views(
            experiments=benchmark_data.experiments,
            data_gt=benchmark_data.data_gt,
            data_raw=benchmark_data.data_raw,
        )
        return {
            'histograms': histograms,
            'configs': configs,
            'metrics': metrics,
            'curves': curves,
            'views': views,
            'titles': titles,
            'modes': modes
        }

    def _save_results(self, artifacts):
        self.saver.save_plot(artifacts['histograms'], 'histograms')
        self.saver.save_text(artifacts['configs'], "config")
        self.saver.save_text(artifacts['metrics'], "metrics")
        self.saver.save_plot(artifacts['curves'], 'curves')
        for plot, title, mode in zip(artifacts['views'], artifacts['titles'], artifacts['modes']):
            if mode == 'plot':
                self.saver.save_plot(plot, title)
            elif mode == 'anim':
                self.saver.save_animation(plot, title, fps=10)
            else:
                raise ValueError(f'The mode {mode} is not valid.')
    
    def _process_results(self, data):
        if self.display_mode:
            artifacts = self._display_results(data)
        if self.save_mode and artifacts is not None:
            self._save_results(artifacts)
    
    def run_experiment(
        self, 
        data_raw: ndarray, 
        data_gt: ndarray,
        methods: list[str], 
        params_grid: dict,
        experiment_config: ExperimentConfig,
    ) -> Tuple[list[ExperimentData], list[float], list[dict]]:
        
        results = []
        for method in methods:
            experiment_config.methods.derivator = method
            best_params, best_score, best_result = self.grid_searcher.search(
                data_raw=data_raw,
                data_gt=data_gt,
                params_grid=params_grid,
                config=experiment_config
            )
            # best_result.
            results.append(deepcopy(best_result))

        return results
    
    @log_time()
    @log_section("Benchmark execution")
    def run(self, 
            benchmark_config: BenchmarkConfig, 
            experiment_config: ExperimentConfig,
            images_raw: list[ndarray],
            images_gt: list[ndarray],
        ) -> list[BenchmarkData]:
        
        # data_raw, data_gt = self._load_data(benchmark_config)
        methods = benchmark_config.methods
        params_grid = benchmark_config.params_grid
        
        results = self.parallelizer.run(
            func=self.run_experiment,
            iterable=(
                (i, data_raw, data_gt, methods, params_grid, experiment_config) 
                for i, (data_raw, data_gt) in enumerate(zip(images_raw, images_gt))
            ),
            show_progress=True,
            unpack_args=True
        )
        
        benchmark_data = BenchmarkData(data_raw=data_raw, data_gt=data_gt, experiments=[], metrics=[])

        results, scores, params = self._run_experiments(benchmark_config, experiment_config, data_raw, data_gt)
        benchmark_data.experiments = results
        benchmark_data.metrics = self._compute_metrics(results, data_gt)
        artifacts = None
        
        
        if self.display_mode:
            artifacts = self._display_results(benchmark_data)
        if self.save_mode and artifacts is not None:
            self._save_results(artifacts)
        return benchmark_data
    
    
    def __init__(self, setup: SetupConfig):
        self.setup = setup
        
        # Logger
        self.logger = setup_logger(log_file=self.setup.log_file, debug_mode=self.setup.debug_mode)
        
        # Loader
        self.loader = Loader(self.setup.input_dir, self.logger)
        
        # Pipeline
        self.setup_pipeline = ConfigBuilder({
            'name': 'pipeline',
            'input_dir': self.setup.input_dir,
            'output_dir': 'outputs/pipeline',
            'log_file': 'pipeline',
            'debug_mode': False,
            'display_mode': False,
            'save_mode': False,
        }, SetupConfig)
        self.pipeline = Pipeline(self.setup_pipeline)
        
        # Analytics
        self.display_mode = self.setup.display_mode
        self.analytics = Analytics(logger=self.logger, display_mode=self.display_mode)

        # Saver
        self.save_mode = self.setup.save_mode
        self.saver = Saver(experiment_name=self.setup.name, output_dir=self.output_dir, logger=self.logger)
        
        self.logger.info(f'[INIT] Benchmark initialized - Experiment {self.setup.name}')

        # GridSearch
        self.grid_searcher = GridSearch(self.pipeline, self.logger)
    
        
    

    def _load_data(self, benchmark_config: BenchmarkConfig) -> Tuple[ndarray, ndarray]:
        data_raw = self.loader.load_data(benchmark_config.raw_file)
        data_gt = self.loader.load_data(benchmark_config.gt_file)
        return data_raw, data_gt

    

    
        
    @log_time()
    @log_section("Benchmark execution for all files")
    def run_all(self, benchmark_config: BenchmarkConfig, experiment_config: ExperimentConfig):
        raw_dir = Path(self.setup.input_dir) / 'raw'
        gt_dir = Path(self.setup.input_dir) / 'gt'

        runs_data: dict[str, BenchmarkData] = {}
        results = {method: {metric: [] for metric in ['dice', 'mcc', 'roc', 'pr']} for method in benchmark_config.methods}

        self.display_mode = False
        self.analytics = Analytics(logger=self.logger, display_mode=self.display_mode)
        
        for raw_file, gt_file in zip(sorted(raw_dir.iterdir()), sorted(gt_dir.iterdir())):
            benchmark_config.raw_file = raw_file.name
            benchmark_config.gt_file = gt_file.name
            self.logger.info(f"[BATCH] Running benchmark for {raw_file.name}")
            run_result = self.run(benchmark_config, experiment_config)
            runs_data[raw_file.name] = run_result

            for metric in run_result.metrics:
                for metric_name in ['dice', 'mcc', 'roc', 'pr']:
                    results[metric['method']][metric_name].append(metric[metric_name])

        results_mean = {
            method: {
                metric: float(np.mean(values)) if values else 0.0
                for metric, values in method_metrics.items()
            }
            for method, method_metrics in results.items()
        }

        results_std = {
            method: {
                metric: float(np.std(values)) if values else 0.0
                for metric, values in method_metrics.items()
            }
            for method, method_metrics in results.items()
        }
        
        summary_metrics = self.analytics.display_summary_metrics(results_mean, results_std)
        if self.save_mode:
            self.saver.save_text(summary_metrics, 'summary_metrics')
            
        return runs_data, results_mean, results_std
