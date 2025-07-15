from numpy import ndarray
from pathlib import Path
from itertools import product
from time import perf_counter
from copy import deepcopy


from pipeline.pipeline import Pipeline
from view.analytics import Analytics
from core.config import ExperimentConfig, BenchmarkConfig, BenchmarkData, ExperimentData, ConfigBuilder, MainConfig
from core.saver import Saver
from core.loader import Loader
from core.config import SetupConfig, ConfigBuilder
from core.logger import setup_logger
from benchmark.metrics import mcc
from utils.decorator import log_time


    
class Benchmark:
    def __init__(self, setup: SetupConfig):
        self.setup = setup
        self.setup_pipeline = ConfigBuilder({
            'name': 'pipeline',
            'input_dir': self.setup.input_dir,
            'output_dir': 'outputs/pipeline',
            'log_dir': 'logs',
            'log_file': 'pipeline',
            'debug_mode': False,
            'display_mode': False,
            'save_mode': False,
        }, SetupConfig)
        
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
        
        # Pipeline
      
        self.pipeline = Pipeline(self.setup_pipeline)
        
        # Analytics
        self.display_mode = self.setup.display_mode
        self.analytics = Analytics(logger=self.logger, display_mode=self.display_mode)

        # Saver
        self.save_mode = self.setup.save_mode
        self.saver = Saver(experiment_name=self.setup.name, output_dir=self.output_dir, logger=self.logger)
        
        self.logger.info(f'[INIT] Benchmark initialized - Experiment {self.setup.name}')

    def grid_search(self, data_raw: ndarray, ground_truth: ndarray, params_grid: dict[str, list], config: ExperimentConfig):
        start = perf_counter()
        self.logger.info(f'[START] Grid Search started for {config.methods.derivator}')
        keys = list(params_grid.keys())
        values = list(params_grid.values())
        combinations = product(*values)
        
        best_score = float('-inf')
        best_params = None
        best_result = None
        
        for combination in combinations:
            t1 = perf_counter()
            params = {}
            for i, key in enumerate(keys):
                params[key] = combination[i]
                config.enhancement[key] = params[key]
            experiment_data = self.pipeline.run(config=config, data_raw=data_raw, ground_truth=ground_truth)
            score = mcc(experiment_data.segmented, ground_truth)
            
            self.logger.debug(f"       params: {params} | score: {score:.4f} | threshold: {experiment_data.config.segmentation.threshold:.4f} | time: {perf_counter()-t1:.2f} s")

            if score > best_score:
                best_score = score
                best_params = params
                best_result = experiment_data
        best_result.config = config
        
        end = perf_counter()
        self.logger.info(f"[END] Grid search executed in {end-start:.2f} seconds."
                         f"\n                           - best params: {best_params}"
                         f"\n                           - best score:  {best_score:.4f}"
                         f"\n                           - threshold:   {best_result.config.segmentation.threshold:.4f}"
                        )
        
        return best_params, best_score, best_result
        
    
        
    @log_time()
    def run(self, config_benchmark: BenchmarkConfig, config_experiment: ExperimentConfig) -> BenchmarkData:
        
        data_raw = self.loader.load_data(config_benchmark.raw_file)
        ground_truth = self.loader.load_data(config_benchmark.gt_file)
        benchmark_data = BenchmarkData(raw_data=data_raw, ground_truth=ground_truth, experiments=[])
        config_experiment.load.input_file = config_benchmark.raw_file
        # Run experiments
        results = []
        scores = []
        params = []
        for method in config_benchmark.methods:
            config_experiment.methods.derivator = method
            best_params, best_score, best_result = self.grid_search(
                data_raw=data_raw, 
                ground_truth=ground_truth,
                params_grid=config_benchmark.params_grid,
                config=config_experiment
            )
            results.append(deepcopy(best_result))
            scores.append(best_score)
            params.append(best_params)
      
        benchmark_data.experiments = results
       
        if self.display_mode:  
            # Histograms
            histograms = self.analytics.display_histograms(
                experiments=benchmark_data.experiments,
                raw_data=benchmark_data.raw_data,
                ground_truth=benchmark_data.ground_truth,
            )
            
            # Metrics
            metrics = self.analytics.display_metrics(
                experiments=benchmark_data.experiments,
                ground_truth=benchmark_data.ground_truth,
            )
            
            # Curves
            curves = self.analytics.display_curves(
                experiments=benchmark_data.experiments,
                ground_truth=benchmark_data.ground_truth,
            )
            
            # Views
            views, titles, modes = self.analytics.display_views(
                experiments=benchmark_data.experiments,
                ground_truth=benchmark_data.ground_truth,
                raw_data = benchmark_data.raw_data,
            )
        
        if self.save_mode:
            self.saver.save_plot(histograms, 'histograms')
            self.saver.save_text(metrics, "metrics")
            self.saver.save_plot(curves, 'curves')
            
            for plot, title, mode in zip(views, titles, modes):
                if mode == 'plot':
                    self.saver.save_plot(plot, title)
                elif mode == 'anim':
                    self.saver.save_animation(plot, title, fps=10)
                else:
                    raise ValueError(f'The mode {mode} is not valid.')
            
        
        return benchmark_data
        
