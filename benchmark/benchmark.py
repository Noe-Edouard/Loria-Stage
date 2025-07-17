from numpy import ndarray
from pathlib import Path
from itertools import product
from time import perf_counter
from copy import deepcopy
from tqdm import tqdm
from typing import Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
from pipeline.pipeline import Pipeline
from view.analytics import Analytics
from core.config import ExperimentConfig, BenchmarkConfig, BenchmarkData, ExperimentData, ConfigBuilder, MainConfig
from core.saver import Saver
from core.loader import Loader
from core.config import SetupConfig, ConfigBuilder
from core.logger import setup_logger
from benchmark.metrics import mcc
from utils.decorator import log_time


def evaluate_combination(
    combination: list, 
    keys: list, 
    config: ExperimentConfig, 
    data_raw:ndarray, 
    ground_truth: ndarray,
    pipeline: Pipeline,
) -> Tuple[float, dict, ExperimentData, float]:
    
    warnings.filterwarnings("ignore")
    start = perf_counter()
    params = {}
    for i, key in enumerate(keys):
        params[key] = combination[i]
        config.enhancement[key] = params[key]
    result = pipeline.run(config=config, data_raw=data_raw, ground_truth=ground_truth)
    score = mcc(result.segmented, ground_truth)
    end = perf_counter()
    time = end - start

    return score, params, result, time

    
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
        combinations = list(product(*values))
        
        best_score: float = float('-inf')
        best_params: dict = {}
        best_result: ExperimentData = None
            
            
        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(evaluate_combination, combination, keys, config, data_raw, ground_truth, self.pipeline)
                for combination in combinations
            ]
            
            progress_bar = tqdm(as_completed(futures), total=len(futures), desc='Grid Search')
            for future in progress_bar:
                score, params, result, time = future.result()
                
                if score > best_score:
                    best_score = score
                    best_params = params
                    best_result = result
                    
                progress_bar.set_postfix({
                    'time': time,
                    'score': score,
                })
        
        for key, value in best_params.items():
            best_result.config[key] = value
        
        end = perf_counter()
        self.logger.info(f"[END] Grid search executed in {end-start:.2f} seconds."
                         f"\n                               - best params: {best_params}"
                         f"\n                               - best score:  {best_score:.4f}"
                         f"\n                               - threshold:   {best_result.config.segmentation.threshold:.4f}"
                        )
        
        return best_params, best_score, best_result
        
    
        
    @log_time()
    def run(self, benchmark_config: BenchmarkConfig, experiment_config: ExperimentConfig) -> BenchmarkData:
        
        data_raw = self.loader.load_data(benchmark_config.raw_file)
        ground_truth = self.loader.load_data(benchmark_config.gt_file)
        benchmark_data = BenchmarkData(raw_data=data_raw, ground_truth=ground_truth, experiments=[])
        experiment_config.load.input_file = benchmark_config.raw_file
        # Run experiments
        results = []
        scores = []
        params = []
        for method in benchmark_config.methods:
            experiment_config.methods.derivator = method
            best_params, best_score, best_result = self.grid_search(
                data_raw=data_raw, 
                ground_truth=ground_truth,
                params_grid=benchmark_config.params_grid,
                config=experiment_config
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
        
