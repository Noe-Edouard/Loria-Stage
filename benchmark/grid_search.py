from numpy import ndarray
from itertools import product
from time import perf_counter
from typing import Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
from tqdm import tqdm
from benchmark.metrics import mcc
from core.config import ExperimentConfig, BenchmarkConfig, BenchmarkData, ExperimentData
from pipeline.pipeline import Pipeline
from core.logger import setup_logger, Logger
from utils.decorator import log_time, log_section  

class GridSearch:
    def __init__(self, pipeline: Pipeline, logger: Logger = setup_logger()):
        self.pipeline = pipeline
        self.logger = logger

    def evaluate_combination(self, combination: list, keys: list, config: ExperimentConfig, data_raw: ndarray, ground_truth: ndarray) -> Tuple[float, dict, object, float]:
        warnings.filterwarnings("ignore")
        start = perf_counter()
        params = {}
        for i, key in enumerate(keys):
            params[key] = combination[i]
            config.enhancement[key] = params[key]
        result = self.pipeline.run(config=config, data_raw=data_raw, ground_truth=ground_truth)
        score = mcc(result.segmented, ground_truth)
        end = perf_counter()
        time = end - start
        return score, params, result, time

    @log_time()
    @log_section("Grid Search")
    def search(self, data_raw: ndarray, ground_truth: ndarray, params_grid: dict, config: ExperimentConfig) -> Tuple[dict, float, BenchmarkData]:
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
                executor.submit(self.evaluate_combination, combination, keys, config, data_raw, ground_truth)
                for combination in combinations
            ]
            progress_bar = tqdm(as_completed(futures), total=len(futures), desc='Grid Search')
            for future in progress_bar:
                try:
                    score, params, result, time = future.result()
                except Exception as e:
                    self.logger.error(f"[ERROR] Failed combination: {e}")
                    continue
                if score > best_score:
                    best_score = score
                    best_params = params
                    best_result = result
                progress_bar.set_postfix({
                    'time': time,
                    'score': score,
                })
        for key, value in best_params.items():
            setattr(best_result.config.enhancement, key, value)
        end = perf_counter()
        self.logger.info(f"[END] Grid search executed in {end-start:.2f} seconds."
                         f"\n                               - best params: {best_params}"
                         f"\n                               - best score:  {best_score:.4f}"
                         f"\n                               - threshold:   {best_result.config.segmentation.threshold:.4f}"
                        )
        return best_params, best_score, best_result
