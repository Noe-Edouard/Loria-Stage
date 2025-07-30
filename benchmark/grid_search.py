from numpy import ndarray
from itertools import product
from time import perf_counter
from typing import Tuple
import warnings

from core.pipeline.pipeline import Pipeline
from core.logger import setup_logger, Logger
from core.config.experiment import ExperimentConfig, Experiment
from core.utils.decorator import log_time, log_section  
from core.utils.parallelizer import Parallelizer
from benchmark.metrics import mcc


class GridSearcher:
    def __init__(self, params_grid: dict, pipeline: Pipeline, logger: Logger = setup_logger('grid_search', True)):
        self.params_grid = params_grid
        self.pipeline = pipeline
        self.logger = logger
        self.parallelizer = Parallelizer()

    def evaluate_combination(self, combination: list, keys: list, config: ExperimentConfig, data_raw: ndarray, data_gt: ndarray) -> Tuple[float, dict, object, float]:
        warnings.filterwarnings("ignore")
        params = {}
        for i, key in enumerate(keys):
            params[key] = combination[i]
            config.enhancement[key] = params[key]
        result = self.pipeline.run(config=config, data_raw=data_raw, data_gt=data_gt)
        score = mcc(result.data_segmented, data_gt)
        
        return score, params, result

    def search(self, 
            data_raw: ndarray, 
            data_gt: ndarray, 
            config: ExperimentConfig
        ) -> Tuple[dict, float, Experiment]:
        
        start = perf_counter()
        self.logger.info(f'[START] Grid Search started for {config.methods.derivator}')
        
        params_grid = self.params_grid
        keys = list(params_grid.keys())
        values = list(params_grid.values())
        combinations = list(product(*values))
        
        best_score: float = float('-inf')
        best_params: dict = {}
        best_result: Experiment = None
        
        results = self.parallelizer.run(
            func=self.evaluate_combination,
            iterable=(
                (combination, keys, config, data_raw, data_gt) 
                for combination in combinations
            ),
            show_progress=True,
            unpack_args=True
        )
        for score, params, result in results:
            if score > best_score:
                best_score = score
                best_params = params
                best_result = result
        
        end = perf_counter()
        self.logger.info(f"[END] Grid search executed in {end-start:.2f} seconds."
                         f"\n                           - best params: {best_params}"
                         f"\n                           - best score:  {best_score:.4f}"
                         f"\n                           - threshold:   {best_result.config.segmentation.threshold:.4f}"
                        )
        return best_params, best_score, best_result
