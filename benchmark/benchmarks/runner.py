import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import fields

from core.pipeline.pipeline import Pipeline
from benchmark.grid_search import GridSearcher
from benchmark.benchmarks.hessian import BenchmarkHessian
from benchmark.benchmarks.enhancement import BenchmarkEnhancement
from benchmark.analytics.runner import AnalyticsRunner
from benchmark.benchmarks.base import BenchmarkBase
from core.io.saver import Saver
from core.io.loader import Loader
from core.io.logger import setup_logger
from core.utils.decorator import log_time, log_section, log_init
from core.utils.parallelizer import Parallelizer
from core.config.benchmark import BenchmarkConfig, RunnerResults, BenchmarkResults, RunnerResultsParsed
from core.config.experiment import ExperimentConfig, Experiment
from core.config.builder import ConfigBuilder 
from core.config.setup import SetupConfig
from core.config.metrics import Metrics
from core.config.figure import FigureData
from configs.args import INPUT_DIR


class BenchmarkRunner():
    
    @log_init()
    def __init__(self, setup: SetupConfig):
        self.setup = setup
        
        self.logger = setup_logger(log_file=self.setup.log_file, debug_mode=self.setup.debug_mode)
        self.loader = Loader(self.setup.input_dir, self.logger)
        self.parallelizer = Parallelizer(max_workers=None, use_processes=True)
        self.saver = Saver(experiment_name=self.setup.name, output_dir=self.setup.output_dir, logger=self.logger)
           
        self.display_mode = self.setup.display_mode
        self.save_mode = self.setup.save_mode
        
        self.analytics = AnalyticsRunner(self.logger)


    def _get_benchmark(self, benchmark_config: BenchmarkConfig) -> BenchmarkBase:
        
        setup_pipeline = ConfigBuilder({
                'name': 'pipeline',
                'input_dir': self.setup.input_dir,
                'output_dir': 'pipeline',
                'log_file': 'pipeline',
                'debug_mode': False,
                'display_mode': False,
                'save_mode': False,
            }, SetupConfig)
        
        pipeline = Pipeline(setup=setup_pipeline)
        
        config = {
            'save_mode': self.save_mode,
            'display_mode': False,
            'pipeline': pipeline,
            'logger': self.logger,
            'loader': self.loader,
            'saver': self.saver,
        }
        if benchmark_config.mode == 'hessian':
            grid_searcher = GridSearcher(
                params_grid=benchmark_config.params_grid,
                pipeline=pipeline, 
                # logger=self.logger
            ) 
            config['grid_searcher'] = grid_searcher
            benchmark = BenchmarkHessian(**config)
        
        elif benchmark_config.mode == 'enhancement':
            benchmark = BenchmarkEnhancement(**config)
        
        else:
            raise ValueError(f'Benchmark mode unknown : {benchmark_config.mode}')
        
        return benchmark
            
    def _get_files(self, images_dir: str = "images", labels_dir: str = "labels"):
        images_dir = Path(f"{INPUT_DIR}/{self.setup.input_dir}") / images_dir
        labels_dir = Path(f"{INPUT_DIR}/{self.setup.input_dir}") / labels_dir
        images_files = []
        labels_files = []
        
        if not images_dir.exists() or not labels_dir.exists():
            raise ValueError(f"The directories 'images' or 'labels' do not exist: {images_dir}, {labels_dir}")
        for image_name in os.listdir(images_dir):
            idx = image_name[len("image_"):]
            images_files.append(f"images/{image_name}")
            label_name = f"label_{idx}"
            labels_files.append(f"labels/{label_name}")
        return images_files, labels_files
    


    def _save_figures(self, figures: list[FigureData]):
        for i, figure in enumerate(figures):
            if figure.name is None:
                figure.name = f"figure_{i}"
                
        for figure in figures:
            if self.display_mode:
                if figure.mode == 'text':
                    self.logger.info(figure.figure)
                plt.show()
            if self.save_mode:
                self.saver.save_figure(figure, self.setup.name)
                
                
    @log_time()
    @log_section("Runner execution")
    def run(self, 
            images_dir: str,
            labels_dir: str,
            benchmark_config: BenchmarkConfig,
            experiment_config: ExperimentConfig,
        ) -> RunnerResults:
        
        # Get files
        files_raw, files_gt = self._get_files(
            images_dir=images_dir, 
            labels_dir=labels_dir,
        )
        
        # Select Benchmark
        benchmark = self._get_benchmark(benchmark_config)
        
        results = self.parallelizer.run(
            func=benchmark.run,
            iterable=(
                (file_raw, file_gt, benchmark_config, experiment_config) 
                for file_raw, file_gt in zip(files_raw, files_gt)
            ),
            show_progress=False,
            unpack_args=True,
        )
        
        # Sauvegarde les results avec pickle
        filename = self.saver.save_results(results, self.setup.name, self.setup.name)
        
        return filename
    
    
    @log_section("Runner analysis")
    def analyse(self, benchmark_config: BenchmarkConfig, results_file: str):
        
        # Load results
        results_raw = self.loader.load_results(results_file)
        
        # Get analysis
        match benchmark_config.mode:
            case 'hessian': 
                figures = self.analytics.get_hessian_figures(results_raw, benchmark_config.params)
            case "enhancement": 
                figures = self.analytics.get_enhancement_figures(results_raw, benchmark_config.params)

        self._save_figures(figures)
        
        
 
###     
       
       
        # results: list[BenchmarkResults] = [images]
        # image = {
        #     'param': {
        #         'value': Experiment
        #     }
        # }
        
        # benchmark: RunnerResultsParsed = {
        #     'metric': {
        #         'param': {
        #             'value': [] # images
        #         }
        #     }
        # }
        
        
        # hessian = {
        #     'dice': {
        #         'derivator' : {
        #             'default':  [],
        #             'gaussian': [],
        #         },
        #     },
        #     'mcc': {
        #         'method' : {
        #             'default':  [],
        #             'gaussian': [],
        #         },
        #     }
        # }
        
        # enhancement = {
        #     'mcc': {
        #         'alpha': {
        #             0.1: [],
        #             0.2: [],
        #             0.5: [],
        #         },
        #         'beta': {
        #             0.1: [],
        #             0.2: [],
        #             0.5: [],
        #         },
        #         'gamma':{
        #             0.1: [],
        #             0.2: [],
        #             0.5: [],
        #         },
        #     }
        # }
        
        # scales = {
        #     'mcc': {
        #         'min': {
        #             1: [],
        #             2: [],
        #             3: [],
        #         },
        #         'min': {
        #             1: [],
        #             2: [],
        #             3: [],
        #         }
        #     }
        # }
        
        
        
        # Hessian : Il faut exporter pour chaque métrique un tableau avec pour chaque méthode les résultats pour chaque images
        
        
        # Enhancement : Il faut exporter pour une seule métrique, pour chaque paramètres (alpha, beta, gamma), les résultats pour chaque valeurs pour chaque images
        
        # Scales : Il faut exporter pour une métrique, pour min et max, pour chaque valeur, les résultats pour les différentes images
        
        # Processing : Il faut exporter les tempts de traitement pour chaque
        
        # On a : pour chaque image une liste de méthodes une liste de métriques
        # On veut : pour chaque méthode, une liste de scores
        # benchmark_data = BenchmarkData(data_raw=data_raw, data_gt=data_gt, experiments=[], metrics=[])

        # results, scores, params = self._run_experiments(benchmark_config, experiment_config, data_raw, data_gt)
        # benchmark_data.results = results
        # benchmark_data.metrics = self._compute_metrics(results, data_gt)
        # artifacts = None
        
    