from abc import ABC, abstractmethod
import os
import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
from copy import deepcopy
from pathlib import Path

from utils.parallelizer import Parallelizer
from utils.decorator import log_time, log_section
from benchmark.metrics import mcc
from core.logger import Logger, setup_logger
from core.loader import Loader
from core.saver import Saver
from core.config import ConfigBuilder, ExperimentConfig, SetupConfig, CSIConfig, AnalyzerConfig, CPIConfig
from pipeline.pipeline import Pipeline
from view.viewer import Viewer
from benchmark.analyzer import BaseAnalyzer


    
class ParamsAnalyzer(BaseAnalyzer):

    def compute_scores(self, 
            data_raw: ndarray, 
            data_gt: ndarray, 
            params: dict[str, list[float]], 
            experiment_config: ExperimentConfig
        ) -> dict[str, list[float]]:

        results = {}
        for param, values in params.items():
            mcc_scores = []
            for value in values:
                # Update parameters
                exp_config = deepcopy(experiment_config)
                setattr(exp_config.enhancement, param, value)

                # Run pipeline
                exp_data = self.pipeline.run(
                    config=exp_config,
                    data_raw=data_raw,
                    data_gt=data_gt,
                )
                
                # Compute scores
                mcc_scores.append(mcc(exp_data.segmented, data_gt))
            
            results[param] = mcc_scores

        return results

    def plot_scores(self, 
            params: dict[str, list[float]],
            results: dict[str, list[float]] | dict[str, dict[str, list[float]]],
            mean_std: bool = True,
        ) -> plt.Figure:
        
        fig = plt.figure(figsize=(16, 8))
        colors = ['red', 'dodgerblue', 'limegreen']

        for i, (param_name, param_values) in enumerate(params.items()):
            plt.subplot(1, len(params), i + 1)

            color = colors[i % len(colors)]

            if mean_std:
                
                mean_vals = np.array(results[param_name]['mean'])
                std_vals = np.array(results[param_name]['std'])
                param_values = np.array(param_values)

                # Mean
                plt.plot(param_values, mean_vals, color=color, label='mean')
                
                # Standard deviation
                plt.fill_between(
                    param_values,
                    mean_vals - std_vals,
                    mean_vals + std_vals,
                    color=color,
                    alpha=0.3,
                    label='± std'
                )
            else:
                plt.plot(param_values, results[param_name], '+-', color=color, label='Scores')

            plt.xlabel(param_name.capitalize())
            plt.ylabel("MCC Score")
            plt.title(f"Influence du paramètre {param_name}")
            plt.grid(True)
            plt.legend()

        plt.tight_layout()
        
        return fig
    
    def process_image(self, 
            name: str, 
            data_raw: ndarray, 
            data_gt: ndarray, 
            params: dict[str, list[float]], 
            experiment_config: ExperimentConfig
        ) -> tuple[dict[str, list[float]], plt.Figure, str]:
        
        # Compute scores
        results = self.compute_scores(data_raw, data_gt, params, experiment_config)
        # Plot scores
        figure = self.plot_scores(params, results, mean_std=False)

        return results, figure, name
    
    @log_time()
    @log_section("Check parameter influence")
    def run(self, 
            config: CPIConfig, 
            experiment_config: ExperimentConfig,
            images_raw: list[ndarray], 
            images_gt: list[ndarray],
        ) -> dict[str, dict[str, list[float]]]:

        # Select parameters
        parameters = {
            'alpha': config.alpha_values,
            'beta': config.beta_values,
            'gamma': config.gamma_values,
        }
        
        # Process images
        results = self.parallelizer.run(
            func=self.process_image,
            iterable=(
                (i, data_raw, data_gt, parameters, experiment_config) 
                for i, (data_raw, data_gt) in enumerate(zip(images_raw, images_gt))
            ),
            show_progress=True,
            unpack_args=True,
        )
        
        # Plot results
        results_sum = {key: [] for key in parameters}
        for result, figure, filename in results:
            self._display_figure(figure, f'params_influence_{filename}')
            for key, scores in result.items():
                results_sum[key].append(scores)
                        
        # Compute mean scores
        results_mean = {
            key: {
                'mean': np.mean(vals, axis=0),
                'std': np.std(vals, axis=0)
            } for key, vals in results_sum.items()
        }
        
        # Plot mean scores
        figure = self.plot_scores(parameters, results_mean)
        self._display_figure(figure, f'params_influence_mean')
            
        return results_mean

class ScalesAnalyzer(BaseAnalyzer):
    def compute_scores(self,
            data_raw: ndarray, 
            data_gt: ndarray, 
            ranges: list[tuple[int]], 
            steps: list[int], 
            experiment_config: ExperimentConfig
        ) -> dict[int, dict[str, list[float]]]:

        results = {}
        for step in steps:
            
            mcc_scores = []
            for r in ranges:
                
                experiment_params = deepcopy(experiment_config)
                
                scales = np.arange(r[0], r[1], step, dtype=int)
                experiment_params.enhancement.scales = scales
                experiment_params.processing.parallelize = True
                # Run pipeline
                experiment_data = self.pipeline.run(
                    config=experiment_params,
                    data_raw=data_raw,
                    data_gt=data_gt,
                )
                
                # Scores
                mcc_scores.append(mcc(experiment_data.segmented, data_gt))

            results[step] = {'scores': mcc_scores, 'ranges': ranges}
        
        return results

    def plot_scores(self, 
            scores_min: dict[int, dict[str, list[float]]], 
            scores_max: dict[int, dict[str, list[float]]], 
            mean: bool =False
        ):
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
        colors = [
            "#022c7aff",
            "#0743b1ff",
            "#175ddfff",
            "#407ff5ff",
            "#6097fcff",
        ]
            
        plot_params = [
            (scores_min, axes[0], "Min range (max=20)",
            "Influence de l'échelle min sur le réhaussement (moyenne)" if mean 
            else "Influence de l'échelle min sur le réhaussement", 0),
            (scores_max, axes[1], "Max range (min=1)",
            "Influence de l'échelle max sur le réhaussement (moyenne)" if mean 
            else "Influence de l'échelle max sur le réhaussement", 1)
        ]
        
        for scores, ax, xlabel, title, mode in plot_params:
            for i, (step, vals) in enumerate(scores.items()):
                xvals = [r[mode] for r in vals['ranges']]
                yvals = vals['scores']
                ax.plot(xvals, yvals, '+-', color=colors[i], label=f'step={step}')
            ax.set_xlabel(xlabel)
            if mode == 0:
                ax.set_ylabel("Score MCC")
            ax.set_title(title)
            ax.grid(True)
            ax.legend()
        plt.tight_layout()
        
        return fig
        
    
    def process_image(self,
            name: str,
            data_raw: ndarray, 
            data_gt: ndarray, 
            min_ranges: list[tuple[int]], 
            max_ranges: list[tuple[int]], 
            steps: list[int], 
            experiment_config: ExperimentConfig
        ):
        

            # Compute scores
            scores_min = self.compute_scores(data_raw, data_gt, min_ranges, steps, experiment_config)
            scores_max = self.compute_scores(data_raw, data_gt, max_ranges, steps, experiment_config)

            # Plot scores
            figure = self.plot_scores(scores_min, scores_max, mean=False)
            
            return scores_min, scores_max, figure, name
    
    @log_time()
    @log_section("Check scales influence")
    def run(self, 
            config: CSIConfig, 
            experiment_config: ExperimentConfig,
            images_raw: list[ndarray],
            images_gt: list[ndarray],
        ) -> tuple[dict, dict, plt.Figure, str]:

        # Select ranges
        min_ranges = [(min_range, 20) for min_range in config.min_ranges]
        max_ranges = [(1, max_range) for max_range in config.max_ranges]
        steps = config.scales_steps

        # Process images
        results = self.parallelizer.run(
            func=self.process_image,
            iterable=(
                (i, data_raw, data_gt, min_ranges, max_ranges, steps, experiment_config) 
                for i, (data_raw, data_gt) in enumerate(zip(images_raw, images_gt))
            ),
            show_progress=True, 
            unpack_args=True,
        )

        # Store scores
        all_scores_min = {step: [] for step in steps}
        all_scores_max = {step: [] for step in steps}
        for scores_min, scores_max, figure, filename in results:
            for step in steps:
                self._display_figure(figure, f'scales_influence_{filename}')
                all_scores_min[step].append(scores_min[step]['scores'])
                all_scores_max[step].append(scores_max[step]['scores'])
        
        # Compute mean scores
        mean_scores_min = {}
        mean_scores_max = {}
        for i, step in enumerate(steps):
            arr_min = np.array(all_scores_min[step])
            arr_max = np.array(all_scores_max[step])
            mean_scores_min[step] = {'scores': np.mean(arr_min, axis=0), 'ranges': min_ranges}
            mean_scores_max[step] = {'scores': np.mean(arr_max, axis=0), 'ranges': max_ranges}

        # Plot mean scores
        figure = self.plot_scores(mean_scores_min, mean_scores_max, mean=True)
        self._display_figure(figure, f'scales_influence_mean')
        
        return mean_scores_min, mean_scores_max
    
    
class EnhancementRunner:
     
    def __init__(self, setup: SetupConfig):
        self.setup = setup
        
        # Logger
        self.logger = setup_logger(log_file=self.setup.log_file, debug_mode=self.setup.debug_mode)
        
        # Loader
        self.loader = Loader(self.setup.input_dir, self.logger)
        
        # Parallelizer
        self.parallelizer = Parallelizer(use_processes=True, max_workers=4)
        
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
        self.viewer = Viewer(display_mode=self.setup.display_mode)
        
        # Saver
        self.saver = Saver(experiment_name=self.setup.name, output_dir=self.setup.output_dir, logger=self.logger)
        
        self.logger.info(f'[INIT] Analyzer initialized - Experiment {self.setup.name}')

    def _get_files(self, images_dir: str = "images", labels_dir: str = "labels"):
        images_dir = Path(self.setup.input_dir) / images_dir
        labels_dir = Path(self.setup.input_dir) / labels_dir
        images_files = []
        labels_files = []
        if not images_dir.exists() or not labels_dir.exists():
            self.logger.warning(f"The directories 'images' or 'labels' do not exist: {images_dir}, {labels_dir}")
            return [], []
        for image_name in os.listdir(images_dir):
            idx = image_name[len("image_"):]
            images_files.append(f"images/{image_name}")
            label_name = f"label_{idx}"
            labels_files.append(f"labels/{label_name}")
        return images_files, labels_files

    @log_time()
    @log_section("Run Enhancement Analyzer")
    def run(self, 
        analyzer_config: AnalyzerConfig, 
        experiment_config: ExperimentConfig
    ):
        
        # Load data
        raw_files, gt_files = self._get_files(
            images_dir=analyzer_config.images_dir, 
            labels_dir=analyzer_config.labels_dir
        )
        
        data_raw = []
        data_gt = []
        for raw, gt in zip(raw_files, gt_files):
            data_raw.append(self.loader.load_data(raw))
            data_gt.append(self.loader.load_data(gt))

        # Setup Analyzers
        params_analyzer = ParamsAnalyzer(
            save_mode=self.setup.save_mode,
            display_mode=self.setup.display_mode,
            pipeline=self.pipeline,
            logger=self.logger,
            parallelizer=self.parallelizer,
            saver=self.saver
        )
        scales_analyzer = ScalesAnalyzer(
            save_mode=self.setup.save_mode,
            display_mode=self.setup.display_mode,
            pipeline=self.pipeline,
            logger=self.logger,
            parallelizer=self.parallelizer,
            saver=self.saver
        )
        
        # Run Analyzers
        params_analyzer.run(
            config=analyzer_config.cpi,
            experiment_config=experiment_config,
            images_raw=data_raw,   
            images_gt=data_gt,
        )
        scales_analyzer.run(
            config=analyzer_config.csi,
            experiment_config=experiment_config,
            images_raw=data_raw,   
            images_gt=data_gt,
        )
        
