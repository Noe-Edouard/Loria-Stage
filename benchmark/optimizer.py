import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from tabulate import tabulate
from core.logger import Logger, setup_logger
from core.config import ConfigBuilder, ExperimentData, ExperimentConfig, SetupConfig, CSIConfig, CAIConfig, CBIConfig, OptimizerConfig, CGIConfig
from utils.decorator import log_time, log_section
from core.loader import Loader
from pathlib import Path
from view.viewer import Viewer
from benchmark.metrics import mcc
from core.saver import Saver
import matplotlib.pyplot as plt
import os
from pipeline.pipeline import Pipeline
from typing import Union


class Optimizer:
     
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
        self.viewer = Viewer(display_mode=self.setup.display_mode)
        
        # Saver
        self.save_mode = self.setup.save_mode
        self.saver = Saver(experiment_name=self.setup.name, output_dir=self.output_dir, logger=self.logger)
        
        self.logger.info(f'[INIT] Optimizer initialized - Experiment {self.setup.name}')

    def _get_files(self):
        images_dir = self.input_dir / "images"
        labels_dir = self.input_dir / "labels"
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

    def _get_colors(self):
        colors = [
            "#022c7aff",
            "#0743b1ff",
            "#175ddfff",
            "#407ff5ff",
            "#6097fcff",
        ]
        return colors
    
    def _check_param_influence(self, 
        config: Union[CSIConfig, CAIConfig, CBIConfig, CGIConfig], 
        experiment_config: ExperimentConfig, 
        param_name: str, 
        param_values: list[float], 
        plot_title: str, 
        plot_label: str, 
    ):
        
        results = {}
        raw_files, gt_files = self._get_files()
        
        for raw_file, gt_file in zip(raw_files, gt_files):
            self.logger.info(f"[PROCESS] Processing {raw_file}")
            data_raw = self.loader.load_data(raw_file)
            data_gt = self.loader.load_data(gt_file)
            
            experiment_params = deepcopy(experiment_config)
          
            mcc_scores = []
            for val in param_values:
                setattr(experiment_params.enhancement, param_name, val)
                experiment_params.processing.parallelize = True
                
                # Run pipeline
                experiment_data = self.pipeline.run(
                    config=experiment_params,
                    data_raw=data_raw,
                    ground_truth=data_gt,
                )
                
                # Scores
                mcc_scores.append(mcc(experiment_data.segmented, data_gt))

            results[raw_file] = mcc_scores
            
        # Plot
        fig = plt.figure()
        colors = self._get_colors()
        for i, scores in enumerate(results.values()):
            plt.plot(param_values, scores, '+-', color=colors[i], label=plot_label)
        plt.xlabel(param_name.capitalize())
        plt.ylabel("Score MCC")
        plt.title(plot_title)
        plt.grid(True)
        plt.legend()
        if self.setup.display_mode:
            plt.show()
        if self.setup.save_mode:
            self.saver.save_plot(fig, config.output_file)
            
        return {param_name: param_values, 'scores': mcc_scores}
    

    @log_time()
    @log_section("Check scales influence")
    def check_scales_influence(self, config: CSIConfig, experiment_config: ExperimentConfig):

        def compute_scores(volume, ground_truth, ranges, steps, experiment_config: ExperimentConfig):
            
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
                        data_raw=volume,
                        ground_truth=ground_truth,
                    )
                    
                    # Scores
                    mcc_scores.append(mcc(experiment_data.segmented, ground_truth))

                self.logger
                    
                results[step] = {'scores': mcc_scores, 'ranges': ranges}
            
            return results

        def plot_scores(scores_min, scores_max, raw_file=None, mean=False, show_std=False, std_min=None, std_max=None):
            fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
            colors = self._get_colors()
            plot_params = [
                (scores_min, std_min, axes[0], "Min range (max=20)",
                 "Influence de l'échelle min sur le réhaussement (moyenne ± écart-type)" if mean and show_std else ("Influence de l'échelle min sur le réhaussement (moyenne)" if mean else "Influence de l'échelle min sur le réhaussement"), 0),
                (scores_max, std_max, axes[1], "Max range (min=1)",
                 "Influence de l'échelle max sur le réhaussement (moyenne ± écart-type)" if mean and show_std else ("Influence de l'échelle max sur le réhaussement (moyenne)" if mean else "Influence de l'échelle max sur le réhaussement"), 1)
            ]
            for scores, stds, ax, xlabel, title, mode in plot_params:
                for i, (step, vals) in enumerate(scores.items()):
                    xvals = [r[mode] for r in vals['ranges']]
                    yvals = vals['scores']
                    if show_std and stds is not None and step in stds:
                        yerr = stds[step]['scores']
                        ax.errorbar(xvals, yvals, yerr=yerr, fmt='o-', color=colors[i], label=f'step={step}')
                    else:
                        ax.plot(xvals, yvals, '+-', color=colors[i], label=f'step={step}')
                ax.set_xlabel(xlabel)
                if mode == 0:
                    ax.set_ylabel("Score MCC")
                ax.set_title(title)
                ax.grid(True)
                ax.legend()
            plt.tight_layout()
            if mean:
                filename = f'{config.output_file}_mean'
            else:
                filename = f'{config.output_file}_{Path(raw_file).stem}'
            if self.setup.display_mode:
                plt.show()
            if self.setup.save_mode:
                self.saver.save_plot(fig, filename)

        # Select ranges
        min_ranges = [(min_range, 20) for min_range in config.min_ranges]
        max_ranges = [(1, max_range) for max_range in config.max_ranges]
        steps = config.scales_steps

        # Store scores for mean
        all_scores_min = {step: [] for step in steps}
        all_scores_max = {step: [] for step in steps}

        # Results per image
        raw_files, gt_files = self._get_files()
        for raw_file, gt_file in zip(raw_files, gt_files):
            self.logger.info(f"[PROCESS] Processing {raw_file}")
            volume = self.loader.load_data(raw_file)
            ground_truth = self.loader.load_data(gt_file)

            # Compute scores
            scores_min = compute_scores(volume, ground_truth, min_ranges, steps, experiment_config)
            scores_max = compute_scores(volume, ground_truth, max_ranges, steps, experiment_config)
            
            # Store scores
            for step in steps:
                all_scores_min[step].append(scores_min[step]['scores'])
                all_scores_max[step].append(scores_max[step]['scores'])

            # Plot (sans std)
            plot_scores(scores_min, scores_max, raw_file=raw_file, show_std=False)

        # Plot (mean + std)
        mean_scores_min = {}
        mean_scores_max = {}
        std_scores_min = {}
        std_scores_max = {}
        for i, step in enumerate(steps):
            arr_min = np.array(all_scores_min[step])
            arr_max = np.array(all_scores_max[step])
            mean_scores_min[step] = {'scores': np.mean(arr_min, axis=0), 'ranges': min_ranges}
            mean_scores_max[step] = {'scores': np.mean(arr_max, axis=0), 'ranges': max_ranges}
            std_scores_min[step] = {'scores': np.std(arr_min, axis=0), 'ranges': min_ranges}
            std_scores_max[step] = {'scores': np.std(arr_max, axis=0), 'ranges': max_ranges}

        plot_scores(mean_scores_min, mean_scores_max, show_std=False, std_min=std_scores_min, std_max=std_scores_max, mean=True)

        return None

    

    @log_time()
    @log_section("Check alpha influence")
    def check_alpha_influence(self, config: CAIConfig, experiment_config: ExperimentConfig):
        return self._check_param_influence(
            config,
            experiment_config,
            param_name='alpha',
            param_values=config.alpha_values,
            plot_title="Influence du paramètre alpha sur le réhaussement",
            plot_label='alpha',
        )

    @log_time()
    @log_section("Check beta influence")
    def check_beta_influence(self, config: CBIConfig, experiment_config: ExperimentConfig):
        return self._check_param_influence(
            config,
            experiment_config,
            param_name='beta',
            param_values=config.beta_values,
            plot_title="Influence du paramètre beta sur le réhaussement",
            plot_label='beta',
        )

    @log_time()
    @log_section("Check gamma influence")
    def check_gamma_influence(self, config: CGIConfig, experiment_config: ExperimentConfig):
        return self._check_param_influence(
            config,
            experiment_config,
            param_name='gamma',
            param_values=config.gamma_values,
            plot_title="Influence du paramètre gamma sur le réhaussement",
            plot_label='gamma',
        )


    def run(self, optimizer_config: OptimizerConfig, experiment_config: ExperimentConfig):
        # CSI (scales influence)
        self.check_scales_influence(optimizer_config.csi, experiment_config)

        # CAI (alpha influence)
        # self.check_alpha_influence(optimizer_config.cai, experiment_config)
        
        # CBI (beta influence)
        # self.check_beta_influence(optimizer_config.cbi, experiment_config)

        # GBI (gamma influence)
        # self.check_gamma_influence(optimizer_config.cgi, experiment_config)
        
