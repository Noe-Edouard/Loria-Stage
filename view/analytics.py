
from benchmark.metrics import dice, mcc, roc, pr, roc_curve, precision_recall_curve
from numpy import ndarray
import numpy as np
import matplotlib.pyplot as plt
from core.logger import setup_logger, Logger
from tabulate import tabulate
from view.viewer import Viewer
from core.config import ExperimentData, ExperimentConfig, MetricsConfig
from utils.helpers import create_error_map


class Analytics():
    
    def __init__(self, logger: Logger = setup_logger(), display_mode: bool = True):
        self.logger = logger
        self.display_mode = display_mode
        self.viewer = Viewer(self.display_mode)
        
    def get_configs(self, experiments: list[ExperimentData]) -> dict[str, ExperimentConfig]:
        configs = {}
        for experiment in experiments:
            configs[experiment.config.methods.derivator] = experiment.config
        return configs
    
    def get_metrics(self, experiments: list[ExperimentData], ground_truth: ndarray) ->list[MetricsConfig]:
        metrics = []
        for experiment in experiments:
            method = experiment.config.methods.derivator
            metrics.append({
                'method': experiment.config.methods.derivator,
                'dice': dice(experiment.segmented, ground_truth),
                'mcc': mcc(experiment.segmented, ground_truth),
                'roc': roc(experiment.enhanced, ground_truth),
                'pr': pr(experiment.enhanced, ground_truth),
            })
        return metrics
        
    def display_configs(self, experiments: list[ExperimentData]):
        configs = self.get_configs(experiments)
        content = "\nEXPERIMENTS CONFIGS\n"
        for method, config in configs.items():
            content += f"{method}: alpha={config.enhancement.alpha:.4f} | beta={config.enhancement.beta:.4f} | gamma={config.enhancement.gamma:.4f} | threshold={config.segmentation.threshold:.4f}"
        content += "\nBENCHMARK CONFIG\n"
        content += f"{config}"
        self.logger.info(content)

        return content
        
    def display_metrics(self, experiments: list[ExperimentData], ground_truth: ndarray):
        metrics = self.get_metrics(experiments, ground_truth)

        best = {}
        for metric_name in ['dice', 'mcc', 'roc', 'pr']:
            best[metric_name] = max(metrics, key=lambda m: m[metric_name])['method']

        best_metric = {'method': 'best'}
        best_metric.update(best)
        metrics.append(best_metric)

        metrics = [{k.upper(): v for k, v in m.items()} for m in metrics]

        # Affichage avec tabulate
        table = tabulate(metrics, headers="keys", tablefmt='github', floatfmt=':.4f')

        content = f'\nBENCHMARK METRICS\n{table}'
        self.logger.info(content)
        return content


    def display_summary_metrics(self, results_mean: dict, results_std: dict):
        headers = ['METHOD', 'DICE', 'MCC', 'ROC', 'PR']
        table = []

        for method in results_mean.keys():
            row = [method]
            for metric in ['dice', 'mcc', 'roc', 'pr']:
                mean = results_mean[method][metric]
                std = results_std[method][metric]
                row.append(f"{mean:.4f} ± {std:.4f}")
            table.append(row)

        table_str = tabulate(table, headers=headers, tablefmt='github')
        content = "\nSUMMARY METRICS ACROSS ALL IMAGES\n" + table_str
        self.logger.info(content)
        return content

  


    def display_histograms(
        self,
        experiments: list[ExperimentData],
        data_raw: np.ndarray,
        data_gt: np.ndarray,
        bins: int = 50,
        density: bool = False,
        color: str = 'dodgerblue',
    ) -> plt.Figure:
        ncols = len(experiments) + 1
        nrows = 2
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4.5 * ncols, 4 * nrows))

        # Add vertical labels
        fig.text(0.015, 0.752, 'Enhanced', va='center', ha='center', rotation='vertical', fontsize=11, fontweight='bold')
        fig.text(0.015, 0.278, 'Segmented', va='center', ha='center', rotation='vertical', fontsize=11, fontweight='bold')

        # Data and titles for each row
        row_data = [
            [data_raw] + [exp.enhanced for exp in experiments],
            [data_gt] + [exp.segmented for exp in experiments],
        ]
        row_titles = [
            ['raw data'] + [exp.config.methods.derivator for exp in experiments],
            ['ground truth'] + [exp.config.methods.derivator for exp in experiments],
        ]

        for row in range(nrows):
            for col in range(ncols):
                axs[row, col].hist(row_data[row][col].ravel(), bins=bins, density=density, color=color)
                axs[row, col].set_title(row_titles[row][col], fontsize=9)
                if row == 0:
                    axs[row, col].set_ylabel('Density' if density else 'Frequency')
                else:
                    axs[row, col].set_xlabel('Intensity')
                    axs[row, col].set_ylabel('Density' if density else 'Frequency')
                axs[row, col].grid(True)
                axs[row, col].tick_params('y', labelsize=8, labelrotation=90)

        plt.subplots_adjust(left=0.06, right=0.98, bottom=0.08, top=0.95, wspace=0.15, hspace=0.2)
        if self.display_mode:
            plt.show()
        return fig


        

    def display_curves(self, experiments: list[ExperimentData], ground_truth: ndarray):
        
        y_true = ground_truth.ravel()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        colors = plt.cm.gist_rainbow(np.linspace(0, 1, len(experiments)))
        for i, experiment in enumerate(experiments):
            y_scores = experiment.enhanced.ravel()

            # ROC
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            ax1.plot(fpr, tpr, label=experiment.config.methods.derivator, color=colors[i])

            # PR
            precision, recall, _ = precision_recall_curve(y_true, y_scores)
            ax2.plot(recall, precision, label=experiment.config.methods.derivator, color=colors[i])

        # ROC subplot
        ax1.plot([0, 1], [0, 1], 'k--', label='random')
        ax1.set_xlabel('False Positive Rate (FPR)')
        ax1.set_ylabel('True Positive Rate (TPR)')
        ax1.set_title('ROC Curve')
        ax1.legend(loc='lower right')
        ax1.grid(True)

        # PR subplot
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curve')
        ax2.legend(loc='lower left')
        ax2.grid(True)

        # Final display
        fig.tight_layout()
        if self.display_mode:
            plt.show()

        return fig
        
    

    def display_views(
        self,
        experiments: list[ExperimentData],
        data_gt: ndarray,
        data_raw: ndarray
    ):
        dim = data_gt.ndim
        data_enhanced = [data_raw] + [exp.enhanced for exp in experiments]
        data_segmented = [data_gt] + [exp.segmented for exp in experiments]
        error_maps = [data_gt] + [create_error_map(data_gt, exp.segmented) for exp in experiments]
        methods = [exp.config.methods.derivator for exp in experiments]

        titles_enhanced = ['raw data'] + methods
        titles_segmented = ['ground truth'] + methods
        titles_error_maps = ['ground truth'] + methods

        if dim == 2:
            plot_enhanced = self.viewer.display_images(data_enhanced, titles=titles_enhanced)
            plot_segmented = self.viewer.display_images(data_segmented, titles=titles_segmented, error_mode=False)
            plot_error_maps = self.viewer.display_images(error_maps, titles=titles_error_maps, error_mode=True)
            return [plot_enhanced, plot_segmented, plot_error_maps], ['images_enhanced', 'images_segmented', 'error_maps'], ['plot', 'plot', 'plot']
        else:
            mip_enhanced = self.viewer.display_mip(data_enhanced, titles=titles_enhanced, cmap='viridis')
            slices_enhanced = self.viewer.display_slices(data_enhanced, titles=titles_enhanced, interval=100, cmap='viridis')
            slices_segmented = self.viewer.display_slices(data_segmented, titles=titles_segmented, interval=100, error_mode=False)
            slices_error_maps = self.viewer.display_slices(error_maps, titles=titles_error_maps, interval=100, error_mode=True)
            return [mip_enhanced, slices_enhanced, slices_segmented, slices_error_maps], ['mip_enhanced', 'slices_enhanced', 'slices_segmented', 'slices_error_maps'], ['plot', 'anim', 'anim', 'anim']

            
            
            
            
    
         
            


