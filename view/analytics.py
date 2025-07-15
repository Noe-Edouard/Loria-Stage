
from benchmark.metrics import dice, mcc, roc, pr, roc_curve, precision_recall_curve
from numpy import ndarray
import numpy as np
import matplotlib.pyplot as plt
from core.logger import setup_logger, Logger
from tabulate import tabulate
from view.viewer import Viewer
from core.config import ExperimentData
from utils.helpers import create_error_map


class Analytics():
    
    def __init__(self, logger: Logger = setup_logger(), display_mode: bool = True):
        self.logger = logger
        self.display_mode = display_mode
        self.viewer = Viewer(self.display_mode)
        
    
    def display_metrics(self, experiments: list[ExperimentData], ground_truth: ndarray):
        methods, dice_scores, mcc_scores, roc_scores, pr_scores = [], [], [], [], []

        for experiment in experiments:
            dice_scores.append(dice(experiment.segmented, ground_truth))
            mcc_scores.append(mcc(experiment.segmented, ground_truth))
            roc_scores.append(roc(experiment.enhanced, ground_truth))
            pr_scores.append(pr(experiment.enhanced, ground_truth))
            methods.append(experiment.config.methods.derivator)

        # Best method
        dice_scores.append(f'{methods[dice_scores.index(max(dice_scores))]}')
        mcc_scores.append(f'{methods[mcc_scores.index(max(mcc_scores))]}')
        roc_scores.append(f'{methods[roc_scores.index(max(roc_scores))]}')
        pr_scores.append(f'{methods[pr_scores.index(max(pr_scores))]}')
        methods.append('BEST')

        headers = ['METHOD', 'DICE', 'MCC', 'ROC', 'PR']
        table_data = list(zip(methods, dice_scores, mcc_scores, roc_scores, pr_scores))
        table = tabulate(table_data, headers, tablefmt='github', floatfmt=':.4f')

        content = f'\nBENCHMARK METRICS\n{table}'
        self.logger.info(content)

        return content

        
  

    def display_histograms(self, experiments: list[ExperimentData], raw_data: np.ndarray, ground_truth: np.ndarray,
                        bins: int = 50, density: bool = False, color='dodgerblue'):
        ncols = len(experiments) + 1  # +2 fot titles and ref
        nrows = 2

        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4.5 * ncols, 4 * nrows))
        # grid = gridspec.GridSpec(nrows=nrows, ncols=ncols, figure=fig)

        # Add vertical labels
        fig.text(0.015, 0.752, 'Enhanced', va='center', ha='center', rotation='vertical', fontsize=11, fontweight='bold')
        fig.text(0.015, 0.278, 'Segmented', va='center', ha='center', rotation='vertical', fontsize=11, fontweight='bold')

        # Raw Data
        # ax = fig.add_subplot(grid[0, 0])
        axs[0, 0].hist(raw_data.ravel(), bins=bins, density=density, color=color)
        axs[0, 0].set_title('Raw Data', fontsize=9)
        axs[0, 0].set_ylabel('Density' if density else 'Frequency')
        axs[0, 0].grid(True)
        axs[0, 0].tick_params('y', labelsize=8, labelrotation=90)
        

        # Ground Truth
        # ax = fig.add_subplot(grid[1, 0])
        axs[1, 0].hist(ground_truth.ravel(), bins=bins, density=density, color=color)
        axs[1, 0].set_title('Ground Truth', fontsize=9)
        axs[1, 0].set_xlabel('Intensity')
        axs[1, 0].set_ylabel('Density' if density else 'Frequency')
        axs[1, 0].grid(True)
        axs[1, 0].tick_params('y', labelsize=8, labelrotation=90)

        for i, experiment in enumerate(experiments):
            # Enhanced
            # ax = fig.add_subplot(grid[0, i + 1])
            axs[0, i+1].hist(experiment.enhanced.ravel(), bins=bins, density=density, color=color)
            axs[0, i+1].set_title(experiment.config.methods.derivator, fontsize=9)
            axs[0, i+1].grid(True)
            axs[0, i+1].tick_params('y', labelsize=8, labelrotation=90)

            # Segmented
            # ax = fig.add_subplot(grid[1, i + 1])
            axs[1, i+1].hist(experiment.segmented.ravel(), bins=bins, density=density, color=color)
            axs[1, i+1].set_title(experiment.config.methods.derivator, fontsize=9)
            axs[1, i+1].set_xlabel('Intensity')
            axs[1, i+1].grid(True)
            axs[1, i+1].tick_params('y', labelsize=8, labelrotation=90)
            
        # plt.tight_layout(rect=[0.035, 0, 0.99, 1])
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
        
    
    def display_views(self, experiments: list[ExperimentData], ground_truth: ndarray, raw_data: ndarray):
        dim = ground_truth.ndim
        
        data_enhanced, data_segmented, error_maps, methods = [raw_data], [ground_truth], [ground_truth], []
        for experiment in experiments:
            data_enhanced.append(experiment.enhanced)
            data_segmented.append(experiment.segmented)
            error_maps.append(create_error_map(ground_truth, experiment.segmented))
            methods.append(experiment.config.methods.derivator)
        
        if dim == 2:
            titles_enhanced   = ['raw data']     + methods.copy()
            titles_segmented  = ['ground truth'] + methods.copy()
            titles_error_maps = ['ground truth'] + methods.copy()

            plot_enhanced   = self.viewer.display_images(data_enhanced, titles=titles_enhanced)
            plot_segmented  = self.viewer.display_images(data_segmented, titles=titles_segmented, error_mode=False)
            plot_error_maps = self.viewer.display_images(error_maps, titles=titles_error_maps, error_mode=True)

            return [plot_enhanced, plot_segmented, plot_error_maps], ['images_enhanced', 'images_segmented', 'error_maps'], ['plot', 'plot', 'plot']

        else:  # dim == 3
            titles_enhanced   = ['raw data']     + methods.copy()
            titles_segmented  = ['ground truth'] + methods.copy()
            titles_error_maps = ['ground truth'] + methods.copy()

            mip_enhanced      = self.viewer.display_mip(data_enhanced, titles=titles_enhanced, cmap='viridis')
            slices_enhanced   = self.viewer.display_slices(data_enhanced, titles=titles_enhanced, interval=100, cmap='viridis')
            slices_segmented  = self.viewer.display_slices(data_segmented, titles=titles_segmented, interval=100, error_mode=False)
            slices_error_maps = self.viewer.display_slices(error_maps, titles=titles_error_maps, interval=100, error_mode=True)

            return [mip_enhanced, slices_enhanced, slices_segmented, slices_error_maps], ['mip_enhanced', 'slices_enhanced', 'slices_segmented', 'slices_error_maps'], ['plot', 'anim', 'anim', 'anim']

            
            
            
            
    
         
            


