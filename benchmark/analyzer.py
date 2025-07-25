from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from core.logger import Logger
from utils.parallelizer import Parallelizer
from pipeline.pipeline import Pipeline
from core.saver import Saver
from pipeline.enhancer import Enhancer

class BaseAnalyzer(ABC):

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
    