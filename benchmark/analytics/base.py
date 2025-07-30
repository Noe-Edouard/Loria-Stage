
import matplotlib.pyplot as plt
from typing import Union

from core.io.viewer import Viewer
from core.config.figure import FigureData, FigureMode
from core.io.logger import setup_logger, Logger


class AnalyticsBase():
    
    def __init__(self, logger: Logger = setup_logger()):
        self.logger = logger
        self.viewer = Viewer(display_mode=False)
        

    def _create_figure(self, figure: Union[plt.Figure, str], name: str, mode: FigureMode) -> FigureData:
        return FigureData(name=name, figure=figure, mode=mode)
    
    
    