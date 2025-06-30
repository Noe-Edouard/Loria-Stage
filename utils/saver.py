import os
from pathlib import Path
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from datetime import datetime
from typing import Literal
from utils.logger import Logger, setup_logger

class Saver:
    def __init__(self, output_dir: str | Path = "outputs", logger: Logger = setup_logger()):
        self.logger = logger
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)


    def _add_timestamp(self, filename: str) -> str:
        basename, extension = os.path.splitext(filename)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        return f"{timestamp}_{basename}{extension}"


    def save_data(self, data: np.ndarray, filename: str, extension: Literal['.nii', '.npz', '.npy', ] = '.npz'):
        filename = self._add_timestamp(filename)
        path = self.output_dir / f'data/{filename}'
        path.parent.mkdir(parents=True, exist_ok=True)

        if extension == '.nii':
            data_nii = nib.Nifti1Image(data.astype(np.float32), affine=np.eye(4))
            nib.save(data_nii, path)
        elif extension in ['.npy', '.npz']:
            if extension == '.npy':
                np.save(path, data)
            else:
                np.savez(path, data=data)
        else:
            raise ValueError('Extension extension invalid.')

        self.logger.info(f'[SAVE] Data file {filename}{extension} saved at {path}.')


    def save_plot(self, fig: plt.Figure, filename: str, dpi: int = 150):
        filename = self._add_timestamp(filename)
        path = self.output_dir / f'plots/{filename}'
        path.parent.mkdir(parents=True, exist_ok=True)
        
        fig.savefig(path, dpi=dpi, bbox_inches='tight')
        self.logger.info(f'[SAVE] Plot saved as {filename}.')


    def save_animation(self, anim: FuncAnimation, filename: str, extension: Literal['.mp4', '.mov', '.avi', '.gif'] = '.gif', fps: int = 30, dpi: int = 150):
        filename = self._add_timestamp(filename)
        path = self.output_dir / f'animations/{filename}{extension}'
        path.parent.mkdir(parents=True, exist_ok=True)

        anim.save(str(path), fps=fps, dpi=dpi)
        self.logger.info(f'[SAVE] Animation saved as {filename}{extension}.')
        
