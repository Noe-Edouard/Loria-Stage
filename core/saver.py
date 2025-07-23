from pathlib import Path
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from datetime import datetime
from typing import Literal
from core.logger import Logger, setup_logger

class Saver:
    def __init__(self, experiment_name: str = "default", output_dir: str | Path = "outputs/results", logger: Logger = setup_logger()):
        self.logger = logger
        
        self.output_dir = Path(f'{output_dir}/{self._get_timestamp()}_{experiment_name}')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        

    def _get_timestamp(self) -> str:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        return timestamp
    
    
    def save_text(self, content: str, filename: str):
        path = self.output_dir / f'text_{filename}'
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)

        self.logger.info(f'[SAVE] Text saved as {filename}.')


    def save_data(self, data: np.ndarray, filename: str, extension: Literal['.nii', '.npz', '.npy', ] = '.npz'):
        path = self.output_dir / f'data_{filename}'
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
        path = self.output_dir / f'plot_{filename}'
        path.parent.mkdir(parents=True, exist_ok=True)
        
        fig.savefig(path, dpi=dpi, bbox_inches='tight')
        self.logger.info(f'[SAVE] Plot saved as {filename}.')


    def save_animation(self, anim: FuncAnimation, filename: str, extension: Literal['.mp4', '.mov', '.avi', '.gif'] = '.gif', fps: int = 30, dpi: int = 150):
        path = self.output_dir / f'animation_{filename}{extension}'
        path.parent.mkdir(parents=True, exist_ok=True)

        anim.save(str(path), fps=fps, dpi=dpi)
        self.logger.info(f'[SAVE] Animation saved as {filename}{extension}.')
        
