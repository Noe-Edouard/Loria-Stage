from pathlib import Path
from src.pipeline import Pipeline
from utils.config import Config
import numpy as np


def test_pipeline():
    config_dict = {
        "experiment": {
            "name": "test_pipeline",
            "debug_mode": True,
            "normalize": True,
            "crop": True,
            "target_shape": [64, 64, 64],
            "input_file": "test.nii",
            "log_file": "tests",
            "input_dir": "data/test",
            "output_dir": "outputs/test",
            "log_dir": "logs"
        },
        "methods": {
            "derivator": "default",
            "enhancer": "frangi",
            "segmenter": "thresholding"
        },
        "hessian": {
            "mode": "reflect",
            "cval": 0.0,
            "use_gaussian_derivatives": True
        },
        "processing": {
            "normalize": True,
            "parallelize": False,
            "chunk_size": [32, 32, 32],
            "overlap_size": 8
        },
        "enhancement": {
            "alpha": 0.5,
            "beta": 0.5,
            "gamma": 15,
            "scales": np.arange(0, 10, 2),
            "scales_number": None,
            "scales_range": None,
            "black_ridges": False,
            "skimage": False,
        },
        "segmentation": {
            "method": "thresholding",
            "threshold": 0.8
        }
    }


    config = Config(config_dict)
    pipeline = Pipeline(config)
    pipeline.run()

    
# ================== REMARQUES ================== # 

# "use_gaussian_derivatives": False donne de mauvais résultats (notamment quand on crop trop fort)
# Lorsque l'on utilise la parallelisation, il faut fixer gamma sinon il y a des problèmes d'intensité
# il est très important d'initialiser les scales np.arange(0, 10, 2) et pas np.arange(1, 10, 2)
# il est impératif de normaliser sinon les résultats sont  incohérents.