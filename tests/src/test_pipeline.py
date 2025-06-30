from pathlib import Path
from src.pipeline import Pipeline
from utils.config import Config
import numpy as np
import nibabel as nib
from utils.viewer import Viewer

def test_pipeline():
    config_dict = {
        "experiment": {
            "name": "test_experiment",
            "debug_mode": True,
            "normalize": True,
            "crop": False,
            "target_shape": [64, 64, 64],
            "input_file": "spiral.nii",
            "output_file": "output.nii",
            "input_dir": "data/test",
            "output_dir": "outputs/test",
            "log_dir": "logs/test"
        },
        "methods": {
            "estimator": "default",
            "enhancer": "frangi",
            "segmenter": "thresholding"
        },
        "hessian": {
            "mode": "reflect",
            "cval": 0.0,
            "use_gaussian_derivatives": False
        },
        "processing": {
            "normalize": True,
            "parallelize": True,
            "chunk_size": [32, 32, 32],
            "overlap_size": 8
        },
        "enhancement": {
            "alpha": 0.5,
            "beta": 0.5,
            "gamma": 15,
            "scales": np.arange(0, 10, 2),
            "scales_number": None,
            "scales_range": None
        },
        "segmentation": {
            "method": "thresholding",
            "threshold": 0.1
        }
    }


    config = Config(config_dict)
    pipeline = Pipeline(config)
    pipeline.run()

    # viewer = Viewer()
    # viewer.display_images([data, enhanced_data, segmented_data], ['data', 'enhanced_data', 'segmented_data'])
    # viewer.display_slices([data, enhanced_data, segmented_data], ['data', 'enhanced_data', 'segmented_data'])