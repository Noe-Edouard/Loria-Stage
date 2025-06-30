import numpy as np 

from benchmark.computational_time import chunk_number_influence

import numpy as np
from utils.viewer import Viewer  # adapte le chemin si besoin

def debug():
    from utils.viewer import Viewer
    from utils.io import DataIO

    io = DataIO("outputs/test")
    data = io.load_data("2025-06-29_18-16-20test_experiment_enhanced_output.nii")
    # Viewer().display_slices(data, 'enhanced')
    # Viewer().display_volume(data, 0.1)
    Viewer().display_mip(data)

DEBUG_MODE = False

if __name__ == "__main__":
    debug()
    # chunk_number_influence([32, 64], 2, (1, 10), 'frangi', [2, 4])