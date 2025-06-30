import numpy as np 

from benchmark.computational_time import chunk_number_influence

import numpy as np
from utils.viewer import Viewer  # adapte le chemin si besoin

def debug():

    data = np.random.rand(32, 32, 62)

    Viewer().display_slices(data)

DEBUG_MODE = False

if __name__ == "__main__":
    debug()
    # chunk_number_influence([32, 64], 2, (1, 10), 'frangi', [2, 4])