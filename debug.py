import numpy as np 
from core.loader import Loader
from benchmark.analyzer_processing import chunk_number_influence

import numpy as np
from view.viewer import Viewer  # adapte le chemin si besoin

def debug():

    loader = Loader("data/test")
    data_raw = loader.load_data('raw.png')
    data_gt = loader.load_data('gt.png')
    Viewer().display_images([data_raw, data_gt])

DEBUG_MODE = False

if __name__ == "__main__":
    debug()
    # chunk_number_influence([32, 64], 2, (1, 10), 'frangi', [2, 4])
    
    
    
