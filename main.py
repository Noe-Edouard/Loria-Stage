from utils.loader import load
from utils.viewer import display_animation, display_volume 
from utils.chunker import *
from time import time
import numpy as np


if __name__ == "__main__":
    
    # Test display animation simple
    # image = load(filename)
    # image = load('./data/test/vessels/VESSEL12_01.mhd', (200, 200, 200))
    # display_animation(image)
    
    # Test dislay volume
    # image = load('./data/test/filtered.nii') # 0.05
    # image = load('./data/processed/data_gt.nii')
    image = load('./data/test/test_filtered_gamma5.nii') # 0.5
    display_volume(image, threshold=0.5)

    
    #Test display animation multi
    # image1 = load('./data/raw/appps1_cc.nii')
    # image2 = load('./data/processed/data_gt.nii')
    # image3 = load('./data/processed/data.nii')
    # display_animation([image1, image2, image3])