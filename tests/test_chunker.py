import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from enhancement.frangi import frangi_filter
from utils.loader import load
from utils.chunker import crop_volume
from pipeline import process_volume, apply_enhancement


def test_chunker():
    volume = crop_volume(load('data/raw/mouse_brain.nii'), (100, 100, 100))
    filtered_image_para = process_volume(volume, (35, 35, 35))
    filtered_image_seq = apply_enhancement(volume)
    # Save
    return filtered_image_para, filtered_image_seq