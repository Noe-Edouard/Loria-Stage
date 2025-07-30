import numpy as np
from core.viewer import Viewer
from utils.helpers import create_error_map
def test_viewer():
    viewer = Viewer()

    # 2D
    img1 = np.random.rand(128, 128)
    img2 = np.random.rand(128, 128)
    bin1 = (img1 > 0.5).astype(np.int8)
    bin2 = (img2 > 0.5).astype(np.int8)
    images_2d = [img1, img2]
    
    error_map = create_error_map(bin1, bin2)

    viewer.display_images(images_2d, titles=["Image A", "Image B"])
    viewer.display_images([bin1, bin2, error_map], titles=["Ground Truth", "Segmented", "Errors"], error_mode=True)
    viewer.display_histograms(images_2d, titles=["Histogramme A", "Histogramme B"])


    # 3D
    volume1 = np.random.rand(64, 64, 64)
    volume2 = np.random.rand(64, 64, 64)
    volumes = [volume1, volume2]
    binvol1 = (volume1 > 0.5).astype(np.int8)
    binvol2 = (volume2 > 0.5).astype(np.int8)
    error_map = create_error_map(binvol1, binvol2)
    
    
    viewer.display_mip(volumes, titles=["Volume 1", "Volume 2"])
    viewer.display_volume(binvol1, threshold=0)
    viewer.display_slices(volumes, titles=["Volume 1", "Volume 2"], interval=100)
    viewer.display_slices([binvol1, binvol2, error_map], titles=["Ground Truth", "Segmented", "Errors"], interval=100, error_mode=True)

