import numpy as np
from utils.viewer import Viewer  # adapte le chemin si besoin

def test_viewer():
    volume1 = np.random.rand(30, 30, 30)
    volume2 = np.random.rand(30, 30, 30) * 0.5
    volume3 = np.zeros((30, 30, 30), dtype=np.uint8)
    volume3[10:20, 10:20, 10:20] = 255

    titles = ['Random Volume 1', 'Random Volume 2', 'Cube Volume']

    viewer = Viewer()
    viewer.display_slices([volume1, volume2, volume3], titles, interval=100)
