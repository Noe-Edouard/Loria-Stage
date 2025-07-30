import numpy as np
import pytest
from core.viewer import Viewer


@pytest.fixture
def viewer() -> Viewer:
    return Viewer(display_mode=True)


def test_display_images(viewer: Viewer):
    imgs = [np.random.rand(50, 50) for _ in range(8)]
    titles = ["Method" for _ in range(8)]
    fig = viewer.display_images(imgs, titles=titles)


def test_display_error_mode(viewer: Viewer):
    img = np.zeros((50, 50), dtype=np.uint8)
    img[10:20, 10:20] = 1
    fig = viewer.display_images(img, titles="Binary", error_mode=True)


def test_display_histograms(viewer: Viewer):
    imgs = [np.random.normal(loc=128, scale=30, size=(100, 100)) for _ in range(4)]
    titles = [f"Img {i}" for i in range(4)]
    fig = viewer.display_histograms(imgs, titles)


def test_display_mip(viewer: Viewer):
    imgs = [np.random.rand(20, 20, 20) for _ in range(8)]
    titles = ["Méthode" for _ in range(8)]
    fig = viewer.display_mip(imgs, titles)


def test_display_volume(viewer: Viewer):
    vol = np.zeros((20, 20, 20))
    vol[5:15, 5:15, 5:15] = np.random.rand(10, 10, 10)
    fig = viewer.display_volume(vol, threshold=0.15)


def test_display_slices(viewer: Viewer):
    vols = [np.random.rand(16, 16, 16) for _ in range(8)]
    titles = ["Ttile" for _ in range(8)]
    anim = viewer.display_slices(vols, titles)

