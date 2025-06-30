import pytest
import numpy as np
from pathlib import Path
from utils.loader import Loader

import imageio.v3 as iio
import nibabel as nib
import tifffile as tiff
import SimpleITK as sitk

def create_jpg(path):
    data = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
    iio.imwrite(path, data)

def create_tiff(path):
    data = (np.random.rand(64, 64) * 255).astype(np.uint8)
    tiff.imwrite(path, data)

def create_nii(path):
    data = (np.random.rand(64, 64, 10)).astype(np.float32)
    nib.save(nib.Nifti1Image(data, affine=np.eye(4)), path)

def create_mhd(path):
    data = (np.random.rand(10, 64, 64) * 255).astype(np.uint8)
    img = sitk.GetImageFromArray(data)
    sitk.WriteImage(img, str(path))

@pytest.fixture
def loader(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    return Loader(input_dir=input_dir)

@pytest.mark.parametrize("ext,creator", [
    ("jpg", create_jpg),
    ("tiff", create_tiff),
    ("nii", create_nii),
    ("mhd", create_mhd),
])
def test_load(loader, tmp_path, ext, creator):
    filename = f"sample.{ext}"
    file_path = tmp_path / "input" / filename
    creator(file_path)

    # Test loading
    data = loader.load_data(filename)
    assert isinstance(data, np.ndarray)

    # Test metadata extraction
    metadata = loader.get_metadata(filename)
    assert isinstance(metadata, dict)
    assert len(metadata) > 0

@pytest.mark.parametrize("creator", [create_nii])
def test_save(loader, tmp_path, creator):
    filename = "test.nii"
    file_path = tmp_path / "input" / filename
    creator(file_path)

    data = loader.load_data(filename)
    loader.save_data(data, "saved_test.nii")

    assert (tmp_path / "output" / "saved_test.nii").exists()
