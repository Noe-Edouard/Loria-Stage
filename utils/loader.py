import os
import numpy as np
import tifffile as tiff
import nibabel as nib
import SimpleITK as sitk
from skimage import io, color
from pathlib import Path

from utils.logger import setup_logger, Logger
from utils.helpers import normalize_data, crop_data


class Loader:
    
    def __init__(self, input_dir: str = "data/raw", logger : Logger = setup_logger()) -> None:
        self.input_dir = input_dir
        self.logger = logger


    def load_data(self, filename: str, normalize: bool = True, crop: bool = False, target_shape: tuple = (64, 64, 64)) -> np.ndarray:
        path = path = Path(self.input_dir) / filename
        suffixes = path.suffixes

        # .JPG, .PNG
        if suffixes[-1] == '.jpg':
            data = io.imread(path).astype(np.float32)
            data = color.rgb2gray(data)

        # .TIF, .TIFF
        elif suffixes[-1] in ['.tif', '.tiff']:
            data = tiff.imread(path).astype(np.float32)
            if data.ndim == 3:  # 3D
                data = np.transpose(data, (2, 1, 0))  # (z, y, x) -> (x, y, z)
            else:  # 2D
                data = np.transpose(data, (1, 0))  # (y, x) -> (x, y)
                
        # .NII, .NII.GZ
        elif suffixes[-2:] == ['.nii', '.gz'] or suffixes[-1] == '.nii':
            data = np.squeeze(nib.load(path).get_fdata().astype(np.float32))

        # .MHD, .RAW
        elif suffixes[-1] == '.mhd':
            sitk_image = sitk.ReadImage(path)
            data = sitk.GetArrayFromImage(sitk_image).astype(np.float32)  # shape: (z, y, x)
            data = np.transpose(data, (2, 1, 0))  # -> (x, y, z)

        else:
            raise ValueError('Unsupported file type. Use .jpg, .tif, .tiff, .nii(.gz), or .mhd')

        # Crop data
        if crop:
            data = crop_data(data, target_shape)
        
        # Normalize data [0, 1]
        if normalize:
            data = normalize_data(data)

        self.logger.info(f'[LOAD] Data {filename} loaded with shape {data.shape} and normalized.')
        return data
        

    def get_metadata(self, filename: str) -> dict:
        path = Path(self.input_dir) / filename
        suffixes = path.suffixes

        # .JPG, .PNG
        if suffixes[-1] in ['.jpg', '.png']:
            name = path.stem
            extension = suffixes[-1]
            shape = io.imread(path).shape
            
            metadata = {
                'name': name,
                'extension': extension,
                'shape': shape,
            }

        # .TIF, .TIFF
        elif suffixes[-1] in ['.tif', '.tiff']:
            with tiff.TiffFile(path) as tif:
                metadata = {}
                for tag in tif.pages[0].tags.values():
                    metadata[tag.name] = tag.value

        # .NII, .NII.GZ
        elif suffixes[-1] == '.nii' or (suffixes[-2:] == ['.nii', '.gz']):
            img = nib.load(path)
            header = img.header
            affine = img.affine
            shape = img.shape
            
            metadata = {
                'shape': shape,
                'datatype': str(header.get_data_dtype()),
                'voxel_size': header.get_zooms(),
                'affine': affine.tolist(), # list pou r la lisibilité
            }
            
        # .MHD, .RAW
        elif suffixes[-1] == '.mhd':
            data = sitk.ReadImage(path)
            size = data.GetSize()  # (x, y, z)
            spacing = data.GetSpacing()
            origin = data.GetOrigin()
            direction = data.GetDirection()
            pixel_type = data.GetPixelIDTypeAsString()

            metadata = {
                'size': size,
                'spacing': spacing,
                'origin': origin,
                'direction': direction,
                'pixel_type': pixel_type,
            }

        else:
            raise ValueError('Unsupported file type. Only the following types are supported: [.jpg, .tif, .tiff, .nii(.gz), or .mhd]')

        self.logger.info(f'[METADATA] "{filename}":' + "\n    ".join([f"  - {k}: {v}" for k, v in metadata.items()]))            
        
        return metadata
