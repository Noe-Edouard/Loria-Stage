import numpy as np
from pathlib import Path
from skimage import io, color
import tifffile as tiff
import nibabel as nib
import SimpleITK as sitk
import os

def crop_center(image: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Crop the center of the image to match the target shape."""
    slices = tuple(
        slice((s - t) // 2, (s - t) // 2 + t)
        for s, t in zip(image.shape, target_shape)
    )
    return image[slices]

def load(filename: str, crop_to: tuple = None) -> np.ndarray:
    path = Path(filename)
    suffixes = path.suffixes
    print(f'Loading file {path.stem} ...')

    # .JPG, .PNG
    if suffixes[-1] == '.jpg':
        image = io.imread(filename).astype(np.float32)
        image = color.rgb2gray(image)

    # .TIF, .TIFF
    elif suffixes[-1] in ['.tif', '.tiff']:
        image = tiff.imread(filename).astype(np.float32)
        if image.ndim == 3:  # 3D
            image = np.transpose(image, (2, 1, 0))  # (z, y, x) -> (x, y, z)
        else:  # 2D
            image = np.transpose(image, (1, 0))  # (y, x) -> (x, y)
            
    # .NII, .NII.GZ
    elif suffixes[-2:] == ['.nii', '.gz'] or suffixes[-1] == '.nii':
        image = np.squeeze(nib.load(filename).get_fdata().astype(np.float32))

    # .MHD, .RAW
    elif suffixes[-1] == '.mhd':
        sitk_image = sitk.ReadImage(str(filename))
        image = sitk.GetArrayFromImage(sitk_image).astype(np.float32)  # shape: (z, y, x)
        image = np.transpose(image, (2, 1, 0))  # -> (x, y, z)

    else:
        raise ValueError('Unsupported file type. Use .jpg, .tif, .tiff, .nii(.gz), or .mhd')

    # Crop image
    if crop_to is not None:
        if len(image.shape) < len(crop_to):
            raise ValueError("Target crop shape has more dimensions than the image.")
        print(f'Cropping to center with shape {crop_to} ...')
        image = crop_center(image, crop_to)

    print(f'File {path.stem} loaded ! Volume shape: {image.shape}.')
    return image


def save(image: np.ndarray, filename: str):
    basename, extension = os.path.splitext(filename)
    print(f'Saving file {basename} ...')

    if extension == '.nii':
        image_nii = nib.Nifti1Image(image, affine=np.eye(4))
        nib.save(image_nii, filename)
    else:
        raise ValueError('Saving only supported for .nii format for now')

    print(f'File {basename} saved!')
    

from pathlib import Path
from skimage import io
import tifffile as tiff
import nibabel as nib

def get_metadata(filename):
    path = Path(filename)
    suffixes = path.suffixes
    print(f'Extracting metadata from {path.name} ...')

    # .JPG, .PNG
    if suffixes[-1] in ['.jpg', '.png']:
        name = path.stem
        extension = suffixes[-1]
        shape = io.imread(filename).shape
        metadata = {
            'name': name,
            'extension': extension,
            'shape': shape,
        }

    # .TIF, .TIFF
    elif suffixes[-1] in ['.tif', '.tiff']:
        with tiff.TiffFile(filename) as tif:
            metadata = {}
            for tag in tif.pages[0].tags.values():
                metadata[tag.name] = tag.value

    # .NII, .NII.GZ
    elif suffixes[-1] == '.nii' or (suffixes[-2:] == ['.nii', '.gz']):
        img = nib.load(filename)
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
    
        image = sitk.ReadImage(filename)
        size = image.GetSize()  # (x, y, z)
        spacing = image.GetSpacing()
        origin = image.GetOrigin()
        direction = image.GetDirection()
        pixel_type = image.GetPixelIDTypeAsString()

        metadata = {
            'size': size,
            'spacing': spacing,
            'origin': origin,
            'direction': direction,
            'pixel_type': pixel_type,
        }

    else:
        raise ValueError('Unsupported file type. Only the following types are supported: [.jpg, .tif, .tiff, .nii(.gz), or .mhd]')

    print(f'Metadata extracted from file {path.stem}')
    
    # Print metadata
    for key, value in metadata.items():
        print(key, value)
        
    return metadata

        
        

    