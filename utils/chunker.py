import numpy as np


# ne pas faire des cubes mais une proportionnelle en fonction des dimensions de l'image 
# Ajouter un overlap

def chunk_volume(volume, chunk_size):
    x, y, z = volume.shape
    cx, cy, cz = chunk_size
    nx, ny, nz = x//cx, y//cy, z//cz
    
    chunks = np.empty((nx, ny, nz, cx, cy, cz), dtype=volume.dtype)
    
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                chunks[i, j, k] = volume[
                    i*cx:(i+1)*cx, 
                    j*cy:(j+1)*cy, 
                    k*cz:(k+1)*cz
                ]        
    return chunks

def unchunk_volume(chunks):
    nx, ny, nz, cx, cy, cz = chunks.shape
    volume = np.empty((nx*cx, ny*cy, nz*cz), dtype=chunks.dtype)

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                volume[
                    i*cx:(i+1)*cx,
                    j*cy:(j+1)*cy,
                    k*cz:(k+1)*cz
                ] = chunks[i, j, k]
    return volume

def pad_volume(volume, chunk_size, mode='reflect'):
    x, y, z = volume.shape
    cx, cy, cz = chunk_size

    pad_x = (cx - x % cx) % cx
    pad_y = (cy - y % cy) % cy
    pad_z = (cz - z % cz) % cz

    padding = (
        (0, pad_x),
        (0, pad_y),
        (0, pad_z)
    )
    padded_volume = np.pad(volume, padding, mode)
    return padded_volume, padding

def unpad_volume(volume, padding):
    (px0, px1), (py0, py1), (pz0, pz1) = padding
    x_end = -px1 if px1 > 0 else None
    y_end = -py1 if py1 > 0 else None
    z_end = -pz1 if pz1 > 0 else None
    return volume[px0:x_end, py0:y_end, pz0:z_end]

def crop_volume(image: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Crop the center of the image to match the target shape."""
    target_shape = [
        image.shape[i] if (target_shape[i] is None or target_shape[i] > image.shape[i]) else target_shape[i]
        for i in range(len(target_shape))
    ]
    slices = tuple(
        slice((s - t) // 2, (s - t) // 2 + t)
        for s, t in zip(image.shape, target_shape)
    )
    return image[slices]