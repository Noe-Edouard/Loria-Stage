import numpy as np
from typing import Optional, Tuple


def normalize_data(data: np.ndarray):
    data_min, data_max = np.min(data), np.max(data)
    if data_max > data_min:  # !div0
        data = (data - data_min) / (data_max - data_min)
    else:
        data = np.zeros_like(data)  # uniform image
    
    return data


def crop_data(data: np.ndarray, target_shape: Tuple[Optional[int], ...]) -> np.ndarray:
    ndim = data.ndim
    if len(target_shape) != ndim:
        raise ValueError(f"Croping failed. data and target_shape must have the same dimension. Current dimensions: {ndim} and {len(target_shape)}.")

    target_shape = [
        data.shape[i] if (target_shape[i] is None or target_shape[i] > data.shape[i]) else target_shape[i]
        for i in range(ndim)
    ]

    slices = tuple(
        slice((s - t) // 2, (s - t) // 2 + t)
        for s, t in zip(data.shape, target_shape)
    )

    return data[slices]
