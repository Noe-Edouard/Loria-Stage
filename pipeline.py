import numpy as np
from utils.chunker import chunk_volume, pad_volume, unchunk_volume, unpad_volume
from concurrent.futures import ProcessPoolExecutor
from src.frangi import frangi_filter
from tqdm import tqdm

def apply_enhancement(volume, params, method: str = 'frangi'):
    if method == 'frangi':
        scales = params.get('scales', np.arange(1, 10, 2))
        alpha = params.get('alpha', 0.5)
        beta = params.get('beta', 0.5)
        gamma = params.get('scales', None)
        mode = params.get('mode', 'reflect')
        cval = params.get('cval', 0)
        return frangi_filter(volume, scales, alpha, beta, gamma, mode, cval)
    else:
        raise ValueError(f'Method {method} not found. Only following methods are supported: "frangi"')
    
def pool_function(args):
    chunk, params = args
    return apply_enhancement(chunk, params, 'frangi')

def process_volume(volume, chunk_size, params):
    
    # Pad volume
    padded_volume, padding = pad_volume(volume, chunk_size)
    
    # Chunk volume 
    chunks = chunk_volume(padded_volume, chunk_size)
    flat_chunks = chunks.reshape(-1, *chunk_size)
    
    args = [(chunk, params) for chunk in flat_chunks]
    with ProcessPoolExecutor() as executor:
        processed_chunks = list(tqdm(executor.map(pool_function, args), total = len(args)))
    
    # Reshape chunks
    processed_chunks = np.stack(processed_chunks)
    processed_chunks = processed_chunks.reshape(chunks.shape)
    
    # Unchunk volume
    processed_volume = unchunk_volume(processed_chunks)
    
    # Unpad volume
    processed_volume = unpad_volume(processed_volume, padding)
    
    return processed_volume