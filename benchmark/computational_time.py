import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter
from src.enhancer import Enhancer
from tabulate import tabulate
from pathlib import Path
from typing import Sequence
from utils.logger import setup_logger

logger = setup_logger(name='benchmark', debug_mode=True)

def compute_time(function, *args, **kargs):
    start = perf_counter()
    function(*args, **kargs)
    end = perf_counter()
    return end - start


def scale_size_influence(
    volume_size:int = 256, 
    scales_range: tuple[int, int] = (1, 10),
    scales_numbers: Sequence[int] = [2, 4, 6, 8, 10], 
    enhancement_method: str = 'frangi', 
    chunk_number: int = 8, 
    output_dir='outputs/benchmark', 
    output_file='benchmark_scale_size.png'
):
    
    enhancer = Enhancer(enhancement_method)
    times_sequential = []
    times_parallel = []
    volume = np.ones((volume_size, volume_size, volume_size), np.float32) * 0.5
    chunk_size = (int(volume_size // chunk_number), int(volume_size // chunk_number), int(volume_size // chunk_number))
    
    for n in scales_numbers:
        scales = np.linspace(scales_range[0], scales_range[1], n, dtype=int)
        
        # Sequential processing
        times_sequential.append(compute_time(
            enhancer.apply_enhancement, 
            data=volume, 
            parallelize=False,
            chunk_size=chunk_size,
            enhancement_params = {'scales': scales}, 
        ))
        
        # Parallel processing
        times_parallel.append(compute_time(
            enhancer.apply_enhancement, 
            data=volume, 
            parallelize=True,
            chunk_size=chunk_size,
            enhancement_params = {'scales': scales}, 
        ))
           
           
    # logger.info times
    logger.info('='*40+' RESUME '+'='*40+'\n')
    logger.info("Influence du nombre d'échelle sur le temps d'exécution du réhaussement\n")
    
    logger.info(f'Scales number:    {scales_numbers}')
    logger.info(f'Times sequential: {times_sequential}')
    logger.info(f'Times parallel:   {times_parallel}')
    
    headers = ['Scales (num)', 'Time sequential (s)', 'Time parallel (s)']
    rows = list(zip(scales_numbers, times_sequential, times_parallel))
    logger.info('\n' + tabulate(rows, headers, tablefmt='github', floatfmt='>.3f', intfmt='^'))
    logger.info('='*88+'\n')


    # Plot times
    plt.figure(figsize=(7, 5))
    
    plt.plot(scales_numbers, times_sequential, '+-', label="Séquentiel", color='red')
    plt.plot(scales_numbers, times_parallel, '+-',  label="Parallèle", color='dodgerblue')
    plt.xlabel("Nombre d'échelles")
    plt.ylabel("Temps (secondes)")
    plt.title("Influence du nombre d'échelle sur le temps d'exécution du réhaussement")
    plt.legend()
    plt.grid(True)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(f"{output_dir}/{output_file}", dpi=300, bbox_inches='tight')
    plt.close()


    return scales_numbers, times_sequential, times_parallel



def volume_size_influence(
    volume_sizes: Sequence[int] = [i**3 for i in range(3, 8)], 
    scales_number: int = 5, 
    scales_range: tuple[int, int] = (1, 10), 
    enhancement_method: str = 'frangi',
    chunk_number: int = 8, 
    output_dir='outputs/benchmark', 
    output_file='benchmark_volume_size.png'
):

    enhancer = Enhancer(enhancement_method)
    scales = np.linspace(scales_range[0], scales_range[1], scales_number, dtype=int)
    
    times_sequential = []
    times_parallel = []

    for size in volume_sizes:
        
        volume = np.ones((size, size, size), np.float32) * 0.5
        chunk_size = (int(size//chunk_number), int(size//chunk_number), int(size//chunk_number))
        
        # Sequential processing
        times_sequential.append(compute_time(
            enhancer.apply_enhancement, 
            data=volume, 
            parallelize=False,
            chunk_size=chunk_size,
            enhancement_params = {'scales': scales}, 
        ))
        
        # Parallel processing
        times_parallel.append(compute_time(
            enhancer.apply_enhancement, 
            data=volume, 
            parallelize=True,
            chunk_size=chunk_size,
            enhancement_params = {'scales': scales}, 
        ))
    
    
    # logger.info times
    logger.info('='*40+' RESUME '+'='*40+'\n')
    logger.info("Influence de la taille du volume sur le temps d'exécution du réhaussement\n")
   
    logger.info(f'Volume sizes:     {volume_sizes}')
    logger.info(f'Times sequential: {times_sequential}')
    logger.info(f'Times parallel:   {times_parallel}')
    
    headers = ['Scales (num)', 'Time sequential (s)', 'Time parallel (s)']
    rows = list(zip(volume_sizes, times_sequential, times_parallel))
    logger.info('\n' + tabulate(rows, headers=headers, tablefmt='github', floatfmt='>.3f', intfmt='^'))
    print('='*88+'\n')
    
    # Linear regression (log scale)
    a, b = np.polyfit(np.log(volume_sizes), np.log(times_sequential), 1)
    c, d = np.polyfit(np.log(volume_sizes), np.log(times_parallel), 1)
    
    # Plot times
    plt.figure(figsize=(16, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(volume_sizes, times_sequential, '+-', label="Séquentiel", color='red')
    plt.plot(volume_sizes, times_parallel, '+-',  label="Parallèle", color='dodgerblue')
    plt.xlabel("Taille du volume (voxels)")
    plt.ylabel("Temps (secondes)")
    plt.title("Influence de la taille du volume sur le temps d'exécution du réhaussement")
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(np.log(volume_sizes), np.log(times_sequential), '+', label='real', color='red')
    plt.plot(np.log(volume_sizes), a * np.log(volume_sizes) + b, '--', label=f'fit: y = {a:.2f} x + {b:.2f}', color='dodgerblue')
    plt.xlabel('log(Taille du volume)')
    plt.ylabel('log(Temps d\'exécution)')
    plt.title('Traitement séquentiel')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(np.log(volume_sizes), np.log(times_parallel), '+', label='real', color='red')
    plt.plot(np.log(volume_sizes), c * np.log(volume_sizes) + d, '--', label=f'fit: y = {a:.2f} x + {b:.2f}', color='dodgerblue')
    plt.xlabel('log(Taille du volume)')
    plt.ylabel('log(Temps d\'exécution)')
    plt.title('Traitement parallèle')
    plt.legend()
    plt.grid(True)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(f"{output_dir}/{output_file}", dpi=300, bbox_inches='tight')
    plt.close()

    
    return volume_sizes, times_sequential, times_parallel



from tqdm import tqdm
import json

def chunk_number_influence(
    volume_sizes: Sequence[int] = [i**3 for i in range(3, 8)], 
    scales_number: int = 5, 
    scales_range: tuple[int, int] = (1, 10), 
    enhancement_method: str = 'frangi',
    chunk_numbers: Sequence[int] = [2, 4, 8, 16], 
    output_dir='outputs/benchmark', 
    output_file='benchmark_chunk_number.png'
):
    
    all_times = []
    all_chunk_sizes = []
    
    enhancer = Enhancer(enhancement_method)
    scales = np.linspace(scales_range[0], scales_range[1], scales_number, dtype=int)

    for volume_size in volume_sizes:
        volume = np.ones((volume_size, volume_size, volume_size), np.float32) * 0.5
        times = []
        chunk_sizes = []
        for chunk_number in chunk_numbers:
            chunk_size = (int(volume_size//chunk_number), int(volume_size//chunk_number), int(volume_size//chunk_number))
            chunk_sizes.append(chunk_size[0])
            # Parallel processing
            times.append(compute_time(
                enhancer.apply_enhancement, 
                data=volume, 
                parallelize=True,
                chunk_size=chunk_size,
                enhancement_params = {'scales': scales}, 
            ))
            
        all_times.append(times)
        all_chunk_sizes.append(chunk_sizes)
    
    
        # logger.info times
        logger.info('='*40+' RESUME '+'='*40+'\n')
        logger.info("Influence du nombre de chunk sur le temps d'exécution du réhaussement\n")
    
        logger.info(f'Volume sizes:     {volume_size}')
        logger.info(f'Chunk number:     {chunk_number}')
        logger.info(f'Chunk sises:      {chunk_sizes}')
        logger.info(f'Times (parallel): {times}')
        
        headers = ['Chunk number', 'Chunk size (vx)', 'Time parallel (s)']
        rows = list(zip(chunk_numbers, chunk_sizes, times))
        logger.info('\n' + tabulate(rows, headers=headers, tablefmt='github', floatfmt='>.3f', intfmt='^'))
        logger.info('='*88+'\n')

                
    # Plot times
    plt.figure(figsize=(7, 5))
    colors = [
        "#022c7aff",
        "#0743b1ff",
        "#175ddfff",
        "#407ff5ff",
        "#6097fcff",
    ]
    
    plt.subplot(1, 2, 1)
    for i, volume_size in enumerate(volume_sizes):
        plt.plot(chunk_numbers, all_times[i], '-+', color=colors[i], label=f"volume size: {volume_size}")
    plt.title(f"Influence du nombre de chunk sur le temps d'éxécution du réhaussement")
    plt.xlabel("Nombre de chunk")
    plt.ylabel("Temps d'exécution (s)")
    
    plt.subplot(1, 2, 2)
    for i, volume_size in enumerate(volume_sizes):
        plt.plot(chunk_numbers, all_chunk_sizes[i], '-+', color=colors[i], label=f"volume size: {volume_size}")
    plt.title(f"Influence du nombre de chunk sur le temps d'éxécution du réhaussement")
    plt.xlabel("Nombre de chunk")
    plt.ylabel("Taille des chunks")
    
    
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(f"{output_dir}/{output_file}", dpi=300, bbox_inches='tight')
    plt.close()
    
    return volume_sizes, chunk_numbers, all_chunk_sizes, all_times
