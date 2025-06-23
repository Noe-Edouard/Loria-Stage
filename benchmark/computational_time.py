import numpy as np
import matplotlib.pyplot as plt
from src.frangi import frangi_filter
from time import time
from pipeline import process_volume

def scale_size_influence():
    scale_steps = [1, 2, 3, 4, 5] 
    scale_sizes = []
    for step in scale_steps:
        
        times_sequential = []
        times_parallel = []
        scales = np.arange(0, 20, step)
        scale_sizes.append(len(scales))
        
        volume_size = 200
        volume = np.ones((volume_size, volume_size, volume_size), np.float32) * 0.5
        chunk_size = (int(volume_size * 0.35), int(volume_size * 0.35), int(volume_size * 0.35))
        
        # Sequential processing
        t1 = time()
        frangi_filter(volume, scales)
        t2 = time()
        
        # Parallel processing
        t3 = time()
        process_volume(volume, chunk_size, scales)
        t4 = time()
        
        # Save times
        times_sequential.append(t2 - t1)   
        times_parallel.append(t4 - t3)   
    
    # Print times
    print('Steps:', scale_steps)
    print('Sizes:', scale_sizes)
    print('Times sequential:', times_sequential)
    print('Times parallel:', times_parallel)
    
    print("\nComparaison temps (taille | séquentiel | parallèle)")
    for ss, ts, tp in zip(scale_sizes, times_sequential, times_parallel):
        print(f"{ss:6} | {ts:.3f}s     | {tp:.3f}s")
    

    # Plot times
    plt.figure(figsize=(7, 5))
    
    plt.plot(scale_sizes, times_sequential, '+-', label="Séquentiel", color='red')
    plt.plot(scale_sizes, times_parallel, '+-',  label="Parallèle", color='dodgerblue')
    plt.xlabel("Nombre d'échelles")
    plt.ylabel("Temps (secondes)")
    plt.title("Influence du nombre d'échelle sur le temps d'exécution")
    plt.legend()
    plt.grid(True)
    

    plt.savefig("benchmark_scale_size.png", dpi=300, bbox_inches='tight')
    plt.close()

    
    return scale_sizes, times_sequential, times_parallel


def volume_size_influence():
    times_sequential = []
    times_parallel = []
    scales = np.arange(0, 20, 2)
    volume_sizes = [i**3 for i in range(3, 8)]  # 27, 64, 125, 216, 343

    for size in volume_sizes:
        
        volume = np.ones((size, size, size), np.float32) * 0.5
        chunk_size = (int(size * 0.35), int(size * 0.35), int(size * 0.35))
        
        # Sequential processing
        t1 = time()
        frangi_filter(volume, scales)
        t2 = time()
        
        # Parallel processing
        t3 = time()
        process_volume(volume, chunk_size, scales)
        t4 = time()
        
        # Save times
        times_sequential.append(t2 - t1)   
        times_parallel.append(t4 - t3)   
    
    # Print times
    print('Sizes:', volume_sizes)
    print('Times sequential:', times_sequential)
    print('Times parallel:', times_parallel)
    
    print("\nComparaison temps (taille | séquentiel | parallèle)")
    for sz, ts, tp in zip(volume_sizes, times_sequential, times_parallel):
        print(f"{sz:6} | {ts:.3f}s     | {tp:.3f}s")
    
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
    plt.title("Influence de la taille du volume sur le temps d'exécution")
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

    plt.savefig("benchmark_volume_size.png", dpi=300, bbox_inches='tight')
    plt.close()

    
    return volume_sizes, times_sequential, times_parallel


import numpy as np
import matplotlib.pyplot as plt
from time import time

def chunk_size_influence():
    volume_sizes = [100, 200, 300, 400, 500]
    
    chunk_ratios = [15, 20, 25, 30, 35, 40, 45]
    all_times = []
    for volume_size in volume_sizes:
        volume = np.ones((volume_size, volume_size, volume_size), dtype=np.float32) * 0.5
        scales = np.arange(1, 10, 2)
        chunk_sizes = [(s * volume_size // 100,) * 3 for s in chunk_ratios]
        times = []

        for chunk_size in chunk_sizes:
            t1 = time()
            process_volume(volume, chunk_size, scales)
            t2 = time()
            times.append(t2 - t1)
        
        # Print times
        print(f'\n### Volume {volume_size} ###')
        for vs, cs, cr, t in zip(volume_sizes, chunk_sizes, chunk_ratios, times):
            print(f'volume size: {vs} | chunk size: {cs} | ratio %: {cr:2d} | time: {t:.2f}s')

        all_times.append(times)
        
    # Plot times
    plt.figure(figsize=(7, 5))
    colors = [
        "#022c7aff",
        "#0743b1ff",
        "#175ddfff",
        "#407ff5ff",
        "#6097fcff",
    ]
    
    for i, volume_size in enumerate(volume_sizes):
        plt.plot(chunk_ratios, all_times[i], '-+', color=colors[i], label=f"volume size: {volume_size}")
    plt.title(f"Influence de la taille des chunk sur le temps d'éxécution")
    plt.xlabel("Taille des chunks (% du volume)")
    plt.ylabel("Temps d'exécution (s)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig("benchmark_chunk_size.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return chunk_size, chunk_ratios, times