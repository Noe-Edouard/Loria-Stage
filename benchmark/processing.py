import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from copy import deepcopy

from core.utils.helpers import compute_time
from core.utils.decorator import log_time, log_section, log_init
from core.logger import setup_logger
from core.loader import Loader
from core.saver import Saver
from core.config.setup import SetupConfig
from core.config.engine import PACConfig, SSIConfig, VSIConfig, CNIConfig, EngineConfig
from core.config.experiment import ExperimentConfig
from core.pipeline.enhancer import Enhancer
from core.pipeline.derivator import Derivator
from core.pipeline.segmenter import Segmenter
from core.viewer import Viewer 

logger = setup_logger(log_file='benchmark', debug_mode=True)

class Engine:
    
    @log_init()
    def __init__(self, setup: SetupConfig):
        self.setup = setup
        
        # Logger
        self.logger = setup_logger(log_file=self.setup.log_file, debug_mode=self.setup.debug_mode)
        
        # Loader
        self.loader = Loader(self.setup.input_dir, self.logger)
        
        # Derivator
        self.derivator = Derivator()
        
        # Enhancer
        self.enhancer = Enhancer()
        
        # Segmenter
        self.segmenter = Segmenter()

        # Analytics
        self.viewer = Viewer(display_mode=self.setup.display_mode)
        
        # Saver
        self.save_mode = self.setup.save_mode
        self.saver = Saver(experiment_name=self.setup.name, output_dir=self.output_dir, logger=self.logger)
        

    @log_time()
    @log_section("Scale size influence")
    def scale_size_influence(self, config: SSIConfig, experiment_config: ExperimentConfig):

        times_sequential = []
        times_parallel = []
        
        processing_params = deepcopy(experiment_config.processing)
        enhancement_params = deepcopy(experiment_config.enhancement)
        hessian_params = deepcopy(experiment_config.hessian)
        
        volume = np.ones((config.volume_size, config.volume_size, config.volume_size), np.float32) * 0.5
        size = config.volume_size
        chunk_size = (size // 2, size //2, size // 2)
        processing_params.chunk_size = chunk_size
        
        for n_scale in config.scales_numbers:
            scales = np.linspace(config.scales_range[0], config.scales_range[1], n_scale, dtype=int)
            enhancement_params.scales = scales
            
            # Sequential processing
            processing_params.parallelize = False
            times_sequential.append(compute_time(
                self.enhancer.enhance_data, 
                data=volume, 
                method=experiment_config.methods.enhancer,
                processing_params=processing_params,
                enhancement_params=enhancement_params,
                hessian_params=hessian_params,
            ))
            
            # Parallel processing
            processing_params.parallelize = True
            times_parallel.append(compute_time(
                self.enhancer.enhance_data, 
                data=volume, 
                method=experiment_config.methods.enhancer,
                processing_params=processing_params,
                enhancement_params=enhancement_params,
                hessian_params=hessian_params,
            ))
            
            
        # logger.info times
        logger.info('='*30+' RESUME '+'='*30+'\n')
        logger.info(f'Scales number:    {config.scales_numbers}')
        logger.info(f'Times sequential: {times_sequential}')
        logger.info(f'Times parallel:   {times_parallel}')
        
        headers = ['Scales (num)', 'Time sequential (s)', 'Time parallel (s)']
        rows = list(zip(config.scales_numbers, times_sequential, times_parallel))
        logger.info('\n' + tabulate(rows, headers, tablefmt='github', floatfmt='>.3f', intfmt='^'))
        logger.info('='*68+'\n')


        # Plot times
        fig = plt.figure(figsize=(7, 5))
        
        plt.plot(config.scales_numbers, times_sequential, '+-', label="Séquentiel", color='red')
        plt.plot(config.scales_numbers, times_parallel, '+-',  label="Parallèle", color='dodgerblue')
        plt.xlabel("Nombre d'échelles")
        plt.ylabel("Temps (secondes)")
        plt.title("Influence du nombre d'échelle sur le temps d'exécution")
        plt.legend()
        plt.grid(True)
        
        if self.setup.display_mode:
            plt.show()
        if self.setup.save_mode:
            self.saver.save_plot(fig, config.output_file)
            
        return config.scales_numbers, times_sequential, times_parallel


    @log_time()
    @log_section("Volume size influence")
    def volume_size_influence(self, config: VSIConfig, experiment_config: ExperimentConfig):
        
        times_sequential = []
        times_parallel = []
        
        volume_sizes = config.volume_sizes
        
        processing_params = deepcopy(experiment_config.processing)
        enhancement_params = deepcopy(experiment_config.enhancement)
        hessian_params = deepcopy(experiment_config.hessian)
        
        
        for size in config.volume_sizes:
            
            volume = np.ones((size, size, size), np.float32) * 0.5
            chunk_size = (size // 2, size // 2, size // 2)
            processing_params.chunk_size = chunk_size
            
            # Sequential processing
            processing_params.parallelize = False
            times_sequential.append(compute_time(
                self.enhancer.enhance_data, 
                data=volume, 
                method=experiment_config.methods.enhancer,
                processing_params=processing_params,
                enhancement_params=enhancement_params,
                hessian_params=hessian_params,
            ))
            
            # Parallel processing
            processing_params.parallelize = True
            times_parallel.append(compute_time(
                self.enhancer.enhance_data, 
                data=volume, 
                method=experiment_config.methods.enhancer,
                processing_params=processing_params,
                enhancement_params=enhancement_params,
                hessian_params=hessian_params,
            ))
        
        # logger.info times
        logger.info('='*30+' RESUME '+'='*30+'\n')
        logger.info(f'Volume sizes:     {volume_sizes}')
        logger.info(f'Times sequential: {times_sequential}')
        logger.info(f'Times parallel:   {times_parallel}')
        
        headers = ['Volume size (px)', 'Time sequential (s)', 'Time parallel (s)']
        rows = list(zip(volume_sizes, times_sequential, times_parallel))
        logger.info('\n' + tabulate(rows, headers=headers, tablefmt='github', floatfmt='>.3f', intfmt='^'))
        print('='*68+'\n')
        
        # Linear regression (log scale)
        a, b = np.polyfit(np.log(volume_sizes), np.log(times_sequential), 1)
        c, d = np.polyfit(np.log(volume_sizes), np.log(times_parallel), 1)
        
        # Plot times
        fig = plt.figure(figsize=(16, 4))
        
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

        if self.setup.display_mode:
            plt.show()
        if self.setup.save_mode:
            self.saver.save_plot(fig, config.output_file)
                    
        return volume_sizes, times_sequential, times_parallel


    @log_time()
    @log_section("Chunk number influence")
    def chunk_number_influence(self, config: CNIConfig, experiment_config: ExperimentConfig):
        all_times = []
        all_chunk_sizes = []
        
        volume_sizes = config.volume_sizes
        chunk_numbers = config.chunk_numbers
        
        enhancement_params = deepcopy(experiment_config.enhancement)
        hessian_params = deepcopy(experiment_config.hessian)
        processing_params = deepcopy(experiment_config.processing)
        processing_params.parallelize = True
        
        
        for volume_size in volume_sizes:
            volume = np.ones((volume_size, volume_size, volume_size), np.float32) * 0.5
            times = []
            chunk_sizes = []
            for chunk_number in chunk_numbers:
                
                chunk_size = (int(volume_size//chunk_number), int(volume_size//chunk_number), int(volume_size//chunk_number))
                chunk_sizes.append(chunk_size[0])
                processing_params.chunk_size = chunk_size
                
                # Parallel processing
                times.append(compute_time(
                    self.enhancer.enhance_data, 
                    data=volume, 
                    method=experiment_config.methods.enhancer,
                    processing_params=processing_params,
                    enhancement_params=enhancement_params,
                    hessian_params=hessian_params,
                ))
                
            all_times.append(times)
            all_chunk_sizes.append(chunk_sizes)
        
            # logger.info times
            logger.info('='*30+' RESUME '+'='*30+'\n')
            logger.info(f'Volume sizes:     {volume_size}')
            logger.info(f'Chunk number:     {chunk_number}')
            logger.info(f'Chunk sises:      {chunk_sizes}')
            logger.info(f'Times (parallel): {times}')
            
            headers = ['Chunk number', 'Chunk size (vx)', 'Time parallel (s)']
            rows = list(zip(chunk_numbers, chunk_sizes, times))
            logger.info('\n' + tabulate(rows, headers=headers, tablefmt='github', floatfmt='>.3f', intfmt='^'))
            logger.info('='*68+'\n')

                    
        # Plot times
        fig = plt.figure(figsize=(7, 5))
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
        plt.title(f"Influence du nombre de chunk sur le temps d'éxécution")
        plt.xlabel("Nombre de chunk")
        plt.ylabel("Temps d'exécution (s)")
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        for i, volume_size in enumerate(volume_sizes):
            plt.plot(chunk_numbers, all_chunk_sizes[i], '-+', color=colors[i], label=f"volume size: {volume_size}")
        plt.title(f"Taille des chunk en fonction de leur nombre par dimension")
        plt.xlabel("Nombre de chunk")
        plt.ylabel("Taille des chunks")
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()

        if self.setup.display_mode:
            plt.show()
        if self.setup.save_mode:
            self.saver.save_plot(fig, config.output_file)
        
        return volume_sizes, chunk_numbers, all_chunk_sizes, all_times


    @log_time()
    @log_section("Parallelization accuracy influence")
    def parallel_accuracy_influence(self, config: PACConfig, experiment_config: ExperimentConfig):
        
        volume = self.loader.load_data(config.input_file)   
             
        processing_params = deepcopy(experiment_config.processing)
        enhancement_params = deepcopy(experiment_config.enhancement)
        hessian_params = deepcopy(experiment_config.hessian)
        
        # Sequential processing
        processing_params.parallelize = False
        
        sequential = self.enhancer.enhance_data( 
            data=volume, 
            method=experiment_config.methods.enhancer,
            processing_params=processing_params,
            enhancement_params=enhancement_params,
            hessian_params=hessian_params,
        )
        
        # Parallel processing
        processing_params.parallelize = True
        parallel = self.enhancer.enhance_data( 
            data=volume, 
            method=experiment_config.methods.enhancer,
            processing_params=processing_params,
            enhancement_params=enhancement_params,
            hessian_params=hessian_params,
        )
            
        fig = self.viewer.display_images([sequential, parallel, sequential-parallel], ['Sequantial', 'Parallel', 'Difference'])
        self.saver.save_plot(fig, config.output_file)
        plt.close()

        mae = np.abs(sequential - parallel).max()
        logger.info(f"MAE (sequential vs parallel): {mae:.4e}")

        
        return sequential, parallel
    
    
    

      
    
    
    @log_time()
    @log_section("Engine run")
    def run(self, config: EngineConfig, experiment_config: ExperimentConfig):
        hessian_function = self.derivator.select_hessian_function(experiment_config.methods.derivator)
        experiment_config.enhancement.hessian_function = hessian_function
        
        self.chunk_number_influence(config.cni, experiment_config)
        self.volume_size_influence(config.vsi, experiment_config)
        self.scale_size_influence(config.ssi, experiment_config)
        self.parallel_accuracy_influence(config.pac, experiment_config)
        self.scale_accuracy_check(config.sac, experiment_config)

        