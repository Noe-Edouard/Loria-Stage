from benchmark.engine import Engine
from benchmark.benchmark import Benchmark
from core.config import EngineConfig, BenchmarkConfig, ConfigBuilder, MainConfig, SetupConfig, ExperimentConfig


class Runner:
    
    def __init__(self, benchmark_setup_file: str = 'configs/benchmark.yaml', engine_setup_file: str = 'configs/engine.yaml'):
        self.benchmark_setup_file = benchmark_setup_file
        self.engine_setup_file = engine_setup_file
    
    
    def run(self, run_benchmark: bool = True, run_engine: bool = True):
        
        if run_engine:
            config: MainConfig = ConfigBuilder('configs/engine.yaml', MainConfig)
            
            setup_config: SetupConfig = config.setup
            experiment_config: ExperimentConfig = config.experiment
            engine_config: EngineConfig = config.runner
            
            engine = Engine(setup_config)
            engine.run(engine_config, experiment_config)
        
        
        if run_benchmark: 
            
            config: MainConfig = ConfigBuilder('./configs/benchmark.yaml', MainConfig)
            
            setup_config: SetupConfig = config.setup
            experiment_config: ExperimentConfig = config.experiment
            benchmark_config: BenchmarkConfig = config.runner
            
            benchmark = Benchmark(setup_config)
            benchmark.run(benchmark_config, experiment_config)