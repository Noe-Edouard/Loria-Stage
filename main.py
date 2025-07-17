from pathlib import Path

from benchmark.engine import Engine
from benchmark.benchmark import Benchmark
from core.config import EngineConfig, BenchmarkConfig, ConfigBuilder, MainConfig, SetupConfig, ExperimentConfig
from pipeline.pipeline import Pipeline



RUN_PIPELINE: bool = False
RUN_TEST: bool = False
RUN_ENGINE: bool = True 
RUN_BENCHMARK: bool = False

SRC_PIPELINE = 'configs/pipeline.yaml'
SRC_TEST = 'configs/test.yaml'
SRC_ENGINE = 'configs/engine.yaml'
SRC_BENCHMARK = 'configs/benchmark.yaml'

FULL_BENCHMARK: bool = True


def main():
    
    if RUN_PIPELINE:
        config: MainConfig = ConfigBuilder(SRC_PIPELINE, MainConfig)
        
        setup_config: SetupConfig = config.setup
        experiment_config: ExperimentConfig = config.experiment
        
        pipeline = Pipeline(setup_config)
        pipeline.run(experiment_config)
    
    if RUN_TEST:
        config: MainConfig = ConfigBuilder(SRC_TEST, MainConfig)
        
        setup_config: SetupConfig = config.setup
        experiment_config: ExperimentConfig = config.experiment
        benchmark_config: BenchmarkConfig = config.runner
        
        benchmark = Benchmark(setup_config)
        benchmark.run(benchmark_config, experiment_config)
    
    
    if RUN_ENGINE:
        
        config: MainConfig = ConfigBuilder(SRC_ENGINE, MainConfig)
        setup_config: SetupConfig = config.setup
        experiment_config: ExperimentConfig = config.experiment
        engine_config: EngineConfig = ConfigBuilder(config.runner, EngineConfig)

        engine = Engine(setup_config)
        engine.run(engine_config, experiment_config)
    
    
    if RUN_BENCHMARK: 
        
        config: MainConfig = ConfigBuilder(SRC_BENCHMARK, MainConfig)
        
        setup_config: SetupConfig = config.setup
        experiment_config: ExperimentConfig = config.experiment
        benchmark_config: BenchmarkConfig = ConfigBuilder(config.runner, BenchmarkConfig)
        
        if FULL_BENCHMARK:
            data_dir = Path(setup_config.input_dir)
            for file in data_dir.iterdir():
                experiment_config.load.input_file = file.name
                benchmark = Benchmark(setup_config)
                benchmark.run(benchmark_config, experiment_config)
            
        else:
            benchmark = Benchmark(setup_config)
            benchmark.run(benchmark_config, experiment_config)
            
            
if __name__ == "__main__":
    main()
    
    
# Gérer le retour des différentes fonctions (notament engine qui ne retourne rien)