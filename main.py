from pathlib import Path

from benchmark.analyzer_processing import Engine
from benchmark.benchmark import BenchmarkDerivator
from core.config import EngineConfig, BenchmarkConfig, BenchmarkData, ConfigBuilder, MainConfig, SetupConfig, ExperimentConfig, AnalyzerConfig
from pipeline.pipeline import Pipeline
from benchmark.analyzer_enhancement import EnhancementRunner 



RUN_PIPELINE: bool = False
RUN_ENGINE: bool = False
RUN_OPTIMIZER: bool = True
RUN_BENCHMARK: bool = False


SRC_PIPELINE = 'configs/pipeline.yaml'
SRC_ENGINE = 'configs/engine.yaml'
SRC_OPTIMIZER = 'configs/analyzer.yaml'
SRC_BENCHMARK = 'configs/benchmark.yaml'
SRC_TEST = 'configs/test.yaml'

TEST: bool = False
FULL_BENCHMARK: bool = False


def main():
    

    if RUN_PIPELINE:
        config_file = SRC_TEST if TEST else SRC_PIPELINE
        config: MainConfig = ConfigBuilder(config_file, MainConfig)
        setup_config: SetupConfig = config.setup
        experiment_config: ExperimentConfig = config.experiment
        
        pipeline = Pipeline(setup_config)
        pipeline.run(experiment_config)

    if RUN_ENGINE:
        config_file = SRC_TEST if TEST else SRC_ENGINE
        config: MainConfig = ConfigBuilder(config_file, MainConfig)
        setup_config: SetupConfig = config.setup
        experiment_config: ExperimentConfig = config.experiment
        engine_config: EngineConfig = ConfigBuilder(config.runner, EngineConfig)
        
        engine = Engine(setup_config)
        engine.run(engine_config, experiment_config)
        
            
    if RUN_OPTIMIZER:
        config_file = SRC_TEST if TEST else SRC_OPTIMIZER
        config: MainConfig = ConfigBuilder(config_file, MainConfig)
        setup_config: SetupConfig = config.setup
        experiment_config: ExperimentConfig = config.experiment
        analyzer_config: AnalyzerConfig = ConfigBuilder(config.runner, AnalyzerConfig)

        analyzer = EnhancementRunner(setup_config)
        analyzer.run(analyzer_config, experiment_config)
        

    if RUN_BENCHMARK:
        config_file = SRC_TEST if TEST else SRC_BENCHMARK
        config: MainConfig = ConfigBuilder(config_file, MainConfig)
        setup_config: SetupConfig = config.setup
        experiment_config: ExperimentConfig = config.experiment
        benchmark_config: BenchmarkConfig = ConfigBuilder(config.runner, BenchmarkConfig)
        
        benchmark = BenchmarkDerivator(setup_config)
        
        if FULL_BENCHMARK:
            benchmark.run_all(benchmark_config, experiment_config)
        elif benchmark_config:
            benchmark.run(benchmark_config, experiment_config)
                
            
if __name__ == "__main__":
    main()
    
    