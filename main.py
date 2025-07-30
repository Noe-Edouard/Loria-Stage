from pathlib import Path
from typing import Literal

from configs.args import get_parser
from benchmark.benchmarks.processing import Engine
from core.pipeline.pipeline import Pipeline
from benchmark.benchmarks.runner import BenchmarkRunner 

from core.config.builder import ConfigBuilder
from core.config.benchmark import RunnerConfig, BenchmarkConfig
from core.config.experiment import ExperimentConfig

parser = get_parser()
args = parser.parse_args()

run_pipeline = args.run_pipeline
run_engine = args.run_engine
run_benchmark = args.run_benchmark
benchmark_type = args.benchmark_type
test = args.test

ROOT = "tests/" if test else "" 

SRC_PIPELINE = ROOT + 'configs/pipeline.yaml'
SRC_ENGINE = ROOT + 'configs/engine.yaml'

SRC_BENCHMARK_RUNNER = ROOT + 'configs/benchmark/runner.yaml'
SRC_BENCHMARK_HESSIAN = ROOT + 'configs/benchmark/hessian.yaml'
SRC_BENCHMARK_ENHANCEMENT = ROOT + 'configs/benchmark/enhancement.yaml'
SRC_BENCHMARK_EXPERIMENT = ROOT + 'configs/benchmark/experiment.yaml'





def main():
    

    # if RUN_PIPELINE:
        
    #     config: MainConfig = ConfigBuilder(SRC_PIPELINE, MainConfig)
    #     setup_config: SetupConfig = config.setup
    #     experiment_config: ExperimentConfig = config.experiment
        
    #     pipeline = Pipeline(setup_config)
    #     pipeline.run(experiment_config)


    # if RUN_ENGINE:
    #     config: MainConfig = ConfigBuilder(SRC_ENGINE, MainConfig)
    #     setup_config: SetupConfig = config.setup
    #     experiment_config: ExperimentConfig = config.experiment
    #     engine_config: EngineConfig = ConfigBuilder(config.runner, EngineConfig)
        
    #     engine = Engine(setup_config)
    #     engine.run(engine_config, experiment_config)

    if run_benchmark:
        runner_config: RunnerConfig = ConfigBuilder(SRC_BENCHMARK_RUNNER, RunnerConfig)
        experiment_config = ConfigBuilder(SRC_BENCHMARK_EXPERIMENT, ExperimentConfig)
        
        if benchmark_type == "hessian":
            benchmark_config: BenchmarkConfig = ConfigBuilder(SRC_BENCHMARK_HESSIAN, BenchmarkConfig)
        elif benchmark_type == "enhancement":
            benchmark_config: BenchmarkConfig = ConfigBuilder(SRC_BENCHMARK_ENHANCEMENT, BenchmarkConfig)
        else:
            raise ValueError(f'Benchmark type unknown: {benchmark_type}')
        
        runner = BenchmarkRunner(runner_config.setup)
        filename = runner.run(
            images_dir=runner_config.images_dir, 
            labels_dir=runner_config.labels_dir,
            benchmark_config=benchmark_config,
            experiment_config=experiment_config,
        )     
        
        runner.analyse(benchmark_config=benchmark_config, results_file=filename)  
        
if __name__ == "__main__":
    main()
    
    