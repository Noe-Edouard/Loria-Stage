from core.config import Config, MainConfig, ConfigBuilder
from pipeline.pipeline import Pipeline 
from benchmark.benchmark import Benchmark

def main():
    # config = Config("configs/default.yaml")
    # pipeline = Pipeline(config)
    # pipeline.run()
    config_main: MainConfig = ConfigBuilder('./configs/benchmark.yaml', MainConfig)
    config_setup = config_main.setup
    config_benchmark = config_main.benchmark
    config_experiment = config_main.experiment
    
    benchmark = Benchmark(config_setup)
    benchmark.run(
        config_benchmark=config_benchmark, 
        config_experiment=config_experiment
    )
    
if __name__ == "__main__":
    main()