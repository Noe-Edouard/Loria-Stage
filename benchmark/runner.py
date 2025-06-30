from utils.config import Config
from src.pipeline import Pipeline 

config_scales_influence = Config({
    'volume_size': 256,
    'scales_number': [2, 4, 6, 8, 10],
    'enhancement_method': 'frangi',
    'chunk_number': 8
})


def runner():
    
    pipeline = Pipeline(config)
    pipeline.run()
    
if __name__ == "__main__":
    runner()