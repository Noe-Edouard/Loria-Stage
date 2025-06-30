from utils.config import Config
from src.pipeline import Pipeline 


def main():
    config = Config("configs/default.yaml")
    pipeline = Pipeline(config)
    pipeline.run()
    
if __name__ == "__main__":
    main()