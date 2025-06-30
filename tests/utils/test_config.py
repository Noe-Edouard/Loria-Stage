from utils.config import Config

def test_load_config():
    config = Config("configs/test.yaml")

    assert isinstance(config, Config)
    assert hasattr(config, "experiment")
    assert hasattr(config, "enhancement")
    assert config.experiment.name == "test_run"

