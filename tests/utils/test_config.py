from core.configs.config import ConfigBase

def test_load_config():
    config = ConfigBase("configs/test.yaml")

    assert isinstance(config, ConfigBase)
    assert hasattr(config, "experiment")
    assert hasattr(config, "enhancement")
    assert config.setup.name == "test_run"

