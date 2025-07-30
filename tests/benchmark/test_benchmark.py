
import pytest
import numpy as np
from core.configs.config import BenchmarkConfig
from benchmark.enhancement import Optimizer
from tests.conftest import DummyEnhancer, DummySegmenter, dummy_config

@pytest.fixture
def optimizer(monkeypatch):
    setup, experiment = dummy_config()
    opt = Optimizer(setup)
    opt.enhancer = DummyEnhancer()
    opt.segmenter = DummySegmenter()
    return opt, experiment

def test_optimizer_init(optimizer):
    opt, _ = optimizer
    assert hasattr(opt, 'enhancer')
    assert hasattr(opt, 'segmenter')
    assert hasattr(opt, 'logger')

def test_optimizer_logger(optimizer):
    opt, _ = optimizer
    assert opt.logger is not None

def test_optimizer_enhancer_segmenter(optimizer):
    opt, _ = optimizer
    assert callable(getattr(opt.enhancer, 'enhance_data', None))
    assert callable(getattr(opt.segmenter, 'segment_data', None))

def test_optimizer_viewer_saver(optimizer):
    opt, _ = optimizer
    assert hasattr(opt, 'viewer')
    assert hasattr(opt, 'saver')

def test_optimizer_loader(optimizer):
    opt, _ = optimizer
    assert hasattr(opt, 'loader')
    assert callable(getattr(opt.loader, 'load_data', None))
