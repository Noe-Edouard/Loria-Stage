import numpy as np
import pytest
from benchmark.benchmark import Benchmark
from core.config import SetupConfig, BenchmarkConfig, ExperimentConfig, LoadConfig, MethodsConfig, ProcessingConfig, HessianConfig, EnhancementConfig, SegmentationConfig

@pytest.fixture
def dummy_data(tmp_path):
    data = np.zeros((64, 64))
    data[20:40, 20:40] = 1

    raw_file = tmp_path / "raw.npy"
    gt_file = tmp_path / "gt.npy"

    np.save(raw_file, data)
    np.save(gt_file, data)

    return raw_file, gt_file


@pytest.fixture
def dummy_setup_config(tmp_path):
    return SetupConfig(
        name="test_benchmark",
        log_file="test",
        input_dir=str(tmp_path),
        output_dir=str(tmp_path / "out"),
        log_dir=str("logs"),
        debug_mode=True,
        display_mode=True,
        save_mode=False,
    )


@pytest.fixture
def dummy_benchmark_config(dummy_data):
    raw_file, gt_file = dummy_data
    return BenchmarkConfig(
        raw_file=str(raw_file),
        gt_file=str(gt_file),
        methods=["default"],
        params_grid={
            "alpha": [0.5, 1.0],
            "beta": [0.5],
            "gamma": [15],
        },
    )


@pytest.fixture
def dummy_experiment_config():
    return ExperimentConfig(
        load=LoadConfig(
            normalize=False,
            crop=False,
            target_shape=[64, 64],
            input_file="dummy_in.npy",
            output_file="dummy_out.npy"
        ),
        methods=MethodsConfig(
            derivator="default",
            enhancer="frangi",
            segmenter="thresholding"
        ),
        processing=ProcessingConfig(
            normalize=False,
            parallelize=False,
            chunk_size=[64, 64],
            overlap_size=0
        ),
        hessian=HessianConfig(
            mode="reflect",
            cval=0.0,
        ),
        enhancement=EnhancementConfig(
            alpha=1.0,
            beta=0.5,
            gamma=15.0,
            scales=[2, 4, 6]
        ),
        segmentation=SegmentationConfig(
            threshold=0.5
        )
    )


def test_benchmark_run(dummy_setup_config, dummy_benchmark_config, dummy_experiment_config):
    benchmark = Benchmark(dummy_setup_config)
    result = benchmark.run(
        config_benchmark=dummy_benchmark_config,
        config_experiment=dummy_experiment_config
    )

    assert result is not None
    assert len(result.experiments) == len(dummy_benchmark_config.methods)
    for experiment in result.experiments:
        assert experiment.segmented.shape == (64, 64)
        assert experiment.enhanced.shape == (64, 64)
