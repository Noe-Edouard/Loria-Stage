from numpy import ndarray
from typing import Literal, Callable, Sequence, Tuple, Optional
from dataclasses import dataclass

from core.config.base import ConfigBase
from core.config.metrics import Metrics


### CONFIGS

@dataclass
class LoadingConfig(ConfigBase):
    normalize: bool
    crop: bool
    target_shape: Sequence[int]
    raw_file: str
    gt_file: str

@dataclass
class MethodsConfig(ConfigBase):
    derivator: Literal['default', 'gaussian', 'farid', 'cubic', 'trigonometric', 'catmull', 'bspline', 'bezier']
    enhancer: Literal['frangi']
    segmenter: Literal['thresholding']

@dataclass
class HessianConfig(ConfigBase):
    mode: Literal['reflect', 'constant', 'nearest', 'mirror', 'wrap']
    cval: float
   
@dataclass
class ProcessingConfig(ConfigBase):
    normalize: bool
    parallelize: bool
    chunk_size: list[int]
    overlap_size: int
    
@dataclass
class EnhancementConfig(ConfigBase):
    alpha: float
    beta: float
    gamma: Optional[float] = None
    scales: Optional[Sequence[int]] = None
    scales_number: Optional[int] = None
    scales_range: Optional[Tuple[int, int]] = None
    black_ridges: Optional[bool] = False
    hessian_function: Optional[Callable[..., list[ndarray]]] = None
    skimage: Optional[bool] = False

@dataclass
class SegmentationConfig(ConfigBase):
    threshold: float
   
   
@dataclass
class ExperimentConfig(ConfigBase):
    loading: LoadingConfig
    methods: MethodsConfig
    processing: ProcessingConfig
    hessian: HessianConfig
    enhancement: EnhancementConfig
    segmentation: SegmentationConfig


### EXPERIMENT
 
@dataclass
class Experiment(ConfigBase):
    data_enhanced: ndarray
    data_segmented: ndarray
    config: ExperimentConfig
    metrics: Optional[Metrics] = None
    id: Optional[str] = None
    