import yaml
from pathlib import Path
from numpy import ndarray
from dataclasses import dataclass, asdict, is_dataclass, fields
from typing import get_origin, get_args
from typing import Any, Union, Literal, Callable, Sequence, Tuple, Optional, Type, TypeVar

### GENERAL

from dataclasses import dataclass, asdict, is_dataclass
from typing import Any

@dataclass
class Config:
    def __getattr__(self, key: str) -> Any:
        try:
            return self.__dict__[key]
        except KeyError:
            raise AttributeError(f"Attribut '{key}' not found in {self.__class__.__name__}")
    
    def __setattr__(self, key: str, value: Any) -> None:
        self.__dict__[key] = value

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    def to_dict(self) -> dict:
        return asdict(self)

    def keys(self):
        return self.to_dict().keys()
    
    def items(self):
        return self.to_dict().items()
    
    def values(self):
        return self.to_dict().values()


C = TypeVar("C", bound=Config)
 
class ConfigBuilder:
    def __new__(cls, config_source: Union[str, Path, dict], config_type: Type[C]) -> C:
        config_dict = cls._get_config(config_source)
        return cls._parse_config(config_type, config_dict)

    @staticmethod
    def _get_config(config_source: Union[str, Path, dict]) -> dict:
        if isinstance(config_source, (str, Path)):
            path = Path(config_source)
            if not path.exists():
                raise FileNotFoundError(f"File {path} not found.")
            with path.open("r", encoding="utf-8") as f:
                config_dict = yaml.safe_load(f)
        elif isinstance(config_source, dict):
            config_dict = config_source
        else:
            raise TypeError("Config must be initialized with: str, Path or dict.")
        return config_dict

    @staticmethod
    def _parse_config(dataclass_type: Type[Config], data: dict) -> Config:
        if not is_dataclass(dataclass_type):
            return data

        kwargs = {}
        for field in fields(dataclass_type):
            value = data.get(field.name)
            field_type = field.type

            if value is None:
                kwargs[field.name] = None
                continue

            origin = get_origin(field_type)
            args = get_args(field_type)

            # Handle Optional[...] or Union[...] where one type is a dataclass
            if origin is Union:
                non_none_args = [arg for arg in args if arg is not type(None)]
                if len(non_none_args) == 1:
                    field_type = non_none_args[0]
                    origin = get_origin(field_type)
                    args = get_args(field_type)

            # Nested dataclass or subclass of Config
            if isinstance(value, dict) and (
                is_dataclass(field_type) or (isinstance(field_type, type) and issubclass(field_type, Config))
            ):
                value = ConfigBuilder._parse_config(field_type, value)

            kwargs[field.name] = value

        return dataclass_type(**kwargs)


 

    
### PIPELINE
@dataclass
class SetupConfig(Config):
    name: str 
    input_dir: str
    output_dir: str
    log_dir: str
    log_file: str
    debug_mode: bool
    display_mode: bool
    save_mode: bool


### EXPERIMENT
@dataclass
class LoadConfig(Config):
    normalize: bool
    crop: bool
    target_shape: Sequence[int]
    input_file: str
    output_file: str

@dataclass
class MethodsConfig(Config):
    derivator: Literal['default', 'gaussian', 'farid', 'cubic', 'trigonometric', 'catmull', 'bspline', 'bezier']
    enhancer: Literal['frangi']
    segmenter: Literal['thresholding']

@dataclass
class HessianConfig(Config):
    mode: Literal['reflect', 'constant', 'nearest', 'mirror', 'wrap']
    cval: float
    
@dataclass
class EnhancementConfig(Config):
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
class ProcessingConfig(Config):
    normalize: bool
    parallelize: bool
    chunk_size: list[int]
    overlap_size: int

@dataclass
class SegmentationConfig(Config):
    threshold: float
   
@dataclass
class ExperimentConfig(Config):
    load: LoadConfig
    methods: MethodsConfig
    processing: ProcessingConfig
    hessian: HessianConfig
    enhancement: EnhancementConfig
    segmentation: SegmentationConfig
    
@dataclass
class ExperimentData(Config):
    enhanced: ndarray
    segmented: ndarray
    config: ExperimentConfig
    
    
### BENCHMARK 

@dataclass
class GridConfig(Config):
    alpha: list[float]
    beta: list[float]
    gamma: list[float] 
 
@dataclass
class BenchmarkConfig(Config): 
    params_grid: GridConfig
    methods: list[str] 
    #search_type: Literal['full', 'same', 'constant', 'gradient']
    raw_file: str
    gt_file: str
       
@dataclass
class BenchmarkData(Config):
    raw_data: ndarray
    ground_truth: ndarray
    experiments: list[ExperimentData] 


### ENGINE


@dataclass
class SSIConfig(Config):
    volume_size: int
    scales_range: tuple[int]
    scales_numbers: list
    output_file: str
    
@dataclass
class VSIConfig(Config):
    volume_sizes: list[int]
    chunk_number: int
    output_file: str
    
    
@dataclass
class CNIConfig(Config):
    volume_sizes: list[int]
    chunk_numbers: list[int]
    output_file: str
    
@dataclass
class PACConfig(Config):
    input_file: str
    output_file: str

@dataclass
class SACConfig(Config):
    input_file: str
    output_file: str
    ground_truth: str 
    scales_numbers: list
    scales_range: tuple


@dataclass
class EngineConfig(Config):
    ssi: SSIConfig
    vsi: VSIConfig
    cni: CNIConfig
    pac: PACConfig
    sac: SACConfig
    

### MAIN

@dataclass
class MainConfig(Config):
    setup: SetupConfig
    experiment: ExperimentConfig
    runner: Optional[Union[BenchmarkConfig, EngineConfig]]