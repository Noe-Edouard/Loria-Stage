import yaml
from pathlib import Path
from numpy import ndarray
from dataclasses import dataclass, asdict, is_dataclass, fields
from typing import get_origin, get_args
from typing import Any, Union, Literal, Callable, Sequence, Tuple, Optional



@dataclass
class Params:
    def __getattr__(self, key: str) -> Any:
        try:
            return self.__dict__[key]
        except KeyError:
            raise AttributeError(f"Attribut '{key}' not found in {self.__class__.__name__}")
    
    def to_dict(self):
        return asdict(self)
    
  
@dataclass
class ExperimentParams(Params):
    name: str
    debug_mode: bool
    normalize: bool
    crop: bool
    target_shape: Sequence[int]
    input_file: str
    output_file: str
    log_file: str
    input_dir: str
    output_dir: str
    log_dir: str


@dataclass
class MethodsParams(Params):
    derivator: Literal['default', 'farid']
    enhancer: Literal['frangi']
    segmenter: Literal['thresholding']


@dataclass
class HessianParams(Params):
    mode: Literal['reflect', 'constant', 'nearest', 'mirror', 'wrap']
    cval: float
    use_gaussian_derivatives: bool
    
    
@dataclass
class EnhancementParams(Params):
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
class ProcessingParams(Params):
    normalize: bool
    parallelize: bool
    chunk_size: list[int]
    overlap_size: int


@dataclass
class SegmentationParams(Params):
    threshold: int
   
    
@dataclass
class ConfigParams(Params):
    experiment: ExperimentParams
    methods: MethodsParams
    processing: ProcessingParams
    hessian: HessianParams
    enhancement: EnhancementParams
    segmentation: SegmentationParams
    

class Config:
    def __init__(self, config_source: Union[str, Path, dict]):
        self.config_source = config_source
        self.config_dict = self._get_config(self.config_source)
        self.config: ConfigParams = self._parse_config(ConfigParams, self.config_dict) 

    def _get_config(self, config_source: Union[str, Path, dict]) -> dict:
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
    
    def _parse_config(self, dataclass_type: type, data: dict):
        if not is_dataclass(dataclass_type):
            return data 

        kwargs = {}
        for field in fields(dataclass_type):
            value = data.get(field.name)

            # If nested dataclass
            if is_dataclass(field.type):
                value = self._parse_config(field.type, value)

            # if Optional
            elif get_origin(field.type) is Union:
                args = get_args(field.type)
                non_none = [a for a in args if a is not type(None)]
                if len(non_none) == 1 and is_dataclass(non_none[0]) and value is not None:
                    value = self._parse_config(non_none[0], value)

            kwargs[field.name] = value

        return dataclass_type(**kwargs)


