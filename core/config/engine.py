from dataclasses import dataclass
from core.config.base import ConfigBase


### ENGINE

@dataclass
class SSIConfig(ConfigBase):
    volume_size: int
    scales_range: tuple[int]
    scales_numbers: list
    
@dataclass
class VSIConfig(ConfigBase):
    volume_sizes: list[int]
    chunk_number: int
    
    
@dataclass
class CNIConfig(ConfigBase):
    volume_sizes: list[int]
    chunk_numbers: list[int]
    
@dataclass
class PACConfig(ConfigBase):
    input_file: str


@dataclass
class EngineConfig(ConfigBase):
    ssi: SSIConfig
    vsi: VSIConfig
    cni: CNIConfig
    pac: PACConfig
    
    