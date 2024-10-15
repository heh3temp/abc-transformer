from dataclasses import dataclass, asdict
from abc import ABC
import yaml


@dataclass
class Config(ABC):

    @classmethod
    def from_yaml(cls, filepath):
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)

        return cls(**config_dict)
    
    def to_dict(self):
        return asdict(self)
    
    def to_yaml(self, filepath):
        config_dict = asdict(self)

        with open(filepath, 'w') as f:
            yaml.safe_dump(config_dict, f, default_flow_style=False, indent=4)
