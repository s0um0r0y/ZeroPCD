import yaml
import argparse
from pathlib import Path

class ConfigNode:
    """
    Recursively converts a dictionary into an object, 
    allowing dot-notation access (e.g., config.model.use_tnet).
    """
    def __init__(self, dictionary) -> None:
        for key, value in dictionary.items():
            if isinstance(value, dict):
                setattr(self, key, ConfigNode(value))
            else:
                setattr(self, key, value)
                
    def __repr__(self) -> str:
        return f"{self.__dict__}"
    
def load_config(config_path: str | Path) -> ConfigNode:
    """
    Loads a YAML configuration file and returns it as a ConfigNode.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(config_path, 'r') as file:
        try:
            config_dict = yaml.safe_load(file)
            return ConfigNode(config_dict)
        except yaml.YAMLError as exc:
            raise ValueError(f"Error parsing YAML file: {exc}")
        
