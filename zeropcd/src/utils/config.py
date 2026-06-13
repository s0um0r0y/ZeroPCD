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
        
def get_args_and_config():
    """
    Parses command line arguments to allow dynamic config loading.
    Usage: python scripts/train.py --config configs/debug.yaml
    """
    parser = argparse.ArgumentParser(description="zero-pcd: Point Cloud Deep Learning")
    parser.add_argument(
        '--config', 
        type=str, 
        default='configs/default.yaml', 
        help='Path to the YAML configuration file'
    )
    
    args = parser.parse_args()
    config = load_config(args.config)
    
    return args, config

if __name__ == "__main__":
    test_config_path = Path("../../configs/default.yaml")
    if test_config_path.exists():
        config = load_config(test_config_path)
        print("Config successfully loaded!")
        print(f"Experiment Name: {config.experiment_name}")
        print(f"Num Points: {config.data.num_points}")
    else:
        print("YAML config not found. Make sure to create it first!")