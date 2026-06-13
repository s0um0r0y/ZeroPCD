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