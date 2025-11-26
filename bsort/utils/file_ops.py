import zipfile
from pathlib import Path
from typing import Dict, Any

import yaml

def extract_zip(zip_path: Path, target_dir: Path, unlink: bool = True) -> None:
    """Extracts a ZIP file to the specified directory and removes the ZIP file.

    Args:
        zip_path (Path): Path to the ZIP file to extract.
        target_dir (Path): Directory path where the contents will be extracted.
    """
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(target_dir)

    if unlink:
        zip_path.unlink()

def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Loads a YAML configuration file from the given path.
    
    Args:
        config_path (str): The file path to the configuration YAML.

    Returns:
        Dict[str, Any]: The loaded configuration dictionary.
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)
