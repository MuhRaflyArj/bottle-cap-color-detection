"""
Data Management Utilities for bsort (CLI).
Handles generic dataset downloading and path sanitization.
"""

import os
import shutil
import zipfile
import logging
import requests
import yaml
from pathlib import Path
from typing import Dict, Union
from roboflow import Roboflow
import gdown

logger = logging.getLogger(__name__)


def get_dataset(config: Dict) -> str:
    """
    Downloads the dataset. IF the dataset folder exists, IT IS DELETED first.
    
    Args:
        config (Dict): The configuration dictionary.
    
    Returns:
        str: The absolute path to the 'data.yaml' file.
    """
    dataset_name = config.get("dataset_name", "dataset")
    yaml_filename = config.get("yaml_path", "data.yaml")
    source = config.get("source", "").lower()
    
    # Resolve paths
    root_path = config.get("datasets_dir", "datasets")
    datasets_root = Path(root_path)
    target_dir = datasets_root / dataset_name

    # 1. If folder exists, delete it to make room for new data
    if target_dir.exists():
        logger.info(f"Removing existing dataset at: {target_dir}")
        shutil.rmtree(target_dir)

    # 2. Create fresh directory and Download
    logger.info(f"Downloading fresh dataset from source: {source}...")
    target_dir.mkdir(parents=True, exist_ok=True)

    try:
        if source == "roboflow":
            download_from_roboflow(config.get("roboflow", {}), target_dir)
        elif source == "gdrive":
            download_from_gdrive(config.get("gdrive", {}), target_dir)
        elif source == "url":
            download_from_url(config.get("url", {}), target_dir)
        else:
            raise ValueError(f"Unknown source: '{source}'")

    except Exception as e:
        logger.error(f"Download failed: {e}")
        raise e

    # 3. Locate YAML (Handle nested folders from unzipping)
    found = list(target_dir.rglob("*.yaml"))
    
    # Filter for likely candidates (data.yaml or dataset.yaml)
    candidates = [f for f in found if f.name in ["data.yaml", "dataset.yaml"]]
    
    if candidates:
        yaml_file = candidates[0]
        logger.info(f"Located config file at: {yaml_file}")
    elif found:
        yaml_file = found[0]
        logger.warning(f"Using fallback config file: {yaml_file}")
    else:
        files_present = [f.name for f in list(target_dir.rglob("*"))[:10]]
        raise FileNotFoundError(
            f"'{yaml_filename}' not found in {target_dir}. \n"
            f"Files found: {files_present}"
        )

    return str(yaml_file.resolve())


def download_from_roboflow(rf_config: Dict, target_dir: Path) -> None:
    if Roboflow is None:
        raise ImportError("pip install roboflow")
    
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        raise ValueError("ROBOFLOW_API_KEY not set.")

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(rf_config["workspace"]).project(rf_config["project"])
    version = project.version(rf_config["version"])
    
    version.download("yolov11", location=str(target_dir))


def download_from_gdrive(gd_config: Dict, target_dir: Path) -> None:
    if gdown is None:
        raise ImportError("pip install gdown")
    
    file_id = gd_config["file_id"]
    output_zip = target_dir / "temp.zip"
    
    gdown.download(id=file_id, output=str(output_zip), quiet=False)
    
    _extract_zip(output_zip, target_dir)


def download_from_url(url_config: Dict, target_dir: Path) -> None:
    response = requests.get(url_config["link"], stream=True)
    output_zip = target_dir / "temp.zip"
    with open(output_zip, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    _extract_zip(output_zip, target_dir)


def _extract_zip(zip_path: Path, target_dir: Path) -> None:
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(target_dir)
    zip_path.unlink()