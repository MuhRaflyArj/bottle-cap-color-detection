import os
import shutil
import logging
from pathlib import Path
from typing import Dict
import requests

from roboflow import Roboflow # pylint: disable=import-error
import gdown

from .file_ops import extract_zip

logger = logging.getLogger(__name__)


def get_dataset(config: Dict) -> str:
    """
    Downloads the dataset and returns the absolute path to the first found .yaml file.
    
    Args:
        config (Dict): The configuration dictionary.
    
    Returns:
        str: The absolute path to the yaml configuration file.
    """
    dataset_name = config.get("dataset_name", "dataset")
    source = config.get("source", "").lower()

    # 1. Resolve Paths
    root_conf = config.get("datasets_dir", "datasets")

    if Path(root_conf).is_absolute():
        datasets_root = Path(root_conf)
    else:
        datasets_root = (Path.cwd() / root_conf).resolve()

    target_dir = datasets_root / dataset_name

    # 2. FORCE CLEAN: If folder exists, delete it
    if target_dir.exists():
        logger.info(f"Removing existing dataset at: {target_dir}")
        shutil.rmtree(target_dir)

    # 3. Download
    logger.info(f"Downloading dataset to: {target_dir}")
    actual_data_dir = target_dir

    try:
        if source == "roboflow":
            actual_path_str = download_from_roboflow(config.get("roboflow", {}), target_dir)
            actual_data_dir = Path(actual_path_str).resolve()

        elif source == "gdrive":
            target_dir.mkdir(parents=True, exist_ok=True)
            download_from_gdrive(config.get("gdrive", {}), target_dir)

        elif source == "url":
            target_dir.mkdir(parents=True, exist_ok=True)
            download_from_url(config.get("url", {}), target_dir)

        else:
            raise ValueError(f"Unknown source: '{source}'")

    except Exception as e:
        logger.error(f"Download failed: {e}")
        raise e

    # 4. Locate YAML (Simplified)
    logger.info(f"Searching for configuration in: {actual_data_dir}")

    # Find ALL yaml files recursively
    found_yamls = list(actual_data_dir.rglob("*.yaml"))

    if not found_yamls:
        raise FileNotFoundError(f"No .yaml configuration found in {actual_data_dir}")

    # Simply take the first one found
    yaml_file = found_yamls[0]
    logger.info(f"Located config file at: {yaml_file}")

    return str(yaml_file.resolve())


def download_from_roboflow(rf_config: Dict, target_dir: Path) -> None:
    """Downloads a dataset from Roboflow and returns the dataset location path.

    Args:
        rf_config (Dict): Configuration dictionary containing Roboflow workspace,
            project, and version details.
        target_dir (Path): Directory path where the dataset will be downloaded.

    Returns:
        str: The absolute path to the downloaded dataset location.

    Raises:
        ImportError: If Roboflow library is not installed.
        ValueError: If ROBOFLOW_API_KEY environment variable is not set.
    """
    if Roboflow is None:
        raise ImportError("pip install roboflow")

    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        raise ValueError("ROBOFLOW_API_KEY not set.")

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(rf_config["workspace"]).project(rf_config["project"])
    version = project.version(rf_config["version"])
    dataset = version.download("yolov11", location=str(target_dir))

    return dataset.location


def download_from_gdrive(gd_config: Dict, target_dir: Path) -> None:
    """Downloads a dataset from Google Drive using gdown and extracts it.

    Args:
        gd_config (Dict): Configuration dictionary containing the Google Drive file ID.
        target_dir (Path): Directory path where the dataset will be downloaded and extracted.

    Raises:
        ImportError: If gdown library is not installed.
    """
    if gdown is None:
        raise ImportError("pip install gdown")

    file_id = gd_config["file_id"]
    output_zip = target_dir / "temp.zip"

    gdown.download(id=file_id, output=str(output_zip), quiet=False)

    extract_zip(output_zip, target_dir)


def download_from_url(url_config: Dict, target_dir: Path) -> None:
    """Downloads a dataset from a URL and extracts the ZIP file.

    Args:
        url_config (Dict): Configuration dictionary containing the download URL link.
        target_dir (Path): Directory path where the dataset will be downloaded and extracted.
    """
    response = requests.get(url_config["link"], stream=True, timeout=15)
    output_zip = target_dir / "temp.zip"
    with open(output_zip, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    extract_zip(output_zip, target_dir)
