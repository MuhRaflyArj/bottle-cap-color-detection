import shutil
import zipfile
from pathlib import Path
from typing import List, Union, Any, Dict, Optional

import yaml
import wandb
from roboflow import Roboflow

from ultralytics.utils.metrics import DetMetrics # pylint: disable=import-error
from bsort.config import settings # pylint: disable=import-error

# Class 0: Others
FILES_OTHERS_TRAIN = [
    "raw-250110_dc_s001_b3_2.jpg",
    "raw-250110_dc_s001_b2_1.jpg",
    "raw-250110_dc_s001_b2_3.jpg",
    "raw-250110_dc_s001_b2_15.jpg",
]

FILES_OTHERS_VALID = [
    "raw-250110_dc_s001_b3_3.jpg",
    "raw-250110_dc_s001_b3_4.jpg",
]

# Class 1: Light Blue
FILES_LIGHT_BLUE_TRAIN = [
    "raw-250110_dc_s001_b4_3.jpg",
    "raw-250110_dc_s001_b4_1.jpg",
]

FILES_LIGHT_BLUE_VALID = [
    "raw-250110_dc_s001_b4_2.jpg",
]

# Class 2: Dark Blue
FILES_DARK_BLUE_TRAIN = [
    "raw-250110_dc_s001_b5_2.jpg",
    "raw-250110_dc_s001_b5_3.jpg",
]

FILES_DARK_BLUE_VALID = [
    "raw-250110_dc_s001_b5_5.jpg",
]

def download_roboflow_dataset(
    workspace: str,
    project: str,
    version_number: int = 1,
    location: Union[str, Path] = None
):
    """
    Downloads a dataset from Roboflow using credentials from .env.
    
    Args:
        workspace (str): The Roboflow workspace ID.
        project (str): The Roboflow project ID.
        version_number (int): The version number to download. Defaults to 1.
        location (Union[str, Path], optional): The target directory for the download. 
                                               If None, uses Roboflow's default location.
    
    Returns:
        dataset: The Roboflow dataset object.
    """
    api_key = settings.ROBOFLOW_API_KEY

    if not api_key:
        raise ValueError("ROBOFLOW_API_KEY not found in environment variables")

    rf = Roboflow(api_key=api_key)
    project_instance = rf.workspace(workspace).project(project)
    version_instance = project_instance.version(version_number)

    if location:
        return version_instance.download("yolov11", location=str(location))

    return version_instance.download("yolov11") # Uses default location


def extract_and_prepare_dataset(
    zip_path: Union[str, Path],
    root_dir: Union[str, Path] = "datasets"
) -> None:
    """Extracts the dataset zip and organizes it into a raw/processed structure.

    This function performs the following steps:
    1. Unzips the source file into `datasets/raw/sample`.
    2. Creates the target directory structure `datasets/processed/sample`.
    3. Splits data into train/valid based on explicit hardcoded lists.
    4. Rewrites label files (txt) to map classes based on filenames.
    5. Generates the YOLO config.yaml file.

    Args:
        zip_path (Union[str, Path]): Path to the source zip file.
        root_dir (Union[str, Path]): The root directory for data storage.
            Defaults to "datasets".

    Raises:
        FileNotFoundError: If no images are found in the extracted zip.
    """
    root_path = Path(root_dir)
    zip_file = Path(zip_path)

    # 1. Define Paths
    raw_dir = root_path / "raw" / "sample"
    processed_dir = root_path / "processed" / "sample"

    if raw_dir.exists():
        shutil.rmtree(raw_dir)
    if processed_dir.exists():
        shutil.rmtree(processed_dir)

    # 2. Extract Zip
    print(f"Extracting {zip_file} to {raw_dir}...")
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(raw_dir)

    # 3. Create Processed Directory Structure
    for split in ["train", "valid"]:
        (processed_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (processed_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    # 4. Find all images in the raw directory (handling internal zip folders)
    source_files = list(raw_dir.rglob("*.jpg")) + list(raw_dir.rglob("*.png"))

    if not source_files:
        raise FileNotFoundError("No images found in the extracted zip file.")

    train_files: List[Path] = []
    valid_files: List[Path] = []

    valid_filenames = (
        FILES_OTHERS_VALID +
        FILES_LIGHT_BLUE_VALID +
        FILES_DARK_BLUE_VALID
    )

    # Sort files into Train or Valid
    for file_path in source_files:
        if file_path.name in valid_filenames:
            valid_files.append(file_path)
        else:
            train_files.append(file_path)

    # Process files into their respective folders
    _process_split(train_files, processed_dir / "train")
    _process_split(valid_files, processed_dir / "valid")

    # 5. Generate Config YAML
    yaml_path =_create_yaml_config(processed_dir)
    print("Dataset preparation complete.")

    return yaml_path


def _determine_class_id(filename: str) -> int:
    """Determines the class ID based on the filename.

    Args:
        filename (str): The name of the image file.

    Returns:
        int: The new class ID (0: others, 1: light_blue, 2: dark_blue).
    """
    is_light_blue = filename in (FILES_LIGHT_BLUE_TRAIN + FILES_LIGHT_BLUE_VALID)
    is_dark_blue = filename in (FILES_DARK_BLUE_TRAIN + FILES_DARK_BLUE_VALID)
    is_others = filename in (FILES_OTHERS_TRAIN + FILES_OTHERS_VALID)

    if is_light_blue:
        return 1
    if is_dark_blue:
        return 2
    if is_others:
        return 0

    return 0

def _process_split(files: List[Path], target_dir: Path) -> None:
    """Moves images and updates labels for a specific data split.

    Args:
        files (List[Path]): List of image file paths to process.
        target_dir (Path): The target directory (e.g., .../train or .../valid).
    """
    for img_path in files:
        filename = img_path.name
        label_path = img_path.with_suffix(".txt")

        new_class_id = _determine_class_id(filename)

        shutil.copy(img_path, target_dir / "images" / filename)

        # Process Label
        if label_path.exists():
            new_lines = []
            with open(label_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        # Replace the first element (original class_id) with new_class_id
                        parts[0] = str(new_class_id)
                        new_lines.append(" ".join(parts))

            # Write new label file
            with open(target_dir / "labels" / label_path.name, 'w', encoding='utf-8') as f:
                f.write("\n".join(new_lines))

def _create_yaml_config(base_path: Path) -> Path:
    """Generates the YOLO config.yaml file.

    Args:
        base_path (Path): The base path of the processed dataset.
    """
    # yaml structure
    config = {
        "path": str(base_path.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "valid/images",
        "names": {
            0: "others",
            1: "light_blue",
            2: "dark_blue"
        }
    }

    yaml_path = base_path / "config.yaml"
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, sort_keys=False)

    return yaml_path


def print_metrics(results: DetMetrics) -> None:
    """Parses the results dictionary and prints a clean text table.

    Args:
        results (DetMetrics): The results object returned by the YOLO training method. 
                              Contains a 'results_dict' attribute with performance metrics.
    """
    # Extract the key metrics
    metrics: Dict[str, float] = results.results_dict

    print(f"{'METRIC':<25}   {'VALUE':<10}")
    print("-" * 40)

    key_map: Dict[str, str] = {
        'metrics/precision(B)': 'Precision',
        'metrics/recall(B)':    'Recall',
        'metrics/mAP50(B)':     'mAP @ 50',
        'metrics/mAP50-95(B)':  'mAP @ 50-95',
        'fitness':              'Fitness Score'
    }

    for key, display_name in key_map.items():
        val: Optional[float] = metrics.get(key)

        if val is not None:
            print(f"{display_name:<25} : {val:.4f}")


def log_metrics_to_wandb(results: Any, run_id: str, project_name: str) -> None:
    """Parses the results and logs specific metrics to the active WandB run.

    This function extracts key performance metrics, renames them for clarity,
    and logs them as a summary dictionary to the currently active Weights & Biases run.
    
    Args:
        results (Any): The results object returned by the YOLO training method
            (typically an instance of ultralytics.utils.metrics.DetMetrics).
            Must contain a 'results_dict' attribute.
        run_id (str): The unique 8-character alphanumeric ID of the W&B run 
            to resume (e.g., 'a1b2c3d4').
        project_name (str): The name of the W&B project where the run exists.

    Returns:
        None: This function logs directly to W&B and does not return a value.

    Raises:
        wandb.Error: If the run ID does not exist or permissions are denied
            (caught and printed within the function).
    """
    metrics: Dict[str, float] = results.results_dict

    key_map: Dict[str, str] = {
        'metrics/precision(B)': 'Precision',
        'metrics/recall(B)':    'Recall',
        'metrics/mAP50(B)':     'mAP_50',
        'metrics/mAP50-95(B)':  'mAP_50_95',
        'fitness':              'Fitness_Score'
    }

    log_payload: Dict[str, float] = {}

    for key, display_name in key_map.items():
        val: Optional[float] = metrics.get(key)
        if val is not None:
            log_payload[display_name] = val

    try:
        # Reactivate the specified WandB run and log metrics
        with wandb.init(id=run_id, project=project_name, resume="must") as run:
            run.log(log_payload)
            print(f"Successfully logged metrics to run {run_id}")

    except wandb.Error as e:
        print(f"Failed to resume run {run_id}: {e}")
