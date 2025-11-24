from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

import yaml


def extract_and_prepare_dataset(zip_path: str | Path, root_dir: str | Path = "datasets") -> Path:
    """Extracts the dataset zip and organizes it into a raw/processed structure.

    Returns:
        Path: The path to the generated data.yaml file.
    """
    root_path = Path(root_dir)
    zip_file = Path(zip_path)

    # 1. Load Split Configuration
    current_dir = Path(__file__).resolve().parent
    split_config_path = current_dir.parent / "sample_data_split.yaml"

    if not split_config_path.exists():
        raise FileNotFoundError(f"Split config not found at {split_config_path}")

    with open(split_config_path) as f:
        split_config = yaml.safe_load(f)

    # 2. Define Paths
    raw_dir = root_path / "raw" / "sample"
    processed_dir = root_path / "processed" / "sample"

    if raw_dir.exists():
        shutil.rmtree(raw_dir)
    if processed_dir.exists():
        shutil.rmtree(processed_dir)

    # 3. Extract Zip
    print(f"Extracting {zip_file} to {raw_dir}...")
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(raw_dir)

    # 4. Create Processed Directory Structure
    for split in ["train", "valid"]:
        (processed_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (processed_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    # 5. Find all images
    source_files = list(raw_dir.rglob("*.jpg")) + list(raw_dir.rglob("*.png"))

    if not source_files:
        raise FileNotFoundError("No images found in the extracted zip file.")

    train_files: list[Path] = []
    valid_files: list[Path] = []

    # Flatten the validation lists for quick lookup
    valid_filenames = (
        split_config["others"]["valid"] + split_config["light_blue"]["valid"] + split_config["dark_blue"]["valid"]
    )

    # Sort files into Train or Valid
    for file_path in source_files:
        if file_path.name in valid_filenames:
            valid_files.append(file_path)
        else:
            train_files.append(file_path)

    # Process files (Pass the config so helper knows class IDs)
    _process_split(train_files, processed_dir / "train", split_config)
    _process_split(valid_files, processed_dir / "valid", split_config)

    # 6. Generate Config YAML
    yaml_path = _create_yaml_config(processed_dir)
    print("Dataset preparation complete.")

    return yaml_path


def _determine_class_id(filename: str, config: dict) -> int:
    """Determines the class ID based on the filename and loaded config."""

    # Helper to check if file is in train or valid list for a key
    def is_in(key):
        return filename in (config[key]["train"] + config[key]["valid"])

    if is_in("light_blue"):
        return 1
    if is_in("dark_blue"):
        return 2
    if is_in("others"):
        return 0

    return 0


def _process_split(files: list[Path], target_dir: Path, config: dict) -> None:
    """Moves images and updates labels for a specific data split."""
    for img_path in files:
        filename = img_path.name
        label_path = img_path.with_suffix(".txt")

        # Pass config to determine ID
        new_class_id = _determine_class_id(filename, config)

        shutil.copy(img_path, target_dir / "images" / filename)

        # Process Label
        if label_path.exists():
            new_lines = []
            with open(label_path, encoding="utf-8") as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        parts[0] = str(new_class_id)
                        new_lines.append(" ".join(parts))

            with open(target_dir / "labels" / label_path.name, "w", encoding="utf-8") as f:
                f.write("\n".join(new_lines))


def _create_yaml_config(base_path: Path) -> Path:
    """Generates the YOLO config.yaml file."""
    config = {
        "path": str(base_path.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "valid/images",
        "names": {0: "others", 1: "light_blue", 2: "dark_blue"},
    }

    yaml_path = base_path / "config.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, sort_keys=False)

    return yaml_path
