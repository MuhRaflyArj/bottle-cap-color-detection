import os
import logging
from pathlib import Path
import yaml
import requests
import wandb
from ultralytics import YOLO

# Configure Logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def infer_model(config_path: str) -> None:
    """Main entry point for the inference pipeline.

    Loads the configuration, retrieves the model from WandB, downloads source media,
    and executes the YOLO inference.

    Args:
        config_path (str): The file path to the inference configuration YAML.
    """
    # 1. Load Configuration
    with open(config_path, 'r', encoding='utf-8') as f:
        full_config = yaml.safe_load(f)

    project_conf = full_config.get("project", {})
    infer_conf = full_config.get("inference", {})

    # Extract nested sections matching YAML structure
    wandb_conf = infer_conf.get("wandb", {})
    source_conf = infer_conf.get("source", {})
    model_args = infer_conf.get("model_args", {})

    # 2. Retrieve Model from WandB
    model_path = _download_model_from_wandb(wandb_conf)

    # 3. Prepare Source Media
    source_path = _prepare_source(
        source_conf.get("type"),
        source_conf.get("location")
    )

    # 4. Run Inference
    logger.info(f"Loading model from: {model_path}")
    model = YOLO(model_path)

    project_dir = project_conf.get("output_dir", "runs")
    run_name = project_conf.get("output_name", "predict")

    logger.info(f"Running inference on {source_path}...")
    logger.info(f"Output will be saved to: {project_dir}/{run_name}")

    # Execute Prediction
    model.predict(
        source=source_path,
        project=project_dir,
        name=run_name,
        exist_ok=True,
        **model_args
    )

    output_path = Path(project_dir) / run_name
    logger.info(f"Inference complete. Results saved to: {output_path.resolve()}")


def _download_model_from_wandb(wandb_conf: dict) -> str:
    """
    Locates the correct Run ID based on the human-readable Run Name,
    downloads the artifact, and returns the path to the .pt file.
    """
    entity = wandb_conf.get("entity")
    project = wandb_conf.get("project")
    target_name = wandb_conf.get("train_run_name")
    version = wandb_conf.get("version", "latest")

    if not all([entity, project, target_name]):
        raise ValueError("Inference config missing WandB entity, project, or train_run_name.")

    logger.info(f"Searching WandB for Run Name: '{target_name}' in {entity}/{project}...")

    api = wandb.Api()

    # Filter runs by the 'display_name'
    runs = api.runs(f"{entity}/{project}", filters={"display_name": target_name})

    if len(runs) == 0:
        raise ValueError(f"No run found with name '{target_name}'.")

    # Get the most recent run if duplicates exist
    target_run = runs[0]
    run_id = target_run.id
    logger.info(f"Resolved Run Name '{target_name}' to Run ID: '{run_id}'")

    # Construct the standard Ultralytics artifact name format
    artifact_name = f"run_{run_id}_model:{version}"
    artifact_path = f"{entity}/{project}/{artifact_name}"

    logger.info(f"Downloading artifact: {artifact_path}")

    try:
        artifact = api.artifact(artifact_path, type='model')
        artifact_dir = artifact.download()

        # Find the .pt file inside the downloaded directory
        model_files = list(Path(artifact_dir).rglob("*.pt"))
        if not model_files:
            raise FileNotFoundError("Artifact downloaded but no .pt file found.")

        return str(model_files[0])

    except Exception as e:
        logger.error(f"Failed to download artifact. Error: {e}")
        raise e


def _prepare_source(source_type: str, source_path: str) -> str:
    """
    Downloads source if URL, or verifies path if local.
    """
    if not source_type or not source_path:
        raise ValueError("Source type or location is missing in settings.")

    if source_type == "local":
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Local source not found: {source_path}")
        return source_path

    if source_type in ["url", "gcs"]:
        temp_dir = Path("temp_inference")
        temp_dir.mkdir(exist_ok=True)

        filename = source_path.split("/")[-1].split("?")[0]
        if not filename:
            filename = "downloaded_source"

        local_path = temp_dir / filename

        logger.info(f"Downloading source from {source_path}...")
        try:
            response = requests.get(source_path, stream=True, timeout=30)
            response.raise_for_status()
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return str(local_path)
        except Exception as e:
            logger.error(f"Failed to download source: {e}")
            raise e

    raise ValueError(f"Unsupported source_type: {source_type}")
