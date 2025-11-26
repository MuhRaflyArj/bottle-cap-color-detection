from typing import Dict, Optional, Any
from pathlib import Path
import gc

from torch.cuda import is_available
import wandb
from ultralytics import YOLO
from ultralytics.utils.metrics import DetMetrics


def print_metrics(results: DetMetrics) -> None:
    """Parses the results dictionary and prints a clean text table.

    Args:
        results (DetMetrics): The results object returned by the YOLO training method. 
                              Contains a 'results_dict' attribute.
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

    Args:
        results (Any): The results object returned by the YOLO training method.
        run_id (str): The unique 8-character alphanumeric ID of the W&B run.
        project_name (str): The name of the W&B project.
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


def export_and_log_tensorrt(run_id: str, project_name: str, run_name: str) -> None:
    """
    Exports the best model to TensorRT and logs the artifact to the active WandB run.
    
    Args:
        run_id (str): The unique ID of the WandB run to attach to.
        project_name (str): The local project directory name (where weights are saved).
        run_name (str): The specific run name (folder name).
    """
    # Locate the best weights
    weights_path = Path(project_name) / run_name / "weights" / "best.pt"

    if not weights_path.exists():
        print(f"Skipping export: Weights not found at {weights_path}")
        return

    try:
        if is_available():
            with wandb.init(id=run_id, project=project_name, resume="must") as run:
                model = YOLO(weights_path)

                exported_path_str = model.export(format="engine", device=0, half=True)
                exported_path = Path(exported_path_str)

                artifact_name = f"{run_name}_tensorrt"
                trt_artifact = wandb.Artifact(
                    name=artifact_name,
                    type="model",
                    description=f"TensorRT FP16 Engine for {run_name}"
                )
                trt_artifact.add_file(str(exported_path))

                run.log_artifact(trt_artifact)
                print(f"Successfully logged {artifact_name} to run {run_id}")

                del model
                gc.collect()
        else:
            print("CUDA is not available. Skipping TensorRT export.")

    except Exception as e: # pylint: disable=broad-exception-caught
        print(f"Failed to export/log TensorRT: {e}")
