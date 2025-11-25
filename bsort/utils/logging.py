from __future__ import annotations

from typing import Any

import wandb  # pylint: disable=import-error

from ultralytics.utils.metrics import DetMetrics


def print_metrics(results: DetMetrics) -> None:
    """Parses the results dictionary and prints a clean text table.

    Args:
        results (DetMetrics): The results object returned by the YOLO training method. Contains a 'results_dict'
            attribute.
    """
    # Extract the key metrics
    metrics: dict[str, float] = results.results_dict

    print(f"{'METRIC':<25}   {'VALUE':<10}")
    print("-" * 40)

    key_map: dict[str, str] = {
        "metrics/precision(B)": "Precision",
        "metrics/recall(B)": "Recall",
        "metrics/mAP50(B)": "mAP @ 50",
        "metrics/mAP50-95(B)": "mAP @ 50-95",
        "fitness": "Fitness Score",
    }

    for key, display_name in key_map.items():
        val: float | None = metrics.get(key)

        if val is not None:
            print(f"{display_name:<25} : {val:.4f}")


def log_metrics_to_wandb(results: Any, run_id: str, project_name: str) -> None:
    """Parses the results and logs specific metrics to the active WandB run.

    Args:
        results (Any): The results object returned by the YOLO training method.
        run_id (str): The unique 8-character alphanumeric ID of the W&B run.
        project_name (str): The name of the W&B project.
    """
    metrics: dict[str, float] = results.results_dict

    key_map: dict[str, str] = {
        "metrics/precision(B)": "Precision",
        "metrics/recall(B)": "Recall",
        "metrics/mAP50(B)": "mAP_50",
        "metrics/mAP50-95(B)": "mAP_50_95",
        "fitness": "Fitness_Score",
    }

    log_payload: dict[str, float] = {}

    for key, display_name in key_map.items():
        val: float | None = metrics.get(key)
        if val is not None:
            log_payload[display_name] = val

    try:
        # Reactivate the specified WandB run and log metrics
        with wandb.init(id=run_id, project=project_name, resume="must") as run:
            run.log(log_payload)
            print(f"Successfully logged metrics to run {run_id}")

    except wandb.Error as e:
        print(f"Failed to resume run {run_id}: {e}")
