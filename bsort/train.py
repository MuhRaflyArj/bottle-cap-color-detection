import yaml
import gc
import wandb
from ultralytics import YOLO
from bsort.utils import get_dataset, log_metrics_to_wandb

def train_model(config_path: str) -> None:
    """
    Executes the training pipeline based on the provided configuration file.

    Args:
        config_path (str): Path to the settings.yaml file.
    """
    # 1. Load Configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    project_conf = config.get("project", {})
    data_conf = config.get("data", {})
    model_conf = config.get("model", {})
    train_args = config.get("training", {})
    wandb_conf = config.get("wandb", {})

    # 2. Prepare Data
    print(f"Preparing dataset from source: {data_conf.get('source')}...")
    data_yaml_path = get_dataset(data_conf)
    
    # 3. Initialize WandB (if enabled)
    run = None
    if wandb_conf.get("enabled", False):
        run = wandb.init(
            project=project_conf.get("project_name", "bsort_project"),
            name=project_conf.get("run_name", "experiment"),
            config=config,
            # entity=wandb_conf.get("entity") # Optional
        )

    try:
        # 4. Initialize Model
        model_name = model_conf.get("name", "yolo11n.pt")
        task = model_conf.get("task", "detect")
        
        print(f"Initializing model: {model_name} (Task: {task})")
        
        # Load model (supports both .pt and custom .yaml like yolo11p.yaml)
        model = YOLO(model_name, task=task)

        # 5. Execute Training
        print(f"Starting training using data config: {data_yaml_path}")
        results = model.train(
            data=data_yaml_path,
            project=project_conf.get("output_dir", "runs"),
            name=project_conf.get("run_name", "exp"),
            **train_args # Unpack all training args from yaml
        )

        # 6. Log Custom Metrics to WandB
        if run:
            print("Logging final metrics to WandB...")
            log_metrics_to_wandb(
                results, 
                run_id=run.id, 
                project_name=project_conf.get("project_name")
            )

    except Exception as e:
        print(f"Training failed: {e}")
        raise e
        
    finally:
        # 7. Cleanup to free GPU memory
        if 'model' in locals():
            del model
        gc.collect()
        if wandb.run:
            wandb.finish()