import argparse
import sys
from pathlib import Path
from bsort.train import train_model
from bsort.infer import infer_model

def main():
    """
    Main CLI entry point.
    """
    # 1. Initialize the ArgumentParser (This was missing/wrong in your snippet)
    parser = argparse.ArgumentParser(
        description="bsort: Bottle Cap Sorting AI Pipeline"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- Train Command ---
    train_parser = subparsers.add_parser("train", help="Train a YOLO model")
    train_parser.add_argument(
        "--config", 
        type=str,
        default="settings.yaml",
        help="Path to the configuration YAML file (default: settings.yaml)"
    )

    # --- Infer Command ---
    infer_parser = subparsers.add_parser("infer", help="Run inference using a WandB model")
    infer_parser.add_argument(
        "--config", 
        type=str,
        default="settings_infer.yaml", # Defaults to the infer settings
        help="Path to the configuration YAML file"
    )
    # Optional: Allow overriding the image source via CLI
    infer_parser.add_argument("--source", type=str, help="Override source image/video path or URL")

    args = parser.parse_args()

    # Dispatch Command
    if args.command == "train":
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"Error: Configuration file '{config_path}' not found.")
            sys.exit(1)

        print(f"Loading configuration from {config_path}...")
        train_model(str(config_path))

    elif args.command == "infer":
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"Error: Configuration file '{config_path}' not found.")
            sys.exit(1)

        print(f"Starting inference with config: {config_path}")
        infer_model(str(config_path))

    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
