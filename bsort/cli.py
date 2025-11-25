import argparse
import sys
from pathlib import Path

from bsort.train import train_model  # pylint: disable=import-error


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="bsort: Bottle Cap Sorting AI Pipeline")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- Train Command ---
    # Usage: python -m bsort train --config settings.yaml
    train_parser = subparsers.add_parser("train", help="Train a YOLO model")
    train_parser.add_argument(
        "--config",
        type=str,
        default="settings.yaml",
        help="Path to the configuration YAML file (default: settings.yaml)",
    )

    # --- Infer Command (Placeholder) ---
    infer_parser = subparsers.add_parser("infer", help="Run inference (Future Implementation)")
    infer_parser.add_argument("--config", type=str, required=True)
    infer_parser.add_argument("--source", type=str, required=True)

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
        print("Inference module is under construction.")

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
