from dotenv import load_dotenv
from pathlib import Path
import os

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

class Settings:
    ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "your_roboflow_api_key")
    WANDB_API_KEY = os.getenv("WANDB_API_KEY", "your_wandb_api_key")
    
settings = Settings()