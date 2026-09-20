"""
Central path config. Every path is relative to the project root (the
folder containing src/, data/, artifacts/), computed from this file's own
location — so the project works the same whether you run it from your
laptop, inside Docker, or on Render/Railway. Nothing here should ever be a
hardcoded absolute path.

Override with environment variables in production (e.g. Docker, Render)
if you want data/artifacts to live outside the repo.
"""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = Path(os.environ.get("PAYGUARD_DATA_DIR", PROJECT_ROOT / "data"))
ARTIFACT_DIR = Path(os.environ.get("PAYGUARD_ARTIFACT_DIR", PROJECT_ROOT / "artifacts"))
DATA_PATH = DATA_DIR / "upi_transactions.csv"

DATA_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
