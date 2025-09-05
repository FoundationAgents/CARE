from pathlib import Path
from huggingface_hub import snapshot_download

# Project-level datasets directory
DATASET_DIR = Path(__file__).resolve().parent.parent.parent / "datasets"
DATASET_DIR.mkdir(parents=True, exist_ok=True)

# Download Qwen model into CARE/datasets
snapshot_download(
    repo_id="Qwen/Qwen2.5-1.5B-Instruct",
    local_dir=str(DATASET_DIR / "Qwen2.5-1.5B-Instruct"),
    local_dir_use_symlinks=False,
)

print("Model downloaded to:", DATASET_DIR / "Qwen2.5-1.5B-Instruct")
