from pathlib import Path
from huggingface_hub import snapshot_download
from datasets import load_dataset

DATASET_DIR = Path(__file__).resolve().parent.parent.parent  / "datasets"
DATASET_DIR.mkdir(parents=True, exist_ok=True)


# Download datasets
musique = load_dataset("sheryc/MuSiQue_processed", split="train")
print(f"MuSiQue loaded, total samples: {len(musique)}")
drop = load_dataset("sheryc/DROP_processed", split="train")
print(f"DROP loaded, total samples: {len(drop)}")

# Save as JSONL
musique.to_json(DATASET_DIR / "musique_processed.jsonl")
drop.to_json(DATASET_DIR / "drop_processed.jsonl")
