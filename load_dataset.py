from datasets import load_dataset
from huggingface_hub import login

# 登录 Hugging Face
login(token="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")

# 下载 MuSiQue_processed 数据集
musique = load_dataset("sheryc/MuSiQue_processed", split="train")
print(f"MuSiQue loaded, total samples: {len(musique)}")

# 下载 DROP_processed 数据集
drop = load_dataset("sheryc/DROP_processed", split="train")
print(f"DROP loaded, total samples: {len(drop)}")

# 保存为本地 jsonl（可选）
musique.to_json("musique_processed.jsonl")
drop.to_json("drop_processed.jsonl")
