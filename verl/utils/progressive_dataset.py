import random
from torch.utils.data import Dataset
from datasets import load_dataset
from typing import Optional
import torch


class ProgressiveMixDataset(Dataset):
    def __init__(
        self,
        train_files: str,
        extra_files: str,
        tokenizer,
        processor=None,
        system_prompt: Optional[str] = None,
        prompt_key: str = "prompt",
        answer_key: str = "answer",
        max_prompt_length: int = 512,
        max_steps: int = 10000,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.system_prompt = system_prompt or ""
        self.prompt_key = prompt_key
        self.answer_key = answer_key
        self.max_prompt_length = max_prompt_length
        self.max_steps = max_steps
        self.global_step = 0

        # Load both datasets
        self.drop = load_dataset("json", data_files=train_files, split="train") \
            if train_files.endswith(".json") or train_files.endswith(".jsonl") \
            else load_dataset(train_files, split="train")

        self.musique = load_dataset("json", data_files=extra_files, split="train") \
            if extra_files.endswith(".json") or extra_files.endswith(".jsonl") \
            else load_dataset(extra_files, split="train")

    def post_init(self, max_steps):
        self.max_steps = max_steps

    def set_global_step(self, step: int):
        self.global_step = step

    def _get_mix_ratio(self):
        return min(0.5, self.global_step / self.max_steps * 0.5)

    def __len__(self):
        return max(len(self.drop), len(self.musique))

    def __getitem__(self, idx):
        p = self._get_mix_ratio()
        use_musique = random.random() < p
        sample = random.choice(
            self.musique) if use_musique else random.choice(self.drop)

        prompt = sample.get(self.prompt_key, "")
        answer = sample.get(self.answer_key, "")

        # Construct full prompt
        full_prompt = self.system_prompt + "\n" + \
            prompt if self.system_prompt else prompt

        # Tokenize inputs
        encoded = self.tokenizer(
            full_prompt,
            text_target=answer,
            truncation=True,
            max_length=self.max_prompt_length,
            padding="max_length",
            return_tensors="pt"
        )

        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        labels = encoded["labels"].squeeze(0)

        # Compute position_ids manually
        position_ids = attention_mask.cumsum(dim=-1) - 1
        position_ids = position_ids.masked_fill(attention_mask == 0, 0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "position_ids": position_ids,
            "raw_prompt_ids": self.tokenizer.encode(full_prompt, add_special_tokens=False),
            "ground_truth": answer,
            "context": prompt,
        }
