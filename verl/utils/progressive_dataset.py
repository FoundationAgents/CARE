import random
from torch.utils.data import Dataset
from datasets import load_dataset
from typing import Optional
import torch
from . import torch_functional as VF


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
        max_prompt_length: int = 2048,
        max_steps: int = 10000,
        truncation: str = "error",
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.system_prompt = system_prompt or ""
        self.prompt_key = prompt_key
        self.answer_key = answer_key
        self.max_prompt_length = max_prompt_length
        self.max_steps = max_steps
        self.global_step = 0
        self.truncation = truncation

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
        row_dict = random.choice(
            self.musique) if use_musique else random.choice(self.drop)

        messages = [{"role": "user", "content": row_dict[self.prompt_key]}]
        if self.system_prompt:
            messages.insert(
                0, {"role": "system", "content": self.system_prompt})

        prompt = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False)

        model_inputs = self.tokenizer(
            [prompt], add_special_tokens=False, return_tensors="pt")
        input_ids = model_inputs.pop("input_ids")[0]
        attention_mask = model_inputs.pop("attention_mask")[0]
        position_ids = torch.clip(attention_mask.cumsum(
            dim=0) - 1, min=0, max=None)  # (seq_length,)

        input_ids, attention_mask, position_ids = VF.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        row_dict["input_ids"] = input_ids
        row_dict["attention_mask"] = attention_mask
        row_dict["position_ids"] = position_ids
        row_dict["raw_prompt_ids"] = self.tokenizer.encode(
            prompt, add_special_tokens=False)
        row_dict["ground_truth"] = row_dict.pop(self.answer_key)
        row_dict["context"] = row_dict[self.prompt_key]
        return row_dict
