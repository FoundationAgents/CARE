# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
CustomRewardManager

This class assigns reward functions based on `compute_score`.

⚠️ In CARE we only use `retrieve` as the reward function.
Other options (`math`, `r1v`) are part of the base framework but not used in CARE.
"""

import torch
from transformers import PreTrainedTokenizer

from ...protocol import DataProto
from ...utils.reward_score import math_compute_score, r1v_compute_score, retrieve_compute_score


class CustomRewardManager:
    def __init__(self, tokenizer: PreTrainedTokenizer, num_examine: int, compute_score: str):
        """
        Args:
            tokenizer: Hugging Face tokenizer for decoding text.
            num_examine: Number of examples to print for inspection.
            compute_score: Which reward function to use.
                - "math":   base framework option (not used in CARE)
                - "r1v":    base framework option (not used in CARE)
                - "retrieve": the only reward function used in CARE
        """
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        if compute_score == "math":
            self.compute_score = math_compute_score
        elif compute_score == "r1v":
            self.compute_score = r1v_compute_score
        elif compute_score == "retrieve":
            self.compute_score = retrieve_compute_score
        else:
            raise NotImplementedError()

    def __call__(self, data: DataProto) -> torch.Tensor:
        """
        Compute reward tensor for a batch of data.

        - Iterates over each DataProtoItem.
        - Decodes prompt and response text.
        - Compares response with ground truth (and context if required).
        - Places the reward score at the last token position of each response.

        In CARE release: only the "retrieve" reward function is expected.
        """
        reward_tensor = torch.zeros_like(
            data.batch["responses"], dtype=torch.float32)
        already_print = 0

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            # Extract prompt tokens
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            # Extract response tokens
            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # Decode strings
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=False)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=False)
            ground_truth = data_item.non_tensor_batch["ground_truth"]

            # Retrieve reward requires context (CARE case)
            if hasattr(self.compute_score, "__code__") and "context" in self.compute_score.__code__.co_varnames:
                context = data_item.non_tensor_batch.get("context", "")
                score = self.compute_score(response_str, ground_truth, context)
            else:
                score = self.compute_score(response_str, ground_truth)

            # Assign score at the last valid response token
            reward_tensor[i, valid_response_length - 1] = score

            # Print a few examples for inspection
            if already_print < self.num_examine:
                already_print += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                print("[score]", score)

        return reward_tensor
