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
Reward configuration class.

In CARE, we only use `retrieve` as the reward computation method.
Other options (`math`, `r1v`) exist in the base framework, but are not
used in CARE release.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class RewardConfig:
    # The reward type (default: function-based reward)
    reward_type: str = "function"

    # Which reward function to compute
    # In CARE, this should always be set to "retrieve"
    compute_score: str = "retrieve"

    # Valid options (for generality, but CARE only uses "retrieve")
    valid_compute_scores: List[str] = field(
        default_factory=lambda: ["math", "r1v", "retrieve"], repr=False
    )

    def __post_init__(self):
        """
        Validate that the chosen compute_score is in the allowed set.
        For CARE release, enforce using "retrieve".
        """
        if self.compute_score not in self.valid_compute_scores:
            raise ValueError(
                f"compute_score must be one of {self.valid_compute_scores}, got {self.compute_score}"
            )
