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

import re


def retrieve_format_reward(predict_str: str) -> float:
    """
    Check if the prediction has the required format structure:
    <think>...</think> followed by Answer: ...
    Within the <think>...</think> part, there needs to exist one or more <retrieval>...</retrieval> blocks.

    Args:
        predict_str: The prediction string to evaluate

    Returns:
        1.0 if the format is correct, 0.0 otherwise
    """
    try:
        # Strip whitespace to ensure consistent matching
        predict_str = predict_str.strip()

        # Check for overall structure: <think>...</think> followed by Answer:
        overall_pattern = re.compile(
            r"<think>.*</think>.*Answer:.*", re.DOTALL)
        if not overall_pattern.fullmatch(predict_str):
            return 0.0

        # Extract content inside <think>...</think>
        think_pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
        think_match = think_pattern.search(predict_str)

        if not think_match:
            return 0.0

        # Check for at least one <retrieval>...</retrieval> block in think content
        think_content = think_match.group(1)
        retrieval_pattern = re.compile(
            r"<retrieval>.*?</retrieval>", re.DOTALL)
        retrieval_matches = retrieval_pattern.findall(think_content)

        if not retrieval_matches:
            return 0.0

        return 1.0
    except Exception:
        # Handle any unexpected errors
        return 0.0


def extract_answer(predict_str: str) -> str:
    """
    Extract the answer from the prediction string, looking for content after "Answer:".

    Args:
        predict_str: The prediction string to evaluate

    Returns:
        The extracted answer or an empty string if not found
    """
    answer_match = re.search(r"Answer:(.*)", predict_str, re.DOTALL)
    if answer_match:
        return answer_match.group(1).strip()
    return ""


def retrieve_accuracy_reward(predict_str: str, ground_truth: str) -> float:
    """
    Check if the answer (After "Answer:") matches the ground truth.

    Args:
        predict_str: The prediction string to evaluate
        ground_truth: The ground truth answer

    Returns:
        1.0 if the answer matches the ground truth, 0.0 otherwise
    """
    answer = extract_answer(predict_str)

    # Normalize both answers for comparison (lowercase, strip spaces)
    answer_norm = answer.lower().strip()
    ground_truth_norm = ground_truth.lower().strip()

    # Check if the normalized answer matches the ground truth
    return 1.0 if answer_norm == ground_truth_norm else 0.0


def retrieval_spans_in_context(predict_str: str, context: str) -> float:
    """
    Check if all retrieval spans in the prediction are found in the context.

    Args:
        predict_str: The prediction string to evaluate
        context: The context string to search in

    Returns:
        1.0 if all retrieval spans are in the context, 0.0 otherwise
    """
    # Extract all retrieval spans
    spans = re.findall(r"<retrieval>(.*?)</retrieval>", predict_str, re.DOTALL)

    # If no retrieval spans were found, return 0.0
    if not spans:
        return 0.0

    # Check if all spans are in the context
    spans_found = 0
    for span in spans:
        # Clean up the span by removing extra whitespace
        cleaned_span = re.sub(r'\s+', ' ', span).strip()
        if not cleaned_span:
            continue
        if cleaned_span in context:
            spans_found += 1

    # Return a score based on the proportion of spans found
    if not spans:
        return 0.0
    return min(1.0, spans_found / len([s for s in spans if s.strip()]))


def retrieve_compute_score(predict_str: str, ground_truth: str, context: str) -> float:
    """
    Compute the combined score for retrieval-based QA evaluation.

    Args:
        predict_str: The prediction string to evaluate
        ground_truth: The ground truth answer
        context: The context from which retrieval should happen

    Returns:
        The combined reward score between 0.0 and 1.0
    """
    # Calculate individual reward components
    format_score = retrieve_format_reward(predict_str)
    accuracy_score = retrieve_accuracy_reward(predict_str, ground_truth)
    retrieval_score = retrieval_spans_in_context(predict_str, context)

    # Combine scores with weights (similar to math.py's weighting)
    return 0.7 * accuracy_score + 0.1 * format_score + 0.2 * retrieval_score
