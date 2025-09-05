import re
import difflib


def retrieve_format_reward(predict_str: str) -> float:
    predict_str = predict_str.strip()
    overall_pattern = re.compile(r"<think>.*</think>.*Answer:.*", re.DOTALL)
    if not overall_pattern.fullmatch(predict_str):
        return 0.0

    think_match = re.search(r"<think>(.*?)</think>", predict_str, re.DOTALL)
    if not think_match:
        return 0.0

    think_content = think_match.group(1)
    retrieval_matches = re.findall(
        r"<retrieval>.*?</retrieval>", think_content, re.DOTALL)

    return 1.0 if retrieval_matches else 0.0


def extract_answer(predict_str: str) -> str:
    answer_match = re.search(r"Answer:(.*)", predict_str, re.DOTALL)
    if answer_match:
        return answer_match.group(1).strip()
    return ""


def retrieve_accuracy_reward(predict_str: str, ground_truth: str) -> float:
    answer = extract_answer(predict_str)
    # Use F1 score for accuracy instead of exact match
    if not answer:
        return 0.0
    # F1 score calculation
    answer_tokens = set(answer.split())
    ground_truth_tokens = set(ground_truth.split())
    common_tokens = answer_tokens.intersection(ground_truth_tokens)
    if not common_tokens:
        return 0.0
    precision = len(common_tokens) / len(answer_tokens)
    recall = len(common_tokens) / len(ground_truth_tokens)
    if precision + recall == 0:
        return 0.0
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score


def soft_contains(span: str, context: str, threshold: float = 0.85) -> bool:
    """
    Check if the span loosely appears in the context using similarity.
    """
    span = span.lower()
    context = context.lower()

    context_words = context.split()
    span_words = span.split()

    if len(span_words) == 0:
        return False

    for i in range(len(context_words) - len(span_words) + 1):
        window = " ".join(context_words[i:i+len(span_words)])
        ratio = difflib.SequenceMatcher(None, window, span).ratio()
        if ratio >= threshold:
            return True
    return False


def retrieval_spans_in_context(predict_str: str, context: str, log=False) -> float:
    spans = re.findall(r"<retrieval>(.*?)</retrieval>", predict_str, re.DOTALL)
    clean_spans = [re.sub(r'\s+', ' ', s).strip() for s in spans if s.strip()]

    if log:
        print(f"[Debug] Retrieval spans: {clean_spans}")

    if not clean_spans:
        return 0.2  # baseline reward for no retrieval spans (instead of 0.0)

    spans_found = 0
    for span in clean_spans:
        if span in context:
            spans_found += 1
        elif soft_contains(span, context):
            spans_found += 0.5  # partial match reward

    return min(1.0, spans_found / len(clean_spans))


def retrieve_compute_score(predict_str: str, ground_truth: str, context: str, log=False) -> float:
    format_score = retrieve_format_reward(predict_str)
    accuracy_score = retrieve_accuracy_reward(predict_str, ground_truth)
    retrieval_score = retrieval_spans_in_context(predict_str, context, log=log)

    # Combined reward: weighted sum
    return 0.7 * accuracy_score + 0.1 * format_score + 0.2 * retrieval_score
