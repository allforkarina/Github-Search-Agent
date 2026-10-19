"""Offline metrics; labels must come from an independent benchmark."""
import math


def retrieval_metrics(relevance: dict[str, int], retrieved: list[str], k: int = 5) -> dict:
    if k <= 0 or any(value < 0 for value in relevance.values()):
        raise ValueError("k must be positive and relevance must be nonnegative")
    ranking = list(dict.fromkeys(retrieved))[:k]
    relevant = {key for key, grade in relevance.items() if grade > 0}
    hits = relevant & set(ranking)
    reciprocal = next((1 / rank for rank, key in enumerate(ranking, 1) if key in relevant), 0.0)
    dcg = sum((2 ** relevance.get(key, 0) - 1) / math.log2(rank + 1)
              for rank, key in enumerate(ranking, 1))
    ideal = sum((2 ** grade - 1) / math.log2(rank + 1)
                for rank, grade in enumerate(sorted(relevance.values(), reverse=True)[:k], 1))
    return {"recall_at_k": len(hits) / len(relevant) if relevant else None,
            "mrr_at_k": reciprocal if relevant else None,
            "ndcg_at_k": dcg / ideal if ideal else None,
            "hit_rate_at_k": float(bool(hits)) if relevant else None}


def judgment_metrics(expected: list[str], predicted: list[str]) -> dict:
    labels = ("satisfied", "not_satisfied", "uncertain")
    if not expected or len(expected) != len(predicted) or any(value not in labels for value in expected + predicted):
        raise ValueError("aligned nonempty judgments with supported labels are required")
    scores = []
    for label in labels:
        tp = sum(gold == label and got == label for gold, got in zip(expected, predicted))
        fp = sum(gold != label and got == label for gold, got in zip(expected, predicted))
        fn = sum(gold == label and got != label for gold, got in zip(expected, predicted))
        if tp + fp + fn:
            scores.append(2 * tp / (2 * tp + fp + fn))
    return {"accuracy": sum(a == b for a, b in zip(expected, predicted)) / len(expected),
            "macro_f1": sum(scores) / len(scores)}


def citation_precision(support_labels: list[bool]) -> float | None:
    """Use human support labels, not just successful URL/ID validation."""
    if any(type(value) is not bool for value in support_labels):
        raise ValueError("independently reviewed boolean support labels are required")
    return sum(support_labels) / len(support_labels) if support_labels else None
