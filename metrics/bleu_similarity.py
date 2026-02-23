"""BLEU score and sentence similarity metrics for text evaluation."""

import torch
from collections import Counter


def compute_bleu(reference: list, hypothesis: list, max_n: int = 4) -> float:
    """Compute corpus-level BLEU score (simplified).

    Args:
        reference: List of reference sentences (list of token lists).
        hypothesis: List of hypothesis sentences (list of token lists).
        max_n: Maximum n-gram order.
    """
    import math

    clipped_counts = [0] * max_n
    total_counts = [0] * max_n
    ref_len = 0
    hyp_len = 0

    for ref, hyp in zip(reference, hypothesis):
        ref_len += len(ref)
        hyp_len += len(hyp)

        for n in range(1, max_n + 1):
            ref_ngrams = Counter()
            for i in range(len(ref) - n + 1):
                ref_ngrams[tuple(ref[i:i + n])] += 1

            hyp_ngrams = Counter()
            for i in range(len(hyp) - n + 1):
                hyp_ngrams[tuple(hyp[i:i + n])] += 1

            for ng, count in hyp_ngrams.items():
                clipped_counts[n - 1] += min(count, ref_ngrams.get(ng, 0))
                total_counts[n - 1] += count

    # Compute BLEU
    if hyp_len == 0:
        return 0.0

    brevity_penalty = min(1.0, math.exp(1.0 - ref_len / hyp_len))

    log_bleu = 0.0
    for n in range(max_n):
        if total_counts[n] == 0 or clipped_counts[n] == 0:
            return 0.0
        log_bleu += (1.0 / max_n) * math.log(clipped_counts[n] / total_counts[n])

    return brevity_penalty * math.exp(log_bleu)


def compute_sentence_similarity(ref_tokens: list, hyp_tokens: list) -> float:
    """Compute simple word-overlap similarity between sentences.

    Args:
        ref_tokens: Reference token list.
        hyp_tokens: Hypothesis token list.

    Returns:
        Proportion of reference tokens found in hypothesis.
    """
    if len(ref_tokens) == 0:
        return 1.0 if len(hyp_tokens) == 0 else 0.0
    ref_set = set(ref_tokens)
    hyp_set = set(hyp_tokens)
    return len(ref_set & hyp_set) / len(ref_set)
