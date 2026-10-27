"""Scoring shared by our model and pix2tex, so both columns of the benchmark use the
exact same tokenizer and metrics. Pure python (nltk only) -- no torch import."""
import math

from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu

COVERAGES = (1.0, 0.8, 0.6, 0.4, 0.2)


def edit_distance(a, b):
    """Token-level Levenshtein, for ExpRate<=k near-miss buckets."""
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev = cur
    return dp[n]


def score(refs, hyps):
    """refs/hyps: lists of token lists -> BLEU + ExpRate buckets."""
    dists = [edit_distance(r, h) for r, h in zip(refs, hyps)]
    n = max(1, len(hyps))
    return {
        "n": len(hyps),
        "bleu": corpus_bleu([[r] for r in refs], hyps,
                            smoothing_function=SmoothingFunction().method4),
        "exprate": sum(d == 0 for d in dists) / n,
        "exprate_le1": sum(d <= 1 for d in dists) / n,
        "exprate_le2": sum(d <= 2 for d in dists) / n,
    }


def selective(correct, confidences, coverages=COVERAGES):
    """Accuracy when only the most-confident fraction of predictions is kept.
    Rising accuracy as coverage drops = the confidence score is informative."""
    ranked = [c for _, c in sorted(zip(confidences, correct), key=lambda x: -x[0])]
    out = {}
    for cov in coverages:
        k = max(1, math.ceil(cov * len(ranked)))
        out[f"exprate@{int(cov * 100)}%"] = sum(ranked[:k]) / k
    right = [p for p, ok in zip(confidences, correct) if ok]
    wrong = [p for p, ok in zip(confidences, correct) if not ok]
    out["mean_conf_correct"] = sum(right) / max(1, len(right))
    out["mean_conf_wrong"] = sum(wrong) / max(1, len(wrong))
    return out
