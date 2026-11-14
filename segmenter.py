"""Region segmentation: label every part of an article as Human (0) or AI (1).

Spec change (2026-11-14): instead of finding only the FIRST Human->AI switch,
we decide for every character whether it is human-written or AI-generated,
and return the resulting regions. An article may contain several regions.

Method
1. Split into 100-char chunks, stride 50 (same as Member 1).
2. Get P(AI) for every chunk from Member 2's model.
3. Character score = mean P(AI) of all chunks covering that character
   (with stride 50 each 50-char block is covered by up to two chunks).
4. score >= threshold -> AI, else Human.
5. Optional smoothing: regions shorter than `min_len` chars are absorbed
   by their neighbours (removes isolated one-chunk mistakes).
"""
from __future__ import annotations

import numpy as np

CHUNK_SIZE = 100
STRIDE = 50
HUMAN, AI = 0, 1


def chunk_spans(n: int, size: int = CHUNK_SIZE, stride: int = STRIDE):
    if n == 0:
        return []
    if n < size:
        return [(0, n)]
    return [(s, s + size) for s in range(0, n - size + 1, stride)]


def chunk_scores(texts, model):
    """P(AI) per chunk; falls back to hard 0/1 labels if the model has no predict_proba."""
    if hasattr(model, "predict_proba"):
        classes = list(model.classes_)
        return model.predict_proba(texts)[:, classes.index(AI)].astype(float)
    return np.asarray(model.predict(texts), dtype=float)


def char_scores(n: int, spans, scores) -> np.ndarray:
    total = np.zeros(n)
    count = np.zeros(n)
    for (s, e), p in zip(spans, scores):
        total[s:e] += p
        count[s:e] += 1
    # characters after the last full chunk: use the last chunk's score
    if spans and spans[-1][1] < n:
        total[spans[-1][1] :] = scores[-1]
        count[spans[-1][1] :] = 1
    return total / np.maximum(count, 1)


def _runs(labels: np.ndarray):
    """[(start, end, label)] for consecutive equal labels."""
    if len(labels) == 0:
        return []
    cut = np.flatnonzero(np.diff(labels)) + 1
    starts = np.r_[0, cut]
    ends = np.r_[cut, len(labels)]
    return [(int(s), int(e), int(labels[s])) for s, e in zip(starts, ends)]


def smooth(labels: np.ndarray, min_len: int) -> np.ndarray:
    """Flip regions shorter than min_len (shortest first) until none remain."""
    labels = labels.copy()
    if min_len <= 0:
        return labels
    while True:
        runs = _runs(labels)
        if len(runs) <= 1:
            return labels
        s, e, lab = min(runs, key=lambda r: r[1] - r[0])
        if e - s >= min_len:
            return labels
        labels[s:e] = 1 - lab


def segment_article(article: str, model, threshold: float = 0.5, min_len: int = 0):
    """Return list of dicts: {start, end, label, score} covering the whole article."""
    n = len(article)
    spans = chunk_spans(n)
    if not spans:
        return []
    scores = chunk_scores([article[s:e] for s, e in spans], model)
    cs = char_scores(n, spans, scores)
    labels = smooth((cs >= threshold).astype(int), min_len)
    return [
        {"start": s, "end": e, "label": lab, "score": float(cs[s:e].mean())}
        for s, e, lab in _runs(labels)
    ]


def char_labels_from_segments(segments, n: int) -> np.ndarray:
    out = np.zeros(n, dtype=int)
    for seg in segments:
        out[seg["start"] : seg["end"]] = seg["label"]
    return out


def true_char_labels(n: int, isfake: int, nchar_real: int | None) -> np.ndarray:
    """Ground truth per character from the dataset labels."""
    if isfake == 0:
        return np.zeros(n, dtype=int)
    if isfake == 2:
        return np.ones(n, dtype=int)
    out = np.zeros(n, dtype=int)
    out[int(nchar_real) :] = 1
    return out
