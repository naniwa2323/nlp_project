"""Integration layer (Member 4).

`analyze_article(text)` is the single entry point used by the Streamlit app.

Spec (updated 2026-11-14): label EVERY region of the article as Human or AI
(an article may switch several times), instead of only the first Human->AI switch.

Components
  - Member 2's trained Pipeline -> models/model.joblib (TF-IDF + LogisticRegression)
  - Region segmentation         -> boundary_detector.segment_article(article, model, ...)
                                   if Member 3 provides it, otherwise segmenter.py
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import joblib

from segmenter import AI, CHUNK_SIZE, HUMAN, STRIDE, chunk_scores, chunk_spans  # noqa: F401

ROOT = Path(__file__).resolve().parent
MODEL_PATH = Path(os.environ.get("MODEL_PATH", ROOT / "models" / "model.joblib"))

DEFAULT_THRESHOLD = 0.5
DEFAULT_MIN_LEN = 100  # see results/compare_rules.csv


def _load_segmenter():
    try:
        from boundary_detector import segment_article  # Member 3 (new interface)

        return segment_article, "boundary_detector.py (Member 3)"
    except ImportError:
        from segmenter import segment_article

        return segment_article, "segmenter.py (Member 4)"


@lru_cache(maxsize=1)
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\n"
            "Put Member 2's model.joblib in models/, or build the placeholder:\n"
            "  python placeholders/train_placeholder_model.py"
        )
    return joblib.load(MODEL_PATH)


def component_info() -> dict:
    _, seg_name = _load_segmenter()
    placeholder_model = MODEL_PATH.with_suffix(".PLACEHOLDER").exists()
    return {
        "model": f"{MODEL_PATH.name}" + (" (placeholder)" if placeholder_model else " (Member 2)"),
        "segmenter": seg_name,
    }


def make_chunks(text: str):
    """[(start, end, chunk_text)] — complete 100-char windows, stride 50 (Member 1's rule)."""
    return [(s, e, text[s:e]) for s, e in chunk_spans(len(text))]


@dataclass
class AnalysisResult:
    text: str
    regions: list = field(default_factory=list)  # [{start, end, label, score}]
    chunks: list = field(default_factory=list)  # [(start, end, text)]
    ai_proba: list = field(default_factory=list)  # P(AI) per chunk
    threshold: float = DEFAULT_THRESHOLD
    min_len: int = DEFAULT_MIN_LEN
    verdict: str = ""  # "human" | "ai" | "mixed" | "empty"
    warnings: list = field(default_factory=list)

    @property
    def chunk_labels(self):
        return [int(p >= self.threshold) for p in self.ai_proba]

    @property
    def ai_chars(self) -> int:
        return sum(r["end"] - r["start"] for r in self.regions if r["label"] == AI)

    @property
    def ai_regions(self):
        return [r for r in self.regions if r["label"] == AI]

    def segments(self):
        """[(text, label)] for highlighting."""
        return [(self.text[r["start"] : r["end"]], r["label"]) for r in self.regions]


def analyze_article(
    text: str, threshold: float = DEFAULT_THRESHOLD, min_len: int = DEFAULT_MIN_LEN
) -> AnalysisResult:
    text = text or ""
    res = AnalysisResult(text=text, threshold=threshold, min_len=min_len)
    if not text.strip():
        res.verdict = "empty"
        res.warnings.append("Please enter some text.")
        return res

    model = load_model()
    segment_article, _ = _load_segmenter()

    res.chunks = make_chunks(text)
    if len(text) < CHUNK_SIZE:
        res.warnings.append(
            f"Text is shorter than {CHUNK_SIZE} characters, so it was judged as a single chunk (less reliable)."
        )
    res.ai_proba = [float(p) for p in chunk_scores([c[2] for c in res.chunks], model)]
    res.regions = segment_article(text, model, threshold=threshold, min_len=min_len)

    labels = {r["label"] for r in res.regions}
    res.verdict = "mixed" if len(labels) > 1 else ("ai" if labels == {AI} else "human")
    return res
