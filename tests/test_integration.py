"""Integration tests for Member 4's checklist (region-segmentation spec).

Run: python -m pytest tests -q
Needs models/model.joblib (Member 2) or models/placeholder_model.joblib and data/processed/test_articles.csv (Member 1).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline import AI, HUMAN, analyze_article, load_model  # noqa: E402
from segmenter import _runs, smooth  # noqa: E402

TEST = pd.read_csv(ROOT / "data" / "processed" / "test_articles.csv", dtype={"id": str})


def _one(isfake):
    return TEST[(TEST.isfake == isfake) & (TEST.context.str.len() > 300)].iloc[0].context


def _check_regions(res):
    # regions tile the whole text without gaps/overlaps, neighbours differ
    assert res.regions[0]["start"] == 0 and res.regions[-1]["end"] == len(res.text)
    for a, b in zip(res.regions, res.regions[1:]):
        assert a["end"] == b["start"] and a["label"] != b["label"]
    assert "".join(s for s, _ in res.segments()) == res.text
    if res.min_len:
        assert len(res.regions) == 1 or all(r["end"] - r["start"] >= res.min_len for r in res.regions)


def test_model_loads():
    assert hasattr(load_model(), "predict")


def test_three_article_types():
    for isfake in (0, 1, 2):
        res = analyze_article(_one(isfake))
        _check_regions(res)
        assert res.verdict in {"human", "mixed", "ai"}


def test_no_smoothing_allows_short_regions():
    res = analyze_article(_one(1), min_len=0)
    _check_regions(res)


def test_smooth_absorbs_short_region():
    lab = np.array([0] * 200 + [1] * 50 + [0] * 200)
    assert _runs(smooth(lab, 100)) == [(0, 450, 0)]
    lab = np.array([0] * 200 + [1] * 300)
    assert _runs(smooth(lab, 100)) == [(0, 200, 0), (200, 500, 1)]


def test_multiple_ai_regions_possible():
    """Human -> AI -> Human must be representable (old spec could not)."""
    lab = np.array([0] * 200 + [1] * 200 + [0] * 200)
    assert [r[2] for r in _runs(smooth(lab, 100))] == [HUMAN, AI, HUMAN]


def test_edge_cases():
    assert analyze_article("").verdict == "empty"
    short = analyze_article("短い文章です。")
    assert short.warnings and len(short.chunks) == 1
    _check_regions(short)


def test_placeholder_used_only_without_member2_model():
    import pipeline

    if pipeline.MODEL_PATH.exists():
        assert pipeline.model_path() == pipeline.MODEL_PATH
        assert "(Member 2)" in pipeline.component_info()["model"]
    else:
        assert pipeline.model_path() == pipeline.PLACEHOLDER_PATH
        assert "(placeholder)" in pipeline.component_info()["model"]
