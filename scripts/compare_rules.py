"""Compare the old rule (first Human->AI switch) with region segmentation on the test split.

Usage: python scripts/compare_rules.py
Writes results/compare_rules.csv and prints a table.

Metrics (per article, then averaged):
  char_acc      fraction of characters labelled correctly (human/AI)
  verdict_acc   article type correct (all-human / mixed / all-AI)
  start_mae     |predicted AI start - nchar_real| for isfake=1 articles predicted as mixed
NOTE: the placeholder model and any setting chosen from this table are evaluated on the
test split; final settings should be chosen on a validation split.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline import load_model  # noqa: E402
from segmenter import (  # noqa: E402
    _runs,
    char_scores,
    chunk_scores,
    chunk_spans,
    smooth,
    true_char_labels,
)

MIN_LENS = [0, 100, 150, 200]


def old_rule(n, spans, chunk_lab):
    """First 0->1 switch; boundary = midpoint of the two chunk centres."""
    for i in range(len(chunk_lab) - 1):
        if chunk_lab[i] == 0 and chunk_lab[i + 1] == 1:
            b = int(((spans[i][0] + spans[i][1]) / 2 + (spans[i + 1][0] + spans[i + 1][1]) / 2) / 2)
            out = np.zeros(n, dtype=int)
            out[b:] = 1
            return out
    return np.full(n, 1 if all(chunk_lab) else 0, dtype=int)


def verdict(labels):
    if labels.min() == labels.max():
        return 2 if labels[0] == 1 else 0
    return 1


def first_ai_start(labels):
    runs = _runs(labels)
    if len(runs) >= 2 and runs[0][2] == 0:
        return runs[1][0]
    return None


def main():
    model = load_model()
    te = pd.read_csv(ROOT / "data" / "processed" / "test_articles.csv", dtype={"id": str})
    te = te[te.context.str.len() >= 100]

    # score all chunks in one batch
    spans_all, texts, owner = [], [], []
    for k, t in enumerate(te.context):
        sp = chunk_spans(len(t))
        spans_all.append(sp)
        texts += [t[s:e] for s, e in sp]
        owner += [k] * len(sp)
    probs = chunk_scores(texts, model)
    owner = np.array(owner)

    rows = []
    for k, r in enumerate(te.itertuples()):
        n = len(r.context)
        sp = spans_all[k]
        p = probs[owner == k]
        truth = true_char_labels(n, r.isfake, r.nchar_real)
        cs = char_scores(n, sp, p)
        preds = {"old: first switch": old_rule(n, sp, (p >= 0.5).astype(int))}
        for m in MIN_LENS:
            preds[f"new: regions (min_len={m})"] = smooth((cs >= 0.5).astype(int), m)
        for name, lab in preds.items():
            st = first_ai_start(lab)
            rows.append({
                "method": name,
                "isfake": r.isfake,
                "char_acc": (lab == truth).mean(),
                "verdict_ok": verdict(lab) == r.isfake,
                "n_regions": len(_runs(lab)),
                "start_err": abs(st - r.nchar_real) if (r.isfake == 1 and st is not None) else np.nan,
            })
    df = pd.DataFrame(rows)
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    df.to_csv(out_dir / "compare_rules_per_article.csv", index=False)

    summary = df.groupby("method", sort=False).agg(
        char_acc=("char_acc", "mean"),
        verdict_acc=("verdict_ok", "mean"),
        start_mae=("start_err", "mean"),
        avg_regions=("n_regions", "mean"),
    )
    by_class = df.pivot_table(index="method", columns="isfake", values="char_acc", sort=False)
    by_class.columns = [f"char_acc_isfake{c}" for c in by_class.columns]
    summary = summary.join(by_class).round(3)
    summary.to_csv(out_dir / "compare_rules.csv")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(f"test articles: {len(te)}")
        print(summary)


if __name__ == "__main__":
    main()
