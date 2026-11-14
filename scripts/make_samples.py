"""Pick demo articles (human / AI / mixed) from the TEST split for the app's sample menu.

Usage: python scripts/make_samples.py
Writes samples/*.txt and samples/answers.csv (ground truth for checking the demo).
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
te = pd.read_csv(ROOT / "data" / "processed" / "test_articles.csv", dtype={"id": str})
te = te[(te.context.str.len() >= 300) & (te.context.str.len() <= 900)]

out = ROOT / "samples"
out.mkdir(exist_ok=True)
picks = [("1_human", 0), ("2_ai", 2), ("3_mixed_a", 1), ("4_mixed_b", 1)]
used, rows = set(), []
for name, label in picks:
    r = te[(te.isfake == label) & (~te.id.isin(used))].sample(1, random_state=7).iloc[0]
    used.add(r.id)
    (out / f"{name}.txt").write_text(r.context, encoding="utf-8")
    rows.append({"sample": name, "id": r.id, "isfake": label, "nchar_real": r.nchar_real, "len": len(r.context)})
pd.DataFrame(rows).to_csv(out / "answers.csv", index=False)
print(pd.DataFrame(rows))
