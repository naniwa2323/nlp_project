"""PLACEHOLDER model so Member 4 can develop/test before Member 2 delivers model.joblib.

Trains on Member 1's data/processed/{train,test}_chunks.csv with the configuration from
Member 2's spec (char TF-IDF 2-5gram + balanced LogisticRegression).
This is NOT the team's official model. Replace models/model.joblib with Member 2's
and delete models/model.PLACEHOLDER.

Usage:
  python src/preprocess.py                       # Member 1 (if data/processed is missing)
  python placeholders/train_placeholder_model.py
"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
SEED = 42


def main():
    tr = pd.read_csv(DATA / "train_chunks.csv")
    te = pd.read_csv(DATA / "test_chunks.csv")
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(2, 5), max_features=50000)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)),
    ])
    print(f"train chunks: {len(tr)}  test chunks: {len(te)}")
    pipe.fit(tr.text, tr.label)
    print(classification_report(te.label, pipe.predict(te.text), digits=3))

    out = ROOT / "models" / "model.joblib"
    out.parent.mkdir(exist_ok=True)
    joblib.dump(pipe, out)
    out.with_suffix(".PLACEHOLDER").write_text("This model.joblib is a placeholder, not Member 2's.\n")
    print("saved", out)


if __name__ == "__main__":
    main()
