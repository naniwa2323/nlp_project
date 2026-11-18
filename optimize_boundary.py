
"""Member 3: Optimize segmentation parameters on validation articles only."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from segmenter import (
    chunk_spans,
    chunk_scores,
    char_scores,
    smooth,
    char_labels_from_segments,
    true_char_labels,
    _runs,
)

THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]
MIN_LENS = [0, 50, 100, 150, 200]

MODEL_PATH = "results/member3_validation/model.joblib"
ARTICLES_PATH = "results/member3_validation/validation_articles.csv"
OUTPUT_DIR = Path("results/member3_validation/optimization")


def prepare_articles(model, articles):
    """Compute character AI probabilities once per article."""
    prepared = []

    for row in articles.itertuples(index=False):
        article = row.context
        n = len(article)

        if n == 0:
            continue

        spans = chunk_spans(n)
        texts = [article[s:e] for s, e in spans]
        scores = chunk_scores(texts, model)
        cs = char_scores(n, spans, scores)

        actual = true_char_labels(
            n, int(row.isfake), int(row.nchar_real)
        )

        prepared.append({
            "id": row.id,
            "isfake": int(row.isfake),
            "true_boundary": (
                int(row.nchar_real) if row.isfake == 1 else None
            ),
            "char_scores": cs,
            "actual": actual,
        })

    return prepared


def evaluate_params(prepared, threshold, min_len):
    accuracies = []
    correct_types = 0
    mixed_count = 0
    detected_count = 0
    boundary_errors = []

    for item in prepared:
        cs = item["char_scores"]
        actual = item["actual"]

        labels = smooth(
            (cs >= threshold).astype(int),
            min_len
        )

        accuracies.append(float(np.mean(labels == actual)))

        unique_labels = set(labels.tolist())
        if unique_labels == {0}:
            predicted_type = 0
        elif unique_labels == {1}:
            predicted_type = 2
        else:
            predicted_type = 1

        if predicted_type == item["isfake"]:
            correct_types += 1

        if item["isfake"] == 1:
            mixed_count += 1

            runs = _runs(labels)
            candidates = [
                s
                for i, (s, e, lab) in enumerate(runs)
                if i > 0
                and runs[i - 1][2] == 0
                and lab == 1
            ]

            if candidates:
                detected_count += 1
                true_boundary = item["true_boundary"]
                best = min(
                    candidates,
                    key=lambda x: abs(x - true_boundary)
                )
                boundary_errors.append(abs(best - true_boundary))

    n = len(prepared)

    return {
        "threshold": threshold,
        "min_len": min_len,
        "mean_article_character_accuracy": float(np.mean(accuracies)),
        "article_type_accuracy": correct_types / n,
        "boundary_detection_rate": (
            detected_count / mixed_count if mixed_count else None
        ),
        "boundary_mae_characters_detected_only": (
            float(np.mean(boundary_errors))
            if boundary_errors else None
        ),
        "detected_mixed_articles": detected_count,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading validation model...")
    model = joblib.load(MODEL_PATH)

    print("Loading validation articles...")
    articles = pd.read_csv(ARTICLES_PATH, dtype={"id": str})

    print("Preparing character probabilities...")
    prepared = prepare_articles(model, articles)
    print("Prepared articles:", len(prepared))

    results = []

    for threshold in THRESHOLDS:
        for min_len in MIN_LENS:
            metrics = evaluate_params(
                prepared, threshold, min_len
            )
            results.append(metrics)

            print(
                f"threshold={threshold:.1f}, "
                f"min_len={min_len:3d}, "
                f"char_acc={metrics['mean_article_character_accuracy']:.4f}, "
                f"type_acc={metrics['article_type_accuracy']:.4f}"
            )

    df = pd.DataFrame(results)

    # Primary objective: maximize mean per-article character accuracy.
    # Tie-breaker: maximize article type accuracy.
    ranked = df.sort_values(
        ["mean_article_character_accuracy", "article_type_accuracy"],
        ascending=[False, False],
        kind="stable",
    )

    best = ranked.iloc[0].to_dict()
    best["min_len"] = int(best["min_len"])
    best["detected_mixed_articles"] = int(
        best["detected_mixed_articles"]
    )

    df.to_csv(OUTPUT_DIR / "parameter_results.csv", index=False)

    with open(
        OUTPUT_DIR / "best_parameters.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(best, f, ensure_ascii=False, indent=2)

    print("\n===== Best Parameters =====")
    print(json.dumps(best, ensure_ascii=False, indent=2))
    print("Saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
