
"""Member 3: Article segmentation and boundary evaluation."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from boundary_detector import segment_article
from segmenter import char_labels_from_segments, true_char_labels


def evaluate(model, articles, threshold, min_len):
    records = []

    for row in articles.itertuples(index=False):
        article = row.context
        n = len(article)

        if n == 0:
            continue

        # Member 3's predicted segments
        segments = segment_article(
            article, model,
            threshold=threshold,
            min_len=min_len,
        )

        predicted = char_labels_from_segments(segments, n)
        actual = true_char_labels(
            n, int(row.isfake), int(row.nchar_real)
        )

        # Character-level accuracy
        char_accuracy = float(np.mean(predicted == actual))

        # Article type: Human / Mixed / AI
        labels = set(predicted.tolist())
        if labels == {0}:
            predicted_type = 0
        elif labels == {1}:
            predicted_type = 2
        else:
            predicted_type = 1

        # Boundary detection for mixed articles
        true_boundary = int(row.nchar_real) if row.isfake == 1 else None
        detected = False
        predicted_boundary = None
        boundary_error = None

        if row.isfake == 1:
            # Human -> AI transitions
            candidates = [
                seg["start"]
                for i, seg in enumerate(segments)
                if i > 0
                and segments[i - 1]["label"] == 0
                and seg["label"] == 1
            ]

            if candidates:
                predicted_boundary = min(
                    candidates,
                    key=lambda x: abs(x - true_boundary)
                )
                detected = True
                boundary_error = abs(predicted_boundary - true_boundary)

        records.append({
            "id": row.id,
            "isfake": int(row.isfake),
            "predicted_type": predicted_type,
            "char_accuracy": char_accuracy,
            "true_boundary": true_boundary,
            "predicted_boundary": predicted_boundary,
            "boundary_detected": detected,
            "boundary_error": boundary_error,
            "num_segments": len(segments),
        })

    return pd.DataFrame(records)


def summarize(df, threshold, min_len):
    mixed = df[df["isfake"] == 1]
    detected = mixed[mixed["boundary_detected"]]

    return {
        "evaluated_articles": len(df),
        "mean_article_character_accuracy": float(df["char_accuracy"].mean()),
        "article_type_accuracy": float(
            (df["isfake"] == df["predicted_type"]).mean()
        ),
        "mixed_articles": len(mixed),
        "detected_mixed_articles": len(detected),
        "boundary_detection_rate": (
            float(len(detected) / len(mixed)) if len(mixed) else None
        ),
        "boundary_mae_characters_detected_only": (
            float(detected["boundary_error"].mean())
            if len(detected) else None
        ),
        "threshold": threshold,
        "min_len": min_len,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/model.joblib")
    parser.add_argument("--articles", default="data/processed/test_articles.csv")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-len", type=int, default=100)
    parser.add_argument("--output", default="results/member3")
    args = parser.parse_args()

    model = joblib.load(args.model)
    articles = pd.read_csv(args.articles)

    results = evaluate(
        model, articles,
        threshold=args.threshold,
        min_len=args.min_len,
    )
    metrics = summarize(results, args.threshold, args.min_len)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    results.to_csv(output / "boundary_predictions.csv", index=False)

    with open(output / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
