"""Train and evaluate Member 2's Japanese AI-generated text chunk classifier.

Performs group-aware hyperparameter validation on train_chunks.csv (grouped by
article ID to prevent chunk-overlap leakage), selects the best char n-gram range
by validation AI-class F1-score, retrains on all training chunks, saves the
sklearn Pipeline to models/model.joblib, and evaluates on test_chunks.csv.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
TRAIN_CHUNKS_PATH = ROOT / "data" / "processed" / "train_chunks.csv"
TEST_CHUNKS_PATH = ROOT / "data" / "processed" / "test_chunks.csv"
MODEL_OUT_PATH = ROOT / "models" / "model.joblib"
RESULTS_DIR = ROOT / "results"
METRICS_OUT_PATH = RESULTS_DIR / "classifier_metrics.json"
CM_OUT_PATH = RESULTS_DIR / "confusion_matrix.png"
ERRORS_OUT_PATH = RESULTS_DIR / "error_analysis.csv"

SEED = 42
NGRAM_CANDIDATES: list[tuple[int, int]] = [(1, 3), (2, 4), (2, 5), (3, 6)]


def load_chunks(path: Path) -> pd.DataFrame:
    """Load chunk CSV and validate required columns and binary integer labels."""
    df = pd.read_csv(path, dtype={"id": str})
    required_cols = {"id", "chunk_index", "start_char", "end_char", "text", "label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")

    if df["text"].isna().any():
        raise ValueError(f"Null text values found in {path}")

    labels = pd.to_numeric(df["label"], errors="coerce")
    if labels.isna().any() or not set(labels.astype(int).unique()).issubset({0, 1}):
        raise ValueError(f"Labels in {path} must be integers in {{0, 1}}")

    df["label"] = labels.astype(int)
    df["text"] = df["text"].astype(str)
    return df


def build_pipeline(ngram_range: tuple[int, int]) -> Pipeline:
    """Create Member 2's char TF-IDF + balanced LogisticRegression pipeline."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=ngram_range,
                    max_features=50000,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=SEED,
                ),
            ),
        ]
    )


def compute_binary_metrics(y_true, y_pred) -> dict:
    """Compute accuracy, per-class metrics, macro averages, and confusion matrix."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "ai_class": {
            "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
            "f1_score": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        },
        "human_class": {
            "precision": float(precision_score(y_true, y_pred, pos_label=0, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
            "f1_score": float(f1_score(y_true, y_pred, pos_label=0, zero_division=0)),
        },
        "macro_avg": {
            "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
            "f1_score": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }


def validate_ngrams(train_df: pd.DataFrame) -> tuple[tuple[int, int], dict]:
    """Compare ngram_range candidates using an article-grouped 80:20 validation split."""
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    tr_idx, val_idx = next(gss.split(train_df, train_df["label"], groups=train_df["id"]))

    tr_split = train_df.iloc[tr_idx]
    val_split = train_df.iloc[val_idx]
    assert set(tr_split["id"]).isdisjoint(set(val_split["id"])), "Article ID leakage detected!"

    print(
        f"Validation split (GroupShuffleSplit by article id): "
        f"train={len(tr_split)} chunks ({tr_split['id'].nunique()} articles), "
        f"val={len(val_split)} chunks ({val_split['id'].nunique()} articles)"
    )
    print("\n--- Hyperparameter Validation (char TF-IDF ngram_range) ---")
    print(f"{'ngram_range':<12} {'AI_F1':>8} {'AI_Prec':>9} {'AI_Rec':>8} {'Macro_F1':>10} {'Acc':>8}")

    validation_results: dict[str, dict] = {}
    best_ngram = NGRAM_CANDIDATES[0]
    best_ai_f1 = -1.0

    for ngram_range in NGRAM_CANDIDATES:
        pipe = build_pipeline(ngram_range)
        pipe.fit(tr_split["text"], tr_split["label"])
        val_pred = pipe.predict(val_split["text"])

        ai_f1 = float(f1_score(val_split["label"], val_pred, pos_label=1, zero_division=0))
        ai_prec = float(precision_score(val_split["label"], val_pred, pos_label=1, zero_division=0))
        ai_rec = float(recall_score(val_split["label"], val_pred, pos_label=1, zero_division=0))
        macro_f1 = float(f1_score(val_split["label"], val_pred, average="macro", zero_division=0))
        acc = float(accuracy_score(val_split["label"], val_pred))

        key = f"{ngram_range[0]}_{ngram_range[1]}"
        validation_results[key] = {
            "ngram_range": list(ngram_range),
            "ai_f1": ai_f1,
            "ai_precision": ai_prec,
            "ai_recall": ai_rec,
            "macro_f1": macro_f1,
            "accuracy": acc,
        }

        print(
            f"{str(ngram_range):<12} {ai_f1:>8.4f} {ai_prec:>9.4f} "
            f"{ai_rec:>8.4f} {macro_f1:>10.4f} {acc:>8.4f}"
        )

        if ai_f1 > best_ai_f1:
            best_ai_f1 = ai_f1
            best_ngram = ngram_range

    print(f"\nSelected best ngram_range={best_ngram} (val AI F1={best_ai_f1:.4f})")
    return best_ngram, validation_results


def save_confusion_matrix(y_true, y_pred, ngram_range: tuple[int, int], out_path: Path) -> None:
    """Plot and save the test set confusion matrix image."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=["Human (0)", "AI (1)"],
        cmap="Blues",
        values_format="d",
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix (char TF-IDF ngram={ngram_range} + LogReg)")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_error_analysis(
    test_df: pd.DataFrame,
    y_pred,
    ai_proba,
    out_path: Path,
    n_per_type: int = 5,
) -> pd.DataFrame:
    """Extract top confident False Positives and False Negatives and save to CSV."""
    eval_df = pd.DataFrame(
        {
            "id": test_df["id"],
            "chunk_index": test_df["chunk_index"],
            "start_char": test_df["start_char"],
            "end_char": test_df["end_char"],
            "true_label": test_df["label"].astype(int),
            "pred_label": pd.Series(y_pred, index=test_df.index).astype(int),
            "ai_prob": pd.Series(ai_proba, index=test_df.index).astype(float),
            "text": test_df["text"],
        }
    )

    fp_df = (
        eval_df[(eval_df["true_label"] == 0) & (eval_df["pred_label"] == 1)]
        .sort_values("ai_prob", ascending=False)
        .head(n_per_type)
        .copy()
    )
    fp_df.insert(0, "error_type", "FP")

    fn_df = (
        eval_df[(eval_df["true_label"] == 1) & (eval_df["pred_label"] == 0)]
        .sort_values("ai_prob", ascending=True)
        .head(n_per_type)
        .copy()
    )
    fn_df.insert(0, "error_type", "FN")

    err_df = pd.concat([fp_df, fn_df], ignore_index=True)
    err_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return err_df


def main() -> None:
    train_df = load_chunks(TRAIN_CHUNKS_PATH)
    test_df = load_chunks(TEST_CHUNKS_PATH)
    print(f"Loaded train_chunks: {len(train_df)}, test_chunks: {len(test_df)}")

    best_ngram, validation_results = validate_ngrams(train_df)

    print(f"\nRetraining final Pipeline with ngram_range={best_ngram} on full train_chunks...")
    final_pipe = build_pipeline(best_ngram)
    final_pipe.fit(train_df["text"], train_df["label"])
    assert list(final_pipe.classes_) == [0, 1], f"Unexpected classes_: {final_pipe.classes_}"

    MODEL_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_pipe, MODEL_OUT_PATH)
    print(f"Saved model to {MODEL_OUT_PATH}")

    y_test = test_df["label"].to_numpy()
    y_pred = final_pipe.predict(test_df["text"])
    ai_class_idx = list(final_pipe.classes_).index(1)
    ai_proba = final_pipe.predict_proba(test_df["text"])[:, ai_class_idx]

    test_metrics = compute_binary_metrics(y_test, y_pred)

    print("\n--- Final Test Evaluation (test_chunks.csv) ---")
    print(f"Accuracy          : {test_metrics['accuracy']:.4f}")
    print(
        f"Macro Avg         : P={test_metrics['macro_avg']['precision']:.4f}  "
        f"R={test_metrics['macro_avg']['recall']:.4f}  "
        f"F1={test_metrics['macro_avg']['f1_score']:.4f}"
    )
    print(
        f"AI Class (1)      : P={test_metrics['ai_class']['precision']:.4f}  "
        f"R={test_metrics['ai_class']['recall']:.4f}  "
        f"F1={test_metrics['ai_class']['f1_score']:.4f}"
    )
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Human (0)", "AI (1)"], digits=4))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_payload = {
        "best_ngram_range": list(best_ngram),
        "validation_results": validation_results,
        "test_metrics": test_metrics,
    }
    METRICS_OUT_PATH.write_text(
        json.dumps(metrics_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Saved metrics to {METRICS_OUT_PATH}")

    save_confusion_matrix(y_test, y_pred, best_ngram, CM_OUT_PATH)
    print(f"Saved confusion matrix to {CM_OUT_PATH}")

    err_df = save_error_analysis(test_df, y_pred, ai_proba, ERRORS_OUT_PATH, n_per_type=5)
    print(
        f"Saved error analysis ({len(err_df)} rows: "
        f"{(err_df['error_type'] == 'FP').sum()} FP, "
        f"{(err_df['error_type'] == 'FN').sum()} FN) to {ERRORS_OUT_PATH}"
    )


if __name__ == "__main__":
    main()
