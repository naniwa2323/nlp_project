"""Prepare identical data splits and 80%-trained TF-IDF baseline for BERT comparison."""
from pathlib import Path
import time
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit

# 既存のモジュールをインポート（既存ロジックと100%一致させるため）
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from train_classifier import build_pipeline, load_chunks, SEED

EXPORT_DIR = ROOT / "bert_export"
EXPORT_DIR.mkdir(exist_ok=True)

def slice_articles_to_chunks(articles_df: pd.DataFrame, window: int = 100, stride: int = 50) -> pd.DataFrame:
    # 本文カラム名を自動検出（context, text, article, body, content などに対応）
    text_col = next((c for c in ["context", "text", "article", "body", "content"] if c in articles_df.columns), None)
    if text_col is None:
        raise KeyError(f"本文カラムが見つかりません。存在する列: {list(articles_df.columns)}")

    rows = []
    for _, row in articles_df.iterrows():
        art_id = str(row["id"])
        text = str(row[text_col])  # ← ここを自動検出した列名に変更！
        n = len(text)
        chunk_idx = 0
        for start in range(0, max(1, n - window + 1), stride):
            end = min(start + window, n)
            chunk_text = text[start:end]
            if len(chunk_text) < window and start > 0:
                continue
            rows.append({
                "id": art_id,
                "chunk_index": chunk_idx,
                "start_char": start,
                "end_char": end,
                "text": chunk_text,
            })
            chunk_idx += 1
    return pd.DataFrame(rows)

def main():
    # 1. 既存のチャンクデータを読み込み
    train_df = load_chunks(ROOT / "data" / "processed" / "train_chunks.csv")
    test_df = load_chunks(ROOT / "data" / "processed" / "test_chunks.csv")

    # 2. 既存と全く同じ記事ID単位の 80:20 分割（GroupShuffleSplit, random_state=42）
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    tr_idx, val_idx = next(gss.split(train_df, train_df["label"], groups=train_df["id"]))
    sub_train_df = train_df.iloc[tr_idx].copy()
    val_df = train_df.iloc[val_idx].copy()
    assert set(sub_train_df["id"]).isdisjoint(set(val_df["id"])), "Leakage detected!"

    # 3. 対照基準：80% Sub-train で学習した TF-IDF のテスト評価
    pipe_80 = build_pipeline((1, 3))
    pipe_80.fit(sub_train_df["text"], sub_train_df["label"])
    y_pred = pipe_80.predict(test_df["text"])
    y_true = test_df["label"].to_numpy()

    print("=== [対照基準] TF-IDF + LogReg (80% Sub-train) Test Results ===")
    print(f"Accuracy : {accuracy_score(y_true, y_pred)*100:.2f}%")
    print(f"AI F1    : {f1_score(y_true, y_pred, pos_label=1)*100:.2f}%")
    print(f"Macro F1 : {f1_score(y_true, y_pred, average='macro')*100:.2f}%")

    # 4. 同一CPU環境での推論レイテンシ（100チャンク ≒ 10記事分）を計測
    sample_100 = test_df["text"].iloc[:100].tolist()
    t0 = time.perf_counter()
    _ = pipe_80.predict_proba(sample_100)
    t1 = time.perf_counter()
    print(f"TF-IDF CPU Latency (100 chunks / ~10 articles): {(t1 - t0)*1000:.2f} ms "
          f"(~{(t1 - t0)*100:.2f} ms / article)")

    # 5. Colab へアップロードするファイルを書き出し
    sub_train_df.to_csv(EXPORT_DIR / "sub_train_chunks.csv", index=False)
    val_df.to_csv(EXPORT_DIR / "val_chunks.csv", index=False)
    test_df.to_csv(EXPORT_DIR / "test_chunks.csv", index=False)

    # Part 3 用の記事データ（train_articles.csv / test_articles.csv）があればチャンク表も書き出し
    train_art_path = ROOT / "data" / "processed" / "train_articles.csv"
    test_art_path = ROOT / "data" / "processed" / "test_articles.csv"
    if train_art_path.exists() and test_art_path.exists():
        train_arts = pd.read_csv(train_art_path, dtype={"id": str})
        test_arts = pd.read_csv(test_art_path, dtype={"id": str})
        val_art_ids = set(val_df["id"].unique())
        val_arts = train_arts[train_arts["id"].isin(val_art_ids)]

        slice_articles_to_chunks(val_arts).to_csv(EXPORT_DIR / "part3_val_article_chunks.csv", index=False)
        slice_articles_to_chunks(test_arts).to_csv(EXPORT_DIR / "part3_test_article_chunks.csv", index=False)
        print("Exported Part 3 article chunks for Validation & Test.")

    print(f"\nAll files saved to: {EXPORT_DIR}")

if __name__ == "__main__":
    main()