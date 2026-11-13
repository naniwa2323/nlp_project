import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pathlib import Path

project_dir = Path(__file__).resolve().parent.parent
output_dir = project_dir / "data" / "processed"
output_dir.mkdir(parents=True, exist_ok=True)
(project_dir / "data" / "raw").mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 100
STRIDE = 50

url = (
    "https://raw.githubusercontent.com/"
    "tanreinama/Japanese-Fakenews-Dataset/master/fakenews.csv"
)

df = pd.read_csv(url)
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["isfake"]
)

train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

def create_chunks(article_df):
    chunks = []
    excluded_chunks = []
    short_articles = []

    for _, row in article_df.iterrows():
        article_id = row["id"]
        text = row["context"]
        article_label = int(row["isfake"])

        if not isinstance(text, str):
            raise ValueError(f"本文が文字列ではありません: id={article_id}")

        if article_label not in (0, 1, 2):
            raise ValueError(f"不正なisfakeです: id={article_id}")

        boundary = None

        if article_label == 1:
            value = pd.to_numeric(row["nchar_real"], errors="coerce")

            if (
                pd.isna(value)
                or not 0 <= value <= len(text)
                or value % 1 != 0
            ):
                raise ValueError(f"不正な境界です: id={article_id}")

            boundary = int(value)

        if len(text) < CHUNK_SIZE:
            short_articles.append({
                "article_id": article_id,
                "isfake": article_label,
                "text_length": len(text),
                "reason": "shorter_than_100"
            })
            continue

        for start in range(0, len(text) - CHUNK_SIZE + 1, STRIDE):
            end = start + CHUNK_SIZE

            record = {
                "article_id": article_id,
                "isfake": article_label,
                "start": start,
                "end": end,
                "text": text[start:end]
            }

            if article_label == 0:
                chunk_label = 0

            elif article_label == 2:
                chunk_label = 1

            else:
                if end <= boundary:
                    chunk_label = 0

                elif start >= boundary:
                    chunk_label = 1

                else:
                    record["boundary"] = boundary
                    record["reason"] = "crosses_boundary"
                    excluded_chunks.append(record)
                    continue

            record["label"] = chunk_label
            chunks.append(record)

    chunk_columns = [
        "article_id", "isfake", "start", "end", "text", "label"
    ]
    excluded_columns = [
        "article_id", "isfake", "start", "end", "text",
        "boundary", "reason"
    ]
    short_columns = [
        "article_id", "isfake", "text_length", "reason"
    ]

    return (
        pd.DataFrame(chunks, columns=chunk_columns),
        pd.DataFrame(excluded_chunks, columns=excluded_columns),
        pd.DataFrame(short_articles, columns=short_columns)
    )

def format_chunks(chunks):
    result = chunks.rename(columns={
        "article_id": "id",
        "start": "start_char",
        "end": "end_char"
    }).copy()

    result["chunk_index"] = (
        result["start_char"] // STRIDE
    ).astype(int)

    return result[[
        "id",
        "chunk_index",
        "start_char",
        "end_char",
        "text",
        "label"
    ]]


def main():

    train_chunks, train_excluded, train_short = create_chunks(train_df)
    test_chunks, test_excluded, test_short = create_chunks(test_df)
    
    train_chunks_output = format_chunks(train_chunks)
    test_chunks_output = format_chunks(test_chunks)

    original_columns = [
    column for column in df.columns
    if column != "text_length"
    ]

    train_articles_output = train_df[original_columns].copy()
    test_articles_output = test_df[original_columns].copy()

    files = {
    "train_chunks.csv": train_chunks_output,
    "test_chunks.csv": test_chunks_output,
    "train_articles.csv": train_articles_output,
    "test_articles.csv": test_articles_output
    }

    for filename, data in files.items():
        data.to_csv(
            output_dir / filename,
            index=False,
            encoding="utf-8-sig"
        )


if __name__ == "__main__":
    main()