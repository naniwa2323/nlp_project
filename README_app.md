# AI-Generated Text Region Detection (Japanese News)

人間が書いた文章と AI（日本語 GPT-2）が生成した文章が混ざった記事について、
**どこが人間が書いた部分で、どこが AI が生成した部分か**を区間ごとに判別し、色分け表示するシステムです。

> 仕様変更（2026-11-14）: 当初は「最初に人間→AI に切り替わる位置（境界 1 点）」を推定していましたが、
> 1 チャンクの誤判定で境界が大きくずれる問題があったため、**全文字を人間 / AI に分類して区間を求める方式**に変更しました。
> 1 記事に複数の AI 区間があってもそのまま表現できます。

## Overview

| Step | Member | Output |
|---|---|---|
| Dataset analysis & preprocessing (article-level split, 100-char chunks / stride 50) | Member 1 | `src/preprocess.py`, `data/processed/*.csv` |
| Chunk classifier: char TF-IDF (2–5gram) + Logistic Regression | Member 2 | `models/model.joblib` |
| Region detection & evaluation | Member 3 | `boundary_detector.py`（新インターフェース `segment_article`） |
| Streamlit app & integration | Member 4 | `app.py`, `pipeline.py`, `segmenter.py` |

Labels: `0` = Human-written, `1` = AI-generated.

## Dataset

[Japanese Fake News Dataset](https://github.com/tanreinama/Japanese-Fakenews-Dataset)（13,040 articles）

| column | meaning |
|---|---|
| `context` | article text |
| `isfake` | 0: all human / 1: human → GPT-2 / 2: all GPT-2 |
| `nchar_real` | number of human-written characters (= true boundary for `isfake=1`) |

文字ごとの正解ラベルは `isfake` と `nchar_real` から作ります（`isfake=1` なら `nchar_real` 文字目より前が人間、以降が AI）。
※「Fake」は虚偽情報ではなく「AI 生成」の意味です。

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## How to run

```bash
python src/preprocess.py                         # Member 1 -> data/processed/*.csv
python placeholders/train_placeholder_model.py   # placeholder, used only until Member 2's model.joblib arrives
python scripts/make_samples.py                   # demo articles with ground truth
streamlit run app.py
python -m pytest tests -q                        # tests
python scripts/compare_rules.py                  # old vs new rule on the test split
```

## How region detection works

1. 記事を 100 文字・50 文字ずらしのチャンクに分割
2. 分類器で各チャンクの P(AI) を計算
3. 各文字の AI スコア = その文字を含むチャンク（最大 2 つ）の P(AI) の平均
4. スコア ≥ しきい値（既定 0.5）の文字を AI、それ以外を人間
5. 平滑化: 最小区間長（既定 100 文字）未満の区間は誤判定とみなし、前後の区間に吸収
6. 連続する同じラベルを 1 区間にまとめて出力

アプリのサイドバーでしきい値と最小区間長を変えられます。
サンプル記事では正解と比較し、文字単位の一致率・食い違った箇所・チャンクごとの正誤を表示します。

## Results (placeholder model, test split 2,605 articles)

| Method | char acc | verdict acc | char acc (isfake 0 / 1 / 2) |
|---|---|---|---|
| Old: first Human→AI switch | 0.801 | 0.664 | 0.801 / 0.801 / 0.800 |
| New: regions, no smoothing | 0.852 | 0.701 | 0.938 / 0.774 / 0.862 |
| **New: regions, min_len = 100** | **0.861** | **0.754** | 0.953 / 0.770 / 0.880 |
| New: regions, min_len = 200 | 0.847 | 0.707 | 0.966 / 0.690 / 0.910 |

- char acc: 文字単位で人間 / AI が正しかった割合（記事ごとに計算して平均）
- verdict acc: 記事タイプ（全文人間 / 混在 / 全文 AI）の正解率
- 混在記事（isfake=1）だけは旧方式がやや良い（0.801 vs 0.770）。旧方式は「人間→AI」という形を前提にしているため。
- ⚠️ 仮モデルでの結果。設定値はテストデータを見て選んでいるため、最終版では検証データで選び直すこと。

<!-- TODO: Member 2 の分類器評価（Accuracy / F1）を追記 -->

## Limitations

- 生成モデルは日本語 GPT-2。ChatGPT / Gemini など最新 LLM の文章への汎化は保証しません。
- 判定の細かさは 50 文字単位（チャンクのずらし幅）です。境界は 50 文字刻みでしか表現できません。
- 100 文字未満の文章は 1 チャンクとして判定するため精度が下がります。
- データセットには「人間→AI」の 1 回切り替わりしかないため、複数区間の検出精度は評価できていません。

## Repository layout

```
app.py                     Streamlit UI (Member 4)
pipeline.py                integration: analyze_article() (Member 4)
segmenter.py               region segmentation (Member 4 版; Member 3 版があればそちらを使用)
boundary_detector.py       Member 3 (when delivered)
models/model.joblib        Member 2 (committed)
models/placeholder_model.joblib  placeholder (git-ignored)
src/preprocess.py          Member 1
data/processed/            Member 1 outputs
placeholders/              temporary model for development (delete at the end)
scripts/make_samples.py    demo article picker
scripts/compare_rules.py   old vs new rule comparison
results/                   comparison results
tests/                     tests
```
