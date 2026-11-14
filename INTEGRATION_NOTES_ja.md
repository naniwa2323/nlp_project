# Member 4 → Member 2 / 3 への連絡事項（インターフェース）

アプリ側は以下の形で受け取れるように作ってあります。この形で渡してもらえれば、コード変更なしで差し替わります。

## Member 2 へ
- `models/model.joblib` に **TF-IDF と LogisticRegression を 1 つの sklearn `Pipeline`** として保存してください。
- `model.predict(list[str]) -> [0/1, ...]`（0 = Human, 1 = AI）
- `model.predict_proba` があればチャンクごとの P(AI) グラフを表示します（`classes_` に 1 が含まれること）。
- 別パスに置く場合は環境変数 `MODEL_PATH` で指定できます。
- そのまま普通にコミットしてください（`.gitignore` の対象は仮モデル `models/placeholder_model.joblib` だけです）。
- `models/model.joblib` があればアプリは自動的にそちらを使います（サイドバー表示が「Member 2」に変わります）。

## Member 3 へ（⚠️ 仕様変更 2026-11-14）
「最初の 人間→AI の切り替わり 1 点」を求める方式から、**全文を人間 / AI の区間に分ける方式**に変更しました。
理由: 1 チャンクの誤判定で境界が大きくずれる（例: 全文人間の記事が「125 文字目から AI」と判定された）。
テストデータ 2,605 記事で比較すると、文字単位の正解率は 0.801 → 0.861、記事タイプの正解率は 0.664 → 0.754 に改善しました
（`python scripts/compare_rules.py`、結果は `results/compare_rules.csv`）。

- 参考実装は `segmenter.py` にあります（文字ごとに重なるチャンクの P(AI) を平均 → しきい値 → 短い区間を吸収）。
- Member 3 版を作る場合は、リポジトリ直下の `boundary_detector.py` に次の関数を用意してください。あればアプリが自動的にそちらを使います。

```python
def segment_article(article: str, model, threshold: float = 0.5, min_len: int = 100) -> list[dict]:
    """[{"start": int, "end": int, "label": 0 or 1, "score": float}, ...]
    区間は 0 から len(article) まで隙間なく並べ、隣り合う区間のラベルは異なること"""
```

- 評価指標の提案: 文字単位の正解率（記事ごとに平均）、記事タイプ正解率、isfake=1 の AI 開始位置 MAE（参考値）。
- しきい値と最小区間長は、テストデータではなく検証データで決め直してください（今の既定値 0.5 / 100 はテストデータを見て選んでいます）。

## Member 1 の成果物（確認済み）
- `src/preprocess.py` をローカルで実行し、`data/processed/` の4ファイルが GitHub 上のものとバイト単位で一致することを確認しました（2026-11-14）。
- アプリ・テストは `data/processed/test_articles.csv` / `*_chunks.csv` を参照します。
- macOS の python.org 版 Python で `CERTIFICATE_VERIFY_FAILED` が出る場合は `pip install certifi` のうえ
  `export SSL_CERT_FILE=$(python -c "import certifi;print(certifi.where())")` で実行できます。

## 現時点の状態
- `placeholders/` に仕様どおりの仮モデルを置いて、アプリと統合テストを先に動かしています。
- 最終提出前に `placeholders/` は削除予定です。
