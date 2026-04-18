# horse-racing-ai

競馬予想AIシステム - JRA全レース対応の期待値ベース馬券抽出AI

## 概要

LightGBMによる複勝圏内確率予測（Stage1）とPlackett-Luceモデルによる組み合わせ確率算出（Stage2）を組み合わせ、単勝・複勝・三連複・三連単の期待値プラス馬券を自動抽出します。

## 免責事項

**このシステムは研究・学習目的であり、馬券の利益を保証するものではありません。馬券購入は自己責任で行ってください。**

## セットアップ

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 実行手順

```bash
# 1. 対象日のレース一覧を取得
python scripts/01_fetch_target.py

# 2. 過去レース結果を取得
python scripts/02_fetch_history.py

# 3. 出走馬の血統情報を取得
python scripts/03_fetch_horses.py

# 4. 特徴量を構築
python scripts/04_build_features.py

# 5. モデル学習（Optuna 30trial）
python scripts/05_train.py

# 6. 予測・馬券抽出
python scripts/06_predict.py

# 7. バックテスト
python scripts/07_backtest.py
```

## バックテスト結果の見方

`outputs/` ディレクトリに以下が出力されます:

- `predictions_YYYYMMDD.csv`: 各レースの予測結果
- `betting_recommendations.csv`: EV閾値を超えた推奨馬券一覧
- `backtest_results.csv`: バックテスト集計結果
- `backtest_*.png`: 日別回収率グラフ

### バックテスト3パターン

| パターン | 説明 |
|---------|------|
| A | 全該当馬券に100円均一 |
| B | 1レース上限1,000円、EV上位から配分 |
| C | 1日予算3,000円、全レースEV降順で合計3,000円以内 |

各パターンで総投資・総回収・回収率・的中数・的中率・馬券種別内訳を表示します。

## 技術スタック

- Python 3.11+
- LightGBM（二値分類）
- Plackett-Luce（組み合わせ確率）
- Optuna（ハイパーパラメータ最適化）
- データソース: netkeiba
