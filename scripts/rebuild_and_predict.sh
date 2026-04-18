#!/bin/bash
# 血統データ取得完了後の一括再実行スクリプト
set -e

echo "=== Step 1: 特徴量再構築（血統込み） ==="
python scripts/04_build_features.py

echo ""
echo "=== Step 2: モデル再学習 ==="
python scripts/05_train.py

echo ""
echo "=== Step 3: 予測再実行 ==="
python scripts/06_predict.py

echo ""
echo "=== Step 4: バックテスト ==="
python scripts/07_backtest.py

echo ""
echo "=== 完了！ ==="
echo "予測結果: outputs/predictions.csv"
echo "バックテスト: outputs/backtest_results.csv"
