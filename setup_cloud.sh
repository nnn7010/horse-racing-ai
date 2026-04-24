#!/bin/bash
# クラウド環境セットアップ: データ取得から学習まで全自動実行
# 使い方: bash setup_cloud.sh
# 所要時間: 約6〜10時間（スクレイピング込み）

set -e
LOG_DIR="logs"
mkdir -p "$LOG_DIR" data/raw data/processed models outputs

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_DIR/setup_cloud.log"
}

log "=== Horse Racing AI クラウドセットアップ開始 ==="
log "Python: $(python3 --version)"

# 依存パッケージ
log "=== パッケージインストール ==="
pip install -r requirements.txt -q
log "パッケージインストール完了"

# 01: 出走表取得（4/25-26）
log "=== 01_fetch_target 開始 ==="
python3 scripts/01_fetch_target.py 2>&1 | tee -a "$LOG_DIR/01_fetch_target.log"
log "=== 01_fetch_target 完了 ==="

# 02: 過去レース結果取得（全会場・3年分）※最長6時間
log "=== 02_fetch_history 開始（全会場・3年分、数時間かかります）==="
python3 scripts/02_fetch_history.py --all-venues 2>&1 | tee -a "$LOG_DIR/02_fetch_history.log"
log "=== 02_fetch_history 完了 ==="

# 03: 競走馬データ取得 ※最長4時間
log "=== 03_fetch_horses 開始（数時間かかります）==="
python3 scripts/03_fetch_horses.py 2>&1 | tee -a "$LOG_DIR/03_fetch_horses.log"
log "=== 03_fetch_horses 完了 ==="

# 04: 特徴量構築
log "=== 04_build_features 開始 ==="
python3 scripts/04_build_features.py 2>&1 | tee -a "$LOG_DIR/04_build_features.log"
log "=== 04_build_features 完了 ==="

# 05: モデル学習
log "=== 05_train 開始 ==="
python3 scripts/05_train.py 2>&1 | tee -a "$LOG_DIR/05_train.log"
log "=== 05_train 完了 ==="

# 06: 予測
log "=== 06_predict 開始 ==="
python3 scripts/06_predict.py 2>&1 | tee -a "$LOG_DIR/06_predict.log"
log "=== 06_predict 完了 ==="

# 07: バックテスト
log "=== 07_backtest 開始 ==="
python3 scripts/07_backtest.py 2>&1 | tee -a "$LOG_DIR/07_backtest.log"
log "=== 07_backtest 完了 ==="

log "=== セットアップ完了 ==="
log "ダッシュボード起動: streamlit run dashboard.py --server.port 8501"
