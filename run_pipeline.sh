#!/bin/bash
# 03→02(新馬戦差分)→04→05 自動パイプライン
# 使い方: bash run_pipeline.sh [02のPID]

set -e
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_DIR/pipeline.log"
}

# 02が実行中なら完了を待つ
WAIT_PID=${1:-}
if [ -n "$WAIT_PID" ]; then
    log "02_fetch_history (PID: $WAIT_PID) の完了を待機中..."
    while kill -0 "$WAIT_PID" 2>/dev/null; do
        sleep 30
    done
    log "02_fetch_history 完了"
fi

# 03: 競走馬データ取得
log "=== 03_fetch_horses 開始 ==="
python3 scripts/03_fetch_horses.py 2>&1 | tee -a "$LOG_DIR/03_fetch_horses.log"
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    log "ERROR: 03_fetch_horses 失敗"
    exit 1
fi
log "=== 03_fetch_horses 完了 ==="

# 02(再): 新馬戦データ差分取得
log "=== 02_fetch_history (新馬戦差分) 開始 ==="
python3 scripts/02_fetch_history.py --all-venues 2>&1 | tee -a "$LOG_DIR/02_fetch_history_debut.log"
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    log "ERROR: 02_fetch_history (新馬戦差分) 失敗"
    exit 1
fi
log "=== 02_fetch_history (新馬戦差分) 完了 ==="

# 04: 特徴量構築
log "=== 04_build_features 開始 ==="
python3 scripts/04_build_features.py 2>&1 | tee -a "$LOG_DIR/04_build_features.log"
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    log "ERROR: 04_build_features 失敗"
    exit 1
fi
log "=== 04_build_features 完了 ==="

# 05: モデル学習
log "=== 05_train 開始 ==="
python3 scripts/05_train.py 2>&1 | tee -a "$LOG_DIR/05_train.log"
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    log "ERROR: 05_train 失敗"
    exit 1
fi
log "=== 05_train 完了 ==="

log "✓ パイプライン完了 (03〜05)"
log "次のステップ:"
log "  python3 scripts/01_fetch_target.py   # 4/25-26 出走表取得"
log "  python3 scripts/06_predict.py        # 予測・買い目抽出"
log "  python3 scripts/07_backtest.py       # バックテスト"
