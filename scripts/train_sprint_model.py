"""ダートスプリント専用モデルの学習・検証スクリプト。

対象: is_turf==0 かつ distance<=1400m の全会場レース
保存先: models_sprint/ (models/ は一切変更しない)
目的: テスト結果を報告し、本番投入はユーザーが判断する
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.models.train import train_model, train_win_model
from src.models.predict import load_model, predict_probabilities
from src.utils.logger import get_logger

logger = get_logger("train_sprint_model")

TRAIN_END   = "2026-02-28"
VALID_START = "2026-03-01"
VALID_END   = "2026-04-24"
MODEL_DIR   = "models_sprint"


def main():
    logger.info("=== ダートスプリント専用モデル学習 ===")
    logger.info(f"保存先: {MODEL_DIR}/ (models/ は変更しない)")

    df = pd.read_parquet("data/processed/features.parquet")

    # ダートスプリント全会場フィルター
    sprint = df[(df["is_turf"] == 0) & (df["distance"] <= 1400)].copy()
    logger.info(f"ダートスプリント全会場: {sprint['race_id'].nunique()} races, {len(sprint)} rows")

    train_sp = sprint[sprint["date"] <= TRAIN_END]
    valid_sp  = sprint[(sprint["date"] >= VALID_START) & (sprint["date"] <= VALID_END)]
    logger.info(f"  学習: {train_sp['race_id'].nunique()} races / 検証: {valid_sp['race_id'].nunique()} races")

    # ── Top3モデル学習 ──────────────────────────────────────
    logger.info("\n--- Top3 モデル (binary) ---")
    top3_model, feature_cols = train_model(
        sprint,
        train_end=TRAIN_END,
        valid_start=VALID_START,
        valid_end=VALID_END,
        n_trials=30,
        seed=42,
        model_dir=MODEL_DIR,
    )

    # ── Winモデル学習 ───────────────────────────────────────
    logger.info("\n--- Win モデル (LambdaRank) ---")
    win_model, _ = train_win_model(
        sprint,
        train_end=TRAIN_END,
        valid_start=VALID_START,
        valid_end=VALID_END,
        n_trials=25,
        seed=42,
        model_dir=MODEL_DIR,
    )

    # ── 検証: バックテスト ──────────────────────────────────
    logger.info("\n=== 検証結果 ===")

    model_sp, feat_sp, calib_sp, win_sp, win_calib_sp = load_model(MODEL_DIR)
    model_cm, feat_cm, calib_cm, win_cm, win_calib_cm = load_model("models")

    valid_data = sprint[
        (sprint["date"] >= VALID_START) &
        (sprint["date"] <= VALID_END) &
        (sprint["finish_position"] > 0)
    ].copy()

    for col in feat_sp:
        if col not in valid_data.columns:
            valid_data[col] = 0.0

    preds_sp = predict_probabilities(
        model_sp, feat_sp, valid_data.sort_values("race_id").reset_index(drop=True),
        calibrator=calib_sp, win_model=win_sp, win_calibrator=win_calib_sp,
    ).rename(columns={"pred_win_prob": "win_prob_sprint", "pred_top3_prob": "top3_prob_sprint"})

    for col in feat_cm:
        if col not in valid_data.columns:
            valid_data[col] = 0.0

    preds_cm = predict_probabilities(
        model_cm, feat_cm, valid_data.sort_values("race_id").reset_index(drop=True),
        calibrator=calib_cm, win_model=win_cm, win_calibrator=win_calib_cm,
    ).rename(columns={"pred_win_prob": "win_prob_common", "pred_top3_prob": "top3_prob_common"})

    # マージ
    meta = valid_data[["race_id", "number", "finish_position", "place_code_num", "distance"]].drop_duplicates()
    cmp = (
        meta
        .merge(preds_sp[["race_id", "number", "win_prob_sprint", "top3_prob_sprint"]], on=["race_id", "number"], how="left")
        .merge(preds_cm[["race_id", "number", "win_prob_common", "top3_prob_common"]], on=["race_id", "number"], how="left")
    )

    def rank1(df, prob_col):
        n = df["race_id"].nunique()
        if n == 0:
            return 0, 0
        hits = (
            df.sort_values(prob_col, ascending=False)
            .groupby("race_id").first()["finish_position"] == 1
        ).sum()
        return int(hits), n

    def precision_at_3(df, prob_col):
        hit, total = 0, 0
        for _, g in df.groupby("race_id"):
            pred3 = set(g.nlargest(3, prob_col)["number"])
            act3  = set(g[g["finish_position"] <= 3]["number"])
            hit  += len(pred3 & act3)
            total += 3
        return hit / total if total > 0 else 0

    logger.info("\n  【全会場 ダートスプリント検証期間 102レース】")
    for label, prob_col in [("スプリント専用モデル", "win_prob_sprint"), ("共通モデル", "win_prob_common")]:
        h, n = rank1(cmp, prob_col)
        p3   = precision_at_3(cmp, prob_col)
        logger.info(f"    {label}: rank-1的中 {h}/{n} = {h/n:.1%}  precision@3 = {p3:.1%}")

    # 会場別内訳
    venue_map = {1:"札幌",2:"函館",3:"福島",4:"新潟",5:"東京",6:"中山",7:"小倉",8:"京都",9:"阪神",10:"中京"}
    logger.info("\n  【会場別 rank-1的中率（スプリント専用モデル）】")
    for code, name in sorted(venue_map.items()):
        sub = cmp[cmp["place_code_num"] == code]
        if sub["race_id"].nunique() < 3:
            continue
        h, n = rank1(sub, "win_prob_sprint")
        hc, nc = rank1(sub, "win_prob_common")
        logger.info(f"    {name}: 専用 {h}/{n}={h/n:.1%}  共通 {hc}/{nc}={hc/nc:.1%}")

    # 距離別
    logger.info("\n  【距離別 rank-1的中率（スプリント専用モデル vs 共通モデル）】")
    for dist in sorted(cmp["distance"].unique()):
        sub = cmp[cmp["distance"] == dist]
        if sub["race_id"].nunique() < 3:
            continue
        h, n  = rank1(sub, "win_prob_sprint")
        hc, nc = rank1(sub, "win_prob_common")
        logger.info(f"    {dist}m: 専用 {h}/{n}={h/n:.1%}  共通 {hc}/{nc}={hc/nc:.1%}")

    logger.info(f"\n保存先: {MODEL_DIR}/  (models/ は変更していません)")
    logger.info("本番への反映はユーザーが確認後に判断してください。")


if __name__ == "__main__":
    main()
