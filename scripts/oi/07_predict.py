"""指定レース or 検証期間の予測を出力する。

Usage:
  # 検証期間バックテスト
  python scripts/oi/07_predict.py --validate
  # 単発レース予測（出馬表から特徴量化済みのDFを別途用意）
  python scripts/oi/07_predict.py --race-id 202644042811
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.oi import load_config
from src.oi.models.predictor import load_models, predict_race, race_betting_table
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate", action="store_true", help="検証期間でバックテスト")
    parser.add_argument("--race-id", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config()
    proc = ROOT / cfg["paths"]["processed"]
    out_dir = ROOT / cfg["paths"]["outputs"]
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(proc / "features.parquet")
    df["date"] = df["date"].astype(str)

    models = load_models(ROOT / cfg["paths"]["models"])

    if args.race_id:
        sub = df[df["race_id"] == args.race_id]
        if len(sub) == 0:
            logger.error(f"race_id {args.race_id} がfeatures中にありません")
            sys.exit(1)
        pred = predict_race(sub, models)
        report = race_betting_table(
            pred,
            win_ev_threshold=cfg["betting"]["win_ev_threshold"],
            partner_count=cfg["betting"]["top3_partner_count"],
        )
        out_path = out_dir / f"prediction_{args.race_id}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        logger.info(f"保存: {out_path}")
        return

    if args.validate:
        v_start = cfg["split"]["valid_start"].replace("-", "")
        v_end = cfg["split"]["valid_end"].replace("-", "")
        sub = df[(df["date"] >= v_start) & (df["date"] <= v_end)]
        pred = predict_race(sub, models)
        out = out_dir / "predictions_validation.parquet"
        pred.to_parquet(out, index=False)
        logger.info(f"検証期間予測: {len(pred)}行 → {out}")

        # 簡易バックテスト指標
        # 単勝EV閾値超え馬の的中率・回収率
        threshold = cfg["betting"]["win_ev_threshold"]
        pred["win_ev"] = pred["pred_win_prob"] * pred["win_odds"].fillna(0)
        picks = pred[pred["win_ev"] > threshold]
        if len(picks):
            hit = (picks["finish_position"] == 1).sum()
            invest = len(picks) * 100
            payout = (picks[picks["finish_position"] == 1]["win_odds"] * 100).sum()
            recovery = payout / invest if invest else 0
            logger.info(
                f"単勝EV>{threshold}: {len(picks)}点 / 的中 {hit} / 投資 {invest:,} / 回収 {payout:,.0f} / 回収率 {recovery:.1%}"
            )
        return

    parser.print_help()


if __name__ == "__main__":
    main()
