"""06: 対象レースの予測・馬券抽出。

最適化版: 事前計算済み特徴量を使い、対象レースのエントリーのみ予測する。
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import yaml

from src.models.predict import load_model, predict_probabilities
from src.probability.plackett_luce import compute_race_probabilities
from src.utils.logger import get_logger

logger = get_logger("06_predict")


def main():
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    raw_dir = Path(config["paths"]["raw"])
    processed_dir = Path(config["paths"]["processed"])
    output_dir = Path(config["paths"]["outputs"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # モデルロード
    model, feature_cols = load_model(config["paths"]["models"])

    # 対象レースを読み込み
    target_file = raw_dir / "target_races.json"
    if not target_file.exists():
        logger.error("target_races.json not found.")
        sys.exit(1)

    with open(target_file, encoding="utf-8") as f:
        target_races = json.load(f)

    # 過去の特徴量データを読み込み（騎手・馬の統計計算用）
    features_file = processed_dir / "features.parquet"
    hist_df = pd.DataFrame()
    if features_file.exists():
        hist_df = pd.read_parquet(features_file)
        logger.info(f"Loaded historical features: {hist_df.shape}")

    # 騎手・調教師・馬の統計を事前計算
    jockey_stats = {}
    trainer_stats = {}
    horse_stats = {}

    if not hist_df.empty:
        # 騎手統計
        if "jockey_id" in hist_df.columns:
            for jid, g in hist_df.groupby("jockey_id"):
                if "jockey_win_rate" in g.columns:
                    last_val = g["jockey_win_rate"].dropna().iloc[-1] if not g["jockey_win_rate"].dropna().empty else 0.0
                    jockey_stats[jid] = {
                        "jockey_win_rate": last_val,
                        "jockey_top3_rate": g["jockey_top3_rate"].dropna().iloc[-1] if "jockey_top3_rate" in g.columns and not g["jockey_top3_rate"].dropna().empty else 0.0,
                        "jockey_rides": len(g),
                    }

        # 調教師統計
        if "trainer_id" in hist_df.columns and "trainer_win_rate" in hist_df.columns:
            for tid, g in hist_df.groupby("trainer_id"):
                last_val = g["trainer_win_rate"].dropna().iloc[-1] if not g["trainer_win_rate"].dropna().empty else 0.0
                trainer_stats[tid] = {
                    "trainer_win_rate": last_val,
                    "trainer_top3_rate": g["trainer_top3_rate"].dropna().iloc[-1] if "trainer_top3_rate" in g.columns and not g["trainer_top3_rate"].dropna().empty else 0.0,
                }

        # 馬の過去成績（近5走）
        if "horse_id" in hist_df.columns:
            for hid, g in hist_df.groupby("horse_id"):
                g = g.sort_values("date")
                recent = g.tail(5)
                stats = {"horse_id": hid}
                for i, (_, row) in enumerate(recent.iterrows(), 1):
                    stats[f"prev{i}_finish_position"] = row.get("finish_position", 0)
                if "avg_finish_5" in g.columns:
                    stats["avg_finish_5"] = g["avg_finish_5"].dropna().iloc[-1] if not g["avg_finish_5"].dropna().empty else 0.0
                if "horse_course_top3_rate" in g.columns:
                    stats["horse_course_top3_rate"] = g["horse_course_top3_rate"].dropna().iloc[-1] if not g["horse_course_top3_rate"].dropna().empty else 0.0
                if "days_since_last" in g.columns:
                    stats["days_since_last"] = g["days_since_last"].dropna().iloc[-1] if not g["days_since_last"].dropna().empty else 0.0
                horse_stats[hid] = stats

    all_predictions = []

    for race in target_races:
        race_id = race["race_id"]
        entries = race.get("entries", [])
        if not entries:
            continue

        # エントリーのDataFrame作成
        rows = []
        for entry in entries:
            row = {
                "race_id": race_id,
                "number": entry.get("number", 0),
                "horse_name": entry.get("horse_name", ""),
                "horse_id": entry.get("horse_id", ""),
                "jockey_name": entry.get("jockey_name", ""),
                "jockey_id": entry.get("jockey_id", ""),
                "trainer_name": entry.get("trainer_name", ""),
                "trainer_id": entry.get("trainer_id", ""),
                "bracket": entry.get("bracket", 0),
                "impost": entry.get("weight", 0.0),
                "distance": race.get("distance", 0),
                "num_runners": len(entries),
                "is_turf": 1 if race.get("surface") == "芝" else 0,
                "place_code_num": int(race.get("place_code", "0")),
                "track_condition_num": 0,
            }

            # 騎手統計をマージ
            jid = entry.get("jockey_id", "")
            if jid in jockey_stats:
                row.update(jockey_stats[jid])

            # 調教師統計をマージ
            tid = entry.get("trainer_id", "")
            if tid in trainer_stats:
                row.update(trainer_stats[tid])

            # 馬の過去成績をマージ
            hid = entry.get("horse_id", "")
            if hid in horse_stats:
                for k, v in horse_stats[hid].items():
                    if k != "horse_id":
                        row[k] = v

            rows.append(row)

        entry_df = pd.DataFrame(rows)

        # 不足列を0で補完
        for col in feature_cols:
            if col not in entry_df.columns:
                entry_df[col] = 0.0

        # 予測
        entry_preds = predict_probabilities(model, feature_cols, entry_df)

        # 結果出力
        logger.info(f"\n{race.get('place_name', '')} {race.get('race_name', '')} ({race.get('surface', '')}{race.get('distance', '')}m)")

        # 確率計算
        probs = compute_race_probabilities(entry_preds)
        win_sorted = sorted(probs["win"].items(), key=lambda x: x[1], reverse=True)
        for num, prob in win_sorted[:5]:
            name = entry_preds[entry_preds["number"] == num]["horse_name"].values
            name = name[0] if len(name) > 0 else "?"
            logger.info(f"  {num:>2}. {name}: {prob:.1%}")

        # 予測結果を保存
        for _, row in entry_preds.iterrows():
            all_predictions.append({
                "race_id": race_id,
                "date": race.get("date", ""),
                "place_name": race.get("place_name", ""),
                "race_name": race.get("race_name", ""),
                "surface": race.get("surface", ""),
                "distance": race.get("distance", 0),
                "number": int(row.get("number", 0)),
                "horse_name": row.get("horse_name", ""),
                "pred_top3_prob": float(row.get("pred_top3_prob", 0)),
                "win_prob": float(probs["win"].get(row.get("number", 0), 0)),
            })

    # 保存
    if all_predictions:
        pred_df = pd.DataFrame(all_predictions)
        pred_df.to_csv(output_dir / "predictions.csv", index=False, encoding="utf-8-sig")
        logger.info(f"\nSaved {len(all_predictions)} predictions to {output_dir / 'predictions.csv'}")

        # レースごとの予測サマリー
        logger.info("\n=== 予測サマリー ===")
        for race_id, race_df in pred_df.groupby("race_id"):
            top = race_df.sort_values("win_prob", ascending=False).head(3)
            place = top.iloc[0]["place_name"] if "place_name" in top.columns else ""
            rname = top.iloc[0]["race_name"] if "race_name" in top.columns else ""
            logger.info(f"{place} {rname}:")
            for _, r in top.iterrows():
                logger.info(f"  ◎ {int(r['number']):>2}. {r['horse_name']} ({r['win_prob']:.1%})")


if __name__ == "__main__":
    main()
