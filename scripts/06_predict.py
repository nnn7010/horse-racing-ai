"""06: 対象レースの予測・馬券抽出。"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import yaml

from src.betting.expected_value import compute_expected_values
from src.features.build import build_features
from src.models.predict import load_model, predict_probabilities
from src.probability.plackett_luce import compute_race_probabilities
from src.utils.logger import get_logger

logger = get_logger("06_predict")


def main():
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    raw_dir = Path(config["paths"]["raw"])
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

    # 過去データも読み込み（特徴量計算用）
    horses_file = raw_dir / "horses.json"
    horses_df = pd.DataFrame()
    if horses_file.exists():
        with open(horses_file, encoding="utf-8") as f:
            horses_df = pd.DataFrame(json.load(f))

    history_file = raw_dir / "historical_results.json"
    historical_rows = []
    if history_file.exists():
        with open(history_file, encoding="utf-8") as f:
            for race in json.load(f):
                for r in race.get("results", []):
                    r["race_id"] = race.get("race_id", "")
                    r["date"] = race.get("date", "")
                    r["surface"] = race.get("surface", "")
                    r["distance"] = race.get("distance", 0)
                    r["track_condition"] = race.get("track_condition", "")
                    r["num_runners"] = race.get("num_runners", 0)
                    r["place_code"] = race.get("race_id", "")[4:6] if len(race.get("race_id", "")) >= 6 else ""
                    historical_rows.append(r)

    all_predictions = []
    all_recommendations = []

    for race in target_races:
        race_id = race["race_id"]
        entries = race.get("entries", [])
        if not entries:
            logger.warning(f"No entries for {race_id}")
            continue

        # エントリーをDataFrame化
        entry_df = pd.DataFrame(entries)
        entry_df["race_id"] = race_id
        entry_df["date"] = race.get("date", "")
        entry_df["surface"] = race.get("surface", "")
        entry_df["distance"] = race.get("distance", 0)
        entry_df["place_code"] = race.get("place_code", "")
        entry_df["finish_position"] = 0  # 未確定

        # 過去データとマージして特徴量計算
        if historical_rows:
            hist_df = pd.DataFrame(historical_rows)
            combined = pd.concat([hist_df, entry_df], ignore_index=True)
        else:
            combined = entry_df

        features = build_features(combined, horses_df)

        # 対象レースのエントリーのみ抽出
        race_features = features[features["race_id"] == race_id].copy()
        if race_features.empty:
            logger.warning(f"No features for {race_id}")
            continue

        # 予測
        race_preds = predict_probabilities(model, feature_cols, race_features)

        # 結果を保存
        for _, row in race_preds.iterrows():
            pred = {
                "race_id": race_id,
                "race_name": race.get("race_name", ""),
                "place_name": race.get("place_name", ""),
                "surface": race.get("surface", ""),
                "distance": race.get("distance", 0),
                "number": int(row.get("number", 0)),
                "horse_name": row.get("horse_name", ""),
                "pred_top3_prob": float(row.get("pred_top3_prob", 0)),
            }
            all_predictions.append(pred)

        # 確率計算
        probs = compute_race_probabilities(race_preds)

        logger.info(f"\n{race.get('place_name', '')} {race.get('race_name', '')} ({race.get('surface', '')}{race.get('distance', '')}m)")
        logger.info("  Win probabilities:")
        win_sorted = sorted(probs["win"].items(), key=lambda x: x[1], reverse=True)
        for num, prob in win_sorted[:5]:
            name = race_preds[race_preds["number"] == num]["horse_name"].values
            name = name[0] if len(name) > 0 else "?"
            logger.info(f"    {num:>2}. {name}: {prob:.3f}")

    # 予測結果を保存
    if all_predictions:
        pred_df = pd.DataFrame(all_predictions)
        pred_df.to_csv(output_dir / "predictions.csv", index=False, encoding="utf-8-sig")
        logger.info(f"\nSaved {len(all_predictions)} predictions to {output_dir / 'predictions.csv'}")

    if all_recommendations:
        rec_df = pd.DataFrame(all_recommendations)
        rec_df.to_csv(output_dir / "betting_recommendations.csv", index=False, encoding="utf-8-sig")
        logger.info(f"Saved {len(all_recommendations)} recommendations")


if __name__ == "__main__":
    main()
