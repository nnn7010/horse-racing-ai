"""04: 特徴量を構築する。"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import yaml

from src.features.build import build_features
from src.utils.logger import get_logger

logger = get_logger("04_build_features")


def main():
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    raw_dir = Path(config["paths"]["raw"])
    processed_dir = Path(config["paths"]["processed"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    # 過去レース結果を読み込み
    history_file = raw_dir / "historical_results.json"
    if not history_file.exists():
        logger.error("historical_results.json not found. Run 02_fetch_history.py first.")
        sys.exit(1)

    with open(history_file, encoding="utf-8") as f:
        historical = json.load(f)

    # フラット化
    rows = []
    for race in historical:
        race_id = race.get("race_id", "")
        race_date = race.get("date", "")
        surface = race.get("surface", "")
        distance = race.get("distance", 0)
        track_condition = race.get("track_condition", "")
        weather = race.get("weather", "")
        num_runners = race.get("num_runners", 0)
        race_name = race.get("race_name", "")
        payouts = race.get("payouts", {})

        # place_code をrace_idから抽出
        place_code = race_id[4:6] if len(race_id) >= 6 else ""

        for r in race.get("results", []):
            row = {
                "race_id": race_id,
                "date": race_date,
                "surface": surface,
                "distance": distance,
                "track_condition": track_condition,
                "weather": weather,
                "num_runners": num_runners,
                "race_name": race_name,
                "place_code": place_code,
                **r,
            }
            rows.append(row)

    results_df = pd.DataFrame(rows)
    logger.info(f"Loaded {len(results_df)} entries from {len(historical)} races")

    # 馬情報を読み込み
    horses_file = raw_dir / "horses.json"
    if horses_file.exists():
        with open(horses_file, encoding="utf-8") as f:
            horses_list = json.load(f)
        horses_df = pd.DataFrame(horses_list)
        logger.info(f"Loaded {len(horses_df)} horses")
    else:
        horses_df = pd.DataFrame()
        logger.warning("No horses data found")

    # 特徴量構築
    features_df = build_features(results_df, horses_df)

    # 保存
    features_df.to_parquet(processed_dir / "features.parquet", index=False)
    logger.info(f"Saved features to {processed_dir / 'features.parquet'}: {features_df.shape}")

    # 統計情報
    logger.info(f"Columns: {list(features_df.columns)}")
    logger.info(f"Date range: {features_df['date'].min()} ~ {features_df['date'].max()}")
    if "finish_position" in features_df.columns:
        target = (features_df["finish_position"].between(1, 3)).astype(int)
        logger.info(f"Target (top3) ratio: {target.mean():.3f}")


if __name__ == "__main__":
    main()
