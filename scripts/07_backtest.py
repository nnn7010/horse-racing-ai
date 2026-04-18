"""07: バックテスト - 検証期間のレースで馬券戦略を評価する。"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import yaml

from src.betting.expected_value import compute_ev_from_results
from src.evaluation.backtest import run_backtest
from src.features.build import build_features
from src.models.predict import load_model, predict_probabilities
from src.utils.logger import get_logger

logger = get_logger("07_backtest")


def main():
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    raw_dir = Path(config["paths"]["raw"])
    processed_dir = Path(config["paths"]["processed"])
    output_dir = Path(config["paths"]["outputs"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # モデルロード
    model, feature_cols = load_model(config["paths"]["models"])

    # 特徴量データ読み込み
    features_file = processed_dir / "features.parquet"
    if not features_file.exists():
        logger.error("features.parquet not found.")
        sys.exit(1)

    df = pd.read_parquet(features_file)
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")

    # 検証期間のデータを抽出
    valid_start = pd.Timestamp(config["split"]["valid_start"])
    valid_end = pd.Timestamp(config["split"]["valid_end"])
    valid_df = df[(df["date"] >= valid_start) & (df["date"] <= valid_end)].copy()

    logger.info(f"Validation period: {valid_start.date()} ~ {valid_end.date()}")
    logger.info(f"Validation data: {len(valid_df)} entries, {valid_df['race_id'].nunique()} races")

    if valid_df.empty:
        logger.error("No validation data found")
        sys.exit(1)

    # 予測
    valid_preds = predict_probabilities(model, feature_cols, valid_df)

    # 過去レース結果（払い戻し情報取得用）
    history_file = raw_dir / "historical_results.json"
    payouts_by_race = {}
    if history_file.exists():
        with open(history_file, encoding="utf-8") as f:
            for race in json.load(f):
                rid = race.get("race_id", "")
                if race.get("payouts"):
                    payouts_by_race[rid] = race["payouts"]

    # 各レースの期待値計算
    all_recs = []
    for race_id, race_df in valid_preds.groupby("race_id"):
        payouts = payouts_by_race.get(race_id, {})
        recs = compute_ev_from_results(race_df, payouts)
        for rec in recs:
            # 日付を追加
            if "date" in race_df.columns:
                rec["date"] = race_df["date"].iloc[0]
        all_recs.extend(recs)

    logger.info(f"Total EV+ recommendations: {len(all_recs)}")

    if all_recs:
        recs_df = pd.DataFrame(all_recs)
        recs_df.to_csv(output_dir / "backtest_recommendations.csv", index=False, encoding="utf-8-sig")

        # バックテスト実行
        results = run_backtest(recs_df, output_dir=str(output_dir))

        logger.info("\nBacktest complete!")
    else:
        logger.warning("No EV+ recommendations found in validation period")


if __name__ == "__main__":
    main()
