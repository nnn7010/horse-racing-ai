"""WIN5予測モジュール。

対象5レースの1着馬を予測し、組み合わせを生成する。
各レースの上位N頭から組み合わせを作り、的中確率順にランキング。
"""

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Win5Combination:
    picks: list  # [(race_id, num, name, prob), ...]
    probability: float
    display: str


def generate_win5(
    race_predictions: list[dict],
    max_per_race: int = 3,
    max_combos: int = 30,
) -> list[Win5Combination]:
    """WIN5の組み合わせを生成する。

    Args:
        race_predictions: 5レース分の予測
            [{"race_id": str, "race_name": str, "picks": [{"num": int, "name": str, "prob": float}, ...]}]
        max_per_race: 各レースの候補頭数（1-3頭）
        max_combos: 出力する組み合わせ数の上限

    Returns:
        確率順の組み合わせリスト
    """
    if len(race_predictions) != 5:
        logger.warning(f"WIN5 requires exactly 5 races, got {len(race_predictions)}")
        return []

    # 各レースの候補
    race_picks = []
    for rp in race_predictions:
        picks = rp["picks"][:max_per_race]
        race_picks.append(picks)

    # 全組み合わせを生成
    combos = []
    for combo in itertools.product(*race_picks):
        prob = 1.0
        for pick in combo:
            prob *= pick["prob"]

        picks_list = [
            (race_predictions[i]["race_id"],
             race_predictions[i]["race_name"],
             combo[i]["num"],
             combo[i]["name"],
             combo[i]["prob"])
            for i in range(5)
        ]

        display_parts = []
        for _, rname, num, name, p in picks_list:
            display_parts.append(f"{num}番{name}({p:.0%})")

        combos.append(Win5Combination(
            picks=picks_list,
            probability=prob,
            display=" × ".join(display_parts),
        ))

    # 確率順にソート
    combos.sort(key=lambda x: x.probability, reverse=True)

    total_points = len(combos)
    logger.info(f"WIN5: {total_points}通り生成、上位{min(max_combos, len(combos))}通り返却")

    return combos[:max_combos]


def analyze_win5_races(preds_df: pd.DataFrame, target_race_ids: list[str]) -> list[dict]:
    """対象5レースの予測を整理する。

    Returns:
        [{"race_id", "race_name", "picks": [{"num", "name", "prob"}, ...]}]
    """
    race_predictions = []

    for race_id in target_race_ids:
        race_preds = preds_df[preds_df["race_id"] == race_id].copy()
        if race_preds.empty:
            continue

        # win_probがなければ計算
        if "win_prob" not in race_preds.columns:
            p = race_preds["pred_top3_prob"].values
            total = p.sum()
            race_preds["win_prob"] = p / total if total > 0 else 1.0 / len(race_preds)

        sorted_df = race_preds.sort_values("win_prob", ascending=False)

        picks = []
        for _, row in sorted_df.iterrows():
            picks.append({
                "num": int(row["number"]),
                "name": row.get("horse_name", ""),
                "prob": row["win_prob"],
            })

        race_name = race_preds.iloc[0].get("race_name", f"{race_id[-2:]}R")

        race_predictions.append({
            "race_id": race_id,
            "race_name": race_name,
            "picks": picks,
        })

    return race_predictions
