"""買い目最適化エンジン v5。

AI予測の的中率で買い目を選定する。
合成オッズ基準は各券種の的中率から動的に算出。
買い目はまとめ表示（フォーメーション等）に対応。

合成オッズ = 総投資額 / Σ(各投資額 / 各オッズ)
合成オッズ基準 = (1 / 合計的中率) × 1.3
"""

import itertools

import numpy as np
import pandas as pd

from src.probability.plackett_luce import compute_race_probabilities
from src.utils.logger import get_logger

logger = get_logger(__name__)


def calc_composite_odds(odds_list: list[float], amounts: list[float] | None = None) -> float:
    """合成オッズ = 総投資額 / Σ(各投資額 / 各オッズ)"""
    if not odds_list:
        return 0.0
    if amounts is None:
        amounts = [1.0] * len(odds_list)
    total_inv = sum(amounts)
    denom = sum(a / o for a, o in zip(amounts, odds_list) if o > 0)
    if denom <= 0:
        return 0.0
    return total_inv / denom


def calc_min_composite_odds(total_prob: float, margin: float = 1.3) -> float:
    """的中率から最低合成オッズ基準を算出。

    基準 = (1 / 合計的中率) × margin
    例: 合計的中率25% → 1/0.25 × 1.3 = 5.2倍以上
    """
    if total_prob <= 0:
        return 999.0
    return (1.0 / total_prob) * margin


def build_recommendations(race_df: pd.DataFrame, all_odds: dict, budget: int) -> dict:
    """AI予測の的中率で買い目を選定する。"""

    probs = compute_race_probabilities(race_df)
    sorted_df = race_df.sort_values("win_prob", ascending=False)
    numbers = sorted_df["number"].astype(int).values
    n = len(numbers)

    def name(num):
        r = race_df[race_df["number"] == num]["horse_name"].values
        return r[0] if len(r) > 0 else "?"

    def top3_prob(num):
        r = race_df[race_df["number"] == num]["pred_top3_prob"].values
        return r[0] if len(r) > 0 else 0

    win_odds = all_odds.get("win", {})
    ticket_groups = []

    # === 1. 単勝（最大2点）===
    win_picks = []
    for num in numbers[:6]:
        ns = str(num).zfill(2)
        odds = win_odds.get(ns, 0)
        prob = probs["win"].get(num, 0)
        if prob >= 0.10:
            win_picks.append({"num": num, "odds": odds, "prob": prob})

    if win_picks:
        win_picks.sort(key=lambda x: x["prob"], reverse=True)
        picks = win_picks[:2]
        total_prob = sum(p["prob"] for p in picks)
        odds_list = [p["odds"] for p in picks if p["odds"] > 0]
        comp_odds = calc_composite_odds(odds_list) if odds_list else 0
        min_odds = calc_min_composite_odds(total_prob)
        ticket_groups.append({
            "bet_type": "単勝",
            "summary": " / ".join(f"{p['num']}番{name(p['num'])}" for p in picks),
            "picks": [{"numbers": str(p["num"]), "names": name(p["num"]),
                       "odds": p["odds"], "prob": p["prob"]} for p in picks],
            "composite_odds": comp_odds,
            "min_composite_odds": min_odds,
            "total_prob": total_prob,
            "n_bets": len(picks),
        })

    axis = numbers[0]

    # === 2. 複勝（最大2点）===
    place_odds_dict = all_odds.get("place", {})
    place_picks = []
    for num in numbers[:6]:
        ns = str(num).zfill(2)
        odds = place_odds_dict.get(ns, 0)
        prob = probs["place"].get(num, 0)
        if prob >= 0.50:
            place_picks.append({"num": num, "odds": odds, "prob": prob})

    if place_picks:
        place_picks.sort(key=lambda x: x["prob"], reverse=True)
        picks = place_picks[:2]
        total_prob = sum(p["prob"] for p in picks)
        odds_list = [p["odds"] for p in picks if p["odds"] > 0]
        comp_odds = calc_composite_odds(odds_list) if odds_list else 0
        min_odds = calc_min_composite_odds(total_prob)
        ticket_groups.append({
            "bet_type": "複勝",
            "summary": " / ".join(f"{p['num']}番{name(p['num'])}" for p in picks),
            "picks": [{"numbers": str(p["num"]), "names": name(p["num"]),
                       "odds": p["odds"], "prob": p["prob"]} for p in picks],
            "composite_odds": comp_odds,
            "min_composite_odds": min_odds,
            "total_prob": total_prob,
            "n_bets": len(picks),
        })

    # === 3. 馬連（1頭軸 × 相手3頭）===
    quin_picks = []
    for partner in numbers[1:8]:
        wi = probs["win"].get(axis, 0)
        wj = probs["win"].get(partner, 0)
        prob = wi * wj / max(1 - wi, 0.01) + wj * wi / max(1 - wj, 0.01)
        prob = min(prob, 0.3)
        if prob >= 0.20:
            quin_picks.append({"partner": partner, "prob": prob})

    if quin_picks:
        quin_picks.sort(key=lambda x: x["prob"], reverse=True)
        picks = quin_picks[:3]
        total_prob = sum(p["prob"] for p in picks)
        min_odds = calc_min_composite_odds(total_prob)
        partners = [p["partner"] for p in picks]
        ticket_groups.append({
            "bet_type": "馬連",
            "summary": f"軸 {axis}番{name(axis)} − {', '.join(f'{p}番' for p in partners)}",
            "picks": [{"numbers": f"{axis}-{p['partner']}",
                       "names": f"{name(axis)}-{name(p['partner'])}",
                       "odds": 0, "prob": p["prob"]} for p in picks],
            "composite_odds": 0,
            "min_composite_odds": min_odds,
            "total_prob": total_prob,
            "n_bets": len(picks),
        })

    # === 4. 馬単（1頭軸 → 相手3頭）===
    exacta_picks = []
    for partner in numbers[1:8]:
        prob = probs["win"].get(axis, 0) * probs["win"].get(partner, 0) / max(1 - probs["win"].get(axis, 0), 0.01)
        if prob >= 0.20:
            exacta_picks.append({"partner": partner, "prob": prob})

    if exacta_picks:
        exacta_picks.sort(key=lambda x: x["prob"], reverse=True)
        picks = exacta_picks[:3]
        total_prob = sum(p["prob"] for p in picks)
        min_odds = calc_min_composite_odds(total_prob)
        partners = [p["partner"] for p in picks]
        ticket_groups.append({
            "bet_type": "馬単",
            "summary": f"1着 {axis}番{name(axis)} → 2着 {', '.join(f'{p}番' for p in partners)}",
            "picks": [{"numbers": f"{axis}→{p['partner']}",
                       "names": f"{name(axis)}→{name(p['partner'])}",
                       "odds": 0, "prob": p["prob"]} for p in picks],
            "composite_odds": 0,
            "min_composite_odds": min_odds,
            "total_prob": total_prob,
            "n_bets": len(picks),
        })

    # === 的中確率 ===
    miss_prob = 1.0
    for g in ticket_groups:
        miss_prob *= (1 - g["total_prob"])
    any_hit = 1 - miss_prob

    return {
        "ticket_groups": ticket_groups,
        "any_hit_probability": any_hit,
    }
