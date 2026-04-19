"""買い目最適化エンジン v4。

AI予測の的中率で買い目を選定する。オッズは単勝のみ表示。
他券種のオッズはユーザーが確認し、合成オッズが基準以上なら購入する。

合成オッズ = 総投資額 / Σ(各投資額 / 各オッズ)

合成オッズ基準:
  単勝: 3.0倍以上
  ワイド: 2.5倍以上
  馬連: 5.0倍以上
  馬単: 8.0倍以上
  三連複: 10.0倍以上
  三連単: 30.0倍以上
"""

import itertools

import numpy as np
import pandas as pd

from src.probability.plackett_luce import compute_race_probabilities
from src.utils.logger import get_logger

logger = get_logger(__name__)

MIN_COMPOSITE_ODDS = {
    "単勝": 3.0,
    "ワイド": 2.5,
    "馬連": 5.0,
    "馬単": 8.0,
    "三連複": 10.0,
    "三連単": 30.0,
}


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
        if prob >= 0.08:
            win_picks.append({"num": num, "odds": odds, "prob": prob})

    if win_picks:
        win_picks.sort(key=lambda x: x["prob"], reverse=True)
        picks = win_picks[:2]
        odds_list = [p["odds"] for p in picks if p["odds"] > 0]
        comp_odds = calc_composite_odds(odds_list) if odds_list else 0
        ticket_groups.append({
            "bet_type": "単勝",
            "picks": [{"numbers": str(p["num"]), "names": name(p["num"]),
                       "odds": p["odds"], "prob": p["prob"]} for p in picks],
            "composite_odds": comp_odds,
            "total_prob": sum(p["prob"] for p in picks),
        })

    # === 2. ワイド（1頭軸流し、最大2点）===
    axis = numbers[0]
    wide_picks = []
    for partner in numbers[1:6]:
        prob = top3_prob(axis) * top3_prob(partner) * 0.8
        if prob >= 0.05:
            wide_picks.append({
                "nums": f"{axis}-{partner}",
                "names": f"{name(axis)}-{name(partner)}",
                "prob": prob,
            })

    if wide_picks:
        wide_picks.sort(key=lambda x: x["prob"], reverse=True)
        picks = wide_picks[:2]
        ticket_groups.append({
            "bet_type": "ワイド",
            "picks": [{"numbers": p["nums"], "names": p["names"],
                       "odds": 0, "prob": p["prob"]} for p in picks],
            "composite_odds": 0,
            "total_prob": sum(p["prob"] for p in picks),
        })

    # === 3. 馬連（1頭軸 × 相手3頭）===
    quin_picks = []
    for partner in numbers[1:8]:
        wi = probs["win"].get(axis, 0)
        wj = probs["win"].get(partner, 0)
        prob = wi * wj / max(1 - wi, 0.01) + wj * wi / max(1 - wj, 0.01)
        prob = min(prob, 0.3)
        if prob >= 0.02:
            quin_picks.append({
                "nums": f"{axis}-{partner}",
                "names": f"{name(axis)}-{name(partner)}",
                "prob": prob,
            })

    if quin_picks:
        quin_picks.sort(key=lambda x: x["prob"], reverse=True)
        picks = quin_picks[:3]
        ticket_groups.append({
            "bet_type": "馬連",
            "picks": [{"numbers": p["nums"], "names": p["names"],
                       "odds": 0, "prob": p["prob"]} for p in picks],
            "composite_odds": 0,
            "total_prob": sum(p["prob"] for p in picks),
        })

    # === 4. 馬単（1頭軸 → 相手3頭）===
    exacta_picks = []
    for partner in numbers[1:8]:
        prob = probs["win"].get(axis, 0) * probs["win"].get(partner, 0) / max(1 - probs["win"].get(axis, 0), 0.01)
        if prob >= 0.01:
            exacta_picks.append({
                "nums": f"{axis}→{partner}",
                "names": f"{name(axis)}→{name(partner)}",
                "prob": prob,
            })

    if exacta_picks:
        exacta_picks.sort(key=lambda x: x["prob"], reverse=True)
        picks = exacta_picks[:3]
        ticket_groups.append({
            "bet_type": "馬単",
            "picks": [{"numbers": p["nums"], "names": p["names"],
                       "odds": 0, "prob": p["prob"]} for p in picks],
            "composite_odds": 0,
            "total_prob": sum(p["prob"] for p in picks),
        })

    # === 5. 三連複（2頭軸 × 相手5頭流し）===
    axis1, axis2 = numbers[0], numbers[1]
    trio_picks = []
    for partner in numbers[2:10]:
        combo = sorted([axis1, axis2, partner])
        combo_fs = frozenset(combo)
        prob = probs["trio"].get(combo_fs, 0)
        if prob >= 0.005:
            trio_picks.append({
                "nums": f"{combo[0]}-{combo[1]}-{combo[2]}",
                "names": f"{name(combo[0])}-{name(combo[1])}-{name(combo[2])}",
                "prob": prob,
            })

    if trio_picks:
        trio_picks.sort(key=lambda x: x["prob"], reverse=True)
        picks = trio_picks[:10]
        ticket_groups.append({
            "bet_type": "三連複",
            "picks": [{"numbers": p["nums"], "names": p["names"],
                       "odds": 0, "prob": p["prob"]} for p in picks],
            "composite_odds": 0,
            "total_prob": sum(p["prob"] for p in picks),
        })

    # === 6. 三連単フォーメーション（最大12点）===
    first_cands = numbers[:3]
    second_cands = numbers[:5]
    third_cands = numbers[:7]
    tri_picks = []
    for a in first_cands:
        for b in second_cands:
            if b == a: continue
            for c in third_cands:
                if c == a or c == b: continue
                prob = probs["trifecta"].get((a, b, c), 0)
                if prob >= 0.002:
                    tri_picks.append({
                        "nums": f"{a}→{b}→{c}",
                        "names": f"{name(a)}→{name(b)}→{name(c)}",
                        "prob": prob,
                    })

    if tri_picks:
        tri_picks.sort(key=lambda x: x["prob"], reverse=True)
        picks = tri_picks[:12]
        ticket_groups.append({
            "bet_type": "三連単",
            "picks": [{"numbers": p["nums"], "names": p["names"],
                       "odds": 0, "prob": p["prob"]} for p in picks],
            "composite_odds": 0,
            "total_prob": sum(p["prob"] for p in picks),
        })

    # === 的中確率 ===
    miss_prob = 1.0
    for g in ticket_groups:
        miss_prob *= (1 - g["total_prob"])
    any_hit = 1 - miss_prob

    return {
        "ticket_groups": ticket_groups,
        "any_hit_probability": any_hit,
        "bets": [],  # 互換用
        "total_investment": 0,
        "n_buy": 0,
        "n_skip": 0,
    }
