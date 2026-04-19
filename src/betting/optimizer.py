"""買い目最適化エンジン。

実際のオッズを使い、予算内で目標回収率（500-1000%）の
達成確率が最も高い馬券の組み合わせを選定する。
"""

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.probability.plackett_luce import compute_race_probabilities
from src.utils.logger import get_logger

logger = get_logger(__name__)

TARGET_ROI_MIN = 5.0   # 500%
TARGET_ROI_MAX = 10.0  # 1000%


@dataclass
class BetCandidate:
    bet_type: str
    numbers: str       # 馬番表記 "3" or "3-4" or "3→4→7"
    names: str         # 馬名表記
    probability: float
    odds: float
    ev: float
    score: float


def generate_candidates(race_df: pd.DataFrame, all_odds: dict) -> list[BetCandidate]:
    """全馬券候補を実オッズで生成する。"""
    probs = compute_race_probabilities(race_df)
    candidates = []

    numbers = sorted(race_df["number"].astype(int).values)
    n = len(numbers)
    top_horses = race_df.sort_values("win_prob", ascending=False).head(min(8, n))
    top_nums = top_horses["number"].astype(int).values

    def get_name(num):
        r = race_df[race_df["number"] == num]["horse_name"].values
        return r[0] if len(r) > 0 else "?"

    # --- 単勝 ---
    win_odds = all_odds.get("win", {})
    for num in numbers:
        ns = str(num).zfill(2)
        if ns not in win_odds:
            continue
        odds = win_odds[ns]
        prob = probs["win"].get(num, 0)
        if prob <= 0 or odds <= 0:
            continue
        candidates.append(BetCandidate(
            bet_type="単勝", numbers=str(num), names=get_name(num),
            probability=prob, odds=odds, ev=prob * odds, score=0,
        ))

    # --- ワイド ---
    wide_odds = all_odds.get("wide", {})
    for i, j in itertools.combinations(top_nums, 2):
        key = f"{min(i,j):02d}-{max(i,j):02d}"
        if key not in wide_odds:
            continue
        odds = wide_odds[key]
        pi = race_df[race_df["number"] == i]["pred_top3_prob"].values[0]
        pj = race_df[race_df["number"] == j]["pred_top3_prob"].values[0]
        prob = pi * pj * 0.8
        if prob <= 0 or odds <= 0:
            continue
        candidates.append(BetCandidate(
            bet_type="ワイド", numbers=f"{i}-{j}", names=f"{get_name(i)}-{get_name(j)}",
            probability=prob, odds=odds, ev=prob * odds, score=0,
        ))

    # --- 馬連 ---
    quinella_odds = all_odds.get("quinella", {})
    for i, j in itertools.combinations(top_nums, 2):
        key = f"{min(i,j):02d}-{max(i,j):02d}"
        if key not in quinella_odds:
            continue
        odds = quinella_odds[key]
        wi = probs["win"].get(i, 0)
        wj = probs["win"].get(j, 0)
        prob = wi * wj / max(1 - wi, 0.01) + wj * wi / max(1 - wj, 0.01)
        prob = min(prob, 0.5)
        if prob <= 0 or odds <= 0:
            continue
        candidates.append(BetCandidate(
            bet_type="馬連", numbers=f"{i}-{j}", names=f"{get_name(i)}-{get_name(j)}",
            probability=prob, odds=odds, ev=prob * odds, score=0,
        ))

    # --- 馬単 ---
    exacta_odds = all_odds.get("exacta", {})
    for i, j in itertools.permutations(top_nums[:6], 2):
        key = f"{i:02d}→{j:02d}"
        if key not in exacta_odds:
            continue
        odds = exacta_odds[key]
        prob = probs["win"].get(i, 0) * probs["win"].get(j, 0) / max(1 - probs["win"].get(i, 0), 0.01)
        if prob <= 0 or odds <= 0:
            continue
        candidates.append(BetCandidate(
            bet_type="馬単", numbers=f"{i}→{j}", names=f"{get_name(i)}→{get_name(j)}",
            probability=prob, odds=odds, ev=prob * odds, score=0,
        ))

    # --- 三連複 ---
    trio_odds = all_odds.get("trio", {})
    for combo in itertools.combinations(top_nums[:6], 3):
        combo_fs = frozenset(combo)
        prob = probs["trio"].get(combo_fs, 0)
        if prob <= 0:
            continue
        s = sorted(combo)
        # 三連複のキーを探す（APIは2頭軸形式なので、3頭の組み合わせを探す）
        # 試行: 01-0203 形式ではなく、0102形式で1頭目固定
        key = f"{s[0]:02d}-{s[1]:02d}"
        odds = trio_odds.get(key, 0)
        if odds <= 0:
            # 別の組み合わせを試す
            key2 = f"{s[0]:02d}-{s[2]:02d}"
            odds = trio_odds.get(key2, 0)
        if odds <= 0:
            continue
        candidates.append(BetCandidate(
            bet_type="三連複", numbers=f"{s[0]}-{s[1]}-{s[2]}",
            names=f"{get_name(s[0])}-{get_name(s[1])}-{get_name(s[2])}",
            probability=prob, odds=odds, ev=prob * odds, score=0,
        ))

    # --- 三連単 ---
    trifecta_odds = all_odds.get("trifecta", {})
    for combo in itertools.permutations(top_nums[:5], 3):
        prob = probs["trifecta"].get(combo, 0)
        if prob <= 0:
            continue
        key = f"{combo[0]:02d}→{combo[1]:02d}→{combo[2]:02d}"
        odds = trifecta_odds.get(key, 0)
        if odds <= 0:
            continue
        candidates.append(BetCandidate(
            bet_type="三連単", numbers=f"{combo[0]}→{combo[1]}→{combo[2]}",
            names=f"{get_name(combo[0])}→{get_name(combo[1])}→{get_name(combo[2])}",
            probability=prob, odds=odds, ev=prob * odds, score=0,
        ))

    # --- スコアリング ---
    for c in candidates:
        hit_score = min(c.probability * 100, 30) * 0.30
        ev_score = min(c.ev, 5.0) / 5.0 * 25 * 0.25

        roi = c.odds
        if TARGET_ROI_MIN <= roi <= TARGET_ROI_MAX:
            roi_score = 25
        elif roi > TARGET_ROI_MAX:
            roi_score = max(0, 25 - (roi - TARGET_ROI_MAX) * 0.5)
        else:
            roi_score = max(0, 25 - (TARGET_ROI_MIN - roi) * 3)
        roi_score *= 0.25

        reliability = min(c.probability * 50, 20) * 0.20
        c.score = hit_score + ev_score + roi_score + reliability

    return sorted(candidates, key=lambda x: x.score, reverse=True)


def optimize_bets(candidates: list[BetCandidate], budget: int) -> dict:
    """予算内で最適な買い目の組み合わせを選定する。"""
    unit = 100
    selected = []
    spent = 0

    for c in candidates:
        if spent >= budget:
            break
        type_count = sum(1 for s in selected if s["bet_type"] == c.bet_type)
        if c.bet_type == "単勝" and type_count >= 3:
            continue
        if c.bet_type in ("ワイド", "馬連") and type_count >= 4:
            continue
        if c.bet_type in ("馬単", "三連複", "三連単") and type_count >= 3:
            continue

        amount = unit
        if c.score >= 15 and budget >= 2000:
            amount = min(300, budget - spent)
        elif c.score >= 10 and budget >= 1500:
            amount = min(200, budget - spent)

        if spent + amount > budget:
            amount = budget - spent
        if amount < unit:
            break

        payout = c.odds * amount
        selected.append({
            "bet_type": c.bet_type,
            "numbers": c.numbers,
            "names": c.names,
            "amount": amount,
            "probability": c.probability,
            "odds": c.odds,
            "payout": payout,
            "ev": c.ev,
            "score": c.score,
        })
        spent += amount

    miss_prob = 1.0
    for s in selected:
        miss_prob *= (1 - s["probability"])
    any_hit_prob = 1 - miss_prob

    payouts = [s["payout"] for s in selected]

    return {
        "bets": selected,
        "total_investment": spent,
        "any_hit_probability": any_hit_prob,
        "min_payout": min(payouts) if payouts else 0,
        "max_payout": max(payouts) if payouts else 0,
        "n_patterns": len(selected),
    }
