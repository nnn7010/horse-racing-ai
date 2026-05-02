"""買い目最適化エンジン v6。

選定基準:
- 単勝: 生win_prob × 実オッズ（EV）> 1.10
- 複勝: pred_top3_prob ≥ 0.60（最大2点）
- 三連複: 軸=win_prob1位、相手=pred_top3_prob上位、EV > 1.30

合成オッズ = 総投資額 / Σ(各投資額 / 各オッズ)
"""

import itertools

import numpy as np
import pandas as pd

from src.probability.plackett_luce import compute_race_probabilities
from src.utils.logger import get_logger

logger = get_logger(__name__)

WIN_EV_THRESHOLD = 1.10
PLACE_PROB_THRESHOLD = 0.70
TRIO_EV_THRESHOLD = 1.30


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
    """的中率から最低合成オッズ基準を算出。"""
    if total_prob <= 0:
        return 999.0
    return (1.0 / total_prob) * margin


def _get_raw_win_prob(race_df: pd.DataFrame, num: int) -> float:
    """キャリブレーション済みの生win_prob（正規化前）を取得。"""
    r = race_df[race_df["number"] == num]["win_prob"].values
    return float(r[0]) if len(r) > 0 else 0.0


def _get_top3_prob(race_df: pd.DataFrame, num: int) -> float:
    r = race_df[race_df["number"] == num]["pred_top3_prob"].values
    return float(r[0]) if len(r) > 0 else 0.0


def _get_name(race_df: pd.DataFrame, num: int) -> str:
    r = race_df[race_df["number"] == num]["horse_name"].values
    return r[0] if len(r) > 0 else "?"


def build_recommendations(race_df: pd.DataFrame, all_odds: dict, budget: int) -> dict:
    """買い目を選定する。"""

    probs = compute_race_probabilities(race_df)
    sorted_df = race_df.sort_values("win_prob", ascending=False)
    numbers = sorted_df["number"].astype(int).values

    win_odds = all_odds.get("win", {})
    trio_odds_dict = all_odds.get("trio", {})
    place_odds_dict = all_odds.get("place", {})
    ticket_groups = []

    # === 1. 単勝: EV = 生win_prob × 実オッズ > 1.10 ===
    win_picks = []
    for num in numbers:
        ns = str(num).zfill(2)
        odds = win_odds.get(ns, 0)
        if odds <= 0:
            continue
        raw_prob = _get_raw_win_prob(race_df, num)
        ev = raw_prob * odds
        if ev > WIN_EV_THRESHOLD:
            win_picks.append({"num": num, "odds": odds, "prob": raw_prob, "ev": ev})

    if win_picks:
        win_picks.sort(key=lambda x: x["ev"], reverse=True)
        picks = win_picks[:3]  # EV上位3点まで
        total_prob = sum(p["prob"] for p in picks)
        odds_list = [p["odds"] for p in picks if p["odds"] > 0]
        comp_odds = calc_composite_odds(odds_list) if odds_list else 0
        ticket_groups.append({
            "bet_type": "単勝",
            "summary": " / ".join(f"{p['num']}番{_get_name(race_df, p['num'])} EV{p['ev']:.2f}" for p in picks),
            "picks": [{"numbers": str(p["num"]), "names": _get_name(race_df, p["num"]),
                       "odds": p["odds"], "prob": p["prob"], "ev": p["ev"]} for p in picks],
            "composite_odds": comp_odds,
            "min_composite_odds": calc_min_composite_odds(total_prob),
            "total_prob": total_prob,
            "n_bets": len(picks),
        })

    # === 2. 複勝: pred_top3_prob ≥ 0.60（最大2点）===
    place_picks = []
    for num in numbers[:6]:
        ns = str(num).zfill(2)
        odds = place_odds_dict.get(ns, 0)
        prob = _get_top3_prob(race_df, num)
        if prob >= PLACE_PROB_THRESHOLD:
            place_picks.append({"num": num, "odds": odds, "prob": prob})

    if place_picks:
        place_picks.sort(key=lambda x: x["prob"], reverse=True)
        picks = place_picks[:2]
        total_prob = sum(p["prob"] for p in picks)
        odds_list = [p["odds"] for p in picks if p["odds"] > 0]
        comp_odds = calc_composite_odds(odds_list) if odds_list else 0
        ticket_groups.append({
            "bet_type": "複勝",
            "summary": " / ".join(f"{p['num']}番{_get_name(race_df, p['num'])}" for p in picks),
            "picks": [{"numbers": str(p["num"]), "names": _get_name(race_df, p["num"]),
                       "odds": p["odds"], "prob": p["prob"]} for p in picks],
            "composite_odds": comp_odds,
            "min_composite_odds": calc_min_composite_odds(total_prob),
            "total_prob": total_prob,
            "n_bets": len(picks),
        })

    # === 3. 三連複: 軸=win_prob1位、相手=pred_top3_prob上位フォーメーション ===
    if len(numbers) >= 3:
        axis_num = int(numbers[0])  # win_prob 1位が軸

        # 相手候補: 軸を除いたpred_top3_prob上位6頭
        form_candidates = [
            (num, _get_top3_prob(race_df, num))
            for num in numbers if int(num) != axis_num
        ]
        form_candidates.sort(key=lambda x: x[1], reverse=True)
        form_nums = [n for n, _ in form_candidates[:6]]

        trio_picks = []
        for j, k in itertools.combinations(form_nums, 2):
            combo = tuple(sorted([axis_num, j, k]))
            key_str = f"{combo[0]:02d}-{combo[1]:02d}-{combo[2]:02d}"
            trio_prob = probs["trio"].get(frozenset(combo), 0)
            trio_odds = trio_odds_dict.get(key_str, 0)
            if trio_odds <= 0 or trio_prob <= 0:
                continue
            ev = trio_prob * trio_odds
            if ev > TRIO_EV_THRESHOLD:
                trio_picks.append({
                    "combo": combo,
                    "prob": trio_prob,
                    "odds": trio_odds,
                    "ev": ev,
                    "key": key_str,
                })

        if trio_picks:
            trio_picks.sort(key=lambda x: x["ev"], reverse=True)
            picks = trio_picks[:6]  # EV上位6点まで
            total_prob = sum(p["prob"] for p in picks)
            odds_list = [p["odds"] for p in picks if p["odds"] > 0]
            comp_odds = calc_composite_odds(odds_list) if odds_list else 0
            ticket_groups.append({
                "bet_type": "三連複",
                "summary": f"{axis_num}番軸 相手フォーメーション",
                "picks": [{
                    "numbers": p["key"].replace("-", "-"),
                    "names": "-".join(_get_name(race_df, n) for n in p["combo"]),
                    "odds": p["odds"],
                    "prob": p["prob"],
                    "ev": p["ev"],
                } for p in picks],
                "composite_odds": comp_odds,
                "min_composite_odds": calc_min_composite_odds(total_prob),
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
