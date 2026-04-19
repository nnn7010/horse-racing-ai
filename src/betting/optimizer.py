"""買い目最適化エンジン。

予算内で目標回収率（500-1000%）の達成確率が最も高い
馬券の組み合わせを選定する。
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
    bet_type: str       # win, place, quinella, exacta, wide, trio, trifecta
    numbers: str        # "3" or "3-4" or "3-4-7"
    probability: float  # 的中確率
    odds: float         # オッズ（推定含む）
    ev: float           # 期待値
    score: float        # 総合スコア
    reason: str         # 推奨理由


def generate_candidates(race_df: pd.DataFrame, odds_data: dict) -> list[BetCandidate]:
    """全馬券候補を生成する。"""
    probs = compute_race_probabilities(race_df)
    candidates = []

    win_odds = odds_data.get("1", {})
    numbers = sorted(race_df["number"].astype(int).values)
    n = len(numbers)

    # 上位馬を特定（勝率上位8頭を候補に）
    top_horses = race_df.sort_values("win_prob", ascending=False).head(min(8, n))
    top_nums = top_horses["number"].astype(int).values

    # --- 単勝 ---
    for num in numbers:
        num_str = str(num).zfill(2)
        if num_str not in win_odds:
            continue
        odds = float(win_odds[num_str][0])
        prob = probs["win"].get(num, 0)
        if prob <= 0:
            continue
        ev = prob * odds
        name = race_df[race_df["number"] == num]["horse_name"].values[0]
        candidates.append(BetCandidate(
            bet_type="単勝", numbers=str(num), probability=prob,
            odds=odds, ev=ev, score=0, reason=f"{name}",
        ))

    # --- ワイド（上位8頭の組み合わせ） ---
    for i, j in itertools.combinations(top_nums, 2):
        prob_i = race_df[race_df["number"] == i]["pred_top3_prob"].values[0]
        prob_j = race_df[race_df["number"] == j]["pred_top3_prob"].values[0]
        # ワイド的中確率の近似: 2頭とも3着以内
        prob = prob_i * prob_j * 0.8  # 独立ではないので補正
        # ワイドオッズ推定: 単勝オッズの積の平方根 × 0.3
        i_str, j_str = str(i).zfill(2), str(j).zfill(2)
        if i_str in win_odds and j_str in win_odds:
            o_i = float(win_odds[i_str][0])
            o_j = float(win_odds[j_str][0])
            odds = max((o_i * o_j) ** 0.5 * 0.3, 1.1)
        else:
            continue
        ev = prob * odds
        name_i = race_df[race_df["number"] == i]["horse_name"].values[0]
        name_j = race_df[race_df["number"] == j]["horse_name"].values[0]
        candidates.append(BetCandidate(
            bet_type="ワイド", numbers=f"{i}-{j}", probability=prob,
            odds=round(odds, 1), ev=ev, score=0, reason=f"{name_i}-{name_j}",
        ))

    # --- 馬連（上位8頭の組み合わせ） ---
    for i, j in itertools.combinations(top_nums, 2):
        prob_ij = probs["win"].get(i, 0) * probs["win"].get(j, 0) / (1 - probs["win"].get(i, 0)) \
                + probs["win"].get(j, 0) * probs["win"].get(i, 0) / (1 - probs["win"].get(j, 0))
        prob_ij = min(prob_ij, 0.5)
        i_str, j_str = str(i).zfill(2), str(j).zfill(2)
        if i_str in win_odds and j_str in win_odds:
            o_i = float(win_odds[i_str][0])
            o_j = float(win_odds[j_str][0])
            odds = max((o_i * o_j) ** 0.5 * 0.7, 2.0)
        else:
            continue
        ev = prob_ij * odds
        name_i = race_df[race_df["number"] == i]["horse_name"].values[0]
        name_j = race_df[race_df["number"] == j]["horse_name"].values[0]
        candidates.append(BetCandidate(
            bet_type="馬連", numbers=f"{i}-{j}", probability=prob_ij,
            odds=round(odds, 1), ev=ev, score=0, reason=f"{name_i}-{name_j}",
        ))

    # --- 馬単（上位6頭の1-2着順列） ---
    for i, j in itertools.permutations(top_nums[:6], 2):
        prob = probs["win"].get(i, 0) * probs["win"].get(j, 0) / (1 - probs["win"].get(i, 0))
        i_str, j_str = str(i).zfill(2), str(j).zfill(2)
        if i_str in win_odds and j_str in win_odds:
            o_i = float(win_odds[i_str][0])
            o_j = float(win_odds[j_str][0])
            odds = max(o_i * o_j * 0.5, 5.0)
        else:
            continue
        ev = prob * odds
        name_i = race_df[race_df["number"] == i]["horse_name"].values[0]
        name_j = race_df[race_df["number"] == j]["horse_name"].values[0]
        candidates.append(BetCandidate(
            bet_type="馬単", numbers=f"{i}→{j}", probability=prob,
            odds=round(odds, 1), ev=ev, score=0, reason=f"{name_i}→{name_j}",
        ))

    # --- 三連複（上位6頭の組み合わせ） ---
    for combo in itertools.combinations(top_nums[:6], 3):
        combo_fs = frozenset(combo)
        prob = probs["trio"].get(combo_fs, 0)
        if prob <= 0:
            continue
        strs = [str(c).zfill(2) for c in combo]
        if all(s in win_odds for s in strs):
            odds_vals = [float(win_odds[s][0]) for s in strs]
            odds = max(np.prod(odds_vals) ** 0.33 * 2.0, 10.0)
        else:
            continue
        ev = prob * odds
        names = [race_df[race_df["number"] == c]["horse_name"].values[0] for c in sorted(combo)]
        candidates.append(BetCandidate(
            bet_type="三連複", numbers="-".join(str(c) for c in sorted(combo)),
            probability=prob, odds=round(odds, 1), ev=ev, score=0,
            reason="-".join(names),
        ))

    # --- 三連単（上位5頭の順列） ---
    for combo in itertools.permutations(top_nums[:5], 3):
        prob = probs["trifecta"].get(combo, 0)
        if prob <= 0:
            continue
        strs = [str(c).zfill(2) for c in combo]
        if all(s in win_odds for s in strs):
            odds_vals = [float(win_odds[s][0]) for s in strs]
            odds = max(np.prod(odds_vals) ** 0.33 * 5.0, 30.0)
        else:
            continue
        ev = prob * odds
        names = [race_df[race_df["number"] == c]["horse_name"].values[0] for c in combo]
        candidates.append(BetCandidate(
            bet_type="三連単", numbers="→".join(str(c) for c in combo),
            probability=prob, odds=round(odds, 1), ev=ev, score=0,
            reason="→".join(names),
        ))

    # --- スコアリング ---
    for c in candidates:
        hit_prob_score = min(c.probability * 100, 30) * 0.30
        ev_score = min(c.ev, 5.0) / 5.0 * 25 * 0.25
        # 回収効率: 的中時に目標レンジ（5-10倍）に入るか
        roi = c.odds
        if TARGET_ROI_MIN <= roi <= TARGET_ROI_MAX:
            roi_score = 25  # ど真ん中
        elif roi > TARGET_ROI_MAX:
            roi_score = max(0, 25 - (roi - TARGET_ROI_MAX) * 2)
        else:
            roi_score = max(0, 25 - (TARGET_ROI_MIN - roi) * 5)
        roi_score *= 0.25

        reliability = min(c.probability * 50, 20) * 0.20

        c.score = hit_prob_score + ev_score + roi_score + reliability

    return sorted(candidates, key=lambda x: x.score, reverse=True)


def optimize_bets(candidates: list[BetCandidate], budget: int) -> list[dict]:
    """予算内で最適な買い目の組み合わせを選定する。"""
    unit = 100
    max_bets = budget // unit

    selected = []
    spent = 0
    used_types = set()

    # 多様性を確保しつつスコア上位から選択
    for c in candidates:
        if spent >= budget:
            break

        # 同じ券種は最大5点まで
        type_count = sum(1 for s in selected if s["bet_type"] == c.bet_type)
        if type_count >= 5:
            continue

        # 単勝は最大3点
        if c.bet_type == "単勝" and type_count >= 3:
            continue

        amount = unit
        # スコアが高い馬券には多く配分（最大500円）
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
            "reason": c.reason,
            "amount": amount,
            "probability": c.probability,
            "odds": c.odds,
            "payout": payout,
            "ev": c.ev,
            "score": c.score,
        })
        spent += amount

    # 的中パターン数と少なくとも1つ当たる確率を計算
    miss_prob = 1.0
    for s in selected:
        miss_prob *= (1 - s["probability"])
    any_hit_prob = 1 - miss_prob

    payouts = [s["payout"] for s in selected]
    min_payout = min(payouts) if payouts else 0
    max_payout = max(payouts) if payouts else 0

    return {
        "bets": selected,
        "total_investment": spent,
        "any_hit_probability": any_hit_prob,
        "min_payout": min_payout,
        "max_payout": max_payout,
        "n_patterns": len(selected),
    }
