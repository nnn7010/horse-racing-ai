"""買い目最適化エンジン v2。

券種別ルールに基づき、的中率とオッズのバランスで買い目を選定する。
妙味がなければ買わない。予算を使い切る義務もない。

券種別上限:
  単勝: 2点
  ワイド: 1頭軸流し 2点
  馬連: 1頭軸 × 相手3頭
  馬単: 1頭軸 → 相手3頭
  三連複: 2頭軸 × 相手5頭流し
  三連単: フォーメーション 12点
"""

import itertools
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.probability.plackett_luce import compute_race_probabilities
from src.utils.logger import get_logger

logger = get_logger(__name__)

# 妙味の最低基準（これ以下は買わない）
MIN_VALUE_SCORE = 0.5  # 的中率 × オッズ のバランス指標


@dataclass
class Bet:
    bet_type: str
    numbers: str
    names: str
    amount: int
    probability: float
    odds: float
    ev: float
    reason: str = ""


def build_recommendations(race_df: pd.DataFrame, all_odds: dict, budget: int) -> dict:
    """レースの予測とオッズから、ルールに基づいた買い目を生成する。"""

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
    wide_odds = all_odds.get("wide", {})
    quinella_odds = all_odds.get("quinella", {})
    exacta_odds = all_odds.get("exacta", {})
    trio_odds = all_odds.get("trio", {})
    trifecta_odds = all_odds.get("trifecta", {})

    bets = []
    spent = 0

    # === 1. 単勝（最大2点）===
    win_candidates = []
    for num in numbers:
        ns = str(num).zfill(2)
        if ns not in win_odds or win_odds[ns] <= 0:
            continue
        odds = win_odds[ns]
        prob = probs["win"].get(num, 0)
        ev = prob * odds
        # 妙味判定: 的中率とオッズのバランス
        # 低オッズ高的中(2倍30%)も高オッズ低的中(50倍1%)も避ける
        # 5-30倍 × 3-15%あたりが狙い目
        value = prob * min(odds, 30)  # オッズ30倍以上は頭打ち
        if value > MIN_VALUE_SCORE and odds >= 3.0:
            win_candidates.append({
                "num": num, "odds": odds, "prob": prob, "ev": ev, "value": value,
            })

    win_candidates.sort(key=lambda x: x["value"], reverse=True)
    for c in win_candidates[:2]:
        if spent + 100 > budget:
            break
        bets.append(Bet(
            bet_type="単勝", numbers=str(c["num"]), names=name(c["num"]),
            amount=100, probability=c["prob"], odds=c["odds"], ev=c["ev"],
            reason=f"勝率{c['prob']:.1%} × {c['odds']:.1f}倍",
        ))
        spent += 100

    # === 2. ワイド（1頭軸流し、最大2点）===
    # 軸: 複勝率最高の馬
    axis_wide = numbers[0]  # AI1位
    wide_candidates = []
    for partner in numbers[1:8]:
        key = f"{min(axis_wide, partner):02d}-{max(axis_wide, partner):02d}"
        if key not in wide_odds or wide_odds[key] <= 0:
            continue
        odds = wide_odds[key]
        prob = top3_prob(axis_wide) * top3_prob(partner) * 0.8
        ev = prob * odds
        value = prob * min(odds, 15)
        if value > MIN_VALUE_SCORE and odds >= 2.0:
            wide_candidates.append({
                "nums": f"{axis_wide}-{partner}", "names": f"{name(axis_wide)}-{name(partner)}",
                "odds": odds, "prob": prob, "ev": ev, "value": value,
            })

    wide_candidates.sort(key=lambda x: x["value"], reverse=True)
    for c in wide_candidates[:2]:
        if spent + 100 > budget:
            break
        bets.append(Bet(
            bet_type="ワイド", numbers=c["nums"], names=c["names"],
            amount=100, probability=c["prob"], odds=c["odds"], ev=c["ev"],
            reason=f"軸{name(axis_wide)}",
        ))
        spent += 100

    # === 3. 馬連（1頭軸 × 相手3頭）===
    axis_quin = numbers[0]
    quin_candidates = []
    for partner in numbers[1:8]:
        key = f"{min(axis_quin, partner):02d}-{max(axis_quin, partner):02d}"
        if key not in quinella_odds or quinella_odds[key] <= 0:
            continue
        odds = quinella_odds[key]
        wi = probs["win"].get(axis_quin, 0)
        wj = probs["win"].get(partner, 0)
        prob = wi * wj / max(1 - wi, 0.01) + wj * wi / max(1 - wj, 0.01)
        prob = min(prob, 0.3)
        ev = prob * odds
        value = prob * min(odds, 30)
        if value > MIN_VALUE_SCORE and odds >= 5.0:
            quin_candidates.append({
                "nums": f"{axis_quin}-{partner}", "names": f"{name(axis_quin)}-{name(partner)}",
                "odds": odds, "prob": prob, "ev": ev, "value": value,
            })

    quin_candidates.sort(key=lambda x: x["value"], reverse=True)
    for c in quin_candidates[:3]:
        if spent + 100 > budget:
            break
        bets.append(Bet(
            bet_type="馬連", numbers=c["nums"], names=c["names"],
            amount=100, probability=c["prob"], odds=c["odds"], ev=c["ev"],
            reason=f"軸{name(axis_quin)}",
        ))
        spent += 100

    # === 4. 馬単（1頭軸 → 相手3頭）===
    axis_exacta = numbers[0]
    exacta_candidates = []
    for partner in numbers[1:8]:
        # 軸が1着 → 相手が2着
        key = f"{axis_exacta:02d}→{partner:02d}"
        if key not in exacta_odds or exacta_odds[key] <= 0:
            continue
        odds = exacta_odds[key]
        prob = probs["win"].get(axis_exacta, 0) * probs["win"].get(partner, 0) / max(1 - probs["win"].get(axis_exacta, 0), 0.01)
        ev = prob * odds
        value = prob * min(odds, 50)
        if value > MIN_VALUE_SCORE and odds >= 8.0:
            exacta_candidates.append({
                "nums": f"{axis_exacta}→{partner}", "names": f"{name(axis_exacta)}→{name(partner)}",
                "odds": odds, "prob": prob, "ev": ev, "value": value,
            })

    exacta_candidates.sort(key=lambda x: x["value"], reverse=True)
    for c in exacta_candidates[:3]:
        if spent + 100 > budget:
            break
        bets.append(Bet(
            bet_type="馬単", numbers=c["nums"], names=c["names"],
            amount=100, probability=c["prob"], odds=c["odds"], ev=c["ev"],
            reason=f"1着固定{name(axis_exacta)}",
        ))
        spent += 100

    # === 5. 三連複（2頭軸 × 相手5頭流し）===
    axis1, axis2 = numbers[0], numbers[1]
    trio_candidates = []
    for partner in numbers[2:10]:
        combo = sorted([axis1, axis2, partner])
        combo_fs = frozenset(combo)
        prob = probs["trio"].get(combo_fs, 0)
        if prob <= 0:
            continue
        # オッズ取得（APIのキー構造に合わせる）
        key = f"{combo[0]:02d}-{combo[1]:02d}"
        odds = trio_odds.get(key, 0)
        if odds <= 0:
            key2 = f"{combo[0]:02d}-{combo[2]:02d}"
            odds = trio_odds.get(key2, 0)
        if odds <= 0:
            continue
        ev = prob * odds
        value = prob * min(odds, 100)
        if value > MIN_VALUE_SCORE and odds >= 10.0:
            trio_candidates.append({
                "nums": f"{combo[0]}-{combo[1]}-{combo[2]}",
                "names": f"{name(combo[0])}-{name(combo[1])}-{name(combo[2])}",
                "odds": odds, "prob": prob, "ev": ev, "value": value,
            })

    trio_candidates.sort(key=lambda x: x["value"], reverse=True)
    for c in trio_candidates[:10]:
        if spent + 100 > budget:
            break
        bets.append(Bet(
            bet_type="三連複", numbers=c["nums"], names=c["names"],
            amount=100, probability=c["prob"], odds=c["odds"], ev=c["ev"],
            reason=f"軸{name(axis1)}-{name(axis2)}",
        ))
        spent += 100

    # === 6. 三連単フォーメーション（最大12点）===
    # 1着候補 × 2着候補 × 3着候補
    first_candidates = numbers[:3]   # AI上位3頭
    second_candidates = numbers[:5]  # AI上位5頭
    third_candidates = numbers[:7]   # AI上位7頭

    tri_candidates = []
    for a in first_candidates:
        for b in second_candidates:
            if b == a:
                continue
            for c in third_candidates:
                if c == a or c == b:
                    continue
                combo = (a, b, c)
                prob = probs["trifecta"].get(combo, 0)
                if prob <= 0:
                    continue
                key = f"{a:02d}→{b:02d}→{c:02d}"
                odds = trifecta_odds.get(key, 0)
                if odds <= 0:
                    continue
                ev = prob * odds
                value = prob * min(odds, 200)
                if value > MIN_VALUE_SCORE and odds >= 20.0:
                    tri_candidates.append({
                        "nums": f"{a}→{b}→{c}",
                        "names": f"{name(a)}→{name(b)}→{name(c)}",
                        "odds": odds, "prob": prob, "ev": ev, "value": value,
                    })

    tri_candidates.sort(key=lambda x: x["value"], reverse=True)
    for c in tri_candidates[:12]:
        if spent + 100 > budget:
            break
        bets.append(Bet(
            bet_type="三連単", numbers=c["nums"], names=c["names"],
            amount=100, probability=c["prob"], odds=c["odds"], ev=c["ev"],
            reason="フォーメーション",
        ))
        spent += 100

    # 的中確率の計算
    miss_prob = 1.0
    for b in bets:
        miss_prob *= (1 - b.probability)
    any_hit = 1 - miss_prob

    payouts = [b.odds * b.amount for b in bets]

    return {
        "bets": [{
            "bet_type": b.bet_type, "numbers": b.numbers, "names": b.names,
            "amount": b.amount, "probability": b.probability, "odds": b.odds,
            "payout": b.odds * b.amount, "ev": b.ev, "reason": b.reason,
        } for b in bets],
        "total_investment": spent,
        "any_hit_probability": any_hit,
        "min_payout": min(payouts) if payouts else 0,
        "max_payout": max(payouts) if payouts else 0,
        "n_patterns": len(bets),
        "unused_budget": budget - spent,
    }


# 後方互換（ダッシュボードから呼ばれる旧インターフェース）
def generate_candidates(race_df, all_odds):
    """旧インターフェース互換。内部でbuild_recommendationsを呼ぶ。"""
    return []  # 使われなくなる


def optimize_bets(candidates, budget):
    """旧インターフェース互換。"""
    return {"bets": [], "total_investment": 0, "any_hit_probability": 0,
            "min_payout": 0, "max_payout": 0, "n_patterns": 0}
