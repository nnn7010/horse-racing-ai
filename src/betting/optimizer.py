"""買い目最適化エンジン v3。

AI予測で買い目を決定し、実オッズで買うかどうかを判断する。
複数点買いの券種は合成オッズで判断。

合成オッズ = 1 / Σ(1/各オッズ)
= どの買い目が的中しても同額になるよう資金配分した場合の倍率

券種別ルール:
  単勝: 最大2点
  ワイド: 1頭軸流し 最大2点
  馬連: 1頭軸 × 相手3頭
  馬単: 1頭軸 → 相手3頭
  三連複: 2頭軸 × 相手5頭流し
  三連単: フォーメーション 最大12点
"""

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.probability.plackett_luce import compute_race_probabilities
from src.utils.logger import get_logger

logger = get_logger(__name__)

# 券種別の合成オッズ最低基準（これ以下なら買わない）
MIN_COMPOSITE_ODDS = {
    "単勝": 3.0,
    "ワイド": 2.5,
    "馬連": 5.0,
    "馬単": 8.0,
    "三連複": 10.0,
    "三連単": 30.0,
}


def calc_composite_odds(odds_list: list[float]) -> float:
    """合成オッズを計算する。

    合成オッズ = 1 / Σ(1/各オッズ)
    """
    if not odds_list:
        return 0.0
    inv_sum = sum(1.0 / o for o in odds_list if o > 0)
    if inv_sum <= 0:
        return 0.0
    return 1.0 / inv_sum


def build_recommendations(race_df: pd.DataFrame, all_odds: dict, budget: int) -> dict:
    """AI予測で買い目を選定し、合成オッズで買い判断する。"""

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

    ticket_groups = []  # 券種ごとのグループ

    # === 1. 単勝（最大2点）===
    win_picks = []
    for num in numbers[:6]:
        ns = str(num).zfill(2)
        if ns not in win_odds or win_odds[ns] <= 0:
            continue
        odds = win_odds[ns]
        prob = probs["win"].get(num, 0)
        # 的中率重視: AI上位で妙味がある馬
        if prob >= 0.08:  # 勝率8%以上
            win_picks.append({"num": num, "odds": odds, "prob": prob})

    if win_picks:
        win_picks.sort(key=lambda x: x["prob"], reverse=True)
        picks = win_picks[:2]
        odds_list = [p["odds"] for p in picks]
        comp_odds = calc_composite_odds(odds_list)
        min_odds = MIN_COMPOSITE_ODDS["単勝"]
        ticket_groups.append({
            "bet_type": "単勝",
            "picks": [{"numbers": str(p["num"]), "names": name(p["num"]),
                       "odds": p["odds"], "prob": p["prob"]} for p in picks],
            "composite_odds": comp_odds,
            "min_odds": min_odds,
            "buy": comp_odds >= min_odds,
            "total_prob": sum(p["prob"] for p in picks),
        })

    # === 2. ワイド（1頭軸流し、最大2点）===
    axis = numbers[0]
    wide_picks = []
    for partner in numbers[1:6]:
        key = f"{min(axis, partner):02d}-{max(axis, partner):02d}"
        if key not in wide_odds or wide_odds[key] <= 0:
            continue
        odds = wide_odds[key]
        prob = top3_prob(axis) * top3_prob(partner) * 0.8
        if prob >= 0.05:
            wide_picks.append({"nums": f"{axis}-{partner}", "names": f"{name(axis)}-{name(partner)}",
                               "odds": odds, "prob": prob})

    if wide_picks:
        wide_picks.sort(key=lambda x: x["prob"], reverse=True)
        picks = wide_picks[:2]
        comp_odds = calc_composite_odds([p["odds"] for p in picks])
        min_odds = MIN_COMPOSITE_ODDS["ワイド"]
        ticket_groups.append({
            "bet_type": "ワイド",
            "picks": [{"numbers": p["nums"], "names": p["names"],
                       "odds": p["odds"], "prob": p["prob"]} for p in picks],
            "composite_odds": comp_odds,
            "min_odds": min_odds,
            "buy": comp_odds >= min_odds,
            "total_prob": sum(p["prob"] for p in picks),
        })

    # === 3. 馬連（1頭軸 × 相手3頭）===
    quin_picks = []
    for partner in numbers[1:8]:
        key = f"{min(axis, partner):02d}-{max(axis, partner):02d}"
        if key not in quinella_odds or quinella_odds[key] <= 0:
            continue
        odds = quinella_odds[key]
        wi = probs["win"].get(axis, 0)
        wj = probs["win"].get(partner, 0)
        prob = wi * wj / max(1 - wi, 0.01) + wj * wi / max(1 - wj, 0.01)
        prob = min(prob, 0.3)
        if prob >= 0.02:
            quin_picks.append({"nums": f"{axis}-{partner}", "names": f"{name(axis)}-{name(partner)}",
                               "odds": odds, "prob": prob})

    if quin_picks:
        quin_picks.sort(key=lambda x: x["prob"], reverse=True)
        picks = quin_picks[:3]
        comp_odds = calc_composite_odds([p["odds"] for p in picks])
        min_odds = MIN_COMPOSITE_ODDS["馬連"]
        ticket_groups.append({
            "bet_type": "馬連",
            "picks": [{"numbers": p["nums"], "names": p["names"],
                       "odds": p["odds"], "prob": p["prob"]} for p in picks],
            "composite_odds": comp_odds,
            "min_odds": min_odds,
            "buy": comp_odds >= min_odds,
            "total_prob": sum(p["prob"] for p in picks),
        })

    # === 4. 馬単（1頭軸 → 相手3頭）===
    exacta_picks = []
    for partner in numbers[1:8]:
        key = f"{axis:02d}→{partner:02d}"
        if key not in exacta_odds or exacta_odds[key] <= 0:
            continue
        odds = exacta_odds[key]
        prob = probs["win"].get(axis, 0) * probs["win"].get(partner, 0) / max(1 - probs["win"].get(axis, 0), 0.01)
        if prob >= 0.01:
            exacta_picks.append({"nums": f"{axis}→{partner}", "names": f"{name(axis)}→{name(partner)}",
                                 "odds": odds, "prob": prob})

    if exacta_picks:
        exacta_picks.sort(key=lambda x: x["prob"], reverse=True)
        picks = exacta_picks[:3]
        comp_odds = calc_composite_odds([p["odds"] for p in picks])
        min_odds = MIN_COMPOSITE_ODDS["馬単"]
        ticket_groups.append({
            "bet_type": "馬単",
            "picks": [{"numbers": p["nums"], "names": p["names"],
                       "odds": p["odds"], "prob": p["prob"]} for p in picks],
            "composite_odds": comp_odds,
            "min_odds": min_odds,
            "buy": comp_odds >= min_odds,
            "total_prob": sum(p["prob"] for p in picks),
        })

    # === 5. 三連複（2頭軸 × 相手5頭流し）===
    axis1, axis2 = numbers[0], numbers[1]
    trio_picks = []
    for partner in numbers[2:10]:
        combo = sorted([axis1, axis2, partner])
        combo_fs = frozenset(combo)
        prob = probs["trio"].get(combo_fs, 0)
        if prob <= 0.005:
            continue
        # オッズ: 2頭軸のオッズを近似値として使う
        key = f"{combo[0]:02d}-{combo[1]:02d}"
        odds = trio_odds.get(key, 0)
        if odds <= 0:
            key2 = f"{combo[0]:02d}-{combo[2]:02d}"
            odds = trio_odds.get(key2, 0)
        if odds <= 0:
            continue
        trio_picks.append({
            "nums": f"{combo[0]}-{combo[1]}-{combo[2]}",
            "names": f"{name(combo[0])}-{name(combo[1])}-{name(combo[2])}",
            "odds": odds, "prob": prob,
        })

    if trio_picks:
        trio_picks.sort(key=lambda x: x["prob"], reverse=True)
        picks = trio_picks[:10]
        comp_odds = calc_composite_odds([p["odds"] for p in picks])
        min_odds = MIN_COMPOSITE_ODDS["三連複"]
        ticket_groups.append({
            "bet_type": "三連複",
            "picks": [{"numbers": p["nums"], "names": p["names"],
                       "odds": p["odds"], "prob": p["prob"]} for p in picks],
            "composite_odds": comp_odds,
            "min_odds": min_odds,
            "buy": comp_odds >= min_odds,
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
                combo = (a, b, c)
                prob = probs["trifecta"].get(combo, 0)
                if prob <= 0.002:
                    continue
                key = f"{a:02d}→{b:02d}→{c:02d}"
                odds = trifecta_odds.get(key, 0)
                if odds <= 0:
                    continue
                tri_picks.append({
                    "nums": f"{a}→{b}→{c}",
                    "names": f"{name(a)}→{name(b)}→{name(c)}",
                    "odds": odds, "prob": prob,
                })

    if tri_picks:
        tri_picks.sort(key=lambda x: x["prob"], reverse=True)
        picks = tri_picks[:12]
        comp_odds = calc_composite_odds([p["odds"] for p in picks])
        min_odds = MIN_COMPOSITE_ODDS["三連単"]
        ticket_groups.append({
            "bet_type": "三連単",
            "picks": [{"numbers": p["nums"], "names": p["names"],
                       "odds": p["odds"], "prob": p["prob"]} for p in picks],
            "composite_odds": comp_odds,
            "min_odds": min_odds,
            "buy": comp_odds >= min_odds,
            "total_prob": sum(p["prob"] for p in picks),
        })

    # === 結果整理 ===
    bets = []
    total_investment = 0
    for group in ticket_groups:
        for pick in group["picks"]:
            bets.append({
                "bet_type": group["bet_type"],
                "numbers": pick["numbers"],
                "names": pick["names"],
                "amount": 100,  # 実際は合成オッズで均等配分するが、表示上は100円
                "probability": pick["prob"],
                "odds": pick["odds"],
                "payout": pick["odds"] * 100,
                "ev": pick["prob"] * pick["odds"],
                "composite_odds": group["composite_odds"],
                "min_odds": group["min_odds"],
                "buy": group["buy"],
            })
            if group["buy"]:
                total_investment += 100

    # 的中確率（買い推奨の券種のみ）
    buy_groups = [g for g in ticket_groups if g["buy"]]
    miss_prob = 1.0
    for g in buy_groups:
        miss_prob *= (1 - g["total_prob"])
    any_hit = 1 - miss_prob

    buy_bets = [b for b in bets if b["buy"]]
    payouts = [b["payout"] for b in buy_bets] if buy_bets else [0]

    return {
        "bets": bets,
        "ticket_groups": ticket_groups,
        "total_investment": total_investment,
        "any_hit_probability": any_hit,
        "min_payout": min(payouts) if payouts else 0,
        "max_payout": max(payouts) if payouts else 0,
        "n_buy": sum(1 for g in ticket_groups if g["buy"]),
        "n_skip": sum(1 for g in ticket_groups if not g["buy"]),
    }
