"""期待値計算と馬券抽出モジュール。"""

import numpy as np
import pandas as pd

from src.probability.plackett_luce import compute_race_probabilities
from src.utils.logger import get_logger

logger = get_logger(__name__)

# EV閾値
EV_THRESHOLDS = {
    "win": 1.10,
    "place": 1.10,
    "trio": 1.30,
    "trifecta": 1.50,
}


def compute_expected_values(race_df: pd.DataFrame, odds: dict | None = None) -> list[dict]:
    """1レース分の期待値を計算し、閾値を超える馬券を抽出する。

    Args:
        race_df: 予測済みレースデータ
        odds: オッズ情報 {"win": {馬番: オッズ}, "place": {馬番: オッズ}, ...}

    Returns:
        推奨馬券のリスト
    """
    probs = compute_race_probabilities(race_df)
    race_id = race_df["race_id"].iloc[0] if "race_id" in race_df.columns else "unknown"

    recommendations = []

    # 単勝
    for number, prob in probs["win"].items():
        if odds and "win" in odds and number in odds["win"]:
            odd = odds["win"][number]
        else:
            # オッズがない場合はスキップ or 推定
            continue
        ev = prob * odd
        if ev > EV_THRESHOLDS["win"]:
            recommendations.append({
                "race_id": race_id,
                "bet_type": "win",
                "numbers": str(int(number)),
                "probability": prob,
                "odds": odd,
                "expected_value": ev,
            })

    # 複勝
    for number, prob in probs["place"].items():
        if odds and "place" in odds and number in odds["place"]:
            odd = odds["place"][number]
        else:
            continue
        ev = prob * odd
        if ev > EV_THRESHOLDS["place"]:
            recommendations.append({
                "race_id": race_id,
                "bet_type": "place",
                "numbers": str(int(number)),
                "probability": prob,
                "odds": odd,
                "expected_value": ev,
            })

    # 三連複（上位の組み合わせのみ）
    trio_sorted = sorted(probs["trio"].items(), key=lambda x: x[1], reverse=True)[:50]
    for combo, prob in trio_sorted:
        combo_list = sorted(combo)
        if odds and "trio" in odds:
            combo_key = "-".join(str(int(c)) for c in combo_list)
            if combo_key in odds["trio"]:
                odd = odds["trio"][combo_key]
            else:
                continue
        else:
            continue
        ev = prob * odd
        if ev > EV_THRESHOLDS["trio"]:
            recommendations.append({
                "race_id": race_id,
                "bet_type": "trio",
                "numbers": "-".join(str(int(c)) for c in combo_list),
                "probability": prob,
                "odds": odd,
                "expected_value": ev,
            })

    # 三連単（上位の組み合わせのみ）
    trifecta_sorted = sorted(probs["trifecta"].items(), key=lambda x: x[1], reverse=True)[:100]
    for combo, prob in trifecta_sorted:
        if odds and "trifecta" in odds:
            combo_key = "-".join(str(int(c)) for c in combo)
            if combo_key in odds["trifecta"]:
                odd = odds["trifecta"][combo_key]
            else:
                continue
        else:
            continue
        ev = prob * odd
        if ev > EV_THRESHOLDS["trifecta"]:
            recommendations.append({
                "race_id": race_id,
                "bet_type": "trifecta",
                "numbers": "-".join(str(int(c)) for c in combo),
                "probability": prob,
                "odds": odd,
                "expected_value": ev,
            })

    return recommendations


def compute_ev_from_results(race_df: pd.DataFrame, payouts: dict) -> list[dict]:
    """過去レースの結果と払い戻しから期待値を事後計算する（バックテスト用）。

    オッズの代わりに実際の払い戻し倍率を使って、
    予測確率×倍率で期待値を計算する。
    """
    probs = compute_race_probabilities(race_df)
    race_id = race_df["race_id"].iloc[0] if "race_id" in race_df.columns else "unknown"
    n_runners = len(race_df)

    recommendations = []

    # 単勝: 各馬のオッズがrace_dfにある場合
    if "win_odds" in race_df.columns:
        for _, row in race_df.iterrows():
            number = row["number"]
            if number in probs["win"] and row["win_odds"] > 0:
                prob = probs["win"][number]
                odd = row["win_odds"]
                ev = prob * odd
                hit = 1 if row.get("finish_position", 0) == 1 else 0
                if ev > EV_THRESHOLDS["win"]:
                    recommendations.append({
                        "race_id": race_id,
                        "bet_type": "win",
                        "numbers": str(int(number)),
                        "probability": prob,
                        "odds": odd,
                        "expected_value": ev,
                        "hit": hit,
                        "payout_per_100": odd * 100 if hit else 0,
                    })

    # 複勝: 簡易推定（3着以内の馬のオッズ推定）
    # バックテスト用に、payoutsから複勝払い戻しを使う
    if payouts and "place" in payouts:
        for p in payouts["place"]:
            nums_str = p.get("numbers", "")
            amount = p.get("amount", 0)
            # 複勝の払い戻し倍率
            place_odds = amount / 100 if amount > 0 else 0

            # 番号を取得
            try:
                num = int(nums_str.strip())
            except ValueError:
                continue

            if num in probs["place"]:
                prob = probs["place"][num]
                ev = prob * place_odds
                if ev > EV_THRESHOLDS["place"]:
                    recommendations.append({
                        "race_id": race_id,
                        "bet_type": "place",
                        "numbers": str(num),
                        "probability": prob,
                        "odds": place_odds,
                        "expected_value": ev,
                        "hit": 1,  # 複勝払い戻しがある=的中
                        "payout_per_100": amount,
                    })

    # 三連複
    if payouts and "trio" in payouts:
        for p in payouts["trio"]:
            nums_str = p.get("numbers", "")
            amount = p.get("amount", 0)
            trio_odds = amount / 100 if amount > 0 else 0

            # 組み合わせの確率を取得
            try:
                nums = [int(x.strip()) for x in nums_str.replace("→", "-").replace("－", "-").replace("-", "-").split("-") if x.strip()]
            except ValueError:
                continue

            if len(nums) == 3:
                combo = frozenset(nums)
                if combo in probs["trio"]:
                    prob = probs["trio"][combo]
                    ev = prob * trio_odds
                    if ev > EV_THRESHOLDS["trio"]:
                        recommendations.append({
                            "race_id": race_id,
                            "bet_type": "trio",
                            "numbers": "-".join(str(n) for n in sorted(nums)),
                            "probability": prob,
                            "odds": trio_odds,
                            "expected_value": ev,
                            "hit": 1,
                            "payout_per_100": amount,
                        })

    # 三連単
    if payouts and "trifecta" in payouts:
        for p in payouts["trifecta"]:
            nums_str = p.get("numbers", "")
            amount = p.get("amount", 0)
            trifecta_odds = amount / 100 if amount > 0 else 0

            try:
                nums = [int(x.strip()) for x in nums_str.replace("→", "-").replace("－", "-").replace("-", "-").split("-") if x.strip()]
            except ValueError:
                continue

            if len(nums) == 3:
                combo = tuple(nums)
                if combo in probs["trifecta"]:
                    prob = probs["trifecta"][combo]
                    ev = prob * trifecta_odds
                    if ev > EV_THRESHOLDS["trifecta"]:
                        recommendations.append({
                            "race_id": race_id,
                            "bet_type": "trifecta",
                            "numbers": "-".join(str(n) for n in nums),
                            "probability": prob,
                            "odds": trifecta_odds,
                            "expected_value": ev,
                            "hit": 1,
                            "payout_per_100": amount,
                        })

    return recommendations
