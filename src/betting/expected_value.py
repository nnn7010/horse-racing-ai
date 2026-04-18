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

    全馬の単勝EVを計算し、的中/不的中を正しく記録する。
    """
    probs = compute_race_probabilities(race_df)
    race_id = race_df["race_id"].iloc[0] if "race_id" in race_df.columns else "unknown"

    recommendations = []

    # 単勝: 全馬のオッズと予測確率からEVを計算
    if "win_odds" in race_df.columns:
        for _, row in race_df.iterrows():
            number = row["number"]
            if number not in probs["win"] or row["win_odds"] <= 0:
                continue
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

    # 複勝: 全馬について推定オッズでEV計算
    # 複勝オッズはwin_oddsの約1/3を推定値として使用
    place_payout_map = {}
    if payouts and "place" in payouts:
        for p in payouts["place"]:
            try:
                num = int(p.get("numbers", "").strip())
                place_payout_map[num] = p.get("amount", 0)
            except ValueError:
                pass

    for _, row in race_df.iterrows():
        number = row["number"]
        if number not in probs["place"]:
            continue
        prob = probs["place"][number]
        # 複勝推定オッズ: 単勝オッズの1/3程度
        est_place_odds = max(row.get("win_odds", 0) * 0.35, 1.1) if row.get("win_odds", 0) > 0 else 0
        if est_place_odds <= 0:
            continue
        ev = prob * est_place_odds
        hit = 1 if row.get("finish_position", 0) <= 3 else 0
        actual_payout = place_payout_map.get(int(number), 0) if hit else 0
        if ev > EV_THRESHOLDS["place"]:
            recommendations.append({
                "race_id": race_id,
                "bet_type": "place",
                "numbers": str(int(number)),
                "probability": prob,
                "odds": est_place_odds,
                "expected_value": ev,
                "hit": hit,
                "payout_per_100": actual_payout if hit else 0,
            })

    # 三連複: 上位確率の組み合わせでEV計算
    trio_sorted = sorted(probs["trio"].items(), key=lambda x: x[1], reverse=True)[:20]
    actual_top3 = set(race_df[race_df["finish_position"].between(1, 3)]["number"].astype(int).values)
    trio_payout = 0
    if payouts and "trio" in payouts and payouts["trio"]:
        trio_payout = payouts["trio"][0].get("amount", 0)
    trio_odds = trio_payout / 100 if trio_payout > 0 else 0

    for combo, prob in trio_sorted:
        combo_set = set(int(c) for c in combo)
        # 推定オッズ: 頭数ベース
        n = len(race_df)
        est_odds = max(1.0 / (prob + 1e-9), 10) if trio_odds == 0 else trio_odds
        ev = prob * est_odds
        hit = 1 if combo_set == actual_top3 else 0
        if ev > EV_THRESHOLDS["trio"]:
            recommendations.append({
                "race_id": race_id,
                "bet_type": "trio",
                "numbers": "-".join(str(int(c)) for c in sorted(combo)),
                "probability": prob,
                "odds": est_odds,
                "expected_value": ev,
                "hit": hit,
                "payout_per_100": trio_payout if hit else 0,
            })

    # 三連単: 上位確率の組み合わせでEV計算
    trifecta_sorted = sorted(probs["trifecta"].items(), key=lambda x: x[1], reverse=True)[:20]
    actual_order = tuple(
        race_df[race_df["finish_position"].between(1, 3)]
        .sort_values("finish_position")["number"]
        .astype(int).values
    )
    trifecta_payout = 0
    if payouts and "trifecta" in payouts and payouts["trifecta"]:
        trifecta_payout = payouts["trifecta"][0].get("amount", 0)
    trifecta_odds = trifecta_payout / 100 if trifecta_payout > 0 else 0

    for combo, prob in trifecta_sorted:
        combo_tuple = tuple(int(c) for c in combo)
        est_odds = max(1.0 / (prob + 1e-9), 50) if trifecta_odds == 0 else trifecta_odds
        ev = prob * est_odds
        hit = 1 if combo_tuple == actual_order else 0
        if ev > EV_THRESHOLDS["trifecta"]:
            recommendations.append({
                "race_id": race_id,
                "bet_type": "trifecta",
                "numbers": "-".join(str(int(c)) for c in combo),
                "probability": prob,
                "odds": est_odds,
                "expected_value": ev,
                "hit": hit,
                "payout_per_100": trifecta_payout if hit else 0,
            })

    return recommendations
