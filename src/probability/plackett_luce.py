"""Plackett-Luceモデルによる組み合わせ確率の算出。

Stage1の複勝圏内確率から、三連複・三連単の組み合わせ確率を計算する。
"""

import itertools
from functools import lru_cache

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def plackett_luce_top3(probs: np.ndarray, indices: np.ndarray) -> dict:
    """Plackett-Luceモデルで上位3頭の全組み合わせ確率を計算する。

    Args:
        probs: 各馬の強さパラメータ（正規化済み確率）
        indices: 馬番号の配列

    Returns:
        trifecta: {(1着, 2着, 3着): 確率} の辞書（三連単）
        trio: {frozenset(1着, 2着, 3着): 確率} の辞書（三連複）
    """
    n = len(probs)
    if n < 3:
        return {"trifecta": {}, "trio": {}}

    # 頭数が多い場合は上位候補に絞る（計算量削減）
    if n > 14:
        top_k = 14
        top_idx = np.argsort(probs)[::-1][:top_k]
        probs = probs[top_idx]
        indices = indices[top_idx]
        n = top_k

    total = probs.sum()
    trifecta = {}
    trio = {}

    # 三連単: P(i,j,k) = p_i/S * p_j/(S-p_i) * p_k/(S-p_i-p_j)
    for i in range(n):
        pi = probs[i]
        s1 = total - pi
        if s1 <= 0:
            continue
        prob1 = pi / total

        for j in range(n):
            if j == i:
                continue
            pj = probs[j]
            s2 = s1 - pj
            if s2 <= 0:
                continue
            prob2 = pj / s1

            for k in range(n):
                if k == i or k == j:
                    continue
                pk = probs[k]
                prob3 = pk / s2

                prob_ijk = prob1 * prob2 * prob3
                key = (indices[i], indices[j], indices[k])
                trifecta[key] = prob_ijk

                # 三連複
                trio_key = frozenset(key)
                trio[trio_key] = trio.get(trio_key, 0.0) + prob_ijk

    return {"trifecta": trifecta, "trio": trio}


def compute_race_probabilities(race_df: pd.DataFrame) -> dict:
    """1レース分の全馬券種確率を計算する。

    Args:
        race_df: 1レースの出走馬データ（pred_top3_prob列が必要）

    Returns:
        win: {馬番: 勝率}
        place: {馬番: 複勝確率}
        trifecta: {(1着,2着,3着): 確率}
        trio: {frozenset: 確率}
    """
    numbers = race_df["number"].values.copy()
    n = len(numbers)

    # Plackett-Luceの強さパラメータ
    # pred_strength（生のodds比）があればそれを使用、なければpred_top3_probにフォールバック
    if "pred_strength" in race_df.columns:
        strengths = race_df["pred_strength"].values.copy()
    else:
        strengths = race_df["pred_top3_prob"].values.copy()
    strengths = np.maximum(strengths, 1e-6)

    # 単勝確率: winモデルがあればpred_win_prob、なければPL強さから導出
    if "pred_win_prob" in race_df.columns:
        win_probs = race_df["pred_win_prob"].values.copy()
        win_probs = np.maximum(win_probs, 1e-6)
        # 正規化（合計=1を保証）
        win_total = win_probs.sum()
        if win_total > 0:
            win_probs = win_probs / win_total
    else:
        total = strengths.sum()
        win_probs = strengths / total if total > 0 else np.ones(n) / n

    # 複勝確率: キャリブレーション済みのpred_top3_probを使用
    place_raw = race_df["pred_top3_prob"].values.copy()
    place_raw = np.maximum(place_raw, 1e-6)
    place_probs = np.minimum(place_raw, 0.99)

    win = dict(zip(numbers, win_probs))
    place = dict(zip(numbers, place_probs))

    # 三連複・三連単: 強さパラメータで計算（winモデルのstrengthを使用）
    pl_result = plackett_luce_top3(strengths, numbers)

    return {
        "win": win,
        "place": place,
        "trifecta": pl_result["trifecta"],
        "trio": pl_result["trio"],
    }
