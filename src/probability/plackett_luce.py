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
    probs = race_df["pred_top3_prob"].values.copy()
    numbers = race_df["number"].values.copy()
    n = len(probs)

    # 確率を正規化（Plackett-Luceの強さパラメータとして使用）
    probs = np.maximum(probs, 1e-6)

    # 単勝確率: PLモデルの1着確率
    total = probs.sum()
    win_probs = probs / total

    # 複勝確率: 3着以内に入る確率（正規化済み）
    place_probs = race_df["pred_top3_prob_norm"].values.copy() if "pred_top3_prob_norm" in race_df.columns else probs / total * 3

    win = dict(zip(numbers, win_probs))
    place = dict(zip(numbers, np.minimum(place_probs, 0.99)))

    # 三連複・三連単
    pl_result = plackett_luce_top3(probs, numbers)

    return {
        "win": win,
        "place": place,
        "trifecta": pl_result["trifecta"],
        "trio": pl_result["trio"],
    }
