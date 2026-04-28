"""コース×馬の適合度スコアと、レース内での3着内率推定。

7軸合算 → softmax → 正規化確率。
EV = 推定勝率 × 想定オッズ - 1。
"""

import math
from typing import Iterable


def _z(x: float | None, mean_: float, std_: float) -> float:
    if x is None or std_ <= 0: return 0.0
    return (x - mean_) / std_


def _safe(x, default=0.0):
    return x if x is not None else default


def score_horse_for_race(
    profile: dict,
    course: dict,
    bracket: int,
    race_distance: int,
    race_track: str,
) -> dict:
    """1頭分のコース適合度スコアを返す。"""
    parts: dict[str, float] = {}

    # 1) ベース能力 (大井での3着内率)
    parts["base"] = _safe(profile.get("oi_top3_rate")) * 100  # 0-100

    # 2) 距離適性: 経験あれば dist_top3、無ければ近い距離で代用
    d = race_distance
    dist_top3 = profile.get("dist_top3", {}).get(d) or profile.get("dist_top3", {}).get(str(d))
    dist_n = profile.get("dist_n", {}).get(d) or profile.get("dist_n", {}).get(str(d), 0) or 0
    if dist_top3 is not None and dist_n >= 2:
        parts["dist"] = dist_top3 * 100
    else:
        # 近い距離(±200m)で代用
        nearby = []
        for dd, top3 in (profile.get("dist_top3") or {}).items():
            try: dd_int = int(dd)
            except: continue
            if abs(dd_int - d) <= 200: nearby.append(top3)
        parts["dist"] = (sum(nearby) / len(nearby)) * 100 if nearby else parts["base"] * 0.7

    # 3) タイム適性: 距離別タイム偏差。マイナス(速い)ほど高得点
    t_dev = (profile.get("dist_time_dev") or {}).get(d) or (profile.get("dist_time_dev") or {}).get(str(d))
    win_std = course.get("win_time_std") or 1.5
    if t_dev is not None:
        parts["time"] = max(-2.0, min(2.0, -t_dev / win_std)) * 25  # ±50点
    else:
        parts["time"] = 0.0

    # 4) 馬場適性
    if race_track in ("重", "稍重", "不良"):
        parts["track"] = _safe(profile.get("track_top3_heavy"), profile.get("track_top3_good") or 0) * 100
    else:
        parts["track"] = _safe(profile.get("track_top3_good")) * 100

    # 5) 枠適性: コース特性 × 馬の枠
    bb = "1-2" if bracket <= 2 else "3-4" if bracket <= 4 else "5-6" if bracket <= 6 else "7-8"
    bracket_rate = (course.get("bracket_top3_rate") or {}).get(bb)
    course_avg = sum(v for v in (course.get("bracket_top3_rate") or {}).values() if v is not None) / 4 if course.get("bracket_top3_rate") else 0.25
    parts["bracket"] = ((bracket_rate or course_avg) - course_avg) * 200  # ±15くらい

    # 6) 直近調子
    parts["form"] = _safe(profile.get("recent_trend")) * 20  # ±20
    days = profile.get("days_since_last")
    if days is not None:
        if days < 14: parts["form"] -= 5  # 連闘気味
        elif days > 90: parts["form"] -= 10  # 長休
        elif 21 <= days <= 60: parts["form"] += 3  # 適度な間隔

    # 7) 1番人気経験(=評価された経験): 期待信頼度
    pop1_top3 = profile.get("pop1_top3")
    pop1_n = profile.get("pop1_count", 0)
    if pop1_n and pop1_top3 is not None:
        parts["proven"] = (pop1_top3 - 0.65) * 30  # 65%超なら+

    total = sum(parts.values())
    return {"total": round(total, 2), "parts": {k: round(v, 2) for k, v in parts.items()}}


def softmax_probs(scores: Iterable[float], temperature: float = 25.0) -> list[float]:
    """スコア → 勝率推定. temperature が大きいほどフラットに。"""
    arr = list(scores)
    if not arr: return []
    m = max(arr)
    exps = [math.exp((s - m) / temperature) for s in arr]
    z = sum(exps)
    return [e / z for e in exps]


def expected_value(prob: float, odds: float) -> float:
    """単勝EV (= prob*odds, 1.0が損益分岐)。"""
    return prob * odds


def plackett_luce_top3(scores: Iterable[float], temperature: float = 25.0) -> tuple[list[float], list[float], list[float]]:
    """Plackett-Luce で勝率・連対率(top2)・複勝率(top3)を計算。

    各馬の強さ θ_i = exp(s_i / T)。
    P(i in topK) = ΣΣΣ over順列 で再帰的に計算する (O(n^2))。
    """
    arr = list(scores)
    n = len(arr)
    if n == 0: return [], [], []

    m = max(arr)
    theta = [math.exp((s - m) / temperature) for s in arr]
    Z = sum(theta)

    # 1着確率
    p_win = [t / Z for t in theta]

    # 2着までに入る確率: P(i topK<=2) = p_win_i + Σ_j!=i p_win_j * theta_i / (Z - theta_j)
    p_top2 = [0.0] * n
    for i in range(n):
        s = p_win[i]
        for j in range(n):
            if j == i: continue
            denom = Z - theta[j]
            if denom > 0:
                s += p_win[j] * (theta[i] / denom)
        p_top2[i] = s

    # 3着までに入る確率: P(i topK<=3) = p_top2_i + Σ_(j,k)!=i p(j 1st) p(k 2nd|j) * theta_i / (Z - theta_j - theta_k)
    p_top3 = [0.0] * n
    for i in range(n):
        s = p_top2[i]
        for j in range(n):
            if j == i: continue
            denom_jk = Z - theta[j]
            if denom_jk <= 0: continue
            for k in range(n):
                if k == i or k == j: continue
                p_jk = p_win[j] * (theta[k] / denom_jk)
                denom_i = Z - theta[j] - theta[k]
                if denom_i > 0:
                    s += p_jk * (theta[i] / denom_i)
        p_top3[i] = s

    return p_win, p_top2, p_top3
