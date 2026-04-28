"""馬の能力ベクトル化(縦軸)とコースとのマッチング(横軸×縦軸)。

過去の大井出走履歴から、各馬を以下のベクトルで表す:
  speed_skill     : タイム偏差の加重平均(マイナス=コース平均より速い)
  finishing_skill : 上り3F偏差の加重平均(マイナス=末脚が鋭い)
  consistency     : 着順率の安定性(低いほど安定)
  power           : 過去の最高水準(=直近5走の最良着順率/最高クラス信用度)
  bracket_pref    : -1..+1, 内枠で好走→+、外枠で好走→-
  heavy_skill     : 重馬場での着順率(良馬場との差)
  distance_curve  : {距離: 着順率} 経験あれば
  jra_grade       : JRA経験値(着順率) ※horse個体ファイルから

コース類似度(距離・馬場の連続)で経験補完:
  ターゲット距離±200m, 同馬場優先 → 加重マージ
"""

import math
from collections import defaultdict
from datetime import date
from statistics import mean, pstdev


def _exp_weight(i: int, n: int, half_life: int = 4) -> float:
    """直近ほど重い指数加重。i=0が最古、i=n-1が最新。"""
    age = n - 1 - i
    return 0.5 ** (age / half_life)


def _bracket_band(b: int) -> str:
    if b <= 2: return "1-2"
    if b <= 4: return "3-4"
    if b <= 6: return "5-6"
    return "7-8"


def build_ability_vector(
    oi_history: list[dict],
    course_profile: dict,
    jra_n: int = 0,
    jra_top3: float | None = None,
) -> dict:
    """大井全出走から能力ベクトルを構築。oi_history は date 昇順前提。"""
    valid = [r for r in oi_history if r["finish"] > 0]
    n = len(valid)

    if n == 0:
        return {
            "speed_skill": None,
            "finishing_skill": None,
            "consistency": None,
            "power": None,
            "bracket_pref": 0.0,
            "heavy_skill": None,
            "distance_curve": {},
            "n_oi": 0,
            "jra_grade": jra_top3 if jra_n >= 3 else None,
            "experience_score": 0.0,
        }

    # タイム偏差・上り3F偏差(全出走、コース平均との差)
    weights = [_exp_weight(i, n) for i in range(n)]
    time_devs, time_ws = [], []
    l3f_devs, l3f_ws = [], []

    for i, r in enumerate(valid):
        key = f"{r['distance']}m_{r['track_condition']}"
        cp = course_profile.get(key) or course_profile.get(f"{r['distance']}m_良")
        if not cp: continue
        w = weights[i]
        if r["time"] > 0 and cp.get("win_time_mean"):
            time_devs.append(r["time"] - cp["win_time_mean"]); time_ws.append(w)
        if r["last_3f"] > 0 and cp.get("win_last3f_mean"):
            l3f_devs.append(r["last_3f"] - cp["win_last3f_mean"]); l3f_ws.append(w)

    speed_skill = sum(d * w for d, w in zip(time_devs, time_ws)) / sum(time_ws) if time_ws else None
    finishing_skill = sum(d * w for d, w in zip(l3f_devs, l3f_ws)) / sum(l3f_ws) if l3f_ws else None

    # 着順率(着順/頭数)
    finish_rates = [r["finish"] / max(r["num_runners"], 1) for r in valid]
    fr_weights = weights
    weighted_fr = sum(f * w for f, w in zip(finish_rates, fr_weights)) / sum(fr_weights)
    consistency = pstdev(finish_rates) if len(finish_rates) > 1 else None

    # power: 過去 (4走以上から3走) の最良着順率(着順1位重視)
    recent = sorted(valid, key=lambda x: x["date"])[-5:]
    if recent:
        recent_fr = [r["finish"] / max(r["num_runners"], 1) for r in recent]
        # 0に近いほど強い → 反転して 0-1
        power = max(0.0, 1.0 - min(recent_fr))
    else:
        power = None

    # 枠選好: 内枠での着順率と外枠での着順率の差
    inside_runs = [r for r in valid if r["bracket"] <= 3]
    outside_runs = [r for r in valid if r["bracket"] >= 6]
    if inside_runs and outside_runs:
        in_fr = mean(r["finish"] / max(r["num_runners"], 1) for r in inside_runs)
        out_fr = mean(r["finish"] / max(r["num_runners"], 1) for r in outside_runs)
        # 内枠で好走(=fr低い) - 外枠で好走 → 正なら外枠選好、反転して内枠選好を+に
        bracket_pref = max(-1.0, min(1.0, (out_fr - in_fr) * 2))
    else:
        bracket_pref = 0.0

    # 重馬場耐性
    heavy = [r for r in valid if r["track_condition"] in ("重", "稍重", "不良")]
    good = [r for r in valid if r["track_condition"] in ("良", "?")]
    if heavy and good:
        h_fr = mean(r["finish"] / max(r["num_runners"], 1) for r in heavy)
        g_fr = mean(r["finish"] / max(r["num_runners"], 1) for r in good)
        # heavy_fr が g_fr より低い→重で巧者。差を反転正規化
        heavy_skill = max(-1.0, min(1.0, (g_fr - h_fr) * 2))
    elif heavy:
        h_fr = mean(r["finish"] / max(r["num_runners"], 1) for r in heavy)
        heavy_skill = max(-1.0, min(1.0, (0.5 - h_fr) * 2))
    else:
        heavy_skill = None

    # 距離別着順率
    by_dist: dict[int, list[dict]] = defaultdict(list)
    for r in valid: by_dist[r["distance"]].append(r)
    distance_curve = {
        d: {
            "n": len(rows),
            "finish_rate": round(mean(r["finish"] / max(r["num_runners"], 1) for r in rows), 3),
            "top3_rate": round(sum(1 for r in rows if r["finish"] <= 3) / len(rows), 3),
        }
        for d, rows in by_dist.items()
    }

    return {
        "n_oi": n,
        "speed_skill": round(speed_skill, 3) if speed_skill is not None else None,
        "finishing_skill": round(finishing_skill, 3) if finishing_skill is not None else None,
        "weighted_finish_rate": round(weighted_fr, 3),
        "consistency": round(consistency, 3) if consistency is not None else None,
        "power": round(power, 3) if power is not None else None,
        "bracket_pref": round(bracket_pref, 3),
        "heavy_skill": round(heavy_skill, 3) if heavy_skill is not None else None,
        "distance_curve": distance_curve,
        "jra_grade": round(jra_top3, 3) if jra_top3 is not None and jra_n >= 3 else None,
        "experience_score": round(min(1.0, n / 10), 2),  # 10走以上で完全
    }


def distance_extrapolation(
    distance_curve: dict,
    target_distance: int,
    target_track: str,
) -> tuple[float | None, float]:
    """指定距離での推定3着内率と信頼度を返す。

    経験あればそのまま、無ければ近接距離(±400m)から重み付きで推定。
    重みは距離差に対し指数減衰。
    """
    if not distance_curve:
        return None, 0.0

    if target_distance in distance_curve:
        d = distance_curve[target_distance]
        return d["top3_rate"], min(1.0, d["n"] / 4)

    # 近接距離からの推定
    weights, vals = [], []
    for d, info in distance_curve.items():
        try: d_int = int(d)
        except: continue
        diff = abs(d_int - target_distance)
        if diff > 400: continue
        w = math.exp(-diff / 200) * info["n"]
        weights.append(w)
        vals.append(info["top3_rate"])
    if not weights:
        return None, 0.0
    estimate = sum(v * w for v, w in zip(vals, weights)) / sum(weights)
    confidence = min(1.0, sum(weights) / 5)
    return estimate, confidence


def match_score(ability: dict, course_capability: dict, bracket: int, target_distance: int, target_track: str) -> dict:
    """馬の能力ベクトル × コースの要求能力 → スコア(0-100)+内訳。

    点数の意味:
      ・0点: 平凡
      ・+30点: コース要求と能力が完全一致
      ・-30点: 完全に不向き
    """
    parts: dict[str, float] = {}

    # 1) スピード適合: 速い馬(speed_skill低い) × speed_focusが高いコース → +
    if ability["speed_skill"] is not None:
        # speed_skill 想定範囲: -3秒(超速) 〜 +3秒(遅い)
        speed_norm = max(-1.0, min(1.0, -ability["speed_skill"] / 2.0))  # 速い=+
        parts["speed_match"] = speed_norm * course_capability["speed_focus"] * 30

    # 2) 末脚適合: 上り速い馬 × finishing_focus高
    if ability["finishing_skill"] is not None:
        l3f_norm = max(-1.0, min(1.0, -ability["finishing_skill"] / 1.5))
        parts["finishing_match"] = l3f_norm * course_capability["finishing_focus"] * 30

    # 3) 枠適合: 内/外選好 × コースの内有利度
    bracket_band = _bracket_band(bracket)
    own_inside = 1.0 if bracket <= 3 else (-1.0 if bracket >= 6 else 0.0)
    parts["bracket_match"] = own_inside * course_capability["bracket_bias"] * 15

    # 4) 馬場適合(良/重)
    if target_track in ("重", "稍重", "不良") and ability["heavy_skill"] is not None:
        parts["track_match"] = ability["heavy_skill"] * 15

    # 5) 基礎能力(power: 過去最高水準)
    if ability["power"] is not None:
        parts["power"] = (ability["power"] - 0.5) * 40  # power=1で+20

    # 6) 距離別実績(類似距離からの外挿)
    est, conf = distance_extrapolation(ability["distance_curve"], target_distance, target_track)
    if est is not None:
        # est が 0.5以上で+方向、3着内率0.5基準
        parts["distance_fit"] = (est - 0.3) * 50 * conf  # ±15点程度

    # 7) 安定性(consistency低い=安定): 安定なら+
    if ability["consistency"] is not None:
        parts["consistency"] = max(0, (0.25 - ability["consistency"])) * 30  # ±5

    # 8) JRA経験補正
    if ability.get("jra_grade") is not None:
        parts["jra"] = (ability["jra_grade"] - 0.3) * 15  # ±5

    total = sum(parts.values())
    return {
        "total": round(total, 2),
        "parts": {k: round(v, 2) for k, v in parts.items()},
    }
