"""コース特性分析（横軸）。

過去レース結果(data/oi/raw/results/*.json)から、コース×距離×馬場別に
レースに求められる能力プロファイルを集計する。

出力指標(各 (distance, track_condition) ごと):
  n_races              レース数
  time_mean / time_std 勝ちタイム平均±σ
  last3f_mean/std      勝ち馬の上り3F平均±σ
  bracket_top3_rate    枠1-2/3-4/5-6/7-8 別の3着内率（内外バイアス）
  popularity_top3_rate 1番人気/2-3番人気/4-6番人気/7+人気の3着内率
  pace_index           勝ちタイム - 上り3F×3 → 前半600m秒数換算（先行有利度の指標）
  weight_change_top3   馬体増減 (-10,-5),(-5,0),(0,5),(5,10) の3着内率
"""

import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev


def _bracket_band(b: int) -> str:
    if b <= 2: return "1-2"
    if b <= 4: return "3-4"
    if b <= 6: return "5-6"
    return "7-8"


def _pop_band(p: int) -> str:
    if p == 1: return "1"
    if p <= 3: return "2-3"
    if p <= 6: return "4-6"
    return "7+"


def _wt_band(c: int) -> str:
    if c <= -6: return "<=-6"
    if c <= -2: return "-5..-2"
    if c <= 2:  return "-1..+1"
    if c <= 5:  return "+2..+5"
    return ">=+6"


def build_course_profile(results_dir: str | Path) -> dict:
    results_dir = Path(results_dir)
    bucket: dict[tuple[int, str], list[dict]] = defaultdict(list)

    for fp in sorted(results_dir.glob("*.json")):
        d = json.loads(fp.read_text())
        if d.get("is_hurdle") or d.get("is_debut"):
            continue
        dist = d.get("distance", 0)
        track = d.get("track_condition", "") or "?"
        if not dist or not d.get("results"):
            continue
        bucket[(dist, track)].append(d)

    profile: dict[str, dict] = {}
    for (dist, track), races in bucket.items():
        key = f"{dist}m_{track}"
        n_races = len(races)

        # 勝ち馬集計
        win_times = []
        win_last3f = []
        for race in races:
            for r in race["results"]:
                if r["finish_position"] == 1:
                    if r["time"] > 0: win_times.append(r["time"])
                    if r["last_3f"] > 0: win_last3f.append(r["last_3f"])
                    break

        # 枠別3着内率
        bracket_run = defaultdict(int)
        bracket_top3 = defaultdict(int)
        # 人気別3着内率
        pop_run = defaultdict(int)
        pop_top3 = defaultdict(int)
        # 馬体増減3着内率
        wt_run = defaultdict(int)
        wt_top3 = defaultdict(int)
        # ペース指標(勝ちタイム - 上り3F)
        pace_vals = []

        for race in races:
            for r in race["results"]:
                pos = r["finish_position"]
                if pos == 0: continue
                bb = _bracket_band(r["bracket"])
                bracket_run[bb] += 1
                if pos <= 3: bracket_top3[bb] += 1
                pop = r.get("popularity", 0)
                if pop > 0:
                    pb = _pop_band(pop)
                    pop_run[pb] += 1
                    if pos <= 3: pop_top3[pb] += 1
                if r.get("horse_weight", 0) > 0:
                    wb = _wt_band(r.get("weight_change", 0))
                    wt_run[wb] += 1
                    if pos <= 3: wt_top3[wb] += 1

            for r in race["results"]:
                if r["finish_position"] == 1 and r["time"] > 0 and r["last_3f"] > 0:
                    pace_vals.append(r["time"] - r["last_3f"])
                    break

        bracket_rates = {b: round(bracket_top3[b] / bracket_run[b], 3) if bracket_run[b] else None for b in ["1-2","3-4","5-6","7-8"]}
        # 内外バイアス指数 = (内枠率 - 外枠率) / 平均  ; 正なら内有利
        valid_brs = [v for v in bracket_rates.values() if v is not None]
        avg_br = mean(valid_brs) if valid_brs else 0.25
        inside = bracket_rates.get("1-2") or avg_br
        outside = bracket_rates.get("7-8") or avg_br
        bracket_bias = round((inside - outside) / max(avg_br, 0.01), 3)

        profile[key] = {
            "distance": dist,
            "track": track,
            "n_races": n_races,
            "win_time_mean": round(mean(win_times), 2) if win_times else None,
            "win_time_std": round(pstdev(win_times), 2) if len(win_times) > 1 else None,
            "win_last3f_mean": round(mean(win_last3f), 2) if win_last3f else None,
            "win_last3f_std": round(pstdev(win_last3f), 2) if len(win_last3f) > 1 else None,
            "pace_remainder_mean": round(mean(pace_vals), 2) if pace_vals else None,
            "pace_remainder_std": round(pstdev(pace_vals), 2) if len(pace_vals) > 1 else None,
            "bracket_top3_rate": bracket_rates,
            "popularity_top3_rate": {p: round(pop_top3[p] / pop_run[p], 3) if pop_run[p] else None for p in ["1","2-3","4-6","7+"]},
            "weight_change_top3_rate": {w: round(wt_top3[w] / wt_run[w], 3) if wt_run[w] else None for w in ["<=-6","-5..-2","-1..+1","+2..+5",">=+6"]},
            "fav_top3_rate": round(pop_top3["1"] / pop_run["1"], 3) if pop_run["1"] else None,
            # 能力ベクトル(0-1正規化、後段でコースのcapabilityで重み付け)
            "capability": _capability_vector(win_times, win_last3f, pace_vals, bracket_bias),
        }

    return profile


def _capability_vector(times: list[float], l3fs: list[float], paces: list[float], bracket_bias: float) -> dict:
    """コースが要求する能力プロファイル。

      speed_focus    : タイム水準が均一(stdが低い)→基礎スピード差で決まる
      finishing_focus: 上り3Fのstdが大きい→末脚差で決まる
      pace_focus     : 前半600m秒数のstdが小さい→先行有利が一定
      bracket_bias   : +1=内枠有利, 0=中立, -1=外枠有利
      heavy_resistance: 後段で良/重比較を入れる（暫定 None）
    """
    def _norm(x, lo, hi):
        if x is None: return None
        return max(0.0, min(1.0, (x - lo) / (hi - lo)))

    t_std = pstdev(times) if len(times) > 1 else None
    l_std = pstdev(l3fs) if len(l3fs) > 1 else None
    p_std = pstdev(paces) if len(paces) > 1 else None

    # 経験的レンジ: 大井ダ標準で時計std 0.8〜2.0、上り3Fstd 0.5〜1.5、ペースstd 0.5〜1.5
    return {
        "speed_focus":      round(1.0 - (_norm(t_std, 0.8, 2.0) or 0.5), 3),  # std小=speed重視
        "finishing_focus":  round(_norm(l_std, 0.5, 1.5) or 0.5, 3),
        "pace_focus":       round(1.0 - (_norm(p_std, 0.5, 1.5) or 0.5), 3),
        "bracket_bias":     round(max(-1.0, min(1.0, bracket_bias)), 3),
    }


def save_course_profile(profile: dict, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(profile, ensure_ascii=False, indent=2))
