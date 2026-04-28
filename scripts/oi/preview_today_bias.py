"""当日バイアス反映のインパクトを「ドライラン」で確認する。

確定済みのレース結果から当日バイアスを推定し、未確定レースのコース特性に
混ぜたら予想がどう変わるか、現行版と並べて表示する。

現行予想ファイルは書き換えない。

使い方:
  python scripts/oi/preview_today_bias.py --date 2026-04-27 --weight 0.3
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.oi.analysis.course_profile import build_course_profile
from src.oi.analysis.horse_profile import index_results_by_horse
from src.oi.analysis.ability import build_ability_vector, match_score
from src.oi.analysis.score import plackett_luce_top3


def estimate_today_bias(finished_races: list[dict]) -> dict:
    """確定済みレース群から当日バイアスを推定。

    Returns dict:
      bracket_top3_rate_today: 当日の枠別3着内率 (1-2/3-4/5-6/7-8)
      fav_top3_rate_today    : 当日の1番人気3着内率
      time_dev_today         : 当日の勝ちタイム偏差(過去平均との差)平均
      l3f_dev_today          : 当日の勝ち上り3F偏差平均
      n_races                : 集計対象数
      details                : レース毎ログ
    """
    bracket_run = defaultdict(int)
    bracket_top3 = defaultdict(int)
    fav_run = 0
    fav_top3 = 0
    time_devs = []
    l3f_devs = []
    details = []

    cp_full = build_course_profile(ROOT / "data/oi/raw/results")

    for d in finished_races:
        if not d.get("results"): continue
        for r in d["results"]:
            pos = r["finish_position"]
            if pos == 0: continue
            b = r["bracket"]
            bb = "1-2" if b <= 2 else "3-4" if b <= 4 else "5-6" if b <= 6 else "7-8"
            bracket_run[bb] += 1
            if pos <= 3: bracket_top3[bb] += 1
            if r.get("popularity") == 1:
                fav_run += 1
                if pos <= 3: fav_top3 += 1

        win = next((r for r in d["results"] if r["finish_position"] == 1), None)
        if not win: continue
        key = f"{d['distance']}m_{d['track_condition'] or '?'}"
        cp = cp_full.get(key) or cp_full.get(f"{d['distance']}m_良") or {}
        if win["time"] > 0 and cp.get("win_time_mean"):
            time_devs.append(win["time"] - cp["win_time_mean"])
        if win["last_3f"] > 0 and cp.get("win_last3f_mean"):
            l3f_devs.append(win["last_3f"] - cp["win_last3f_mean"])

        details.append({
            "race_no": d["race_no"],
            "winner_bracket": win["bracket"],
            "winner_pop": win.get("popularity"),
            "win_time_dev": round(win["time"] - (cp.get("win_time_mean") or win["time"]), 2) if cp.get("win_time_mean") else None,
            "win_l3f_dev": round(win["last_3f"] - (cp.get("win_last3f_mean") or win["last_3f"]), 2) if cp.get("win_last3f_mean") else None,
        })

    bracket_rates = {
        b: round(bracket_top3[b] / bracket_run[b], 3) if bracket_run[b] else None
        for b in ["1-2","3-4","5-6","7-8"]
    }

    return {
        "n_races": len(finished_races),
        "bracket_top3_rate_today": bracket_rates,
        "bracket_n": dict(bracket_run),
        "fav_top3_rate_today": round(fav_top3 / fav_run, 3) if fav_run else None,
        "fav_n": fav_run,
        "time_dev_today": round(mean(time_devs), 2) if time_devs else None,
        "l3f_dev_today": round(mean(l3f_devs), 2) if l3f_devs else None,
        "details": details,
    }


def apply_bias_to_capability(capability: dict, today: dict, course_bracket: dict, weight: float) -> dict:
    """capabilityに当日バイアスを weight 比率で混ぜる。"""
    if today["n_races"] < 2: return capability  # サンプル小さすぎ
    new = dict(capability)

    # bracket_bias の更新: 過去の bracket_bias と 当日の (内枠率 - 外枠率)/平均 を加重平均
    today_bracket = today["bracket_top3_rate_today"]
    valid = [v for v in today_bracket.values() if v is not None]
    if len(valid) >= 3:
        avg_t = mean(valid)
        inside_t = today_bracket.get("1-2") or avg_t
        outside_t = today_bracket.get("7-8") or avg_t
        today_bias = max(-1.0, min(1.0, (inside_t - outside_t) / max(avg_t, 0.01)))
        new["bracket_bias"] = round((1 - weight) * capability["bracket_bias"] + weight * today_bias, 3)

    return new


def predict_with_capability(shutuba: dict, capability: dict, by_horse, course_profile, horse_dir):
    """capabilityを差し替えて予想を再計算。"""
    dist = shutuba["distance"]
    track = shutuba.get("track_condition", "") or "良"
    course = course_profile.get(f"{dist}m_{track}") or course_profile.get(f"{dist}m_良") or {}

    rows = []
    for entry in shutuba["entries"]:
        ab = build_ability_vector(by_horse.get(entry["horse_id"], []), course_profile)
        sc = match_score(ab, capability, entry["bracket"], dist, track)
        rows.append({
            "number": entry["number"],
            "bracket": entry["bracket"],
            "horse_name": entry["horse_name"],
            "win_odds_est": entry["win_odds"],
            "score": sc["total"],
        })
    p_win, _, p_top3 = plackett_luce_top3([r["score"] for r in rows], temperature=12.0)
    for r, pw, p3 in zip(rows, p_win, p_top3):
        r["prob_win"] = pw
        r["prob_top3"] = p3
        r["ev"] = pw * (r["win_odds_est"] or 0)
    rows.sort(key=lambda x: -x["score"])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--weight", type=float, default=0.3, help="当日バイアスの重み (0-1)")
    args = ap.parse_args()
    yyyymmdd = args.date.replace("-", "")

    # 1. 確定済レース
    finished = []
    for fp in sorted((ROOT / "data/oi/raw/results").glob(f"*44{yyyymmdd[4:8]}*.json")):
        d = json.loads(fp.read_text())
        if d.get("date") == yyyymmdd and d.get("num_runners", 0) > 0:
            finished.append(d)

    today = estimate_today_bias(finished)

    print(f"\n{'='*92}")
    print(f"  当日バイアス推定 (確定 {today['n_races']}R, 重み={args.weight})")
    print(f"{'='*92}")
    bb = today["bracket_top3_rate_today"]
    bn = today["bracket_n"]
    print(f"  枠別3着内率(本日): " + " | ".join(f"{b}:{bb[b]}({bn[b]}走)" if bb[b] is not None else f"{b}:-" for b in ["1-2","3-4","5-6","7-8"]))
    print(f"  1番人気3着内率(本日): {today['fav_top3_rate_today']} ({today['fav_n']}走)")
    print(f"  勝ちタイム偏差: {today['time_dev_today']:+.2f}s (マイナス=過去より速い)" if today['time_dev_today'] is not None else "  勝ちタイム偏差: -")
    print(f"  勝ち上り3F偏差: {today['l3f_dev_today']:+.2f}s" if today['l3f_dev_today'] is not None else "  勝ち上り3F偏差: -")

    # 2. 未確定レース
    pred_path = ROOT / "data/oi/predictions" / f"{args.date}.json"
    preds = json.loads(pred_path.read_text())
    finished_nos = {d["race_no"] for d in finished}
    pending = [p for p in preds if p["race_no"] not in finished_nos]
    print(f"\n  未確定 {len(pending)}R: {[p['race_no'] for p in pending]}")
    print()

    # 3. 各未確定レースを再計算
    shutubas = {}
    for fp in sorted((ROOT / "data/oi/raw/shutuba").glob(f"*44{yyyymmdd[4:8]}*.json")):
        d = json.loads(fp.read_text())
        shutubas[d["race_no"]] = d
    course_profile = build_course_profile(ROOT / "data/oi/raw/results")
    by_horse = index_results_by_horse(ROOT / "data/oi/raw/results")
    horse_dir = ROOT / "data/oi/raw/horses"

    for pred in pending:
        rno = pred["race_no"]
        sh = shutubas.get(rno)
        if not sh: continue
        old_cap = pred["course_capability"]
        new_cap = apply_bias_to_capability(old_cap, today, None, args.weight)
        new_rows = predict_with_capability(sh, new_cap, by_horse, course_profile, horse_dir)

        old_rows = pred["rows"]  # スコア降順
        old_axis = old_rows[0]
        new_axis = new_rows[0]
        old_partners = sorted(old_rows[1:], key=lambda x: -(x.get("prob_top3") or 0))[:4]
        new_partners = sorted(new_rows[1:], key=lambda x: -x["prob_top3"])[:4]

        old_partner_nums = [r["number"] for r in old_partners]
        new_partner_nums = [r["number"] for r in new_partners]

        axis_changed = old_axis["number"] != new_axis["number"]
        partner_diff_added = set(new_partner_nums) - set(old_partner_nums)
        partner_diff_removed = set(old_partner_nums) - set(new_partner_nums)

        # 上位5頭の順位変化
        old_top5 = [(r["number"], r["horse_name"], r["score"], r["prob_win"]) for r in old_rows[:5]]
        new_by_num = {r["number"]: r for r in new_rows}
        old_by_num = {r["number"]: r for r in old_rows}

        bracket_change = ""
        if abs(new_cap["bracket_bias"] - old_cap["bracket_bias"]) > 0.01:
            bracket_change = f"  [bracket_bias {old_cap['bracket_bias']:+.2f} → {new_cap['bracket_bias']:+.2f}]"

        marker = "★" if axis_changed or partner_diff_added or partner_diff_removed else " "
        print(f"━━━ {marker} {rno}R {pred['race_name']} {pred['distance']}m {pred['track']} ━━━{bracket_change}")
        print(f"  軸:  現{old_axis['number']:>2}番({old_axis['horse_name'][:10]:<10}) → 新{new_axis['number']:>2}番({new_axis['horse_name'][:10]:<10}) {'CHANGE' if axis_changed else 'same'}")
        print(f"  相手 現:{old_partner_nums}  新:{new_partner_nums}  追加:{sorted(partner_diff_added) or '-'}  削除:{sorted(partner_diff_removed) or '-'}")
        # スコア順位上位5頭の比較
        new_score_rank = {r["number"]: i for i, r in enumerate(new_rows)}
        old_score_rank = {r["number"]: i for i, r in enumerate(old_rows)}
        print(f"  上位5(現)→新スコア:")
        for rank, (num, name, sc, pw) in enumerate(old_top5, 1):
            new_r = new_by_num[num]
            sc_diff = new_r["score"] - sc
            new_rank = new_score_rank[num] + 1
            mark = "↑" if new_rank < rank else ("↓" if new_rank > rank else "→")
            print(f"   {rank}位 {num:>2}番 {name[:14]:<14} スコア {sc:>6.1f} → {new_r['score']:>6.1f} ({sc_diff:+.1f}) 順位{rank}{mark}{new_rank}")
        print()


if __name__ == "__main__":
    main()
