"""4/27の大井全レース予想を、コース能力ベクトル × 馬能力ベクトルで出力。

使い方:
  python scripts/oi/predict_today_quick.py --date 2026-04-27
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.oi.analysis.course_profile import build_course_profile, save_course_profile
from src.oi.analysis.horse_profile import index_results_by_horse
from src.oi.analysis.ability import build_ability_vector, match_score
from src.oi.analysis.score import plackett_luce_top3
from scripts.oi.preview_today_bias import estimate_today_bias, apply_bias_to_capability


JRA_PLACES = {"札幌","函館","福島","新潟","東京","中山","中京","京都","阪神","小倉"}


def _is_jra(place: str) -> bool:
    place_kanji = re.sub(r"\d+", "", place or "")
    return place_kanji in JRA_PLACES


def load_jra_summary(horse_id: str, horse_dir: Path) -> tuple[int, float | None]:
    """horse個体ファイルから JRA出走数とJRA3着内率を取り出す。"""
    fp = horse_dir / f"{horse_id}.json"
    if not fp.exists():
        return 0, None
    h = json.loads(fp.read_text())
    past = h.get("past_results", [])
    jra = [r for r in past if _is_jra(r.get("place", ""))]
    valid = [r for r in jra if r.get("finish_position", 0) > 0]
    if not valid:
        return len(jra), None
    top3 = sum(1 for r in valid if r["finish_position"] <= 3) / len(valid)
    return len(jra), top3


def predict_race(shutuba: dict, course_profile: dict, abilities: dict[str, dict], jra_info: dict[str, tuple[int, float | None]], today_bias: dict | None = None, today_weight: float = 0.0) -> dict:
    dist = shutuba["distance"]
    track = shutuba.get("track_condition", "") or "良"
    course = course_profile.get(f"{dist}m_{track}") or course_profile.get(f"{dist}m_良") or {}
    capability = course.get("capability", {"speed_focus":0.5,"finishing_focus":0.5,"pace_focus":0.5,"bracket_bias":0.0})
    if today_bias and today_weight > 0:
        capability = apply_bias_to_capability(capability, today_bias, None, today_weight)

    rows = []
    for entry in shutuba["entries"]:
        ab = abilities.get(entry["horse_id"], {})
        sc = match_score(ab, capability, entry["bracket"], dist, track)
        jra_n, jra_top3 = jra_info.get(entry["horse_id"], (0, None))
        rows.append({
            "number": entry["number"],
            "bracket": entry["bracket"],
            "horse_id": entry["horse_id"],
            "horse_name": entry["horse_name"],
            "sex_age": entry["sex_age"],
            "impost": entry["impost"],
            "jockey_name": entry["jockey_name"],
            "win_odds_est": entry["win_odds"],
            "popularity_est": entry["popularity"],
            "score": sc["total"],
            "score_parts": sc["parts"],
            "n_oi": ab.get("n_oi", 0),
            "speed_skill": ab.get("speed_skill"),
            "finishing_skill": ab.get("finishing_skill"),
            "power": ab.get("power"),
            "consistency": ab.get("consistency"),
            "bracket_pref": ab.get("bracket_pref"),
            "jra_n": jra_n,
            "jra_top3": jra_top3,
        })

    p_win, p_top2, p_top3 = plackett_luce_top3([r["score"] for r in rows], temperature=12.0)
    for r, pw, p2, p3 in zip(rows, p_win, p_top2, p_top3):
        r["prob_win"] = round(pw, 4)
        r["prob_top2"] = round(p2, 4)
        r["prob_top3"] = round(p3, 4)
        r["ev"] = round(pw * (r["win_odds_est"] or 0), 3) if r["win_odds_est"] else None

    rows.sort(key=lambda x: -x["score"])
    return {
        "race_id": shutuba["race_id"],
        "race_no": shutuba["race_no"],
        "race_name": shutuba["race_name"],
        "distance": dist,
        "track": track,
        "n_runners": shutuba["num_runners"],
        "course_capability": capability,
        "rows": rows,
    }


def format_race(p: dict) -> str:
    out = []
    cap = p["course_capability"]
    out.append(f"\n━━━ {p['race_no']:>2}R {p['race_name']} {p['distance']}m {p['track']} ({p['n_runners']}頭) ━━━")
    out.append(f"  コース性質: speed={cap['speed_focus']:.2f} finishing={cap['finishing_focus']:.2f} bracket_bias={cap['bracket_bias']:+.2f}")
    out.append(f"  {'番':>2} {'枠':>2} {'馬名':<14} {'騎手':<10} {'スコア':>6} {'勝率(推)':>7} {'複率(推)':>7} {'想オ':>5} {'EV':>5}  {'大井n':>4} {'速力':>5} {'末脚':>5} {'地力':>5} {'JRA':>4}")
    for r in p["rows"]:
        sp = r["speed_skill"]
        l3 = r["finishing_skill"]
        pw = r["power"]
        jr = f"{int((r['jra_top3'] or 0)*100):>2}%" if r["jra_n"] >= 3 else f"{r['jra_n']}走" if r["jra_n"] else " - "
        out.append(
            f"  {r['number']:>2} {r['bracket']:>2} {r['horse_name'][:14]:<14} {r['jockey_name'][:10]:<10} "
            f"{r['score']:>6.1f}  {r['prob_win']*100:>5.1f}%  {r['prob_top3']*100:>5.1f}% {r['win_odds_est']:>5.1f} "
            f"{(r['ev'] if r['ev'] is not None else 0):>5.2f}  "
            f"{r['n_oi']:>4} "
            f"{(f'{sp:+.2f}' if sp is not None else '  - '):>5} "
            f"{(f'{l3:+.2f}' if l3 is not None else '  - '):>5} "
            f"{(f'{pw:.2f}' if pw is not None else '  -'):>5} "
            f"{jr:>4}"
        )
    ev_high = [r for r in p["rows"] if r["ev"] and r["ev"] > 1.15]
    if ev_high:
        out.append(f"  → 単勝EV>1.15: " + ", ".join(f"{r['number']}({r['ev']:.2f})" for r in ev_high[:3]))
    # 軸=スコア1位、相手=「軸を除いた複勝率上位4頭」
    if p["rows"]:
        axis = p["rows"][0]
        partners_pool = sorted(p["rows"][1:], key=lambda x: -x["prob_top3"])[:4]
        partners = ",".join(str(r["number"]) for r in partners_pool)
        out.append(f"  → 軸={axis['number']}番  相手={partners} (複勝率上位)")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="YYYY-MM-DD")
    ap.add_argument("--today-weight", type=float, default=0.0, help="当日バイアスを混ぜる重み (0で無効)")
    ap.add_argument("--save-both", action="store_true", help="バイアスなし版も {date}_no_bias.json として保存")
    args = ap.parse_args()

    target = datetime.strptime(args.date, "%Y-%m-%d").date()
    yyyymmdd = target.strftime("%Y%m%d")

    # 常に再計算(コース特性は変わるので)
    course_profile = build_course_profile(ROOT / "data/oi/raw/results")
    save_course_profile(course_profile, ROOT / "data/oi/processed/course_profile.json")

    shutuba_dir = ROOT / "data/oi/raw/shutuba"
    shutubas = []
    for fp in sorted(shutuba_dir.glob(f"*44{yyyymmdd[4:8]}*.json")):
        d = json.loads(fp.read_text())
        if d.get("date") == yyyymmdd:
            shutubas.append(d)
    if not shutubas:
        print(f"出馬表が見つかりません: {yyyymmdd}", file=sys.stderr); sys.exit(1)

    horse_ids = {e["horse_id"] for s in shutubas for e in s["entries"] if e["horse_id"]}

    by_horse = index_results_by_horse(ROOT / "data/oi/raw/results")

    horse_dir = ROOT / "data/oi/raw/horses"
    abilities: dict[str, dict] = {}
    jra_info: dict[str, tuple[int, float | None]] = {}
    for hid in horse_ids:
        jra_n, jra_top3 = load_jra_summary(hid, horse_dir)
        jra_info[hid] = (jra_n, jra_top3)
        abilities[hid] = build_ability_vector(by_horse.get(hid, []), course_profile, jra_n, jra_top3)

    # 当日バイアス: 確定済みレースから推定
    today_bias = None
    if args.today_weight > 0:
        finished = []
        for fp in sorted((ROOT / "data/oi/raw/results").glob(f"*44{yyyymmdd[4:8]}*.json")):
            d = json.loads(fp.read_text())
            if d.get("date") == yyyymmdd and d.get("num_runners", 0) > 0:
                finished.append(d)
        if finished:
            today_bias = estimate_today_bias(finished)
            print(f"  当日バイアス推定 ({today_bias['n_races']}R) bracket={today_bias['bracket_top3_rate_today']} weight={args.today_weight}")
        finished_nos = {d["race_no"] for d in finished}
    else:
        finished_nos = set()

    # 確定済レースは元の予想を保つ、未確定だけバイアス反映
    results = []
    for s in shutubas:
        use_weight = args.today_weight if s["race_no"] not in finished_nos else 0.0
        results.append(predict_race(s, course_profile, abilities, jra_info, today_bias, use_weight))
    results.sort(key=lambda x: x["race_no"])

    out_dir = ROOT / "data/oi/predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.date}.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))

    # --save-both: バイアスなし版も別ファイルに保存
    if args.save_both and args.today_weight > 0:
        results_nb = [predict_race(s, course_profile, abilities, jra_info, None, 0.0) for s in shutubas]
        results_nb.sort(key=lambda x: x["race_no"])
        nb_path = out_dir / f"{args.date}_no_bias.json"
        nb_path.write_text(json.dumps(results_nb, ensure_ascii=False, indent=2))
        print(f"→ バイアスなし版も保存: {nb_path}")

    print(f"\n{'='*92}")
    print(f"  大井 {args.date} 予想  (コース能力ベクトル × 馬能力ベクトル, n_courses={len(course_profile)})")
    cached = sum(1 for hid in horse_ids if (horse_dir / f"{hid}.json").exists())
    has_jra = sum(1 for v in jra_info.values() if v[0] >= 3)
    print(f"  馬個体取得済 {cached}/{len(horse_ids)} (うちJRA経験3走以上: {has_jra}頭)")
    print(f"{'='*92}")
    for p in results:
        print(format_race(p))
    print(f"\n→ 保存: {out_path}")


if __name__ == "__main__":
    main()
