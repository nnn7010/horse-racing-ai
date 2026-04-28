"""当日バイアスを確定済みレースから再推定し {date}.json を更新する。

レース確定後に実行すると、残りレースの予想がバイアス反映版に更新される。
--fetch フラグを付けると結果を HTTP から再取得してから更新する。

使い方:
  python scripts/oi/update_today_bias.py --date 2026-04-28
  python scripts/oi/update_today_bias.py --date 2026-04-28 --fetch
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.oi.analysis.course_profile import build_course_profile
from src.oi.analysis.horse_profile import index_results_by_horse
from src.oi.analysis.ability import build_ability_vector
from src.oi.scraping.race import fetch_race_result
from scripts.oi.predict_today_quick import predict_race, load_jra_summary
from scripts.oi.preview_today_bias import estimate_today_bias


def _fetch_result(race_id: str, force: bool) -> dict | None:
    cache = ROOT / "data/oi/raw/results" / f"{race_id}.json"
    if cache.exists() and not force:
        d = json.loads(cache.read_text())
        if d.get("num_runners", 0) > 0:
            return d

    url = f"https://nar.netkeiba.com/race/result.html?race_id={race_id}"
    http_cache = ROOT / "data/oi/cache" / f"{hashlib.md5(url.encode()).hexdigest()}.html"
    if http_cache.exists():
        http_cache.unlink()

    try:
        d = fetch_race_result(race_id)
        if d.get("num_runners", 0) == 0:
            return None
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(d, ensure_ascii=False, indent=2))
        return d
    except Exception as e:
        print(f"  [warn] {race_id}: {e}", file=sys.stderr)
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="YYYY-MM-DD")
    ap.add_argument("--weight", type=float, default=0.3, help="バイアス重み (default: 0.3)")
    ap.add_argument("--fetch", action="store_true", help="HTTPから結果を再取得してから更新")
    args = ap.parse_args()

    yyyymmdd = args.date.replace("-", "")
    out_dir = ROOT / "data/oi/predictions"
    out_path = out_dir / f"{args.date}.json"

    if not out_path.exists():
        print(f"予想ファイルがありません: {out_path}", file=sys.stderr)
        sys.exit(1)

    preds = json.loads(out_path.read_text())

    # 結果取得（--fetch 時は全レース再スクレイプ）
    if args.fetch:
        print("  結果を再取得中...")
        for p in preds:
            _fetch_result(p["race_id"], force=True)

    # 確定済みレースを収集
    results_dir = ROOT / "data/oi/raw/results"
    finished: list[dict] = []
    finished_nos: set[int] = set()
    for p in preds:
        cache = results_dir / f"{p['race_id']}.json"
        if cache.exists():
            d = json.loads(cache.read_text())
            if d.get("num_runners", 0) > 0 and d.get("date") == yyyymmdd:
                finished.append(d)
                finished_nos.add(p["race_no"])

    if not finished:
        print("  確定済みレースがありません。更新をスキップします。")
        return

    today_bias = estimate_today_bias(finished)
    print(f"  確定 {today_bias['n_races']}R からバイアス推定")
    print(f"  枠別3着内率: {today_bias['bracket_top3_rate_today']}")
    print(f"  1番人気3着内率: {today_bias['fav_top3_rate_today']}")

    # 能力ベクトル再構築
    shutuba_dir = ROOT / "data/oi/raw/shutuba"
    shutubas: dict[int, dict] = {}
    for fp in sorted(shutuba_dir.glob(f"*44{yyyymmdd[4:8]}*.json")):
        d = json.loads(fp.read_text())
        if d.get("date") == yyyymmdd:
            shutubas[d["race_no"]] = d

    course_profile = build_course_profile(results_dir)
    by_horse = index_results_by_horse(results_dir)
    horse_dir = ROOT / "data/oi/raw/horses"
    horse_ids = {e["horse_id"] for s in shutubas.values() for e in s["entries"] if e["horse_id"]}

    abilities: dict[str, dict] = {}
    jra_info: dict[str, tuple] = {}
    for hid in horse_ids:
        jn, jt = load_jra_summary(hid, horse_dir)
        jra_info[hid] = (jn, jt)
        abilities[hid] = build_ability_vector(by_horse.get(hid, []), course_profile, jn, jt)

    # 未確定レースのみバイアス反映して更新
    updated = []
    changed = []
    for p in preds:
        rno = p["race_no"]
        s = shutubas.get(rno)
        if not s:
            updated.append(p)
            continue
        if rno in finished_nos:
            # 確定済みはバイアスなしで固定
            updated.append(predict_race(s, course_profile, abilities, jra_info, None, 0.0))
        else:
            new_p = predict_race(s, course_profile, abilities, jra_info, today_bias, args.weight)
            old_axis = p["rows"][0]["number"]
            new_axis = new_p["rows"][0]["number"]
            if old_axis != new_axis:
                changed.append(f"{rno}R: 軸 {old_axis}→{new_axis}番")
            updated.append(new_p)

    updated.sort(key=lambda x: x["race_no"])
    out_path.write_text(json.dumps(updated, ensure_ascii=False, indent=2))

    pending = [p["race_no"] for p in updated if p["race_no"] not in finished_nos]
    print(f"  更新完了: 確定{len(finished_nos)}R / 未確定{len(pending)}R {pending}")
    if changed:
        print(f"  軸馬変化: {', '.join(changed)}")
    print(f"→ 保存: {out_path}")


if __name__ == "__main__":
    main()
