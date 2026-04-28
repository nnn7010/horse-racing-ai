"""当日バイアスを確定済みレースから再推定し {date}.json を更新する。

データリーク防止:
  - コースプロファイル・馬能力ベクトルは当日結果を除外して構築
  - 確定済みレースの予想は _no_bias.json から凍結コピー（書き換えない）
  - 未確定レースのみ当日バイアスを適用して更新

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


def _fetch_result(race_id: str) -> dict | None:
    cache = ROOT / "data/oi/raw/results" / f"{race_id}.json"
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
    ap.add_argument("--date",   required=True, help="YYYY-MM-DD")
    ap.add_argument("--weight", type=float, default=0.3)
    ap.add_argument("--fetch",  action="store_true", help="HTTPから結果を再取得してから更新")
    args = ap.parse_args()

    yyyymmdd = args.date.replace("-", "")
    pred_dir = ROOT / "data/oi/predictions"
    out_path  = pred_dir / f"{args.date}.json"
    nb_path   = pred_dir / f"{args.date}_no_bias.json"

    if not nb_path.exists():
        print(f"_no_bias.json がありません: {nb_path}", file=sys.stderr)
        print("先に --save-all で3バリアントを生成してください。", file=sys.stderr)
        sys.exit(1)

    # _no_bias.json を「確定済みレース凍結用のベース」として読み込む
    nb_preds: dict[int, dict] = {p["race_no"]: p for p in json.loads(nb_path.read_text())}

    # 現在の当日バイアスファイル（なければno_biasで初期化）
    current: dict[int, dict] = (
        {p["race_no"]: p for p in json.loads(out_path.read_text())}
        if out_path.exists()
        else dict(nb_preds)
    )

    # 結果取得
    if args.fetch:
        print("  結果を再取得中...")
        for p in nb_preds.values():
            _fetch_result(p["race_id"])

    # 確定済みレースを収集
    results_dir = ROOT / "data/oi/raw/results"
    finished: list[dict] = []
    finished_nos: set[int] = set()
    for p in nb_preds.values():
        cache = results_dir / f"{p['race_id']}.json"
        if not cache.exists():
            continue
        d = json.loads(cache.read_text())
        if d.get("num_runners", 0) > 0 and d.get("date") == yyyymmdd:
            finished.append(d)
            finished_nos.add(p["race_no"])

    if not finished:
        print("  確定済みレースがありません。更新をスキップします。")
        return

    pending_nos = sorted(set(nb_preds.keys()) - finished_nos)
    if not pending_nos:
        print("  全レース確定済みです。更新をスキップします。")
        return

    today_bias = estimate_today_bias(finished)
    print(f"  確定 {len(finished_nos)}R / 未確定 {len(pending_nos)}R {pending_nos}")
    print(f"  枠別3着内率: {today_bias['bracket_top3_rate_today']}")
    print(f"  1番人気3着内率: {today_bias['fav_top3_rate_today']}")

    # ── モデル構築: 当日確定済み結果を含めて残りレースを再予想 ──────────────
    course_profile = build_course_profile(results_dir)
    by_horse       = index_results_by_horse(results_dir)

    shutuba_dir = ROOT / "data/oi/raw/shutuba"
    shutubas: dict[int, dict] = {}
    for fp in sorted(shutuba_dir.glob(f"*44{yyyymmdd[4:8]}*.json")):
        d = json.loads(fp.read_text())
        if d.get("date") == yyyymmdd:
            shutubas[d["race_no"]] = d

    horse_dir = ROOT / "data/oi/raw/horses"
    horse_ids = {e["horse_id"] for rno in pending_nos for e in shutubas[rno]["entries"] if e.get("horse_id")}
    abilities: dict[str, dict] = {}
    jra_info:  dict[str, tuple] = {}
    for hid in horse_ids:
        jn, jt = load_jra_summary(hid, horse_dir)
        jra_info[hid] = (jn, jt)
        abilities[hid] = build_ability_vector(by_horse.get(hid, []), course_profile, jn, jt)

    # ── 更新: 確定済み=凍結、未確定=当日バイアス適用 ─────────────────────
    updated: dict[int, dict] = {}
    changed: list[str] = []

    for rno, p in nb_preds.items():
        if rno in finished_nos:
            updated[rno] = nb_preds[rno]   # _no_bias.json から凍結コピー
        else:
            s = shutubas.get(rno)
            if not s:
                updated[rno] = current.get(rno, p)
                continue
            new_p = predict_race(s, course_profile, abilities, jra_info, today_bias, args.weight)
            old_axis = current.get(rno, p)["rows"][0]["number"]
            new_axis = new_p["rows"][0]["number"]
            if old_axis != new_axis:
                changed.append(f"{rno}R: 軸 {old_axis}→{new_axis}番")
            updated[rno] = new_p

    result_list = sorted(updated.values(), key=lambda x: x["race_no"])
    out_path.write_text(json.dumps(result_list, ensure_ascii=False, indent=2))

    if changed:
        print(f"  軸馬変化: {', '.join(changed)}")
    print(f"→ 保存: {out_path}")


if __name__ == "__main__":
    main()
