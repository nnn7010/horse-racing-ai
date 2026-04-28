"""馬プロファイル（縦軸）。

各出走馬について、過去レース結果(大井1年分)+ 馬個体データ(JRA含む) から
能力プロファイルを集計する。

出力指標:
  oi_n              大井出走数
  oi_top3_rate      大井3着内率
  oi_win_rate       大井勝率
  recent_finish_avg 直近5走の (着順/頭数) 平均
  recent_trend      直近5走の傾向 (-1..+1, 正=上昇)
  dist_n[d]         大井×距離別出走数
  dist_top3[d]      大井×距離別3着内率
  dist_time_dev[d]  大井×距離別タイム偏差(コース平均との差; マイナスほど速い)
  dist_last3f_dev   上り3F偏差(マイナスほど速い)
  track_top3_good   良馬場3着内率
  track_top3_heavy  重・不良馬場3着内率
  days_since_last   前走からの日数
  jra_experience    JRA出走数 (horse個体ファイルから)
  jra_top3_rate     JRA3着内率
  weight_avg        平均馬体重
  pop1_count        1番人気経験数
  pop1_top3         1番人気時の3着内率
"""

import json
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from statistics import mean


JRA_PLACES = {"札幌","函館","福島","新潟","東京","中山","中京","京都","阪神","小倉"}


def _parse_date(s: str) -> date | None:
    s = (s or "").replace("/", "").replace("-", "")
    if len(s) != 8 or not s.isdigit(): return None
    try:
        return date(int(s[:4]), int(s[4:6]), int(s[6:8]))
    except ValueError:
        return None


def _is_jra(place: str) -> bool:
    import re
    place_kanji = re.sub(r"\d+", "", place or "")
    return place_kanji in JRA_PLACES


def _is_oi(place: str) -> bool:
    import re
    place_kanji = re.sub(r"\d+", "", place or "")
    return place_kanji == "大井"


def index_results_by_horse(results_dir: str | Path) -> dict[str, list[dict]]:
    """大井結果データを horse_id でインデックス化（メタ込み）。"""
    results_dir = Path(results_dir)
    by_horse: dict[str, list[dict]] = defaultdict(list)
    for fp in sorted(results_dir.glob("*.json")):
        d = json.loads(fp.read_text())
        if d.get("is_hurdle") or d.get("is_debut"):
            continue
        meta = {
            "race_id": d["race_id"],
            "date": d["date"],
            "distance": d["distance"],
            "track_condition": d.get("track_condition", "") or "?",
            "num_runners": d.get("num_runners", 0),
            "race_class": d.get("race_class", ""),
        }
        for r in d.get("results", []):
            hid = r.get("horse_id")
            if not hid: continue
            row = dict(meta)
            row.update({
                "finish": r["finish_position"],
                "bracket": r["bracket"],
                "number": r["number"],
                "popularity": r.get("popularity", 0),
                "win_odds": r.get("win_odds", 0.0),
                "time": r.get("time", 0.0),
                "last_3f": r.get("last_3f", 0.0),
                "horse_weight": r.get("horse_weight", 0),
                "weight_change": r.get("weight_change", 0),
                "impost": r.get("impost", 0.0),
                "jockey_id": r.get("jockey_id", ""),
                "jockey_name": r.get("jockey_name", ""),
            })
            by_horse[hid].append(row)
    for hid in by_horse:
        by_horse[hid].sort(key=lambda x: x["date"])
    return by_horse


def build_horse_profile(
    horse_id: str,
    oi_history: list[dict],
    horse_file_dir: str | Path,
    course_profile: dict,
    today: date | None = None,
) -> dict:
    today = today or date.today()
    prof: dict = {"horse_id": horse_id}

    # ── 大井集計 ──────────────────────────────────────────
    valid = [r for r in oi_history if r["finish"] > 0]
    prof["oi_n"] = len(valid)
    if valid:
        prof["oi_win_rate"] = round(sum(1 for r in valid if r["finish"] == 1) / len(valid), 3)
        prof["oi_top3_rate"] = round(sum(1 for r in valid if r["finish"] <= 3) / len(valid), 3)
        prof["oi_finish_rate_avg"] = round(mean(r["finish"] / max(r["num_runners"], 1) for r in valid), 3)
    else:
        prof["oi_win_rate"] = prof["oi_top3_rate"] = prof["oi_finish_rate_avg"] = None

    # 直近5走 (大井のみ、新→旧の順で last 5)
    recent = sorted(valid, key=lambda x: x["date"])[-5:]
    if recent:
        prof["recent5_finish_rate_avg"] = round(mean(r["finish"] / max(r["num_runners"], 1) for r in recent), 3)
        if len(recent) >= 3:
            # 線形傾向: 着順率の傾き(過去→現在で減少すればトレンド+)
            xs = list(range(len(recent)))
            ys = [r["finish"] / max(r["num_runners"], 1) for r in recent]
            mx, my = mean(xs), mean(ys)
            num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
            den = sum((x - mx) ** 2 for x in xs)
            slope = -num / den if den else 0  # 下降=好調なので符号反転
            prof["recent_trend"] = round(max(-1.0, min(1.0, slope * 5)), 3)
        else:
            prof["recent_trend"] = 0.0
    else:
        prof["recent5_finish_rate_avg"] = None
        prof["recent_trend"] = 0.0

    # 距離別
    by_dist: dict[int, list[dict]] = defaultdict(list)
    for r in valid:
        by_dist[r["distance"]].append(r)
    prof["dist_n"] = {d: len(rows) for d, rows in by_dist.items()}
    prof["dist_top3"] = {d: round(sum(1 for r in rows if r["finish"] <= 3) / len(rows), 3) for d, rows in by_dist.items()}

    # 距離別タイム偏差(コース平均との差)
    dist_time_dev: dict[int, float] = {}
    dist_last3f_dev: dict[int, float] = {}
    for d, rows in by_dist.items():
        # 着順問わず使う
        time_devs = []
        l3f_devs = []
        for r in rows:
            key = f"{d}m_{r['track_condition']}"
            cp = course_profile.get(key) or course_profile.get(f"{d}m_良")
            if not cp: continue
            if r["time"] > 0 and cp.get("win_time_mean"):
                time_devs.append(r["time"] - cp["win_time_mean"])
            if r["last_3f"] > 0 and cp.get("win_last3f_mean"):
                l3f_devs.append(r["last_3f"] - cp["win_last3f_mean"])
        if time_devs:
            dist_time_dev[d] = round(mean(time_devs), 2)
        if l3f_devs:
            dist_last3f_dev[d] = round(mean(l3f_devs), 2)
    prof["dist_time_dev"] = dist_time_dev
    prof["dist_last3f_dev"] = dist_last3f_dev

    # 馬場別
    good = [r for r in valid if r["track_condition"] in ("良", "?")]
    heavy = [r for r in valid if r["track_condition"] in ("重", "稍重", "不良")]
    prof["track_top3_good"] = round(sum(1 for r in good if r["finish"] <= 3) / len(good), 3) if good else None
    prof["track_top3_heavy"] = round(sum(1 for r in heavy if r["finish"] <= 3) / len(heavy), 3) if heavy else None

    # 1番人気時成績
    pop1 = [r for r in valid if r["popularity"] == 1]
    prof["pop1_count"] = len(pop1)
    prof["pop1_top3"] = round(sum(1 for r in pop1 if r["finish"] <= 3) / len(pop1), 3) if pop1 else None

    # 馬体重
    weights = [r["horse_weight"] for r in valid if r["horse_weight"] > 0]
    prof["weight_avg"] = round(mean(weights)) if weights else None
    prof["weight_last"] = weights[-1] if weights else None

    # 休み明け日数
    if valid:
        last_date = _parse_date(valid[-1]["date"])
        prof["days_since_last"] = (today - last_date).days if last_date else None
    else:
        prof["days_since_last"] = None

    # ── 馬個体ファイル(あれば) からの拡張 ────────────
    horse_path = Path(horse_file_dir) / f"{horse_id}.json"
    if horse_path.exists():
        h = json.loads(horse_path.read_text())
        prof["horse_name"] = h.get("horse_name", "")
        prof["sire"] = h.get("sire", "")
        prof["dam_sire"] = h.get("dam_sire", "")
        past = h.get("past_results", [])
        jra_runs = [r for r in past if _is_jra(r.get("place", ""))]
        prof["jra_n"] = len(jra_runs)
        if jra_runs:
            valid_jra = [r for r in jra_runs if r.get("finish_position", 0) > 0]
            prof["jra_top3_rate"] = round(sum(1 for r in valid_jra if r["finish_position"] <= 3) / len(valid_jra), 3) if valid_jra else None
        else:
            prof["jra_top3_rate"] = None
    else:
        prof["horse_name"] = ""
        prof["jra_n"] = 0
        prof["jra_top3_rate"] = None

    return prof


def build_all_profiles(
    horse_ids: list[str],
    results_dir: str | Path,
    horse_file_dir: str | Path,
    course_profile: dict,
    today: date | None = None,
) -> dict[str, dict]:
    by_horse = index_results_by_horse(results_dir)
    profiles: dict[str, dict] = {}
    for hid in horse_ids:
        profiles[hid] = build_horse_profile(
            hid,
            by_horse.get(hid, []),
            horse_file_dir,
            course_profile,
            today=today,
        )
    return profiles
