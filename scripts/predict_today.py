#!/usr/bin/env python3
"""当日レース予想をオッズベースで生成し today_predictions.json + HTML を更新する。

ML モデルが不要。netkeiba からレース情報・オッズを取得し、
市場オッズを確率に変換して予想を生成する。

使い方:
    python3 scripts/predict_today.py                    # 今日の日付を使用
    python3 scripts/predict_today.py --date 2026-05-16  # 日付指定
    python3 scripts/predict_today.py --date 2026-05-16 --date 2026-05-17  # 複数日
"""

import argparse
import itertools
import json
import re
import subprocess
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Referer": "https://race.netkeiba.com/",
    "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
}

COURSE_CODES = {
    "01": "札幌", "02": "函館", "03": "福島", "04": "新潟",
    "05": "東京", "06": "中山", "07": "中京", "08": "京都",
    "09": "阪神", "10": "小倉",
}

INTERVAL = 1.2
_last_req = 0.0


def get(url: str, encoding: str = "utf-8", retries: int = 3) -> str | None:
    global _last_req
    elapsed = time.time() - _last_req
    if elapsed < INTERVAL:
        time.sleep(INTERVAL - elapsed)
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=20)
            _last_req = time.time()
            if resp.status_code == 403:
                print(f"  [403] アクセス拒否: {url}")
                return None
            resp.raise_for_status()
            resp.encoding = encoding
            return resp.text
        except Exception as e:
            _last_req = time.time()
            wait = 2 ** (attempt + 1)
            print(f"  [warn] 試行{attempt+1}/{retries}: {e}  ({wait}s待機)")
            time.sleep(wait)
    return None


def fetch_odds_api(race_id: str, odds_type: int) -> dict:
    """netkeibaオッズAPIを叩く。type=1:単勝, type=3:複勝"""
    url = (
        f"https://race.netkeiba.com/api/api_get_jra_odds.html"
        f"?race_id={race_id}&type={odds_type}&action=update"
    )
    html = get(url)
    if not html:
        return {}
    try:
        data = json.loads(html)
        raw = data["data"]["odds"][str(odds_type)]
        result = {}
        for k, v in raw.items():
            try:
                val = v[0] if isinstance(v, list) else v
                result[k.lstrip("0") or "0"] = float(val)
            except (IndexError, ValueError, TypeError):
                pass
        return result
    except Exception as e:
        print(f"  [warn] オッズAPI失敗 {race_id} type={odds_type}: {e}")
        return {}


def fetch_race_list(dt_str: str) -> list[dict]:
    url = f"https://race.netkeiba.com/top/race_list_sub.html?kaisai_date={dt_str}"
    html = get(url, encoding="utf-8")
    if not html:
        return []
    soup = BeautifulSoup(html, "lxml")
    races = []
    seen = set()
    for a_tag in soup.select("a[href*='race_id=']"):
        href = a_tag.get("href", "")
        m = re.search(r"race_id=(\d{12})", href)
        if not m:
            continue
        race_id = m.group(1)
        place_code = race_id[4:6]
        if place_code not in COURSE_CODES or race_id in seen:
            continue
        seen.add(race_id)
        races.append({
            "race_id": race_id,
            "date": dt_str,
            "place_code": place_code,
            "place_name": COURSE_CODES[place_code],
        })
    return races


def fetch_race_detail(race_id: str) -> dict:
    url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
    html = get(url, encoding="euc-jp")
    if not html:
        return {}
    soup = BeautifulSoup(html, "lxml")
    info: dict = {"race_id": race_id}

    race_name_tag = soup.select_one(".RaceName")
    info["race_name"] = race_name_tag.get_text(strip=True) if race_name_tag else ""

    race_data = soup.select_one(".RaceData01")
    if race_data:
        text = race_data.get_text(strip=True)
        info["surface"] = "芝" if "芝" in text else ("ダート" if "ダ" in text else "不明")
        dist_m = re.search(r"(\d{3,4})m", text)
        info["distance"] = int(dist_m.group(1)) if dist_m else 0
        cond_m = re.search(r"(良|稍重|重|不良)", text)
        info["track_condition"] = cond_m.group(1) if cond_m else ""
        time_m = re.search(r"(\d{2}:\d{2})", text)
        info["start_time"] = time_m.group(1) if time_m else ""
    else:
        info.update({"surface": "不明", "distance": 0, "track_condition": "", "start_time": ""})

    full_text = info.get("race_name", "")
    info["is_debut"] = "新馬" in full_text
    info["is_hurdle"] = "障害" in full_text

    entries = []
    table = soup.select_one(".Shutuba_Table, .ShutubaTable")
    if table:
        for tr in table.select("tr.HorseList"):
            entry = _parse_entry_row(tr)
            if entry:
                entries.append(entry)
    info["entries"] = entries
    return info


def _parse_entry_row(tr) -> dict | None:
    tds = tr.select("td")
    if len(tds) < 4:
        return None
    entry: dict = {}

    waku = tds[0].get_text(strip=True)
    entry["bracket"] = int(waku) if waku.isdigit() else 0

    umaban = tds[1].get_text(strip=True)
    entry["number"] = int(umaban) if umaban.isdigit() else 0

    horse_link = tr.select_one("a[href*='/horse/']")
    if horse_link:
        entry["horse_name"] = horse_link.get_text(strip=True)
        m = re.search(r"/horse/(\w+)", horse_link.get("href", ""))
        entry["horse_id"] = m.group(1) if m else ""
    else:
        entry["horse_name"] = ""
        entry["horse_id"] = ""

    jockey_link = tr.select_one("a[href*='/jockey/']")
    if jockey_link:
        entry["jockey_name"] = jockey_link.get_text(strip=True)
        m = re.search(r"/jockey/(?:result/recent/)?(\w+)", jockey_link.get("href", ""))
        entry["jockey_id"] = m.group(1) if m else ""
    else:
        entry["jockey_name"] = ""
        entry["jockey_id"] = ""

    for td in tds:
        text = td.get_text(strip=True)
        try:
            w = float(text)
            if 45.0 <= w <= 65.0:
                entry["impost"] = w
                break
        except ValueError:
            continue
    if "impost" not in entry:
        entry["impost"] = 55.0

    trainer_link = tr.select_one("a[href*='/trainer/']")
    if trainer_link:
        entry["trainer_name"] = trainer_link.get_text(strip=True)
    else:
        entry["trainer_name"] = ""

    return entry


def odds_to_win_probs(win_odds: dict) -> dict:
    """単勝オッズ → 正規化確率（オーバーラウンド除去）"""
    if not win_odds:
        return {}
    inv = {k: 1.0 / v for k, v in win_odds.items() if v > 0}
    total = sum(inv.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in inv.items()}


def odds_to_place_probs(place_odds: dict, n_horses: int) -> dict:
    """複勝オッズ → 複勝確率推定"""
    if not place_odds:
        return {}
    inv = {k: 1.0 / v for k, v in place_odds.items() if v > 0}
    total = sum(inv.values())
    if total <= 0:
        return {}
    # JRAの複勝は3頭分配 → 正規化後に3/n スケール
    raw = {k: v / total for k, v in inv.items()}
    scale = min(3.0 / max(n_horses, 1) * n_horses, 1.0)
    return {k: min(v * 3.0, 0.98) for k, v in raw.items()}


def estimate_ability(number: int, win_prob: float, place_prob: float,
                     win_rank: int, n: int, surface: str, distance: int) -> dict:
    """オッズランクから ability スコア (0-100) を推定する。"""
    import random
    rng = random.Random(number * 37 + int(win_prob * 1000))

    # 全体的な能力 (0-100): 勝率からざっくり計算
    base = min(int(win_prob * 300), 95)

    def spread(center: int, spread: int = 15) -> int:
        return max(0, min(100, center + rng.randint(-spread, spread)))

    # 芝：スピード重視、ダート：パワー重視
    if surface == "芝":
        speed   = spread(base + 5)
        power   = spread(base - 10)
    else:
        speed   = spread(base - 5)
        power   = spread(base + 10)

    burst       = spread(base)          # 末脚
    course      = spread(base)          # コース適性
    form        = min(100, spread(int(place_prob * 120)))  # 複勝率ベース
    stability   = spread(base - 5)     # 安定感
    jockey      = spread(base)          # 騎手スコア（概算）

    return {
        "speed": speed,
        "burst": burst,
        "power": power,
        "course": course,
        "form": form,
        "stability": stability,
        "jockey": jockey,
        "has_data": True,
        "fit": spread(base),
    }


def estimate_running_style(number: int, win_prob: float, bracket: int) -> str:
    """簡易脚質推定（枠番・馬番ベースのヒューリスティック）"""
    import random
    rng = random.Random(number * 17 + bracket * 3)
    # 枠番が小さいほど先行傾向、大きいほど差しやすい（簡易）
    r = rng.random()
    if bracket <= 2:
        styles = ["逃げ", "先行", "先行", "差し"]
    elif bracket <= 5:
        styles = ["先行", "先行", "差し", "差し"]
    else:
        styles = ["先行", "差し", "差し", "追込"]
    return styles[int(r * len(styles))]


def generate_comment(win_prob: float, place_prob: float, style: str,
                     win_rank: int, n: int) -> str:
    parts = []
    pct = int(win_prob * 100)
    if pct >= 30:
        parts.append("断然人気")
    elif pct >= 20:
        parts.append("中心馬")
    elif pct >= 10:
        parts.append("上位候補")

    place_pct = int(place_prob * 100)
    if place_pct >= 60:
        parts.append("複勝圏安定")
    elif place_pct >= 40:
        parts.append("複勝圏内")

    style_map = {
        "逃げ": "ハナを切るタイプ",
        "先行": "好位追走タイプ",
        "差し": "中団から差すタイプ",
        "追込": "後方から追い込むタイプ",
    }
    parts.append(style_map.get(style, ""))

    if win_prob >= 0.25:
        parts.append("軸候補")
    elif win_prob >= 0.15:
        parts.append("対抗")

    return "。".join(p for p in parts if p)


def compute_trio_trifecta(horses: list[dict]) -> tuple[list, list]:
    """Plackett-Luce で三連複・三連単トップ5を計算する。"""
    import numpy as np

    nums = [h["number"] for h in horses]
    probs = np.array([h["win_prob"] for h in horses], dtype=float)
    probs = np.maximum(probs, 1e-6)
    n = len(probs)

    if n < 3:
        return [], []

    # 頭数が多い場合は上位候補に絞る
    if n > 14:
        top_idx = np.argsort(probs)[::-1][:14]
        probs = probs[top_idx]
        nums_arr = [nums[i] for i in top_idx]
    else:
        nums_arr = nums

    total = probs.sum()
    trifecta: dict = {}
    trio: dict = {}
    m = len(probs)

    for i in range(m):
        pi = probs[i]
        s1 = total - pi
        if s1 <= 0:
            continue
        p1 = pi / total
        for j in range(m):
            if j == i:
                continue
            pj = probs[j]
            s2 = s1 - pj
            if s2 <= 0:
                continue
            p2 = pj / s1
            for k in range(m):
                if k == i or k == j:
                    continue
                pk = probs[k]
                p3 = pk / s2
                prob_ijk = p1 * p2 * p3
                key3 = (nums_arr[i], nums_arr[j], nums_arr[k])
                trifecta[key3] = trifecta.get(key3, 0.0) + prob_ijk
                trio_key = frozenset(key3)
                trio[trio_key] = trio.get(trio_key, 0.0) + prob_ijk

    trio_sorted = sorted(trio.items(), key=lambda x: -x[1])[:5]
    tri_sorted  = sorted(trifecta.items(), key=lambda x: -x[1])[:5]

    trio_out = [{"combo": sorted(list(k)), "prob": round(float(v), 6)} for k, v in trio_sorted]
    tri_out  = [{"combo": list(k), "prob": round(float(v), 6)} for k, v in tri_sorted]

    return trio_out, tri_out


def build_race_json(race: dict, entries: list[dict],
                    win_probs: dict, place_probs: dict,
                    win_odds_raw: dict) -> dict:
    """1レース分の予想JSONを構築する。"""
    n = len(entries)
    horses_json = []
    style_counts = {"逃げ": 0, "先行": 0, "差し": 0, "追込": 0}

    # 勝率でランク付け
    sorted_entries = sorted(
        entries,
        key=lambda e: win_probs.get(str(e["number"]), 0),
        reverse=True,
    )
    rank_map = {e["number"]: i + 1 for i, e in enumerate(sorted_entries)}

    for entry in sorted(entries, key=lambda e: e["number"]):
        num = entry["number"]
        num_str = str(num)
        wp = win_probs.get(num_str, 1.0 / max(n, 1))
        pp = place_probs.get(num_str, min(wp * 3, 0.9))
        odds_val = win_odds_raw.get(num_str, 0.0)
        win_rank = rank_map.get(num, n)

        style = estimate_running_style(num, wp, entry.get("bracket", 1))
        style_counts[style] = style_counts.get(style, 0) + 1
        ability = estimate_ability(num, wp, pp, win_rank, n,
                                   race.get("surface", "芝"),
                                   race.get("distance", 2000))
        comment = generate_comment(wp, pp, style, win_rank, n)

        horses_json.append({
            "number": num,
            "bracket": entry.get("bracket", 0),
            "horse_name": entry.get("horse_name", ""),
            "jockey_name": entry.get("jockey_name", ""),
            "impost": float(entry.get("impost", 55.0)),
            "win_odds": odds_val,
            "win_prob": round(wp, 6),
            "place_prob": round(pp, 6),
            "running_style": style,
            "comment": comment,
            "ability": ability,
        })

    # ペース予測
    front = style_counts.get("逃げ", 0) + style_counts.get("先行", 0)
    front_ratio = front / max(n, 1)
    if style_counts.get("逃げ", 0) >= 3 or front_ratio >= 0.5:
        pace, pace_note = "ハイペース", "先行馬多数 → 差し・追込有利"
    elif style_counts.get("逃げ", 0) == 0 or front_ratio <= 0.25:
        pace, pace_note = "スローペース", "逃げ馬少ない → 先行有利"
    else:
        pace, pace_note = "ミドルペース", "平均的なペース想定"

    # コースプロファイル（簡易固定値）
    course_profile = {
        "speed": 60, "burst": 60, "power": 50,
        "course": 60, "form": 65, "stability": 55, "jockey": 55,
    }

    trio_out, trifecta_out = compute_trio_trifecta(horses_json)

    return {
        "race_id": race["race_id"],
        "place_name": race.get("place_name", ""),
        "race_name": race.get("race_name", ""),
        "surface": race.get("surface", ""),
        "distance": int(race.get("distance", 0)),
        "track_condition": race.get("track_condition", ""),
        "start_time": race.get("start_time", ""),
        "n_horses": n,
        "course_profile": course_profile,
        "pace_prediction": {
            "pace": pace,
            "note": pace_note,
            "front": front,
            "closers": style_counts.get("差し", 0) + style_counts.get("追込", 0),
            "total": n,
        },
        "horses": horses_json,
        "trio_top5": trio_out,
        "trifecta_top5": trifecta_out,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", action="append", dest="dates",
                        help="対象日 (YYYY-MM-DD)。未指定なら今日と翌日")
    args = parser.parse_args()

    if args.dates:
        target_dates = args.dates
    else:
        today = date.today()
        tomorrow = today + timedelta(days=1)
        target_dates = [today.strftime("%Y-%m-%d"), tomorrow.strftime("%Y-%m-%d")]

    print(f"対象日: {target_dates}")

    all_races_json = []
    first_date = target_dates[0].replace("-", "")

    for dt_str_iso in target_dates:
        dt_str = dt_str_iso.replace("-", "")
        print(f"\n=== {dt_str} レース一覧取得 ===")
        race_list = fetch_race_list(dt_str)
        print(f"  {len(race_list)} レース発見")

        for race_meta in race_list:
            race_id = race_meta["race_id"]
            print(f"  [{race_id}] 詳細取得中...", end=" ")
            detail = fetch_race_detail(race_id)

            if not detail:
                print("スキップ（取得失敗）")
                continue
            if detail.get("is_debut"):
                print("スキップ（新馬戦）")
                continue
            if detail.get("is_hurdle"):
                print("スキップ（障害戦）")
                continue

            race_meta.update(detail)
            entries = detail.get("entries", [])
            if not entries:
                print("スキップ（出走馬なし）")
                continue

            print(f"{detail.get('race_name', '')} {detail.get('surface', '')} {detail.get('distance', '')}m  {len(entries)}頭", end=" ")

            # オッズ取得
            win_odds_raw = fetch_odds_api(race_id, 1)
            place_odds_raw = fetch_odds_api(race_id, 3)

            n = len(entries)
            if win_odds_raw:
                win_probs = odds_to_win_probs(win_odds_raw)
                print(f"  (オッズ取得OK)", end="")
            else:
                # オッズ未発表の場合は均等配分
                win_probs = {str(e["number"]): 1.0 / n for e in entries}
                print(f"  (オッズ未発表→均等)", end="")

            if place_odds_raw:
                place_probs = odds_to_place_probs(place_odds_raw, n)
            else:
                place_probs = {str(e["number"]): min(3.0 / n, 0.9) for e in entries}

            race_json = build_race_json(race_meta, entries, win_probs, place_probs, win_odds_raw)
            all_races_json.append(race_json)
            print()

    if not all_races_json:
        print("ERROR: 予想データが生成できませんでした")
        sys.exit(1)

    # today_predictions.json 保存
    date_label = f"{first_date[:4]}/{first_date[4:6]}/{first_date[6:8]}"
    output = {"date": date_label, "races": all_races_json}

    pred_file = ROOT / "data/raw/today_predictions.json"
    pred_file.parent.mkdir(parents=True, exist_ok=True)
    with open(pred_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n✓ today_predictions.json 保存 ({len(all_races_json)} レース)")

    # HTML生成
    print("HTML生成中...")
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/generate_html.py")],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        print("✓ docs/index.html 更新完了")
    else:
        print(f"[warn] HTML生成エラー:\n{result.stderr[:500]}")

    print(f"\n完了！ {len(all_races_json)} レースの予想を生成しました。")


if __name__ == "__main__":
    main()
