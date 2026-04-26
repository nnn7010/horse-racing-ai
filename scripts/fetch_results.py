#!/usr/bin/env python3
"""当日レース結果を取得し、コース傾向を分析して HTML を更新する。

使い方:
    python3 scripts/fetch_results.py          # 結果取得 + HTML更新
    python3 scripts/fetch_results.py --push   # 上記 + git push
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).parent.parent
PRED_FILE = ROOT / "data/raw/today_predictions.json"
RESULTS_FILE = ROOT / "data/raw/today_results.json"

HEADERS = {"User-Agent": "Mozilla/5.0", "Referer": "https://race.netkeiba.com/"}


# ────────────────────────────────────────────
# 結果取得
# ────────────────────────────────────────────

def fetch_result(race_id: str) -> dict | None:
    """1レースの確定着順を取得する。"""
    url = f"https://race.netkeiba.com/race/result.html?race_id={race_id}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.encoding = "euc-jp"
        soup = BeautifulSoup(resp.text, "lxml")
    except Exception as e:
        print(f"  [warn] 取得失敗 {race_id}: {e}")
        return None

    table = soup.select_one(".RaceTable01, .ResultTable")
    if not table:
        return None

    top3 = []
    for tr in table.select("tr")[1:]:
        tds = tr.select("td")
        if len(tds) < 4:
            continue
        try:
            pos = int(tds[0].get_text(strip=True))
            num = int(tds[2].get_text(strip=True))
            name = tds[3].get_text(strip=True)
            if pos <= 3:
                top3.append({"pos": pos, "num": num, "name": name})
        except ValueError:
            continue

    if len(top3) < 3:
        return None

    top3.sort(key=lambda x: x["pos"])

    # 払い戻し取得
    payouts = {}
    for div in soup.select(".Payout_Detail_Table, .PayoutTable"):
        for tr in div.select("tr"):
            tds = tr.select("td, th")
            if len(tds) >= 2:
                label = tds[0].get_text(strip=True)
                val = tds[-1].get_text(strip=True).replace(",", "").replace("円", "")
                try:
                    if "単勝" in label:
                        payouts["win"] = int(val)
                    elif "複勝" in label:
                        payouts["place"] = int(val)
                except ValueError:
                    pass

    return {
        "1st": top3[0]["num"], "1st_name": top3[0]["name"],
        "2nd": top3[1]["num"], "2nd_name": top3[1]["name"],
        "3rd": top3[2]["num"], "3rd_name": top3[2]["name"],
        "payouts": payouts,
        "status": "confirmed",
    }


# ────────────────────────────────────────────
# 傾向分析
# ────────────────────────────────────────────

def analyze_trends(predictions: dict, results: dict, date_map: dict | None = None) -> dict:
    """完了レースからコース・芝ダート別傾向を計算する。

    date_map を渡すと {date: {venue: {...}}} 形式で日別に出力。
    """
    by_date: dict[str, dict[str, list]] = {}

    for race in predictions["races"]:
        race_id = race["race_id"]
        if race_id not in results:
            continue

        result = results[race_id]
        surface = race.get("surface", "芝")
        venue = f"{race['place_name']} {surface}"
        # 日付特定: date_map優先、なければrace_id先頭8文字
        if date_map and race_id in date_map:
            d = date_map[race_id]
        else:
            d = race_id[:8] if len(race_id) >= 8 else "unknown"

        by_date.setdefault(d, {}).setdefault(venue, [])
        venue_data = by_date[d]

        winner_num = result["1st"]
        horses = {h["number"]: h for h in race["horses"]}
        winner = horses.get(winner_num)
        if not winner:
            continue

        # 人気順（オッズでランク付け）
        sorted_by_odds = sorted(
            [h for h in race["horses"] if h["win_odds"] > 0],
            key=lambda h: h["win_odds"]
        )
        popularity = {h["number"]: i + 1 for i, h in enumerate(sorted_by_odds)}
        winner_pop = popularity.get(winner_num, 99)

        venue_data[venue].append({
            "race_id": race_id,
            "race_name": race["race_name"],
            "winner_num": winner_num,
            "winner_name": result["1st_name"],
            "winner_bracket": winner.get("bracket", 0),
            "winner_odds": winner.get("win_odds", 0),
            "winner_pop": winner_pop,
            "n_horses": race["n_horses"],
            "surface": surface,
            "distance": race["distance"],
            "2nd": result["2nd"],
            "3rd": result["3rd"],
        })

    # 日別 × venue 別に集計
    out: dict = {}
    for date, venue_dict in by_date.items():
        date_trends = {}
        for venue, races in venue_dict.items():
            n = len(races)
            if n == 0:
                continue

            inner_wins = sum(1 for r in races if r["winner_bracket"] in range(1, 5))
            outer_wins = sum(1 for r in races if r["winner_bracket"] in range(5, 9))

            pop_wins = {1: 0, 2: 0, 3: 0, "other": 0}
            for r in races:
                p = r["winner_pop"]
                if p in (1, 2, 3):
                    pop_wins[p] += 1
                else:
                    pop_wins["other"] += 1

            avg_winner_odds = sum(r["winner_odds"] for r in races if r["winner_odds"] > 0) / max(n, 1)
            fav_trust = "高" if pop_wins[1] / n >= 0.4 else "中" if pop_wins[1] / n >= 0.25 else "低"

            if n >= 3:
                if inner_wins / n >= 0.6:
                    bracket_bias = "内枠有利"
                elif outer_wins / n >= 0.6:
                    bracket_bias = "外枠有利"
                else:
                    bracket_bias = "枠差なし"
            else:
                bracket_bias = "データ不足"

            date_trends[venue] = {
                "completed": n,
                "bracket_bias": bracket_bias,
                "inner_wins": inner_wins,
                "outer_wins": outer_wins,
                "pop_wins": pop_wins,
                "fav_trust": fav_trust,
                "avg_winner_odds": round(avg_winner_odds, 1),
                "races": races,
            }
        if date_trends:
            out[date] = date_trends

    return out


# ────────────────────────────────────────────
# メイン
# ────────────────────────────────────────────

def main(push: bool = False):
    if not PRED_FILE.exists():
        print(f"ERROR: {PRED_FILE} が見つかりません")
        return

    with open(PRED_FILE, encoding="utf-8") as f:
        predictions = json.load(f)

    # 既存の結果を読み込む
    existing = {}
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE, encoding="utf-8") as f:
            existing = json.load(f)

    race_ids = [r["race_id"] for r in predictions["races"]]
    pending = [rid for rid in race_ids if rid not in existing]

    print(f"取得済み: {len(existing)}/{len(race_ids)} レース")
    print(f"未取得: {len(pending)} レース")

    new_count = 0
    for race_id in pending:
        race_info = next((r for r in predictions["races"] if r["race_id"] == race_id), {})
        label = f"{race_info.get('place_name','')} {race_info.get('race_name','')}"
        print(f"取得中: {label} ({race_id})", end=" ... ", flush=True)

        result = fetch_result(race_id)
        if result:
            existing[race_id] = result
            print(f"✓ {result['1st']}着={result['1st_name']}")
            new_count += 1
        else:
            print("未確定")
        time.sleep(1.5)

    if new_count > 0 or not RESULTS_FILE.exists():
        with open(RESULTS_FILE, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)
        print(f"\n✓ {len(existing)} レース保存 → {RESULTS_FILE.name}")

    # 傾向分析（日別）
    target_file = ROOT / "data/raw/target_races.json"
    date_map = {}
    if target_file.exists():
        with open(target_file, encoding="utf-8") as f:
            for r in json.load(f):
                date_map[r["race_id"]] = r.get("date", "")
    trends_by_date = analyze_trends(predictions, existing, date_map=date_map)
    trends_file = ROOT / "data/raw/today_trends.json"
    with open(trends_file, "w", encoding="utf-8") as f:
        json.dump(trends_by_date, f, ensure_ascii=False, indent=2)

    # 傾向を表示（最新日付のみ）
    print("\n=== 本日のコース傾向 ===")
    if not trends_by_date:
        return
    latest_date = sorted(trends_by_date.keys())[-1]
    trends = trends_by_date[latest_date]
    print(f"日付: {latest_date}")
    for venue, t in trends.items():
        print(f"\n【{venue}】{t['completed']}R完了")
        print(f"  枠順: {t['bracket_bias']} (内{t['inner_wins']}-外{t['outer_wins']})")
        pw = t["pop_wins"]
        print(f"  人気: 1番人気{pw[1]}勝 2番人気{pw[2]}勝 3番人気{pw[3]}勝 その他{pw['other']}勝")
        print(f"  1番人気信頼度: {t['fav_trust']}  平均勝ちオッズ: {t['avg_winner_odds']}倍")

    # HTML再生成
    result = subprocess.run(
        ["python3", "scripts/generate_html.py"],
        cwd=ROOT, capture_output=True, text=True
    )
    print(f"\n{result.stdout.strip()}")

    if push and new_count > 0:
        subprocess.run(["git", "add",
                        "data/raw/today_results.json",
                        "data/raw/today_trends.json",
                        "docs/index.html"], cwd=ROOT)
        subprocess.run(["git", "commit", "-m",
                        f"chore: update results and trends"], cwd=ROOT)
        subprocess.run(["git", "push", "origin", "main"], cwd=ROOT)
        print("✓ GitHub Pages に push しました")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()
    main(push=args.push)
