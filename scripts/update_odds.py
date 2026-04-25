#!/usr/bin/env python3
"""オッズを再取得して today_predictions.json と docs/index.html を更新する。

使い方:
    python3 scripts/update_odds.py          # 全レース更新
    python3 scripts/update_odds.py --push   # 更新後 git push も実行
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

import requests

ROOT = Path(__file__).parent.parent
PRED_FILE = ROOT / "data/raw/today_predictions.json"
CACHE_DIR = ROOT / "data/cache"

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://race.netkeiba.com/",
}


def fetch_win_odds(race_id: str) -> dict[str, float]:
    """単勝オッズを取得。{馬番文字列: オッズ}"""
    url = f"https://race.netkeiba.com/api/api_get_jra_odds.html?race_id={race_id}&type=1&action=update"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return {k: float(v[0]) for k, v in data["data"]["odds"]["1"].items()}
    except Exception as e:
        print(f"  [warn] 単勝オッズ取得失敗 {race_id}: {e}")
        return {}


def fetch_place_odds(race_id: str) -> dict[str, float]:
    """複勝オッズを取得。{馬番文字列: オッズ（最小値）}"""
    url = f"https://race.netkeiba.com/api/api_get_jra_odds.html?race_id={race_id}&type=3&action=update"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        # 複勝オッズは {"01": ["1.5", "2.0", ...]} または {"01": "1.5"} の形式
        result = {}
        for k, v in data["data"]["odds"]["3"].items():
            try:
                val = v[0] if isinstance(v, list) else v
                result[k] = float(val)
            except (IndexError, ValueError, TypeError):
                pass
        return result
    except Exception as e:
        print(f"  [warn] 複勝オッズ取得失敗 {race_id}: {e}")
        return {}


def update_odds(push: bool = False):
    if not PRED_FILE.exists():
        print(f"ERROR: {PRED_FILE} が見つかりません")
        return

    with open(PRED_FILE, encoding="utf-8") as f:
        data = json.load(f)

    races = data["races"]
    updated = 0

    for race in races:
        race_id = race["race_id"]
        race_name = f"{race['place_name']} {race['race_name']}"
        print(f"取得中: {race_name} ({race_id})")

        win_odds = fetch_win_odds(race_id)
        time.sleep(1.0)
        place_odds = fetch_place_odds(race_id)
        time.sleep(1.0)

        if not win_odds:
            print(f"  → スキップ（オッズ未取得）")
            continue

        for horse in race["horses"]:
            num_str = str(horse["number"]).zfill(2)
            if num_str in win_odds:
                horse["win_odds"] = win_odds[num_str]
            if num_str in place_odds:
                horse["place_odds"] = place_odds[num_str]

        updated += 1
        print(f"  → {len(win_odds)}頭分更新")

    # JSON保存
    with open(PRED_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\n✓ {updated}/{len(races)} レース更新 → {PRED_FILE.name}")

    # HTML再生成
    result = subprocess.run(
        ["python3", "scripts/generate_html.py"],
        cwd=ROOT, capture_output=True, text=True
    )
    print(result.stdout.strip())

    # git push
    if push:
        subprocess.run(["git", "add",
                        "data/raw/today_predictions.json",
                        "docs/index.html"], cwd=ROOT)
        subprocess.run(["git", "commit", "-m",
                        f"chore: update odds {data.get('date', '')}"],
                       cwd=ROOT)
        subprocess.run(["git", "push", "origin", "main"], cwd=ROOT)
        print("✓ GitHub Pages に push しました")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--push", action="store_true", help="更新後 git push する")
    args = parser.parse_args()
    update_odds(push=args.push)
