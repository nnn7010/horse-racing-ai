#!/usr/bin/env python3
"""各コースの能力プロファイルを過去データから算出する。

04_build_features.py の後に実行する。
出力: data/processed/course_profiles.json
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np

FEATURES_FILE = ROOT / "data/processed/features.parquet"
OUT_FILE = ROOT / "data/processed/course_profiles.json"

VENUE_NAMES = {
    1: "札幌", 2: "函館", 3: "福島", 4: "新潟", 5: "東京",
    6: "中山", 7: "中京", 8: "京都", 9: "阪神", 10: "小倉",
}

# (key, column, invert)  invert=True: 値が低いほど良い
ABILITY_AXES = [
    ("speed",     "avg_speed_3",            True),
    ("burst",     "avg_last_3f_rank_3",     True),
    ("power",     "horse_tough_top3_rate",  False),
    ("course",    "horse_course_top3_rate", False),
    ("form",      "avg_finish_5",           True),
    ("stability", "finish_std_5",           True),
    ("jockey",    "jockey_top3_rate",       False),
]
MIN_SAMPLES = 15  # コース別最低top3サンプル数


def compute():
    if not FEATURES_FILE.exists():
        print(f"ERROR: {FEATURES_FILE} が見つかりません。04_build_features.py を先に実行してください。")
        sys.exit(1)

    df = pd.read_parquet(FEATURES_FILE)

    # グローバル基準値（外れ値除外のため1-99パーセンタイル）
    global_stats: dict[str, dict] = {}
    for key, col, invert in ABILITY_AXES:
        if col not in df.columns:
            continue
        vals = df[col].replace(0, np.nan).dropna()  # 0は「データなし」扱いで除外
        global_stats[key] = {
            "min": float(vals.quantile(0.05)),   # 5-95pctで外れ値を除外
            "max": float(vals.quantile(0.95)),
            "median": float(vals.median()),
            "invert": invert,
            "col": col,
        }

    def to_score(value: float, stats: dict) -> float:
        mn, mx = stats["min"], stats["max"]
        if mx == mn:
            return 50.0
        s = (value - mn) / (mx - mn) * 100
        s = max(0.0, min(100.0, s))
        return 100.0 - s if stats["invert"] else s

    df["course_key"] = (
        df["place_code_num"].astype(str) + "_"
        + df["is_turf"].astype(str) + "_"
        + df["distance"].astype(str)
    )

    profiles: dict = {}

    for course_key, group in df.groupby("course_key"):
        top3 = group[group["finish_position"] <= 3]
        if len(top3) < MIN_SAMPLES:
            continue

        parts = course_key.split("_")
        venue_num = int(parts[0])
        is_turf = parts[1] == "1"
        distance = int(parts[2])
        surface = "芝" if is_turf else "ダート"
        label = f"{VENUE_NAMES.get(venue_num, str(venue_num))} {surface} {distance}m"

        scores: dict[str, float] = {}
        for key, col, _ in ABILITY_AXES:
            if key not in global_stats or col not in top3.columns:
                scores[key] = 50.0
                continue
            vals = top3[col].replace(0, np.nan).dropna()
            if len(vals) < 5:
                scores[key] = 50.0
            else:
                scores[key] = round(to_score(float(vals.median()), global_stats[key]), 1)

        profiles[course_key] = {
            "label": label,
            "n_top3": int(len(top3)),
            "scores": scores,
        }

    OUT_FILE.parent.mkdir(exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(profiles, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(profiles)} course profiles → {OUT_FILE}")
    # サンプル表示
    for k, v in sorted(profiles.items(), key=lambda x: -x[1]["n_top3"])[:8]:
        s = v["scores"]
        print(f"  {v['label']:18s} (n={v['n_top3']:4d})  "
              f"S={s['speed']:4.0f} 瞬={s['burst']:4.0f} パ={s['power']:4.0f} "
              f"C={s['course']:4.0f} F={s['form']:4.0f} 安={s['stability']:4.0f} 騎={s['jockey']:4.0f}")

    return global_stats  # 06_predict.pyから参照用


if __name__ == "__main__":
    compute()
