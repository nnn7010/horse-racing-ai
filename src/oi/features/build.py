"""特徴量生成。

入力:
  - data/oi/raw/results/{race_id}.json    (レース結果)
  - data/oi/raw/horses/{horse_id}.json    (馬個体: 血統+過去成績)
  - data/oi/processed/bias_daily.csv      (日次バイアス)

出力:
  - data/oi/processed/features.parquet    (1行=1出走)
  - data/oi/processed/labels.parquet      (win/top3 ラベル)

ポイント:
  - 過去成績は出走日より前のものだけを使う（リーク防止）
  - JRA成績と地方成績を分離して別特徴量にする
  - 同じレースに出る他馬との横比較は予測時にレース内ランクで実施（学習時は不要）
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from src.oi.scraping.horse import classify_past_result
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _to_date(yyyymmdd: str) -> date | None:
    if not yyyymmdd or len(yyyymmdd) != 8 or not yyyymmdd.isdigit():
        return None
    try:
        return date(int(yyyymmdd[:4]), int(yyyymmdd[4:6]), int(yyyymmdd[6:8]))
    except ValueError:
        return None


def _summarize_runs(rows: list[dict], lookback_days: int | None = None, asof: date | None = None) -> dict:
    """過去成績行リストから集約特徴量を計算する。"""
    if asof is not None:
        rows = [r for r in rows if (d := _to_date(r.get("date", ""))) and d < asof]
    if lookback_days is not None and asof is not None:
        cutoff = asof - timedelta(days=lookback_days)
        rows = [r for r in rows if (d := _to_date(r.get("date", ""))) and d >= cutoff]

    n = len(rows)
    if n == 0:
        return {"runs": 0, "win_rate": 0.0, "top3_rate": 0.0, "avg_last_3f": 0.0}

    wins = sum(1 for r in rows if r.get("finish_position") == 1)
    top3 = sum(1 for r in rows if 1 <= r.get("finish_position", 0) <= 3)
    valid_3f = [r["last_3f"] for r in rows if r.get("last_3f", 0)]
    avg_3f = sum(valid_3f) / len(valid_3f) if valid_3f else 0.0

    return {
        "runs": n,
        "win_rate": wins / n,
        "top3_rate": top3 / n,
        "avg_last_3f": avg_3f,
    }


def _course_specific(rows: list[dict], distance: int, asof: date) -> dict:
    """同じ距離帯（±200m）での成績を集約する。"""
    if not distance:
        return {"course_runs": 0, "course_top3_rate": 0.0, "course_win_rate": 0.0}
    similar = [r for r in rows if abs(r.get("distance", 0) - distance) <= 200]
    similar = [r for r in similar if (d := _to_date(r.get("date", ""))) and d < asof]
    if not similar:
        return {"course_runs": 0, "course_top3_rate": 0.0, "course_win_rate": 0.0}
    wins = sum(1 for r in similar if r.get("finish_position") == 1)
    top3 = sum(1 for r in similar if 1 <= r.get("finish_position", 0) <= 3)
    return {
        "course_runs": len(similar),
        "course_top3_rate": top3 / len(similar),
        "course_win_rate": wins / len(similar),
    }


def _build_runner_row(
    race: dict,
    runner: dict,
    horse_info: dict | None,
    bias_today: dict | None,
    jockey_stats: dict | None,
    trainer_stats: dict | None,
) -> dict:
    """1出走の特徴量行を構築する。"""
    asof = _to_date(race.get("date", ""))
    distance = race.get("distance", 0)

    feat: dict = {
        "race_id": race["race_id"],
        "date": race.get("date", ""),
        "horse_id": runner.get("horse_id", ""),
        "horse_name": runner.get("horse_name", ""),
        # レース条件
        "distance": distance,
        "surface": race.get("surface", "ダート"),
        "track_condition": race.get("track_condition", ""),
        "weather": race.get("weather", ""),
        "num_runners": race.get("num_runners", 0),
        # 馬・出走条件
        "bracket": runner.get("bracket", 0),
        "number": runner.get("number", 0),
        "impost": runner.get("impost", 0.0),
        "horse_weight": runner.get("horse_weight", 0),
        "weight_change": runner.get("weight_change", 0),
        "sex_age": runner.get("sex_age", ""),
        # ラベル
        "finish_position": runner.get("finish_position", 0),
        "is_win": int(runner.get("finish_position", 0) == 1),
        "is_top3": int(1 <= runner.get("finish_position", 0) <= 3),
        # オッズ（学習時は使わない可能性あり、予測時は必須）
        "win_odds": runner.get("win_odds", 0.0),
    }

    # 過去成績由来
    if horse_info and asof:
        past = horse_info.get("past_results", [])

        # 全体
        feat.update({f"all_{k}": v for k, v in _summarize_runs(past, asof=asof).items()})
        # 直近5走（日数指定なし、最新5走を抽出するためのソート前提）
        past_sorted = sorted(
            [r for r in past if (d := _to_date(r.get("date", ""))) and d < asof],
            key=lambda r: r.get("date", ""),
            reverse=True,
        )
        recent5 = past_sorted[:5]
        feat.update({f"recent5_{k}": v for k, v in _summarize_runs(recent5).items()})

        # 前走情報
        if past_sorted:
            last = past_sorted[0]
            last_d = _to_date(last.get("date", ""))
            feat["last_finish"] = last.get("finish_position", 0)
            feat["last_distance"] = last.get("distance", 0)
            feat["last_last_3f"] = last.get("last_3f", 0.0)
            feat["last_win_odds"] = last.get("win_odds", 0.0)
            feat["days_since_last"] = (asof - last_d).days if last_d else 999
        else:
            feat["last_finish"] = 0
            feat["last_distance"] = 0
            feat["last_last_3f"] = 0.0
            feat["last_win_odds"] = 0.0
            feat["days_since_last"] = 999

        # 場系列ごと分離
        from src.oi.scraping.horse import split_past_results
        buckets = split_past_results([r for r in past if (d := _to_date(r.get("date", ""))) and d < asof])

        # JRA成績（中央経験馬の能力推定）
        jra_rows = buckets["jra"]
        feat["jra_runs"] = len(jra_rows)
        feat["has_jra_experience"] = int(len(jra_rows) > 0)
        if jra_rows:
            feat["jra_top3_rate"] = sum(1 for r in jra_rows if 1 <= r.get("finish_position", 0) <= 3) / len(jra_rows)
            feat["jra_win_rate"] = sum(1 for r in jra_rows if r.get("finish_position") == 1) / len(jra_rows)
            # 直近2年JRA
            cutoff = asof - timedelta(days=730)
            recent_jra = [r for r in jra_rows if (d := _to_date(r.get("date", ""))) and d >= cutoff]
            feat["jra_recent2y_runs"] = len(recent_jra)
            feat["jra_recent2y_top3_rate"] = (
                sum(1 for r in recent_jra if 1 <= r.get("finish_position", 0) <= 3) / len(recent_jra)
                if recent_jra else 0.0
            )
        else:
            feat["jra_top3_rate"] = 0.0
            feat["jra_win_rate"] = 0.0
            feat["jra_recent2y_runs"] = 0
            feat["jra_recent2y_top3_rate"] = 0.0

        # 大井成績
        oi_rows = buckets["oi"]
        feat["oi_runs"] = len(oi_rows)
        feat["oi_top3_rate"] = (
            sum(1 for r in oi_rows if 1 <= r.get("finish_position", 0) <= 3) / len(oi_rows) if oi_rows else 0.0
        )
        feat.update({f"oi_{k}": v for k, v in _course_specific(oi_rows, distance, asof).items()})

        # 南関他場・地方他場
        feat["nankan_other_runs"] = len(buckets["nankan_other"])
        feat["nar_other_runs"] = len(buckets["nar_other"])

        # 血統
        feat["sire"] = horse_info.get("sire", "")
        feat["dam_sire"] = horse_info.get("dam_sire", "")
        feat["dam_dam_sire"] = horse_info.get("dam_dam_sire", "")
    else:
        for k in [
            "all_runs", "all_win_rate", "all_top3_rate", "all_avg_last_3f",
            "recent5_runs", "recent5_win_rate", "recent5_top3_rate", "recent5_avg_last_3f",
            "last_finish", "last_distance", "last_last_3f", "last_win_odds", "days_since_last",
            "jra_runs", "has_jra_experience", "jra_top3_rate", "jra_win_rate",
            "jra_recent2y_runs", "jra_recent2y_top3_rate",
            "oi_runs", "oi_top3_rate", "oi_course_runs", "oi_course_top3_rate", "oi_course_win_rate",
            "nankan_other_runs", "nar_other_runs",
        ]:
            feat[k] = 0
        feat["sire"] = feat["dam_sire"] = feat["dam_dam_sire"] = ""

    # 当日バイアス
    if bias_today:
        feat["bias_inner"] = bias_today.get("bias_inner", 0.0)
        feat["bias_front"] = bias_today.get("bias_front", 0.0)
        feat["bias_sample_size"] = bias_today.get("sample_size", 0)
    else:
        feat["bias_inner"] = 0.0
        feat["bias_front"] = 0.0
        feat["bias_sample_size"] = 0

    # 騎手・調教師（事前計算した直近成績辞書から引く）
    feat["jockey_top3_rate"] = jockey_stats.get("top3_rate", 0.0) if jockey_stats else 0.0
    feat["jockey_runs"] = jockey_stats.get("runs", 0) if jockey_stats else 0
    feat["trainer_top3_rate"] = trainer_stats.get("top3_rate", 0.0) if trainer_stats else 0.0
    feat["trainer_runs"] = trainer_stats.get("runs", 0) if trainer_stats else 0

    return feat


def build_features(
    raw_dir: Path,
    processed_dir: Path,
) -> pd.DataFrame:
    """全レース・全出走の特徴量DataFrameを構築する。"""
    results_dir = raw_dir / "results"
    horses_dir = raw_dir / "horses"
    bias_path = processed_dir / "bias_daily.csv"

    bias_by_date: dict[str, dict] = {}
    if bias_path.exists():
        df_bias = pd.read_csv(bias_path, dtype={"date": str})
        for _, row in df_bias.iterrows():
            bias_by_date[row["date"]] = row.to_dict()

    # 馬個体ロード
    horse_cache: dict[str, dict] = {}
    for jp in horses_dir.glob("*.json"):
        with open(jp, encoding="utf-8") as f:
            horse_cache[jp.stem] = json.load(f)

    # 騎手・調教師の直近成績を集約（簡易版: 全期間）
    jockey_runs: dict[str, list[dict]] = defaultdict(list)
    trainer_runs: dict[str, list[dict]] = defaultdict(list)

    rows: list[dict] = []
    race_files = sorted(results_dir.glob("*.json"))

    # 1パス目: 騎手・調教師成績を蓄積
    for jp in race_files:
        with open(jp, encoding="utf-8") as f:
            race = json.load(f)
        for r in race.get("results", []):
            if r.get("jockey_id"):
                jockey_runs[r["jockey_id"]].append(r)
            if r.get("trainer_id"):
                trainer_runs[r["trainer_id"]].append(r)

    def _stats(rows_: list[dict]) -> dict:
        if not rows_:
            return {"runs": 0, "top3_rate": 0.0}
        valid = [r for r in rows_ if r.get("finish_position", 0) > 0]
        if not valid:
            return {"runs": 0, "top3_rate": 0.0}
        top3 = sum(1 for r in valid if 1 <= r.get("finish_position", 0) <= 3)
        return {"runs": len(valid), "top3_rate": top3 / len(valid)}

    jockey_stats = {jid: _stats(rs) for jid, rs in jockey_runs.items()}
    trainer_stats = {tid: _stats(rs) for tid, rs in trainer_runs.items()}

    # 2パス目: 特徴量行生成
    for jp in race_files:
        with open(jp, encoding="utf-8") as f:
            race = json.load(f)
        # 障害・新馬は除外
        if race.get("is_hurdle") or race.get("is_debut"):
            continue
        bias = bias_by_date.get(race.get("date", ""))
        for r in race.get("results", []):
            hid = r.get("horse_id", "")
            horse_info = horse_cache.get(hid)
            row = _build_runner_row(
                race, r, horse_info, bias,
                jockey_stats.get(r.get("jockey_id", "")),
                trainer_stats.get(r.get("trainer_id", "")),
            )
            rows.append(row)

    df = pd.DataFrame(rows)
    logger.info(f"特徴量: {len(df)}行 / {df['race_id'].nunique() if len(df) else 0}レース")
    return df
