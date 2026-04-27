"""06: 対象レースの予測・馬券抽出。

過去の特徴量データから各馬・騎手・調教師の最新統計を取得し、
学習済みモデルで予測する。
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import yaml

from src.models.predict import load_model, predict_probabilities
from src.probability.plackett_luce import compute_race_probabilities
from src.utils.logger import get_logger

logger = get_logger("06_predict")


def build_stats_lookup(hist_df: pd.DataFrame) -> dict:
    """過去データから各エンティティの最新統計を辞書化する。"""
    lookup = {"horse": {}, "jockey": {}, "trainer": {}, "combo": {}, "sire": {}, "dam_sire": {}}

    if hist_df.empty:
        return lookup

    hist_df = hist_df.sort_values("date")

    # 馬ごとの最新統計（全特徴量列の最終行）
    for hid, g in hist_df.groupby("horse_id"):
        last = g.iloc[-1]
        stats = {}
        for col in g.columns:
            if col in ["race_id", "horse_id", "date", "horse_name", "jockey_name",
                        "trainer_name", "jockey_id", "trainer_id", "finish_position", "number"]:
                continue
            val = last.get(col)
            if pd.notna(val):
                stats[col] = val
        # 近5走着順を明示的にセット
        recent = g.tail(5)
        for i, (_, row) in enumerate(recent.iloc[::-1].iterrows(), 1):
            stats[f"prev{i}_finish_position"] = row.get("finish_position", 0)
            if "time" in row:
                stats[f"prev{i}_time"] = row["time"]
            if "last_3f" in row:
                stats[f"prev{i}_last_3f"] = row["last_3f"]
        if len(recent) > 0:
            stats["avg_finish_5"] = recent["finish_position"].mean()
        # 最後の歴史レースの日付・距離・馬体重を保存（今日基準で再計算用）
        stats["_last_race_date"] = last.get("date")
        stats["_last_race_distance"] = last.get("distance")
        stats["_last_race_horse_weight"] = last.get("horse_weight")
        lookup["horse"][hid] = stats

    # 騎手ごとの最新統計
    if "jockey_id" in hist_df.columns:
        for jid, g in hist_df.groupby("jockey_id"):
            last = g.iloc[-1]
            lookup["jockey"][jid] = {
                "jockey_win_rate": last.get("jockey_win_rate", 0.0) if pd.notna(last.get("jockey_win_rate")) else 0.0,
                "jockey_top3_rate": last.get("jockey_top3_rate", 0.0) if pd.notna(last.get("jockey_top3_rate")) else 0.0,
                "jockey_rides": last.get("jockey_rides", 0) if pd.notna(last.get("jockey_rides")) else 0,
            }

    # 調教師ごとの最新統計
    if "trainer_id" in hist_df.columns:
        for tid, g in hist_df.groupby("trainer_id"):
            last = g.iloc[-1]
            lookup["trainer"][tid] = {
                "trainer_win_rate": last.get("trainer_win_rate", 0.0) if pd.notna(last.get("trainer_win_rate")) else 0.0,
                "trainer_top3_rate": last.get("trainer_top3_rate", 0.0) if pd.notna(last.get("trainer_top3_rate")) else 0.0,
            }

    # コンビ統計
    if "jockey_id" in hist_df.columns:
        for (jid, hid), g in hist_df.groupby(["jockey_id", "horse_id"]):
            last = g.iloc[-1]
            lookup["combo"][(jid, hid)] = {
                "combo_top3_rate": last.get("combo_top3_rate", 0.0) if pd.notna(last.get("combo_top3_rate")) else 0.0,
                "combo_rides": last.get("combo_rides", 0) if pd.notna(last.get("combo_rides")) else 0,
            }

    # 父・母父の統計（features.parquetに計算済みの場合）
    for sire_col, key in [("sire", "sire"), ("dam_sire", "dam_sire")]:
        win_col = f"{sire_col}_win_rate" if sire_col == "sire" else "dam_sire_win_rate"
        t3_col = f"{sire_col}_top3_rate" if sire_col == "sire" else "dam_sire_top3_rate"
        course_col = "sire_course_top3_rate" if sire_col == "sire" else "damsire_course_top3_rate"
        if sire_col in hist_df.columns and win_col in hist_df.columns:
            for sname, g in hist_df[hist_df[sire_col].notna()].groupby(sire_col):
                last = g.iloc[-1]
                lookup[key][sname] = {
                    win_col: float(last.get(win_col, 0.08)) if pd.notna(last.get(win_col)) else 0.08,
                    t3_col: float(last.get(t3_col, 0.21)) if pd.notna(last.get(t3_col)) else 0.21,
                }

    return lookup


def main():
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    raw_dir = Path(config["paths"]["raw"])
    processed_dir = Path(config["paths"]["processed"])
    output_dir = Path(config["paths"]["outputs"])
    output_dir.mkdir(parents=True, exist_ok=True)

    model, feature_cols, calibrator, win_model, win_calibrator = load_model(config["paths"]["models"])

    with open(raw_dir / "target_races.json", encoding="utf-8") as f:
        target_races = json.load(f)

    # 対象日のレースに絞る
    import datetime
    target_date_strs = set()
    for r in target_races:
        d = r.get("date", "")
        if d:
            target_date_strs.add(d)
    # 日付指定がない場合は今日のみ
    if not target_date_strs:
        today_str = datetime.date.today().strftime("%Y%m%d")
        target_races = [r for r in target_races if r.get("date", "") == today_str]
    logger.info(f"Target dates: {sorted(target_date_strs)}, {len(target_races)} races")

    # 過去特徴量から統計を構築
    hist_df = pd.read_parquet(processed_dir / "features.parquet")
    logger.info(f"Loaded historical features: {hist_df.shape}")
    lookup = build_stats_lookup(hist_df)
    logger.info(f"Stats lookup: {len(lookup['horse'])} horses, {len(lookup['jockey'])} jockeys, {len(lookup['trainer'])} trainers")

    # 血統データ読み込み
    horses_file = raw_dir / "horses.json"
    pedigree_lookup = {}
    if horses_file.exists():
        with open(horses_file, encoding="utf-8") as f:
            for h in json.load(f):
                pedigree_lookup[h["horse_id"]] = {
                    "sire": h.get("sire", ""),
                    "dam_sire": h.get("dam_sire", ""),
                    "dam_dam_sire": h.get("dam_dam_sire", ""),
                }
        logger.info(f"Pedigree lookup: {len(pedigree_lookup)} horses")

    # 能力パラメータ読み込み
    ability_file = processed_dir / "ability_features.csv"
    ability_lookup = {}
    if ability_file.exists():
        import pandas as _pd
        ability_df = _pd.read_csv(ability_file)
        for _, row in ability_df.iterrows():
            ability_lookup[row["horse_id"]] = {
                k: row[k] for k in row.index if k.startswith("ability_")
            }
        logger.info(f"Ability lookup: {len(ability_lookup)} horses")

    # コースプロファイル読み込み
    course_profiles = {}
    course_profiles_file = processed_dir / "course_profiles.json"
    if course_profiles_file.exists():
        with open(course_profiles_file, encoding="utf-8") as f:
            course_profiles = json.load(f)
        logger.info(f"Course profiles: {len(course_profiles)} courses loaded")

    # 単勝オッズ取得用
    import hashlib as _hl
    import time as _time
    import requests as _requests

    _odds_headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://race.netkeiba.com/",
    }

    def _get_win_odds_for_race(race_id):
        url = f"https://race.netkeiba.com/api/api_get_jra_odds.html?race_id={race_id}&type=1&action=update"
        cache_path = Path(f"data/cache/{_hl.md5(url.encode()).hexdigest()}.html")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            # キャッシュがあれば使用（1時間以内のもの）
            if cache_path.exists() and (_time.time() - cache_path.stat().st_mtime) < 3600:
                with open(cache_path) as f:
                    data = json.loads(f.read())
            else:
                resp = _requests.get(url, headers=_odds_headers, timeout=10)
                resp.raise_for_status()
                data = resp.json()
                with open(cache_path, "w") as f:
                    json.dump(data, f)
                _time.sleep(1.5)
            return {k: float(v[0]) for k, v in data["data"]["odds"]["1"].items()}
        except Exception as e:
            logger.warning(f"Failed to fetch odds for {race_id}: {e}")
            return {}

    all_predictions = []
    today_races = []

    for race in target_races:
        race_id = race["race_id"]
        entries = race.get("entries", [])
        if not entries:
            continue

        # オッズ取得
        race_win_odds = _get_win_odds_for_race(race_id)

        rows = []
        for entry in entries:
            row = {
                "race_id": race_id,
                "number": entry.get("number", 0),
                "horse_name": entry.get("horse_name", ""),
                "horse_id": entry.get("horse_id", ""),
                "bracket": entry.get("bracket", 0),
                "impost": entry.get("weight", 0.0),
                "distance": race.get("distance", 0),
                "num_runners": len(entries),
                "is_turf": 1 if race.get("surface") == "芝" else 0,
                "place_code_num": int(race.get("place_code", "0")),
                "track_condition_num": 0,
                "horse_weight": 0,
                "weight_change": 0,
                "has_history": 0,
            }

            # オッズ特徴量
            import math
            num_str = str(entry.get("number", 0)).zfill(2)
            odds_val = race_win_odds.get(num_str, 0)
            if odds_val > 0:
                row["win_odds"] = odds_val
                row["log_odds"] = math.log(max(odds_val, 1.0))
                row["market_prob"] = 1.0 / max(odds_val, 1.0)
            else:
                row["win_odds"] = 0
                row["log_odds"] = 0
                row["market_prob"] = 0

            # クラスレベル
            class_map = {
                "新馬": 1, "未勝利": 2, "1勝クラス": 3, "2勝クラス": 4,
                "3勝クラス": 5, "オープン": 6, "リステッド": 7,
                "GIII": 8, "GII": 9, "GI": 10, "G3": 8, "G2": 9, "G1": 10,
            }
            race_class = race.get("class", "")
            for key, val in class_map.items():
                if key in race_class:
                    row["class_num"] = val
                    break
            else:
                row["class_num"] = 3

            # 馬の過去統計をマージ
            hid = entry.get("horse_id", "")
            if hid in lookup["horse"]:
                row["has_history"] = 1
                for k, v in lookup["horse"][hid].items():
                    if k not in row:  # race条件系は上書きしない
                        row[k] = v

                # === 今日のレース基準で前走関連を再計算 ===
                stats = lookup["horse"][hid]
                last_date = stats.get("_last_race_date")
                last_dist = stats.get("_last_race_distance")
                last_weight = stats.get("_last_race_horse_weight")

                # 今日のレース日付
                race_date_str = race.get("date", "")
                if race_date_str and last_date is not None:
                    try:
                        race_ts = pd.Timestamp(race_date_str)
                        last_ts = pd.Timestamp(last_date)
                        row["days_since_last"] = float((race_ts - last_ts).days)
                    except Exception:
                        pass
                # 距離変化: 今日 - 前走
                today_dist = race.get("distance", 0)
                if last_dist is not None and today_dist:
                    row["prev_distance"] = float(last_dist)
                    row["distance_change"] = float(today_dist - last_dist)
                # 前走馬体重
                if last_weight is not None:
                    row["prev_weight"] = float(last_weight)

                # 内部マーカー削除（モデル特徴量に流れないように）
                for k in ["_last_race_date", "_last_race_distance", "_last_race_horse_weight"]:
                    row.pop(k, None)

            # 騎手統計をマージ
            jid = entry.get("jockey_id", "")
            if jid in lookup["jockey"]:
                row.update(lookup["jockey"][jid])

            # 調教師統計をマージ
            tid = entry.get("trainer_id", "")
            if tid in lookup["trainer"]:
                row.update(lookup["trainer"][tid])

            # コンビ統計
            if (jid, hid) in lookup["combo"]:
                row.update(lookup["combo"][(jid, hid)])

            # 能力パラメータをマージ
            if hid in ability_lookup:
                row.update(ability_lookup[hid])

            # 血統情報をマージ
            if hid in pedigree_lookup:
                ped = pedigree_lookup[hid]
                sire_name = ped.get("sire", "")
                dam_sire_name = ped.get("dam_sire", "")
                row["sire"] = sire_name
                row["dam_sire"] = dam_sire_name
                # 父の統計
                if sire_name in lookup["sire"]:
                    row.update(lookup["sire"][sire_name])
                # 母父の統計
                if dam_sire_name in lookup["dam_sire"]:
                    row.update(lookup["dam_sire"][dam_sire_name])

            rows.append(row)

        entry_df = pd.DataFrame(rows)

        # odds_rank（レース内の人気順位）
        if "win_odds" in entry_df.columns:
            entry_df["odds_rank"] = entry_df["win_odds"].rank(method="min")
            entry_df["odds_rank"] = entry_df["odds_rank"].fillna(0)

        # 不足列を0で補完
        for col in feature_cols:
            if col not in entry_df.columns:
                entry_df[col] = 0.0

        # 予測
        entry_preds = predict_probabilities(
            model, feature_cols, entry_df,
            calibrator=calibrator,
            win_model=win_model,
            win_calibrator=win_calibrator,
        )
        probs = compute_race_probabilities(entry_preds)

        # 結果出力
        logger.info(f"\n{race.get('place_name', '')} {race.get('race_name', '')} ({race.get('surface', '')}{race.get('distance', '')}m)")
        win_sorted = sorted(probs["win"].items(), key=lambda x: x[1], reverse=True)
        for num, prob in win_sorted[:5]:
            name = entry_preds[entry_preds["number"] == num]["horse_name"].values
            name = name[0] if len(name) > 0 else "?"
            logger.info(f"  {num:>2}. {name}: {prob:.1%}")

        for _, row in entry_preds.iterrows():
            all_predictions.append({
                "race_id": race_id,
                "date": race.get("date", ""),
                "place_name": race.get("place_name", ""),
                "race_name": race.get("race_name", ""),
                "surface": race.get("surface", ""),
                "distance": race.get("distance", 0),
                "number": int(row.get("number", 0)),
                "horse_name": row.get("horse_name", ""),
                "pred_top3_prob": float(row.get("pred_top3_prob", 0)),
                "win_prob": float(probs["win"].get(row.get("number", 0), 0)),
            })

        # today_predictions.json 用データ収集
        trio_sorted = sorted(probs["trio"].items(), key=lambda x: x[1], reverse=True)[:5]
        trifecta_sorted = sorted(probs["trifecta"].items(), key=lambda x: x[1], reverse=True)[:5]

        # 能力スコア計算（レース内0-100正規化）
        def minmax(series, invert=False):
            mn, mx = series.min(), series.max()
            if mx == mn:
                return pd.Series([50.0] * len(series), index=series.index)
            s = (series - mn) / (mx - mn) * 100
            return 100 - s if invert else s

        n_h = len(entry_preds)
        # スピード: avg_speed_3（低いほど速い → invert）
        spd_col = "avg_speed_3" if "avg_speed_3" in entry_preds.columns else "speed_index"
        if spd_col in entry_preds.columns:
            spd_med = entry_preds[spd_col].replace(0, float("nan")).median()
            speed_score = minmax(entry_preds[spd_col].fillna(spd_med if pd.notna(spd_med) else 61.0), invert=True)
        else:
            speed_score = pd.Series([50.0]*n_h, index=entry_preds.index)

        # コース適性: horse_course_top3_rate / max(horse_top3_rate, 0.01)
        if "horse_course_top3_rate" in entry_preds.columns and "horse_top3_rate" in entry_preds.columns:
            course_ratio = entry_preds["horse_course_top3_rate"] / entry_preds["horse_top3_rate"].clip(lower=0.01)
            course_score = minmax(course_ratio.fillna(1.0))
        else:
            course_score = pd.Series([50.0]*n_h, index=entry_preds.index)

        # 近走勢い: 小さい着順ほど良い → invert
        form_score = minmax(entry_preds["avg_finish_5"].fillna(entry_preds["avg_finish_5"].median()), invert=True) if "avg_finish_5" in entry_preds.columns else pd.Series([50.0]*n_h, index=entry_preds.index)

        # 安定性: finish_std_5 が小さいほど安定 → invert
        stab_score = minmax(entry_preds["finish_std_5"].fillna(entry_preds["finish_std_5"].median()), invert=True) if "finish_std_5" in entry_preds.columns else pd.Series([50.0]*n_h, index=entry_preds.index)

        # パワー: タフコース実績(50%) + 消耗戦での好走歴(50%)
        has_tough = "horse_tough_top3_rate" in entry_preds.columns
        has_slow = "horse_slow_race_top3_rate" in entry_preds.columns
        if has_tough and has_slow:
            tough_s = minmax(entry_preds["horse_tough_top3_rate"].fillna(0.21))
            slow_s = minmax(entry_preds["horse_slow_race_top3_rate"].fillna(0.21))
            power_score = 0.5 * tough_s + 0.5 * slow_s
        elif has_tough:
            power_score = minmax(entry_preds["horse_tough_top3_rate"].fillna(0.21))
        elif has_slow:
            power_score = minmax(entry_preds["horse_slow_race_top3_rate"].fillna(0.21))
        else:
            power_score = pd.Series([50.0]*n_h, index=entry_preds.index)

        # 瞬発力: 近3走の上がり3F順位平均（小さいほど良い → invert）
        if "avg_last_3f_rank_3" in entry_preds.columns:
            burst_score = minmax(entry_preds["avg_last_3f_rank_3"].fillna(entry_preds["avg_last_3f_rank_3"].median()), invert=True)
        elif "avg_last_3f_5" in entry_preds.columns:
            burst_score = minmax(entry_preds["avg_last_3f_5"].fillna(entry_preds["avg_last_3f_5"].median()), invert=True)
        else:
            burst_score = pd.Series([50.0]*n_h, index=entry_preds.index)

        # 騎手相性: combo_top3_rate > 0なら使用、なければjockey_top3_rate
        if "combo_top3_rate" in entry_preds.columns:
            jockey_base = entry_preds["combo_top3_rate"].where(entry_preds.get("combo_rides", pd.Series(0, index=entry_preds.index)) >= 3, entry_preds.get("jockey_top3_rate", pd.Series(0.21, index=entry_preds.index)))
        else:
            jockey_base = entry_preds.get("jockey_top3_rate", pd.Series(0.21, index=entry_preds.index))
        jockey_score = minmax(jockey_base.fillna(0.21))

        # コーナー通過順位由来の脚質シグナル（過去5走平均）
        early_pos_series = entry_preds.get("avg_early_pos_ratio_5", pd.Series(np.nan, index=entry_preds.index))
        lead_rate_series = entry_preds.get("early_lead_rate", pd.Series(np.nan, index=entry_preds.index))
        closer_rate_series = entry_preds.get("closer_rate", pd.Series(np.nan, index=entry_preds.index))

        ability_scores_df = pd.DataFrame({
            "number": entry_preds["number"].astype(int),
            "speed": speed_score.round(0).astype(int),
            "burst": burst_score.round(0).astype(int),
            "power": power_score.round(0).astype(int),
            "course": course_score.round(0).astype(int),
            "form": form_score.round(0).astype(int),
            "stability": stab_score.round(0).astype(int),
            "jockey": jockey_score.round(0).astype(int),
            "early_pos_ratio": early_pos_series.values,
            "early_lead_rate": lead_rate_series.values,
            "closer_rate": closer_rate_series.values,
            "has_history": entry_preds.get("has_history", pd.Series(1, index=entry_preds.index)).astype(int),
        }).set_index("number")

        # コースプロファイル取得
        place_code_num = int(race.get("place_code", "0"))
        is_turf = 1 if race.get("surface") == "芝" else 0
        distance = int(race.get("distance", 0))
        course_key = f"{place_code_num}_{is_turf}_{distance}"
        course_profile = course_profiles.get(course_key, {}).get("scores", {})

        def estimate_running_style(
            early_pos_ratio: float,
            early_lead_rate: float,
            closer_rate: float,
            burst: int,
            speed: int,
            has_data: bool,
        ) -> str:
            """脚質推定。

            一次優先: コーナー通過順位ベース（過去5走の序盤位置 / 頭数）。
            データ不足時は burst/speed ヒューリスティックにフォールバック。
            """
            if not has_data:
                return ""

            # コーナー通過データがある場合
            if pd.notna(early_pos_ratio):
                lead = early_lead_rate if pd.notna(early_lead_rate) else 0.0
                close = closer_rate if pd.notna(closer_rate) else 0.0
                if lead >= 0.6 or early_pos_ratio < 0.18:
                    return "逃げ"
                if early_pos_ratio < 0.40:
                    return "先行"
                if close >= 0.5 or early_pos_ratio > 0.70:
                    return "追込"
                return "差し"

            # フォールバック: 旧ヒューリスティック
            if burst >= 70 and speed <= 40:
                return "追込"
            elif burst >= 60 and speed <= 55:
                return "差し"
            elif burst <= 35 and speed >= 60:
                return "逃げ"
            elif burst <= 45 and speed >= 50:
                return "先行"
            elif burst >= 55:
                return "差し"
            else:
                return "先行"

        def generate_comment(row, ab: dict, win_prob: float, place_prob: float, style: str) -> str:
            """各馬の特徴を短文コメントとして生成する。"""
            parts = []

            # 初出走の場合は「初出走」のみ表示してreturn
            if not ab.get("has_data", False):
                return "初出走（過去データなし）"

            # 近走成績
            avg_f = float(row.get("avg_finish_5", 0) or 0)
            if avg_f > 0:
                if avg_f <= 2.5:
                    parts.append("近走絶好調")
                elif avg_f <= 4.0:
                    parts.append("近走安定")
                elif avg_f >= 7.0:
                    parts.append("近走不振")

            # コース適性
            course_rate = float(row.get("horse_course_top3_rate", 0) or 0)
            course_runs_raw = row.get("horse_course_runs", 0)
            course_runs = int(course_runs_raw) if pd.notna(course_runs_raw) else 0
            if course_runs >= 3:
                if course_rate >= 0.55:
                    parts.append(f"このコース得意({int(course_rate*100)}%)")
                elif course_rate <= 0.15:
                    parts.append("コース実績薄")

            # 前走間隔
            days = float(row.get("days_since_last", 0) or 0)
            if days >= 90:
                parts.append("休み明け")
            elif days <= 14:
                parts.append("中1週")

            # 距離変化
            dist_chg = float(row.get("distance_change", 0) or 0)
            if dist_chg >= 200:
                parts.append("距離延長")
            elif dist_chg <= -200:
                parts.append("距離短縮")

            # 騎手
            jockey_rate = float(row.get("jockey_top3_rate", 0) or 0)
            combo_rides_raw = row.get("combo_rides", 0)
            combo_rides = int(combo_rides_raw) if pd.notna(combo_rides_raw) else 0
            combo_rate = float(row.get("combo_top3_rate", 0) or 0)
            if combo_rides >= 5 and combo_rate >= 0.5:
                parts.append("コンビ好相性")
            elif jockey_rate >= 0.40:
                parts.append("騎手上位")
            elif jockey_rate <= 0.15:
                parts.append("騎手苦戦中")

            # 脚質コメント
            style_map = {
                "逃げ": "ハナを切るタイプ",
                "先行": "好位追走タイプ",
                "差し": "中団から差すタイプ",
                "追込": "後方から追い込むタイプ",
            }
            parts.append(style_map.get(style, ""))

            # 期待値
            if win_prob >= 0.25:
                parts.append("軸候補")
            elif win_prob >= 0.15:
                parts.append("対抗")

            return "。".join(p for p in parts if p)

        horses_json = []
        style_counts = {"逃げ": 0, "先行": 0, "差し": 0, "追込": 0}

        for _, row in entry_preds.sort_values("number").iterrows():
            num = int(row.get("number", 0))
            num_str = str(num).zfill(2)
            odds_val = race_win_odds.get(num_str, 0)
            bracket = next(
                (e.get("bracket", 0) for e in entries if e.get("number") == num), 0
            )
            ab_match = ability_scores_df.loc[ability_scores_df.index == num]
            ab = ab_match.iloc[0] if len(ab_match) > 0 else None
            ab_dict = {
                "speed": int(ab["speed"]) if ab is not None else 50,
                "burst": int(ab["burst"]) if ab is not None else 50,
                "power": int(ab["power"]) if ab is not None else 50,
                "course": int(ab["course"]) if ab is not None else 50,
                "form": int(ab["form"]) if ab is not None else 50,
                "stability": int(ab["stability"]) if ab is not None else 50,
                "jockey": int(ab["jockey"]) if ab is not None else 50,
                "has_data": bool(ab["has_history"]) if ab is not None else False,
            }
            # コーナー位置（脚質判定用、JSONには出さず内部のみ）
            _early_pos = float(ab["early_pos_ratio"]) if ab is not None and pd.notna(ab.get("early_pos_ratio", np.nan)) else float("nan")
            _lead_rate = float(ab["early_lead_rate"]) if ab is not None and pd.notna(ab.get("early_lead_rate", np.nan)) else float("nan")
            _closer_rate = float(ab["closer_rate"]) if ab is not None and pd.notna(ab.get("closer_rate", np.nan)) else float("nan")
            # コース適性スコア
            if course_profile:
                keys = ["speed", "burst", "power", "course", "form", "stability", "jockey"]
                shortfalls = [max(0, course_profile.get(k, 50) - ab_dict[k]) for k in keys]
                fit_score = int(round(100 - sum(shortfalls) / len(keys)))
                fit_score = max(0, min(100, fit_score))
            else:
                fit_score = 50
            ab_dict["fit"] = fit_score

            # 脚質推定（コーナー通過順位優先、データ無は burst/speed フォールバック）
            style = estimate_running_style(
                _early_pos, _lead_rate, _closer_rate,
                ab_dict["burst"], ab_dict["speed"], ab_dict.get("has_data", False),
            )
            style_counts[style] = style_counts.get(style, 0) + 1

            # コメント生成
            wp = float(probs["win"].get(num, 0))
            pp = float(probs["place"].get(num, 0))
            comment = generate_comment(row, ab_dict, wp, pp, style)

            horses_json.append({
                "number": num,
                "bracket": int(bracket),
                "horse_name": row.get("horse_name", ""),
                "jockey_name": row.get("jockey_name", ""),
                "impost": float(row.get("impost", 0)),
                "win_odds": float(odds_val),
                "win_prob": round(wp, 6),
                "place_prob": round(pp, 6),
                "running_style": style,
                "comment": comment,
                "ability": ab_dict,
            })

        # 展開予測（コーナー通過順位ベースの脚質を集計）
        front_count = style_counts.get("逃げ", 0) + style_counts.get("先行", 0)
        n_total = len(horses_json)
        front_ratio = front_count / max(n_total, 1)
        # 「真の逃げ馬」: early_lead_rate >= 0.5（過去5走の半数以上で序盤2位以内）
        true_leaders = int((ability_scores_df["early_lead_rate"].fillna(0) >= 0.5).sum())
        if style_counts.get("逃げ", 0) >= 3 or front_ratio >= 0.5 or true_leaders >= 3:
            pace = "ハイペース"
            pace_note = "先行馬多数 → 差し・追込有利"
        elif true_leaders == 0 and style_counts.get("逃げ", 0) <= 1:
            pace = "スローペース"
            pace_note = "逃げ馬少ない → 先行有利"
        elif front_ratio <= 0.25:
            pace = "スローペース"
            pace_note = "前に行く馬が少ない → 先行有利"
        else:
            pace = "ミドルペース"
            pace_note = "平均的なペース想定"
        pace_prediction = {
            "pace": pace,
            "note": pace_note,
            "front": front_count,
            "true_leaders": true_leaders,
            "closers": style_counts.get("差し", 0) + style_counts.get("追込", 0),
            "total": n_total,
        }

        today_races.append({
            "race_id": race_id,
            "place_name": race.get("place_name", ""),
            "race_name": race.get("race_name", ""),
            "surface": race.get("surface", ""),
            "distance": int(race.get("distance", 0)),
            "track_condition": race.get("track_condition", ""),
            "start_time": race.get("start_time", ""),
            "n_horses": len(entries),
            "course_profile": course_profile or {},
            "pace_prediction": pace_prediction,
            "horses": horses_json,
            "trio_top5": [
                {"combo": [int(x) for x in sorted(k)], "prob": round(float(v), 6)}
                for k, v in trio_sorted
            ],
            "trifecta_top5": [
                {"combo": [int(x) for x in k], "prob": round(float(v), 6)}
                for k, v in trifecta_sorted
            ],
        })

    if all_predictions:
        pred_df = pd.DataFrame(all_predictions)
        pred_df.to_csv(output_dir / "predictions.csv", index=False, encoding="utf-8-sig")
        logger.info(f"\nSaved {len(all_predictions)} predictions to {output_dir / 'predictions.csv'}")

    if today_races:
        import datetime
        date_str = today_races[0]["race_id"][0:4] + "/" + today_races[0]["race_id"][4:6] + "/" + today_races[0]["race_id"][6:8] if today_races else ""
        today_json = {"date": date_str, "races": today_races}
        json_path = raw_dir / "today_predictions.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(today_json, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved today_predictions.json ({len(today_races)} races)")


if __name__ == "__main__":
    main()
