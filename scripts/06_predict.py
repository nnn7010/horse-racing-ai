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
    lookup = {"horse": {}, "jockey": {}, "trainer": {}, "combo": {}}

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

    return lookup


def main():
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    raw_dir = Path(config["paths"]["raw"])
    processed_dir = Path(config["paths"]["processed"])
    output_dir = Path(config["paths"]["outputs"])
    output_dir.mkdir(parents=True, exist_ok=True)

    model, feature_cols, calibrator = load_model(config["paths"]["models"])

    with open(raw_dir / "target_races.json", encoding="utf-8") as f:
        target_races = json.load(f)

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

    # 血統系統分類
    from src.features.pedigree_dict import classify_sire_line
    from src.features.pedigree import encode_sire_lines

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

    # 単勝オッズ取得用
    import hashlib as _hl

    def _get_win_odds_for_race(race_id):
        cache = f"data/cache/{_hl.md5(f'https://race.netkeiba.com/api/api_get_jra_odds.html?race_id={race_id}&type=1&action=update'.encode()).hexdigest()}.html"
        try:
            with open(cache) as f:
                data = json.loads(f.read())
            return {k: float(v[0]) for k, v in data["data"]["odds"]["1"].items()}
        except:
            return {}

    all_predictions = []

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
                for ped_col in ["sire", "dam_sire", "dam_dam_sire"]:
                    sire_name = ped.get(ped_col, "")
                    line = classify_sire_line(sire_name)
                    # 系統one-hot
                    for line_name in ["サンデーサイレンス系", "ノーザンダンサー系", "ミスプロ系", "ネイティヴダンサー系", "ナスルーラ系", "その他"]:
                        prefix = {"sire": "sire_line", "dam_sire": "damsire_line", "dam_dam_sire": "damdamsire_line"}[ped_col]
                        row[f"{prefix}_{line_name}"] = 1 if line == line_name else 0

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
        entry_preds = predict_probabilities(model, feature_cols, entry_df, calibrator=calibrator)
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

    if all_predictions:
        pred_df = pd.DataFrame(all_predictions)
        pred_df.to_csv(output_dir / "predictions.csv", index=False, encoding="utf-8-sig")
        logger.info(f"\nSaved {len(all_predictions)} predictions to {output_dir / 'predictions.csv'}")


if __name__ == "__main__":
    main()
