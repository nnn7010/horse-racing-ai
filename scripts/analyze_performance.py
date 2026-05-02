"""過去レースの遡及予測 + 5/2当日予測の性能分析。

対象: 2026/04/25, 2026/04/26, 2026/05/02
- 4/25-26: result.html からエントリーを取得し遡及予測
- 5/2: today_predictions.json の予測を使用
"""

import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import requests
import yaml
from bs4 import BeautifulSoup

from src.models.predict import load_model, predict_probabilities
from src.utils.logger import get_logger

logger = get_logger("analyze_performance")

HEADERS = {"User-Agent": "Mozilla/5.0", "Referer": "https://race.netkeiba.com/"}
SLEEP = 1.5

SURFACE_MAP = {"芝": "芝", "ダ": "ダート", "障": "障害"}
COND_MAP = {"良": 0, "稍重": 1, "稍": 1, "重": 2, "不良": 3}
CLASS_MAP = {
    "新馬": 1, "未勝利": 2, "1勝": 3, "2勝": 4, "3勝": 5,
    "オープン": 6, "リステッド": 7, "GIII": 8, "GII": 9, "GI": 10,
    "G3": 8, "G2": 9, "G1": 10,
}
COURSE_CODES = {
    "01": "札幌", "02": "函館", "03": "福島", "04": "新潟",
    "05": "東京", "06": "中山", "07": "中京", "08": "京都",
    "09": "阪神", "10": "小倉",
}


# ─────────────────────────────────────────────
# スクレイピング
# ─────────────────────────────────────────────

def fetch_result_page(race_id: str) -> dict | None:
    """result.html からエントリー + 着順 + 払い戻しをパースする。"""
    url = f"https://race.netkeiba.com/race/result.html?race_id={race_id}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.encoding = "euc-jp"
        soup = BeautifulSoup(resp.text, "lxml")
    except Exception as e:
        logger.warning(f"fetch failed {race_id}: {e}")
        return None

    # レース名・コース条件
    race_name = (soup.select_one(".RaceName") or soup.select_one("h1")).get_text(strip=True) if soup.select_one(".RaceName") else ""
    race_data_text = soup.select_one(".RaceData01").get_text(strip=True) if soup.select_one(".RaceData01") else ""
    race_data2 = soup.select_one(".RaceData02")
    race_class_text = race_data2.get_text(strip=True) if race_data2 else ""

    surface = "ダート" if "ダ" in race_data_text else "芝"
    dist_m = re.search(r"(\d{3,4})m", race_data_text)
    distance = int(dist_m.group(1)) if dist_m else 0
    track_cond = 0
    for k, v in COND_MAP.items():
        if k in race_data_text:
            track_cond = v
            break
    race_cls = 2
    for k, v in CLASS_MAP.items():
        if k in race_name or k in race_class_text:
            race_cls = v
            break

    place_code = race_id[4:6]
    place_name = COURSE_CODES.get(place_code, "?")

    # 出走馬テーブル (着順, 枠, 馬番, 馬名, 性齢, 斤量, 騎手, タイム, 着差, 人気, 単勝, 後3F, コーナー, 厩舎, 馬体重)
    table = soup.select_one(".RaceTable01")
    if not table:
        return None

    entries = []
    results = {}
    for tr in table.select("tr")[1:]:
        tds = tr.select("td")
        if len(tds) < 14:
            continue
        try:
            pos_text = tds[0].get_text(strip=True)
            pos = int(pos_text) if pos_text.isdigit() else 0
            bracket = int(tds[1].get_text(strip=True)) if tds[1].get_text(strip=True).isdigit() else 0
            number = int(tds[2].get_text(strip=True)) if tds[2].get_text(strip=True).isdigit() else 0
            horse_name = tds[3].get_text(strip=True)
            impost_text = tds[5].get_text(strip=True)
            impost = float(impost_text) if impost_text else 0.0
            win_odds_text = tds[10].get_text(strip=True).replace(",", "")
            win_odds = float(win_odds_text) if win_odds_text else 0.0
            wt_text = tds[14].get_text(strip=True)  # 馬体重(増減)
            wt_m = re.match(r"(\d+)\(([+-]?\d+)\)", wt_text)
            horse_weight = int(wt_m.group(1)) if wt_m else 0
            weight_change = int(wt_m.group(2)) if wt_m else 0

            horse_link = tds[3].select_one("a[href*='/horse/']")
            hid = re.search(r"/horse/(\w+)", horse_link["href"]).group(1) if horse_link else ""
            jockey_link = tds[6].select_one("a[href*='/jockey/']")
            jid = re.search(r"/jockey/(?:result/recent/)?(\w+)", jockey_link["href"]).group(1) if jockey_link else ""
            trainer_link = tds[13].select_one("a[href*='/trainer/']")
            tid = re.search(r"/trainer/(?:result/recent/)?(\w+)", trainer_link["href"]).group(1) if trainer_link else ""

            entry = {
                "number": number, "bracket": bracket, "horse_name": horse_name,
                "horse_id": hid, "jockey_id": jid, "trainer_id": tid,
                "impost": impost, "win_odds": win_odds,
                "horse_weight": horse_weight, "weight_change": weight_change,
                "finish_position": pos,
            }
            entries.append(entry)
            if 1 <= pos <= 3:
                results[pos] = number
        except Exception:
            continue

    # 払い戻し (単勝)
    win_payout = 0
    for pt in soup.select(".Payout_Detail_Table"):
        for tr in pt.select("tr"):
            tds = tr.select("td")
            if len(tds) >= 2:
                label = tds[0].get_text(strip=True)
                val_text = tds[1].get_text(strip=True).replace(",", "").replace("円", "")
                if "単勝" in label:
                    try:
                        win_payout = int(val_text.split()[0])
                    except Exception:
                        pass

    return {
        "race_id": race_id,
        "date": race_id[4:6],  # 仮、後で上書き
        "place_name": place_name,
        "place_code": place_code,
        "race_name": race_name,
        "surface": surface,
        "distance": distance,
        "track_condition_num": track_cond,
        "class_num": race_cls,
        "entries": entries,
        "results": results,  # {1: num, 2: num, 3: num}
        "win_payout": win_payout,
    }


def fetch_race_list_for_date(date_str: str) -> list[str]:
    """指定日のJRAレースIDリストを取得する（target_races_0425.jsonを使用）。"""
    # target_races_0425.json から該当日のレースIDを取得
    tgt_file = Path("data/raw/target_races_0425.json")
    if tgt_file.exists():
        with open(tgt_file) as f:
            races = json.load(f)
        ids = [r["race_id"] for r in races if r.get("date", "") == date_str]
        if ids:
            return ids

    # フォールバック: netkeiba から取得
    url = f"https://race.netkeiba.com/top/race_list_sub.html?kaisai_date={date_str}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.encoding = "utf-8"
        soup = BeautifulSoup(resp.text, "lxml")
        ids = []
        for a in soup.select("a[href*='race_id=']"):
            m = re.search(r"race_id=(\d{12})", a.get("href", ""))
            if m and m.group(1)[4:6] in COURSE_CODES and m.group(1) not in ids:
                ids.append(m.group(1))
        time.sleep(SLEEP)
        return ids
    except Exception as e:
        logger.warning(f"race_list fetch failed for {date_str}: {e}")
        return []


# ─────────────────────────────────────────────
# 遡及予測生成（06_predict.py のロジックを再現）
# ─────────────────────────────────────────────

def build_stats_lookup(hist_df: pd.DataFrame) -> dict:
    lookup = {"horse": {}, "jockey": {}, "trainer": {}, "combo": {}, "sire": {}, "dam_sire": {}}
    if hist_df.empty:
        return lookup
    hist_df = hist_df.sort_values("date")
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
        recent = g.tail(5)
        for i, (_, row) in enumerate(recent.iloc[::-1].iterrows(), 1):
            stats[f"prev{i}_finish_position"] = row.get("finish_position", 0)
            if "time" in row:
                stats[f"prev{i}_time"] = row["time"]
            if "last_3f" in row:
                stats[f"prev{i}_last_3f"] = row["last_3f"]
        if len(recent) > 0:
            stats["avg_finish_5"] = recent["finish_position"].mean()
        stats["_last_race_date"] = last.get("date")
        stats["_last_race_distance"] = last.get("distance")
        stats["_last_race_horse_weight"] = last.get("horse_weight")
        lookup["horse"][hid] = stats

    for jid, g in hist_df.groupby("jockey_id"):
        last = g.iloc[-1]
        lookup["jockey"][jid] = {
            k: last[k] for k in ["jockey_win_rate", "jockey_top3_rate", "jockey_rides"]
            if k in last.index and pd.notna(last[k])
        }
    for tid, g in hist_df.groupby("trainer_id"):
        last = g.iloc[-1]
        lookup["trainer"][tid] = {
            k: last[k] for k in ["trainer_win_rate", "trainer_top3_rate"]
            if k in last.index and pd.notna(last[k])
        }
    # コンビ
    for (jid, hid), g in hist_df.groupby(["jockey_id", "horse_id"]):
        last = g.iloc[-1]
        combo_stats = {
            k: last[k] for k in ["combo_top3_rate", "combo_rides"]
            if k in last.index and pd.notna(last[k])
        }
        if combo_stats:
            lookup["combo"][(jid, hid)] = combo_stats
    # 血統
    for sire, g in hist_df.groupby("sire_win_rate") if "sire_win_rate" in hist_df.columns else []:
        pass  # sire lookupはpedigree_lookupから別途処理

    return lookup


def predict_race_retroactive(race_data: dict, lookup: dict, pedigree_lookup: dict,
                              ability_lookup: dict, model, feature_cols, calibrator,
                              win_model, win_calibrator) -> pd.DataFrame | None:
    """1レース分の遡及予測を生成する。"""
    entries = race_data.get("entries", [])
    if not entries:
        return None

    race_id = race_data["race_id"]
    date_str = race_data.get("date", "")
    rows = []

    for entry in entries:
        row = {
            "race_id": race_id,
            "number": entry.get("number", 0),
            "horse_name": entry.get("horse_name", ""),
            "horse_id": entry.get("horse_id", ""),
            "bracket": entry.get("bracket", 0),
            "impost": entry.get("impost", 0.0),
            "distance": race_data.get("distance", 0),
            "num_runners": len(entries),
            "is_turf": 1 if race_data.get("surface") == "芝" else 0,
            "place_code_num": int(race_data.get("place_code", "0")),
            "track_condition_num": race_data.get("track_condition_num", 0),
            "horse_weight": entry.get("horse_weight", 0),
            "weight_change": entry.get("weight_change", 0),
            "win_odds": entry.get("win_odds", 0),
            "class_num": race_data.get("class_num", 2),
            "has_history": 0,
        }
        import math
        odds_val = entry.get("win_odds", 0)
        if odds_val > 0:
            row["log_odds"] = math.log(max(odds_val, 1.0))
            row["market_prob"] = 1.0 / max(odds_val, 1.0)

        hid = entry.get("horse_id", "")
        if hid in lookup["horse"]:
            row["has_history"] = 1
            stats = lookup["horse"][hid]
            for k, v in stats.items():
                if k not in row:
                    row[k] = v
            last_date = stats.get("_last_race_date")
            last_dist = stats.get("_last_race_distance")
            last_weight = stats.get("_last_race_horse_weight")
            if date_str and last_date is not None:
                try:
                    row["days_since_last"] = float((pd.Timestamp(date_str) - pd.Timestamp(last_date)).days)
                except Exception:
                    pass
            if last_dist is not None and race_data.get("distance"):
                row["prev_distance"] = float(last_dist)
                row["distance_change"] = float(race_data["distance"] - last_dist)
            if last_weight is not None:
                row["prev_weight"] = float(last_weight)
            for k in ["_last_race_date", "_last_race_distance", "_last_race_horse_weight"]:
                row.pop(k, None)

        jid = entry.get("jockey_id", "")
        if jid in lookup["jockey"]:
            row.update(lookup["jockey"][jid])
        tid = entry.get("trainer_id", "")
        if tid in lookup["trainer"]:
            row.update(lookup["trainer"][tid])
        if (jid, hid) in lookup["combo"]:
            row.update(lookup["combo"][(jid, hid)])
        if hid in ability_lookup:
            row.update(ability_lookup[hid])
        if hid in pedigree_lookup:
            ped = pedigree_lookup[hid]
            row["sire"] = ped.get("sire", "")
            row["dam_sire"] = ped.get("dam_sire", "")

        rows.append(row)

    entry_df = pd.DataFrame(rows)
    if "win_odds" in entry_df.columns:
        entry_df["odds_rank"] = entry_df["win_odds"].rank(method="min").fillna(0)
    for col in feature_cols:
        if col not in entry_df.columns:
            entry_df[col] = 0.0

    return predict_probabilities(model, feature_cols, entry_df,
                                 calibrator=calibrator, win_model=win_model,
                                 win_calibrator=win_calibrator)


# ─────────────────────────────────────────────
# 分析
# ─────────────────────────────────────────────

def analyze_results(all_records: list[dict]) -> None:
    """予測 vs 実際結果の分析レポートを出力する。"""
    df = pd.DataFrame(all_records)
    if df.empty:
        logger.warning("分析データなし")
        return

    logger.info(f"\n{'='*60}")
    logger.info(f"対象: {df['date'].unique()} 合計 {df['race_id'].nunique()} レース, {len(df)} 頭")

    # ── 日付別 rank-1 的中率 ──
    logger.info(f"\n{'─'*50}")
    logger.info("【日付別 Rank-1 的中率（1着予測）】")
    for date, gd in df.groupby("date"):
        n_races = gd["race_id"].nunique()
        hits = gd.sort_values("win_prob", ascending=False).groupby("race_id").first()["is_1st"].sum()
        logger.info(f"  {date}: {hits}/{n_races} = {hits/n_races:.1%}")

    # 全体
    n_total = df["race_id"].nunique()
    total_hits = df.sort_values("win_prob", ascending=False).groupby("race_id").first()["is_1st"].sum()
    logger.info(f"  全体: {total_hits}/{n_total} = {total_hits/n_total:.1%}")

    # ── top-3 precision@3 ──
    logger.info(f"\n{'─'*50}")
    logger.info("【Top-3 precision@3（pred_top3_prob上位3頭の的中率）】")
    for date, gd in df.groupby("date"):
        total_p, total_h = 0, 0
        for _, g in gd.groupby("race_id"):
            pred3 = set(g.nlargest(3, "pred_top3_prob")["number"])
            act3 = set(g[g["is_top3"] == 1]["number"])
            total_h += len(pred3 & act3)
            total_p += 3
        logger.info(f"  {date}: {total_h}/{total_p} = {total_h/total_p:.1%}")

    total_p, total_h = 0, 0
    for _, g in df.groupby("race_id"):
        pred3 = set(g.nlargest(3, "pred_top3_prob")["number"])
        act3 = set(g[g["is_top3"] == 1]["number"])
        total_h += len(pred3 & act3)
        total_p += 3
    logger.info(f"  全体: {total_h}/{total_p} = {total_h/total_p:.1%}")

    # ── win_prob キャリブレーション ──
    logger.info(f"\n{'─'*50}")
    logger.info("【win_prob キャリブレーション】")
    bins = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 1.01]
    labels = ["<5%", "5-10%", "10-15%", "15-20%", "20-30%", "30%+"]
    df["win_band"] = pd.cut(df["win_prob"], bins=bins, labels=labels, right=False)
    calib = df.groupby("win_band", observed=False).agg(
        n=("is_1st", "count"),
        actual_win=("is_1st", "mean"),
        avg_pred=("win_prob", "mean"),
    ).reset_index()
    calib["error"] = (calib["actual_win"] - calib["avg_pred"]).round(4)
    for _, row in calib.iterrows():
        logger.info(f"  {row['win_band']:8s}: n={row['n']:4d} 実際={row['actual_win']:.1%} 予測={row['avg_pred']:.1%} 誤差={row['error']:+.3f}")

    # ── 芝ダート別 rank-1 的中率 ──
    logger.info(f"\n{'─'*50}")
    logger.info("【芝/ダート別 Rank-1 的中率】")
    for surf, gd in df.groupby("surface"):
        n_r = gd["race_id"].nunique()
        h = gd.sort_values("win_prob", ascending=False).groupby("race_id").first()["is_1st"].sum()
        logger.info(f"  {surf}: {h}/{n_r} = {h/n_r:.1%}")

    # ── 予測確率帯別の勝率 ──
    logger.info(f"\n{'─'*50}")
    logger.info("【win_prob 1位馬の勝率（レースの勝ち馬をどれだけ当てているか）】")
    top1_df = df.sort_values("win_prob", ascending=False).groupby("race_id").first().reset_index()
    logger.info(f"  win_prob 1位 → 実際に1着: {top1_df['is_1st'].mean():.1%} ({int(top1_df['is_1st'].sum())}/{len(top1_df)})")
    logger.info(f"  win_prob 1位 → 実際に3着以内: {top1_df['is_top3'].mean():.1%} ({int(top1_df['is_top3'].sum())}/{len(top1_df)})")

    # win_prob 2位馬
    top2_df = df.sort_values("win_prob", ascending=False).groupby("race_id").nth(1).reset_index()
    if not top2_df.empty:
        logger.info(f"  win_prob 2位 → 実際に1着: {top2_df['is_1st'].mean():.1%} ({int(top2_df['is_1st'].sum())}/{len(top2_df)})")
        logger.info(f"  win_prob 2位 → 実際に3着以内: {top2_df['is_top3'].mean():.1%} ({int(top2_df['is_top3'].sum())}/{len(top2_df)})")

    # ── pred_top3_prob 上位3頭の内訳 ──
    logger.info(f"\n{'─'*50}")
    logger.info("【pred_top3_prob 上位3頭の平均的中頭数】")
    hits_by_race = []
    for _, g in df.groupby("race_id"):
        pred3 = set(g.nlargest(3, "pred_top3_prob")["number"])
        act3 = set(g[g["is_top3"] == 1]["number"])
        hits_by_race.append(len(pred3 & act3))
    hit_series = pd.Series(hits_by_race)
    for k in [0, 1, 2, 3]:
        n = (hit_series == k).sum()
        logger.info(f"  {k}頭的中: {n}レース ({n/len(hit_series):.1%})")

    logger.info(f"\n{'='*60}")


# ─────────────────────────────────────────────
# メイン
# ─────────────────────────────────────────────

def main():
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    raw_dir = Path(config["paths"]["raw"])
    processed_dir = Path(config["paths"]["processed"])
    output_dir = Path(config["paths"]["outputs"])
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("モデル読み込み...")
    model, feature_cols, calibrator, win_model, win_calibrator = load_model(config["paths"]["models"])

    logger.info("履歴特徴量読み込み...")
    hist_df = pd.read_parquet(processed_dir / "features.parquet")
    lookup = build_stats_lookup(hist_df)
    logger.info(f"  horse: {len(lookup['horse'])}, jockey: {len(lookup['jockey'])}, trainer: {len(lookup['trainer'])}")

    # 血統・能力データ
    pedigree_lookup = {}
    horses_file = raw_dir / "horses.json"
    if horses_file.exists():
        with open(horses_file) as f:
            for h in json.load(f):
                pedigree_lookup[h["horse_id"]] = {"sire": h.get("sire", ""), "dam_sire": h.get("dam_sire", "")}

    ability_lookup = {}
    ability_file = processed_dir / "ability_features.csv"
    if ability_file.exists():
        ab_df = pd.read_csv(ability_file)
        for _, row in ab_df.iterrows():
            ability_lookup[row["horse_id"]] = {k: row[k] for k in row.index if k.startswith("ability_")}

    all_records = []

    # ── 4/25, 4/26, 5/2: 遡及予測（新モデルで統一）──
    for date_str in ["20260425", "20260426", "20260502"]:
        logger.info(f"\n{'='*40}")
        logger.info(f"=== {date_str} 遡及予測 ===")
        race_ids = fetch_race_list_for_date(date_str)
        logger.info(f"対象レース: {len(race_ids)}件")

        for race_id in race_ids:
            logger.info(f"  fetching {race_id} ...")
            race_data = fetch_result_page(race_id)
            time.sleep(SLEEP)
            if not race_data or not race_data.get("entries"):
                logger.warning(f"  skip {race_id}: no entries")
                continue
            race_data["date"] = date_str

            preds = predict_race_retroactive(
                race_data, lookup, pedigree_lookup, ability_lookup,
                model, feature_cols, calibrator, win_model, win_calibrator,
            )
            if preds is None:
                continue

            results = race_data.get("results", {})
            actual_1st = results.get(1, -1)
            actual_top3 = set(results.values())

            for _, row in preds.iterrows():
                num = int(row.get("number", 0))
                all_records.append({
                    "date": date_str,
                    "race_id": race_id,
                    "place_name": race_data.get("place_name", ""),
                    "race_name": race_data.get("race_name", ""),
                    "surface": race_data.get("surface", ""),
                    "distance": race_data.get("distance", 0),
                    "number": num,
                    "horse_name": row.get("horse_name", ""),
                    "win_prob": float(row.get("pred_win_prob", row.get("win_prob", 0))),
                    "pred_top3_prob": float(row.get("pred_top3_prob", 0)),
                    "is_1st": 1 if num == actual_1st else 0,
                    "is_top3": 1 if num in actual_top3 else 0,
                    "finish_position": next((p for p, n in results.items() if n == num), 99),
                    "source": "retroactive",
                })

    # 5/2 は上のループで遡及予測済みのため、today_predictions.json は使用しない

    logger.info(f"\n合計レコード: {len(all_records)}")
    if not all_records:
        logger.error("分析データが空です")
        return

    # CSV保存
    result_df = pd.DataFrame(all_records)
    result_df.to_csv(output_dir / "performance_analysis.csv", index=False, encoding="utf-8-sig")
    logger.info("outputs/performance_analysis.csv に保存しました")

    # 分析レポート
    analyze_results(all_records)


if __name__ == "__main__":
    main()
