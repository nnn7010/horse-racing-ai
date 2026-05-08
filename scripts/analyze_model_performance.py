"""予測モデルの予想成績分析スクリプト。

today_predictions.json と today_results.json を突合し、以下を分析する:
- Rank-1 的中率 (win_prob 最高馬が1着に来る割合)
- Top-3 precision@3 (place_prob 上位3頭中の実際の3着以内馬数)
- win_prob キャリブレーション
- 人気順との比較
- 馬券EV分析 (単勝・複勝)
- 日付別 / 競馬場別 / 芝ダート別の内訳
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("analyze_model_performance")

PLACE_CODES = {
    "01": "札幌", "02": "函館", "03": "福島", "04": "新潟",
    "05": "東京", "06": "中山", "07": "中京", "08": "京都",
    "09": "阪神", "10": "小倉",
}


def _build_date_map(target_path: Path) -> dict[str, str]:
    """target_races.json から race_id → 実カレンダー日付 のマップを返す。"""
    date_map: dict[str, str] = {}
    if target_path.exists():
        with open(target_path, encoding="utf-8") as f:
            for r in json.load(f):
                if r.get("race_id") and r.get("date"):
                    date_map[r["race_id"]] = str(r["date"])
    return date_map


def load_data(pred_path: str, result_path: str,
              extra_pred_json: str | None = None,
              extra_date_map: dict[str, str] | None = None) -> pd.DataFrame:
    """予測と結果を突合してDataFrameを返す。

    extra_pred_json: 追加の予測JSONテキスト（例: git履歴から取得した過去分）
    extra_date_map: 追加予測分の race_id → 実日付 マップ
    """
    pred_path = Path(pred_path)
    with open(pred_path, encoding="utf-8") as f:
        pred_data = json.load(f)
    with open(result_path, encoding="utf-8") as f:
        result_data = json.load(f)

    # race_id の先頭8文字は YYYYVVHH (場コード+開催番号) であり日付ではない
    # target_races.json から実カレンダー日付を取得する
    date_map = _build_date_map(pred_path.parent / "target_races.json")
    if extra_date_map:
        date_map.update(extra_date_map)
    fallback_date = pred_data.get("date", "").replace("/", "")

    # 通常の予測データ + 追加予測データ をまとめる
    all_races = list(pred_data["races"])
    if extra_pred_json:
        extra_data = json.loads(extra_pred_json)
        all_races.extend(extra_data.get("races", []))
        if not fallback_date:
            fallback_date = extra_data.get("date", "").replace("/", "")

    rows = []
    skipped = 0
    seen_race_ids: set[str] = set()
    for race in all_races:
        rid = race["race_id"]
        if rid in seen_race_ids:
            continue
        seen_race_ids.add(rid)
        if rid not in result_data:
            skipped += 1
            continue
        res = result_data[rid]
        if res.get("status") != "confirmed" or not res.get("1st"):
            skipped += 1
            continue

        actual_1st = res["1st"]
        actual_2nd = res.get("2nd", -1)
        actual_3rd = res.get("3rd", -1)
        actual_top3 = {actual_1st, actual_2nd, actual_3rd} - {-1, 0, None}

        date_str = date_map.get(rid, fallback_date)
        place_code = rid[4:6]
        place_name = PLACE_CODES.get(place_code, "?")

        for h in race["horses"]:
            num = h["number"]
            win_odds = h.get("win_odds", 0) or 0
            win_prob = h.get("win_prob", 0) or 0
            place_prob = h.get("place_prob", 0) or 0
            market_prob = 1.0 / win_odds if win_odds > 0 else 0

            rows.append({
                "date": date_str,
                "race_id": rid,
                "place_name": place_name,
                "race_name": race.get("race_name", ""),
                "surface": race.get("surface", ""),
                "distance": race.get("distance", 0),
                "n_horses": race.get("n_horses", 0),
                "number": num,
                "horse_name": h.get("horse_name", ""),
                "win_odds": win_odds,
                "win_prob": win_prob,
                "place_prob": place_prob,
                "market_prob": market_prob,
                "is_1st": 1 if num == actual_1st else 0,
                "is_top3": 1 if num in actual_top3 else 0,
                "finish_position": (
                    1 if num == actual_1st else
                    2 if num == actual_2nd else
                    3 if num == actual_3rd else 99
                ),
                "payout_win": res.get("payouts", {}).get("win", 0) or 0,
            })

    if skipped:
        logger.info(f"スキップしたレース: {skipped}件（結果未確定 or 未マッチ）")

    return pd.DataFrame(rows)


def section(title: str):
    logger.info(f"\n{'─'*55}")
    logger.info(f"  {title}")
    logger.info(f"{'─'*55}")


def analyze(df: pd.DataFrame):
    n_races = df["race_id"].nunique()
    n_horses = len(df)

    logger.info(f"\n{'='*55}")
    logger.info("  モデル予想成績分析レポート")
    logger.info(f"{'='*55}")
    logger.info(f"  対象期間 : {sorted(df['date'].unique())}")
    logger.info(f"  レース数 : {n_races}")
    logger.info(f"  出走頭数 : {n_horses}")

    # ── 1. Rank-1 的中率 ──
    section("1. Rank-1 的中率 (win_prob 最高馬 → 実際に1着)")
    rank1 = df.sort_values("win_prob", ascending=False).groupby("race_id").first()
    hit1 = rank1["is_1st"].sum()
    hit3 = rank1["is_top3"].sum()
    logger.info(f"  1着的中: {hit1}/{n_races} = {hit1/n_races:.1%}")
    logger.info(f"  3着以内: {hit3}/{n_races} = {hit3/n_races:.1%}")
    logger.info(f"  ※参考: ランダム期待値 ≈ {1/df.groupby('race_id')['number'].count().mean():.1%}")

    # 人気1番手 (最低オッズ) との比較
    fav = df.sort_values("win_odds").groupby("race_id").first()
    fav_hit = fav["is_1st"].sum()
    logger.info(f"\n  [比較] 単勝1番人気の1着的中率: {fav_hit}/{n_races} = {fav_hit/n_races:.1%}")

    # ── 2. Top-3 precision@3 ──
    section("2. Top-3 precision@3 (place_prob 上位3頭 vs 実際3着以内)")
    total_pred, total_hit = 0, 0
    hits_dist = defaultdict(int)
    for _, g in df.groupby("race_id"):
        pred3 = set(g.nlargest(3, "place_prob")["number"])
        act3 = set(g[g["is_top3"] == 1]["number"])
        h = len(pred3 & act3)
        hits_dist[h] += 1
        total_hit += h
        total_pred += 3
    p3 = total_hit / total_pred if total_pred > 0 else 0
    logger.info(f"  precision@3 : {total_hit}/{total_pred} = {p3:.1%}")
    logger.info(f"  ※参考: ランダム期待値 ≈ {3 / df.groupby('race_id')['number'].count().mean():.1%}")
    logger.info("  的中頭数分布:")
    for k in [0, 1, 2, 3]:
        n = hits_dist[k]
        logger.info(f"    {k}頭的中: {n}レース ({n/n_races:.1%})")

    # ── 3. win_prob キャリブレーション ──
    section("3. win_prob キャリブレーション")
    bins = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 1.01]
    labels = ["<5%", "5-10%", "10-15%", "15-20%", "20-30%", "30%+"]
    df["win_band"] = pd.cut(df["win_prob"], bins=bins, labels=labels, right=False)
    calib = df.groupby("win_band", observed=False).agg(
        n=("is_1st", "count"),
        actual=("is_1st", "mean"),
        pred_avg=("win_prob", "mean"),
    ).reset_index()
    calib["diff"] = calib["actual"] - calib["pred_avg"]
    header = f"  {'帯':>8}  {'n':>5}  {'実勝率':>7}  {'予測平均':>8}  {'差':>7}"
    logger.info(header)
    for _, row in calib.iterrows():
        if row["n"] == 0:
            continue
        mark = "▲" if row["diff"] > 0.03 else ("▼" if row["diff"] < -0.03 else " ")
        logger.info(
            f"  {str(row['win_band']):>8}  {int(row['n']):>5}  "
            f"{row['actual']:>6.1%}  {row['pred_avg']:>8.1%}  "
            f"{mark}{row['diff']:>+6.3f}"
        )

    # ── 4. 日付別成績 ──
    section("4. 日付別成績")
    for date, gd in df.groupby("date"):
        nr = gd["race_id"].nunique()
        h1 = gd.sort_values("win_prob", ascending=False).groupby("race_id").first()["is_1st"].sum()
        tp, th = 0, 0
        for _, g in gd.groupby("race_id"):
            pred3 = set(g.nlargest(3, "place_prob")["number"])
            act3 = set(g[g["is_top3"] == 1]["number"])
            th += len(pred3 & act3)
            tp += 3
        logger.info(
            f"  {date}: {nr}レース  "
            f"Rank-1={h1}/{nr}({h1/nr:.0%})  "
            f"prec@3={th}/{tp}({th/tp:.0%})"
        )

    # ── 5. 芝/ダート別 ──
    section("5. 芝/ダート別成績")
    for surf, gd in df[df["surface"].notna()].groupby("surface"):
        nr = gd["race_id"].nunique()
        if nr == 0:
            continue
        h1 = gd.sort_values("win_prob", ascending=False).groupby("race_id").first()["is_1st"].sum()
        tp, th = 0, 0
        for _, g in gd.groupby("race_id"):
            pred3 = set(g.nlargest(3, "place_prob")["number"])
            act3 = set(g[g["is_top3"] == 1]["number"])
            th += len(pred3 & act3)
            tp += 3
        logger.info(
            f"  {surf}: {nr}レース  "
            f"Rank-1={h1}/{nr}({h1/nr:.0%})  "
            f"prec@3={th}/{tp}({th/tp:.0%})"
        )

    # ── 6. 競馬場別 ──
    section("6. 競馬場別成績")
    place_rows = []
    for place, gd in df.groupby("place_name"):
        nr = gd["race_id"].nunique()
        if nr == 0:
            continue
        h1 = gd.sort_values("win_prob", ascending=False).groupby("race_id").first()["is_1st"].sum()
        tp, th = 0, 0
        for _, g in gd.groupby("race_id"):
            pred3 = set(g.nlargest(3, "place_prob")["number"])
            act3 = set(g[g["is_top3"] == 1]["number"])
            th += len(pred3 & act3)
            tp += 3
        place_rows.append({
            "競馬場": place, "レース数": nr,
            "rank1_hit": h1, "rank1_rate": h1/nr,
            "prec3_hit": th, "prec3_total": tp, "prec3_rate": th/tp,
        })
    place_df = pd.DataFrame(place_rows).sort_values("rank1_rate", ascending=False)
    for _, row in place_df.iterrows():
        logger.info(
            f"  {row['競馬場']:>4}: {int(row['レース数']):>3}レース  "
            f"Rank-1={int(row['rank1_hit'])}/{int(row['レース数'])}({row['rank1_rate']:.0%})  "
            f"prec@3={int(row['prec3_hit'])}/{int(row['prec3_total'])}({row['prec3_rate']:.0%})"
        )

    # ── 7. 馬券EV分析 (単勝) ──
    section("7. 馬券EV分析 (win_prob × オッズ ベース)")
    # EV = win_prob * odds
    df["win_ev"] = df["win_prob"] * df["win_odds"]
    df["place_ev"] = df["place_prob"] * (df["win_odds"] ** 0.5 * 0.6).clip(lower=1.1)

    ev_thresholds = [1.0, 1.1, 1.2, 1.3, 1.5]
    logger.info("  [単勝 EV別 回収率シミュレーション]")
    logger.info(f"  {'EV閾値':>7}  {'対象馬':>6}  {'的中':>5}  {'投資':>8}  {'回収(推定)':>10}  {'回収率':>7}")
    for thr in ev_thresholds:
        sub = df[df["win_ev"] >= thr]
        if sub.empty:
            continue
        n = len(sub)
        hits = sub["is_1st"].sum()
        invest = n * 100
        # 払い戻し = 実際のオッズ × 100 (オッズ情報がある場合)
        # 払い戻し額推定: 的中馬のオッズ×100
        ret = (sub["win_odds"] * sub["is_1st"] * 100).sum()
        roi = ret / invest * 100 if invest > 0 else 0
        logger.info(
            f"  ≥{thr:>5.1f}  {n:>6}頭  {int(hits):>5}的中  "
            f"¥{invest:>7,}  ¥{int(ret):>9,}  {roi:>6.1f}%"
        )

    logger.info("\n  [複勝 EV別 回収率シミュレーション]")
    logger.info(f"  {'EV閾値':>7}  {'対象馬':>6}  {'的中':>5}  {'投資':>8}  {'回収(推定)':>10}  {'回収率':>7}")
    for thr in [1.0, 1.1, 1.2]:
        sub = df[df["place_ev"] >= thr]
        if sub.empty:
            continue
        n = len(sub)
        hits = sub["is_top3"].sum()
        invest = n * 100
        # 複勝推定オッズ (単勝^0.5 × 0.6)
        est_place_odds = (sub["win_odds"] ** 0.5 * 0.6).clip(lower=1.1)
        ret = (est_place_odds * sub["is_top3"] * 100).sum()
        roi = ret / invest * 100 if invest > 0 else 0
        logger.info(
            f"  ≥{thr:>5.1f}  {n:>6}頭  {int(hits):>5}的中  "
            f"¥{invest:>7,}  ¥{int(ret):>9,}  {roi:>6.1f}%"
        )

    # ── 8. 市場確率 vs モデル確率の乖離上位 ──
    section("8. モデル vs 市場確率の乖離（穴馬発掘）")
    df["prob_diff"] = df["win_prob"] - df["market_prob"]
    # モデルが市場より高く評価している馬
    overrated_by_model = df[df["win_prob"] > 0.05].nlargest(15, "prob_diff")[
        ["date", "place_name", "race_name", "horse_name", "number",
         "win_odds", "market_prob", "win_prob", "prob_diff", "is_1st", "finish_position"]
    ]
    logger.info("  モデルが市場より高評価な馬 TOP15 (win_prob - market_prob):")
    logger.info(f"  {'日付':>9}  {'場':>4}  {'馬名':>12}  {'馬番':>4}  {'オッズ':>6}  {'市場確率':>8}  {'モデル確率':>10}  {'乖離':>6}  {'着順':>4}")
    for _, row in overrated_by_model.iterrows():
        logger.info(
            f"  {row['date']}  {row['place_name']:>4}  {row['horse_name']:>12}  "
            f"{int(row['number']):>4}  {row['win_odds']:>6.1f}倍  "
            f"{row['market_prob']:>7.1%}  {row['win_prob']:>9.1%}  "
            f"{row['prob_diff']:>+5.3f}  {int(row['finish_position']):>4}着"
        )

    logger.info(f"\n{'='*55}")
    logger.info("  分析完了")
    logger.info(f"{'='*55}\n")

    return df


def _git_show(git_ref: str, filepath: str) -> str | None:
    """git show でファイル内容を取得する。失敗時は None を返す。"""
    import subprocess
    result = subprocess.run(
        ["git", "show", f"{git_ref}:{filepath}"],
        capture_output=True, text=True, encoding="utf-8",
    )
    return result.stdout if result.returncode == 0 else None


def main():
    base = Path(__file__).resolve().parents[1]
    pred_path = base / "data/raw/today_predictions.json"
    result_path = base / "data/raw/today_results.json"

    if not pred_path.exists():
        logger.error(f"予測ファイルが見つかりません: {pred_path}")
        return
    if not result_path.exists():
        logger.error(f"結果ファイルが見つかりません: {result_path}")
        return

    # ── 過去の予測データを git 履歴から取得 ──
    # 4/25-26 予測: commit 6e4d49f の today_predictions.json
    # 4/25-26 日付マップ: commit 717e159 の target_races.json
    PAST_SNAPSHOTS = [
        {"pred_ref": "6e4d49f", "target_ref": "717e159", "label": "4/25-26"},
    ]

    extra_pred_json = None
    extra_date_map: dict[str, str] = {}

    for snap in PAST_SNAPSHOTS:
        pred_json = _git_show(snap["pred_ref"], "data/raw/today_predictions.json")
        target_json = _git_show(snap["target_ref"], "data/raw/target_races.json")
        if pred_json:
            extra_pred_json = pred_json
            logger.info(f"過去予測読み込み: {snap['label']} ({snap['pred_ref']})")
        else:
            logger.warning(f"git show 失敗: {snap['pred_ref']}")
        if target_json:
            for r in json.loads(target_json):
                if r.get("race_id") and r.get("date"):
                    extra_date_map[r["race_id"]] = str(r["date"])
            logger.info(f"過去日付マップ読み込み: {snap['label']} ({snap['target_ref']}, {len(extra_date_map)}件)")
        else:
            logger.warning(f"git show 失敗: {snap['target_ref']}")

    logger.info("データ読み込み中...")
    df = load_data(str(pred_path), str(result_path),
                   extra_pred_json=extra_pred_json,
                   extra_date_map=extra_date_map)
    if df.empty:
        logger.error("マッチするデータがありません")
        return

    result_df = analyze(df)

    out_dir = base / "outputs"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "model_performance.csv"
    result_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info(f"詳細データ保存: {out_path}")


if __name__ == "__main__":
    main()
