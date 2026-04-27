"""08: 予想 vs 実結果を照合して、現行モデルの改善点を診断する。

入力:
  data/raw/today_predictions.json  (各レースの horses[] に win_prob/place_prob/win_odds/ability)
  data/raw/today_results.json      (race_id -> 1st/2nd/3rd 馬番)

出力:
  outputs/diagnose_summary.csv     (主要指標まとめ)
  outputs/diagnose_calibration.csv (確率ビン別キャリブレーション)
  outputs/diagnose_segments.csv    (セグメント別 hit@1 / hit@3)
  outputs/diagnose_failures.csv    (本命を外したレース一覧)
  outputs/diagnose_report.md       (人が読むサマリ)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import yaml

from src.utils.logger import get_logger

logger = get_logger("08_diagnose")


def _build_horse_table(predictions: dict, results: dict) -> pd.DataFrame:
    """予想と結果を1馬1行のDataFrameに展開する。"""
    rows: list[dict[str, Any]] = []
    for race in predictions.get("races", []):
        rid = race["race_id"]
        result = results.get(rid)
        if not result or result.get("status") != "confirmed":
            continue
        finish = {
            int(result.get("1st", -1)): 1,
            int(result.get("2nd", -1)): 2,
            int(result.get("3rd", -1)): 3,
        }
        finish.pop(-1, None)
        n = race.get("n_horses", len(race.get("horses", [])))
        cp = race.get("course_profile") or {}
        for h in race.get("horses", []):
            num = int(h["number"])
            ab = h.get("ability") or {}
            rows.append({
                "race_id": rid,
                "place_name": race.get("place_name"),
                "surface": race.get("surface"),
                "distance": race.get("distance"),
                "n_horses": n,
                "number": num,
                "bracket": h.get("bracket"),
                "horse_name": h.get("horse_name"),
                "win_odds": h.get("win_odds") or 0.0,
                "win_prob": h.get("win_prob") or 0.0,
                "place_prob": h.get("place_prob") or 0.0,
                "running_style": h.get("running_style"),
                "ability_speed": ab.get("speed"),
                "ability_burst": ab.get("burst"),
                "ability_power": ab.get("power"),
                "ability_course": ab.get("course"),
                "ability_form": ab.get("form"),
                "ability_stability": ab.get("stability"),
                "ability_jockey": ab.get("jockey"),
                "ability_fit": ab.get("fit"),
                "course_speed": cp.get("speed"),
                "course_form": cp.get("form"),
                "finish": finish.get(num, 0),  # 0=4着以下
                "is_top1": int(finish.get(num, 0) == 1),
                "is_top3": int(1 <= finish.get(num, 0) <= 3),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # レース内ランクを付ける
    df["pred_rank_win"] = df.groupby("race_id")["win_prob"].rank(method="first", ascending=False).astype(int)
    df["pred_rank_place"] = df.groupby("race_id")["place_prob"].rank(method="first", ascending=False).astype(int)
    df["pop_rank"] = df.groupby("race_id")["win_odds"].rank(method="first", ascending=True).astype(int)
    return df


def _hit_summary(df: pd.DataFrame) -> dict[str, float]:
    """全体ヒット率と人気予想との比較。"""
    races = df["race_id"].nunique()
    pick1 = df[df["pred_rank_win"] == 1]
    top3 = df[df["pred_rank_place"] <= 3]
    fav = df[df["pop_rank"] == 1]

    hit_at_1 = pick1["is_top1"].mean() if not pick1.empty else 0.0
    hit_at_1_top3 = pick1["is_top3"].mean() if not pick1.empty else 0.0
    # 予想Top3と実Top3の重複率（馬数ベース）
    top3_overlap = top3.groupby("race_id")["is_top3"].sum().mean() / 3.0 if not top3.empty else 0.0
    fav_at_1 = fav["is_top1"].mean() if not fav.empty else 0.0

    return {
        "races": int(races),
        "horses": int(len(df)),
        "model_hit@1": float(hit_at_1),
        "model_pick1_top3": float(hit_at_1_top3),
        "model_top3_overlap": float(top3_overlap),
        "favorite_hit@1": float(fav_at_1),
    }


def _calibration(df: pd.DataFrame, prob_col: str, target_col: str,
                 bins: list[float]) -> pd.DataFrame:
    """確率ビン別の予測 vs 実勝率。"""
    labels = [f"{bins[i]*100:.0f}-{bins[i+1]*100:.0f}%" for i in range(len(bins)-1)]
    df = df.copy()
    df["bin"] = pd.cut(df[prob_col], bins=bins, labels=labels, include_lowest=True)
    grp = df.groupby("bin", observed=True).agg(
        n=("race_id", "count"),
        pred_mean=(prob_col, "mean"),
        actual_rate=(target_col, "mean"),
    ).reset_index()
    grp["gap"] = grp["actual_rate"] - grp["pred_mean"]
    grp["prob"] = prob_col
    grp["target"] = target_col
    return grp


def _segment_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    """セグメント別の本命的中率。"""
    pick1 = df[df["pred_rank_win"] == 1].copy()
    top3 = df[df["pred_rank_place"] <= 3].copy()

    rows = []

    def add(name: str, value: Any, sub_pick1: pd.DataFrame, sub_top3: pd.DataFrame):
        if len(sub_pick1) == 0:
            return
        rows.append({
            "segment": name,
            "value": value,
            "races": int(len(sub_pick1)),
            "hit@1": float(sub_pick1["is_top1"].mean()),
            "pick1_top3": float(sub_pick1["is_top3"].mean()),
            "top3_overlap": float(sub_top3.groupby("race_id")["is_top3"].sum().mean() / 3.0)
                if not sub_top3.empty else 0.0,
        })

    for v in sorted(df["surface"].dropna().unique()):
        add("surface", v, pick1[pick1["surface"] == v], top3[top3["surface"] == v])
    for v in sorted(df["place_name"].dropna().unique()):
        add("place", v, pick1[pick1["place_name"] == v], top3[top3["place_name"] == v])
    # 距離レンジ
    bins = [0, 1400, 1800, 2200, 9999]
    labels = ["~1400", "1401-1800", "1801-2200", "2201+"]
    pick1["dist_band"] = pd.cut(pick1["distance"], bins=bins, labels=labels)
    top3["dist_band"] = pd.cut(top3["distance"], bins=bins, labels=labels)
    for v in labels:
        add("distance", v, pick1[pick1["dist_band"] == v], top3[top3["dist_band"] == v])
    # 頭数レンジ
    bins_n = [0, 9, 13, 16, 99]
    labels_n = ["~9", "10-13", "14-16", "17+"]
    pick1["n_band"] = pd.cut(pick1["n_horses"], bins=bins_n, labels=labels_n)
    top3["n_band"] = pd.cut(top3["n_horses"], bins=bins_n, labels=labels_n)
    for v in labels_n:
        add("n_horses", v, pick1[pick1["n_band"] == v], top3[top3["n_band"] == v])

    return pd.DataFrame(rows)


def _ev_bias(df: pd.DataFrame) -> pd.DataFrame:
    """EV(=win_prob*win_odds) 帯別のヒット率と回収率。"""
    work = df[(df["win_prob"] > 0) & (df["win_odds"] > 0)].copy()
    if work.empty:
        return pd.DataFrame()
    work["ev"] = work["win_prob"] * work["win_odds"]
    bins = [0, 0.5, 0.8, 1.0, 1.1, 1.3, 1.6, 100]
    labels = ["<0.5", "0.5-0.8", "0.8-1.0", "1.0-1.1", "1.1-1.3", "1.3-1.6", "1.6+"]
    work["ev_band"] = pd.cut(work["ev"], bins=bins, labels=labels)
    grp = work.groupby("ev_band", observed=True).agg(
        n=("race_id", "count"),
        win_rate=("is_top1", "mean"),
        place_rate=("is_top3", "mean"),
        avg_odds=("win_odds", "mean"),
        avg_prob=("win_prob", "mean"),
    ).reset_index()
    # 100円賭けたときの回収率（単勝）
    work["payout100"] = np.where(work["is_top1"] == 1, work["win_odds"] * 100, 0)
    roi = work.groupby("ev_band", observed=True)["payout100"].mean().rename("roi_per_100").reset_index()
    grp = grp.merge(roi, on="ev_band")
    return grp


def _ability_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """ability_* 列と is_top3 / is_top1 の点相関。"""
    cols = [c for c in df.columns if c.startswith("ability_")]
    rows = []
    for c in cols:
        sub = df[[c, "is_top1", "is_top3"]].dropna()
        if sub.empty or sub[c].std() == 0:
            continue
        rows.append({
            "feature": c,
            "n": int(len(sub)),
            "corr_top1": float(sub[c].corr(sub["is_top1"])),
            "corr_top3": float(sub[c].corr(sub["is_top3"])),
        })
    return pd.DataFrame(rows).sort_values("corr_top3", ascending=False)


def _failure_table(df: pd.DataFrame) -> pd.DataFrame:
    """本命を外したレース。何位に来たか、1着馬は何番人気か。"""
    pick1 = df[df["pred_rank_win"] == 1].copy()
    losers = pick1[pick1["is_top3"] == 0].copy()
    if losers.empty:
        return pd.DataFrame()

    winners = df[df["finish"] == 1][["race_id", "number", "horse_name",
                                      "win_odds", "pop_rank", "win_prob"]]
    winners = winners.rename(columns={
        "number": "winner_num", "horse_name": "winner_name",
        "win_odds": "winner_odds", "pop_rank": "winner_pop",
        "win_prob": "winner_pred_prob",
    })
    out = losers.merge(winners, on="race_id", how="left")
    return out[[
        "race_id", "place_name", "surface", "distance", "n_horses",
        "horse_name", "number", "win_prob", "win_odds", "pop_rank",
        "winner_name", "winner_num", "winner_pop", "winner_odds",
        "winner_pred_prob",
    ]].sort_values("winner_pop", ascending=False)


def _writeline(buf: list[str], s: str = "") -> None:
    buf.append(s)


def _format_report(summary: dict, calib_win: pd.DataFrame, calib_place: pd.DataFrame,
                   segments: pd.DataFrame, ev: pd.DataFrame,
                   ab_corr: pd.DataFrame, fail: pd.DataFrame) -> str:
    buf: list[str] = []
    _writeline(buf, "# 予想モデル診断レポート")
    _writeline(buf)
    _writeline(buf, f"- 対象レース数: {summary['races']}")
    _writeline(buf, f"- 対象馬数: {summary['horses']}")
    _writeline(buf, f"- モデル本命の1着的中率 (hit@1): {summary['model_hit@1']*100:.1f}%")
    _writeline(buf, f"- モデル本命の3着内率: {summary['model_pick1_top3']*100:.1f}%")
    _writeline(buf, f"- 予想Top3と実Top3の重複率: {summary['model_top3_overlap']*100:.1f}%")
    _writeline(buf, f"- 1番人気の1着的中率（参考値）: {summary['favorite_hit@1']*100:.1f}%")
    _writeline(buf)

    _writeline(buf, "## 1. キャリブレーション (win_prob)")
    _writeline(buf, "| ビン | n | 予測平均 | 実勝率 | 乖離 |")
    _writeline(buf, "|---|---:|---:|---:|---:|")
    for _, r in calib_win.iterrows():
        _writeline(buf, f"| {r['bin']} | {r['n']} | {r['pred_mean']*100:.1f}% | "
                        f"{r['actual_rate']*100:.1f}% | {r['gap']*100:+.1f}pt |")
    _writeline(buf)

    _writeline(buf, "## 2. キャリブレーション (place_prob)")
    _writeline(buf, "| ビン | n | 予測平均 | 実3着内率 | 乖離 |")
    _writeline(buf, "|---|---:|---:|---:|---:|")
    for _, r in calib_place.iterrows():
        _writeline(buf, f"| {r['bin']} | {r['n']} | {r['pred_mean']*100:.1f}% | "
                        f"{r['actual_rate']*100:.1f}% | {r['gap']*100:+.1f}pt |")
    _writeline(buf)

    _writeline(buf, "## 3. セグメント別 本命精度")
    _writeline(buf, "| 軸 | 値 | レース数 | hit@1 | 本命3着内 | Top3重複 |")
    _writeline(buf, "|---|---|---:|---:|---:|---:|")
    for _, r in segments.iterrows():
        _writeline(buf, f"| {r['segment']} | {r['value']} | {r['races']} | "
                        f"{r['hit@1']*100:.1f}% | {r['pick1_top3']*100:.1f}% | "
                        f"{r['top3_overlap']*100:.1f}% |")
    _writeline(buf)

    _writeline(buf, "## 4. EV帯別の単勝回収率")
    _writeline(buf, "（EV = win_prob × win_odds。100円賭けの平均払い戻し）")
    _writeline(buf, "| EV帯 | n | 平均オッズ | 平均予測勝率 | 実勝率 | 実3着内率 | 100円ROI |")
    _writeline(buf, "|---|---:|---:|---:|---:|---:|---:|")
    for _, r in ev.iterrows():
        _writeline(buf, f"| {r['ev_band']} | {r['n']} | {r['avg_odds']:.1f} | "
                        f"{r['avg_prob']*100:.1f}% | {r['win_rate']*100:.1f}% | "
                        f"{r['place_rate']*100:.1f}% | {r['roi_per_100']:.0f}円 |")
    _writeline(buf)

    _writeline(buf, "## 5. 能力指標と実結果の相関")
    _writeline(buf, "| 特徴量 | n | corr(1着) | corr(3着内) |")
    _writeline(buf, "|---|---:|---:|---:|")
    for _, r in ab_corr.iterrows():
        _writeline(buf, f"| {r['feature']} | {r['n']} | {r['corr_top1']:+.3f} | "
                        f"{r['corr_top3']:+.3f} |")
    _writeline(buf)

    _writeline(buf, "## 6. 本命を外したレース（穴党順 = 勝ち馬の人気が低い順）")
    _writeline(buf, "| race_id | コース | 距離 | 本命名 | 本命人気 | 勝ち馬 | 勝ち馬人気 | 勝ちオッズ |")
    _writeline(buf, "|---|---|---:|---|---:|---|---:|---:|")
    for _, r in fail.head(20).iterrows():
        _writeline(buf, f"| {r['race_id']} | {r['place_name']}{r['surface']} | "
                        f"{r['distance']} | {r['horse_name']} | {int(r['pop_rank'])} | "
                        f"{r['winner_name']} | {int(r['winner_pop']) if pd.notna(r['winner_pop']) else '-'} | "
                        f"{r['winner_odds']:.1f} |")
    _writeline(buf)

    _writeline(buf, "## 改善ヒントの読み方")
    _writeline(buf, "- §1/§2 で `乖離` がプラスに大きい確率ビン = モデルが過小評価。"
                    "マイナス = 過信。Isotonic / Platt 再キャリブレーションの候補。")
    _writeline(buf, "- §3 で hit@1 が極端に低いセグメントは、その条件向けの特徴量が不足。")
    _writeline(buf, "- §4 で EV>1.1 帯のROIが100円割れなら、確率推定が高EV側で甘い。")
    _writeline(buf, "- §5 で相関が低い ability_* 列は、"
                    "重み付けの見直し or 削除候補。逆に高いものは信号として有効。")
    _writeline(buf, "- §6 で勝ち馬人気が低いレースが多い = 穴狙いが弱い。"
                    "近走不振の巻き返しシグナル（休み明け、距離短縮、騎手乗替り）を再評価。")
    return "\n".join(buf)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default=None)
    parser.add_argument("--results", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    raw_dir = Path(config["paths"]["raw"])
    out_dir = Path(args.output_dir or config["paths"]["outputs"])
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_path = Path(args.predictions or raw_dir / "today_predictions.json")
    res_path = Path(args.results or raw_dir / "today_results.json")

    with open(pred_path, encoding="utf-8") as f:
        predictions = json.load(f)
    with open(res_path, encoding="utf-8") as f:
        results = json.load(f)

    df = _build_horse_table(predictions, results)
    if df.empty:
        logger.error("No matched races between predictions and results")
        sys.exit(1)
    logger.info(f"Matched horses: {len(df)} / races: {df['race_id'].nunique()}")

    summary = _hit_summary(df)
    calib_win = _calibration(df, "win_prob", "is_top1",
                             [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.01])
    calib_place = _calibration(df, "place_prob", "is_top3",
                               [0.0, 0.10, 0.20, 0.35, 0.50, 0.70, 1.01])
    segments = _segment_accuracy(df)
    ev = _ev_bias(df)
    ab_corr = _ability_correlation(df)
    fail = _failure_table(df)

    pd.DataFrame([summary]).to_csv(out_dir / "diagnose_summary.csv", index=False)
    pd.concat([calib_win, calib_place], ignore_index=True).to_csv(
        out_dir / "diagnose_calibration.csv", index=False)
    segments.to_csv(out_dir / "diagnose_segments.csv", index=False)
    ev.to_csv(out_dir / "diagnose_ev.csv", index=False)
    ab_corr.to_csv(out_dir / "diagnose_ability_corr.csv", index=False)
    fail.to_csv(out_dir / "diagnose_failures.csv", index=False)

    report = _format_report(summary, calib_win, calib_place, segments, ev, ab_corr, fail)
    (out_dir / "diagnose_report.md").write_text(report, encoding="utf-8")

    logger.info(f"Wrote report to {out_dir / 'diagnose_report.md'}")
    logger.info(f"hit@1={summary['model_hit@1']*100:.1f}% "
                f"vs favorite={summary['favorite_hit@1']*100:.1f}% "
                f"top3_overlap={summary['model_top3_overlap']*100:.1f}%")


if __name__ == "__main__":
    main()
