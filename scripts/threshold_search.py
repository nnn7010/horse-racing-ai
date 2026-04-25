"""閾値グリッドサーチ: 単勝/複勝/馬連の最適閾値を探索する。

検証期間(2026/3/1-4/17)のすべての買い目候補(閾値なし)を生成し、
各閾値組み合わせでROIを計算して最適値を求める。
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

ROOT = Path(__file__).resolve().parents[1]


def collect_all_bets(valid_preds: pd.DataFrame, race_results: dict, race_payouts: dict) -> pd.DataFrame:
    """全買い目候補を確率付きで収集する（閾値なし）。"""
    rows = []

    for race_id, race_df in valid_preds.groupby("race_id"):
        if race_id not in race_results:
            continue

        result = race_results[race_id]
        payouts = race_payouts.get(race_id, {})
        actual_top3 = result.get("finish_position", {})
        actual_1st = actual_top3.get(1, 0)
        actual_2nd = actual_top3.get(2, 0)
        actual_3rd = actual_top3.get(3, 0)
        date = race_df["date"].iloc[0]

        # Plackett-Luce確率を計算
        try:
            probs = compute_race_probabilities(race_df)
        except Exception:
            continue

        win_probs = probs.get("win", {})
        place_probs = probs.get("place", {})

        # 単勝オッズを構築
        win_odds_dict = {}
        for _, row in race_df.iterrows():
            num = int(row["number"])
            odds = row.get("win_odds", 0)
            if odds > 0:
                win_odds_dict[num] = odds

        numbers = sorted(race_df["number"].astype(int).values)

        # === 単勝 ===
        for num in numbers:
            prob = win_probs.get(num, 0)
            odds = win_odds_dict.get(num, 0)
            if prob <= 0:
                continue
            hit = 1 if num == actual_1st else 0
            payout_per_100 = odds if hit else 0
            rows.append({
                "race_id": race_id, "date": date,
                "bet_type": "単勝", "numbers": str(num),
                "prob": prob, "odds": odds,
                "hit": hit, "payout_per_100": payout_per_100,
            })

        # === 複勝 ===
        for num in numbers:
            prob = place_probs.get(num, 0)
            odds = win_odds_dict.get(num, 0)
            if prob <= 0:
                continue
            place_odds = max(odds ** 0.5 * 0.6, 1.1) if odds > 0 else 0
            hit = 1 if num in {actual_1st, actual_2nd, actual_3rd} else 0
            # 実際の払い戻しがあれば使用
            if hit and payouts.get("place"):
                payout_list = payouts["place"]
                # 複勝は着順の人気に対応する払い戻し (単純に平均)
                payout_per_100 = sum(payout_list) / len(payout_list) / 100 if payout_list else place_odds
            else:
                payout_per_100 = place_odds if hit else 0
            rows.append({
                "race_id": race_id, "date": date,
                "bet_type": "複勝", "numbers": str(num),
                "prob": prob, "odds": place_odds,
                "hit": hit, "payout_per_100": payout_per_100,
            })

        # === 馬連（軸=1位予測、相手=2-8位）===
        sorted_nums = sorted(numbers, key=lambda x: -win_probs.get(x, 0))
        axis = sorted_nums[0]
        axis_wp = win_probs.get(axis, 0)

        for partner in sorted_nums[1:8]:
            if partner == axis:
                continue
            wj = win_probs.get(partner, 0)
            # 馬連確率の推定
            prob = axis_wp * wj / max(1 - axis_wp, 0.01) + wj * axis_wp / max(1 - wj, 0.01)
            prob = min(prob, 0.5)
            if prob <= 0:
                continue
            oi = win_odds_dict.get(axis, 0)
            oj = win_odds_dict.get(partner, 0)
            est_odds = max((oi * oj) ** 0.5 * 0.7, 2.0) if oi > 0 and oj > 0 else 0
            pair = sorted([axis, partner])
            hit = 1 if set(pair) == {actual_1st, actual_2nd} else 0
            payout_per_100 = est_odds if hit else 0
            rows.append({
                "race_id": race_id, "date": date,
                "bet_type": "馬連", "numbers": f"{pair[0]}-{pair[1]}",
                "prob": prob, "odds": est_odds,
                "hit": hit, "payout_per_100": payout_per_100,
            })

    return pd.DataFrame(rows)


def grid_search(df: pd.DataFrame) -> pd.DataFrame:
    """各閾値組み合わせでROIを計算する。"""
    results = []

    win_thresholds = np.arange(0.05, 0.35, 0.025)
    place_thresholds = np.arange(0.30, 0.80, 0.05)
    quinella_thresholds = np.arange(0.05, 0.45, 0.025)

    win_df = df[df["bet_type"] == "単勝"]
    place_df = df[df["bet_type"] == "複勝"]
    quin_df = df[df["bet_type"] == "馬連"]

    total_combos = len(win_thresholds) * len(place_thresholds) * len(quinella_thresholds)
    print(f"Grid search: {total_combos} combinations × 3 bet types")

    for wt in win_thresholds:
        w_sel = win_df[win_df["prob"] >= wt]
        for pt in place_thresholds:
            p_sel = place_df[place_df["prob"] >= pt]
            for qt in quinella_thresholds:
                q_sel = quin_df[quin_df["prob"] >= qt]

                combined = pd.concat([w_sel, p_sel, q_sel])
                if len(combined) == 0:
                    continue

                inv = len(combined) * 100  # 各買い目100円
                ret = combined["payout_per_100"].sum() * 100
                roi = ret / inv * 100 if inv > 0 else 0
                hits = combined["hit"].sum()
                n_bets = len(combined)

                results.append({
                    "win_thresh": round(wt, 3),
                    "place_thresh": round(pt, 3),
                    "quinella_thresh": round(qt, 3),
                    "n_bets": n_bets,
                    "n_win": len(w_sel),
                    "n_place": len(p_sel),
                    "n_quin": len(q_sel),
                    "hits": int(hits),
                    "hit_rate": hits / n_bets * 100,
                    "invest": inv,
                    "return": ret,
                    "roi": roi,
                })

    return pd.DataFrame(results).sort_values("roi", ascending=False)


def main():
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    model_dir = config["paths"]["models"]
    raw_dir = Path(config["paths"]["raw"])
    processed_dir = Path(config["paths"]["processed"])

    print("Loading model...")
    model, feature_cols, calibrator, win_model, win_calibrator = load_model(model_dir)

    print("Loading features...")
    df = pd.read_parquet(processed_dir / "features.parquet")
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")

    valid_start = pd.Timestamp(config["split"]["valid_start"])
    valid_end = pd.Timestamp(config["split"]["valid_end"])
    valid_df = df[(df["date"] >= valid_start) & (df["date"] <= valid_end)].copy()
    print(f"Validation: {valid_start.date()} ~ {valid_end.date()}, {valid_df['race_id'].nunique()} races")

    print("Running predictions...")
    valid_preds = predict_probabilities(
        model, feature_cols, valid_df,
        calibrator=calibrator,
        win_model=win_model,
        win_calibrator=win_calibrator,
    )

    if "pred_win_prob" in valid_preds.columns:
        valid_preds["win_prob"] = valid_preds["pred_win_prob"]
    else:
        for race_id, race_grp in valid_preds.groupby("race_id"):
            strengths = valid_preds.loc[race_grp.index, "pred_strength"].values if "pred_strength" in valid_preds.columns else valid_preds.loc[race_grp.index, "pred_top3_prob"].values
            total = strengths.sum()
            valid_preds.loc[race_grp.index, "win_prob"] = strengths / total if total > 0 else 1.0 / len(race_grp)

    print("Loading historical results...")
    race_results = {}
    race_payouts = {}
    history_file = raw_dir / "historical_results.json"
    if history_file.exists():
        with open(history_file, encoding="utf-8") as f:
            for race in json.load(f):
                rid = race.get("race_id", "")
                fp_map = {}
                for r in race.get("results", []):
                    pos = r.get("finish_position", 0)
                    num = r.get("number", 0)
                    if 1 <= pos <= 3:
                        fp_map[pos] = num
                if fp_map:
                    race_results[rid] = {"finish_position": fp_map}
                payouts = race.get("payouts", {})
                if payouts:
                    p = {}
                    if payouts.get("win"):
                        p["win"] = payouts["win"][0].get("amount", 0)
                    if payouts.get("place"):
                        p["place"] = [x.get("amount", 0) for x in payouts["place"]]
                    if payouts.get("trio"):
                        p["trio"] = payouts["trio"][0].get("amount", 0)
                    if payouts.get("trifecta"):
                        p["trifecta"] = payouts["trifecta"][0].get("amount", 0)
                    race_payouts[rid] = p

    print("Collecting all bet candidates...")
    all_bets = collect_all_bets(valid_preds, race_results, race_payouts)
    print(f"Total candidates: {len(all_bets)} ({all_bets['bet_type'].value_counts().to_dict()})")

    out_dir = Path(config["paths"]["outputs"])
    out_dir.mkdir(parents=True, exist_ok=True)
    all_bets.to_csv(out_dir / "all_bet_candidates.csv", index=False, encoding="utf-8-sig")
    print(f"Saved: {out_dir / 'all_bet_candidates.csv'}")

    print("\nRunning grid search...")
    grid_df = grid_search(all_bets)

    grid_df.to_csv(out_dir / "threshold_search.csv", index=False, encoding="utf-8-sig")
    print(f"Saved: {out_dir / 'threshold_search.csv'}")

    print("\n=== TOP 20 combinations by ROI ===")
    print(grid_df.head(20).to_string(index=False))

    print("\n=== TOP 20 by ROI with n_bets >= 100 ===")
    filtered = grid_df[grid_df["n_bets"] >= 100]
    print(filtered.head(20).to_string(index=False))

    # 券種別の最適閾値
    print("\n=== 券種別 単独最適閾値 ===")
    for bet_type, col in [("単勝", "win_thresh"), ("複勝", "place_thresh"), ("馬連", "quinella_thresh")]:
        btype_df = all_bets[all_bets["bet_type"] == bet_type]
        rows = []
        for thresh in np.arange(0.05, 0.80, 0.025):
            sel = btype_df[btype_df["prob"] >= thresh]
            if len(sel) < 10:
                continue
            inv = len(sel) * 100
            ret = sel["payout_per_100"].sum() * 100
            roi = ret / inv * 100
            rows.append({"thresh": round(thresh, 3), "n": len(sel), "roi": roi, "hits": sel["hit"].sum()})
        if rows:
            best_df = pd.DataFrame(rows).sort_values("roi", ascending=False)
            print(f"\n{bet_type} (top10):")
            print(best_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
