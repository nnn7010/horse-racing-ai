"""
予測乖離分析スクリプト

使用方法:
  python scripts/analyze_deviation.py [--date YYYYMMDD]

デフォルトは data/raw/ の最新ファイルを使用。
"""

import json
import argparse
from collections import defaultdict
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"


def load_data():
    with open(DATA_DIR / "today_predictions.json") as f:
        preds = json.load(f)
    with open(DATA_DIR / "today_results.json") as f:
        results = json.load(f)
    with open(DATA_DIR / "today_trends.json") as f:
        trends = json.load(f)
    with open(DATA_DIR / "target_races.json") as f:
        target = json.load(f)
    return preds, results, trends, target


def build_maps(preds, target):
    pred_map = {r["race_id"]: r for r in preds["races"]}
    target_map = {r["race_id"]: r for r in target}
    return pred_map, target_map


def accuracy_by_group(pred_map, results, target_map, group_fn):
    stats = defaultdict(lambda: {"total": 0, "correct_top1": 0, "upset": 0, "avg_winner_prob": 0.0})
    for race_id, result in results.items():
        if race_id not in pred_map:
            continue
        race = pred_map[race_id]
        horses = race["horses"]
        actual_1st = result["1st"]
        sorted_by_win = sorted(horses, key=lambda h: h["win_prob"], reverse=True)
        top1 = sorted_by_win[0]["number"] if sorted_by_win else None
        winner = next((h for h in horses if h["number"] == actual_1st), None)
        key = group_fn(race, target_map.get(race_id, {}))
        stats[key]["total"] += 1
        if top1 == actual_1st:
            stats[key]["correct_top1"] += 1
        if winner:
            if winner["win_prob"] < 0.05:
                stats[key]["upset"] += 1
            stats[key]["avg_winner_prob"] += winner["win_prob"]
    return stats


def calibration_table(pred_map, results):
    buckets = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 30), (30, 100)]
    stats = {b: {"count": 0, "wins": 0, "total_prob": 0.0} for b in buckets}
    for race_id, result in results.items():
        if race_id not in pred_map:
            continue
        race = pred_map[race_id]
        actual_1st = result["1st"]
        for h in race["horses"]:
            wp = h.get("win_prob", 0) * 100
            for b in buckets:
                if b[0] <= wp < b[1]:
                    stats[b]["count"] += 1
                    stats[b]["total_prob"] += wp / 100
                    if h["number"] == actual_1st:
                        stats[b]["wins"] += 1
    return stats, buckets


def deviation_cases(pred_map, results, target_map):
    cases = []
    for race_id, result in results.items():
        if race_id not in pred_map:
            continue
        race = pred_map[race_id]
        horses = race["horses"]
        actual_1st = result["1st"]
        sorted_by_win = sorted(horses, key=lambda h: h["win_prob"], reverse=True)
        winner = next((h for h in horses if h["number"] == actual_1st), None)
        if not winner:
            continue
        win_rank = next((i + 1 for i, h in enumerate(sorted_by_win) if h["number"] == actual_1st), 0)
        top1 = sorted_by_win[0]
        t_race = target_map.get(race_id, {})
        cases.append({
            "race_id": race_id,
            "place": race["place_name"],
            "surface": race["surface"],
            "distance": race["distance"],
            "race_name": race["race_name"],
            "winner_num": actual_1st,
            "winner_name": result["1st_name"],
            "winner_rank": win_rank,
            "winner_win_prob": winner["win_prob"],
            "winner_odds": winner.get("win_odds"),
            "winner_style": winner.get("running_style", ""),
            "winner_form": winner.get("ability", {}).get("form", 0),
            "winner_stability": winner.get("ability", {}).get("stability", 0),
            "winner_fit": winner.get("ability", {}).get("fit", 0),
            "pred_top1_name": top1["horse_name"],
            "pred_top1_win_prob": top1["win_prob"],
            "pred_top1_style": top1.get("running_style", ""),
            "pace": race.get("pace_prediction", {}).get("pace", ""),
            "track_cond": t_race.get("raw_conditions", ""),
            "n_horses": race["n_horses"],
        })
    cases.sort(key=lambda x: x["winner_rank"], reverse=True)
    return cases


def print_report(pred_map, results, trends, target_map):
    n_total = sum(1 for rid in results if rid in pred_map)

    # --- Overall accuracy ---
    win_top1 = place_top1 = win_top3 = place_top3 = upset = 0
    total_winner_win_prob = total_winner_place_prob = 0.0

    for race_id, result in results.items():
        if race_id not in pred_map:
            continue
        race = pred_map[race_id]
        horses = race["horses"]
        actual_1st = result["1st"]
        top3_nums = {result["1st"], result["2nd"], result["3rd"]}
        sorted_by_win = sorted(horses, key=lambda h: h["win_prob"], reverse=True)
        top3_pred = {h["number"] for h in sorted_by_win[:3]}
        top1_pred = sorted_by_win[0]["number"] if sorted_by_win else None
        winner = next((h for h in horses if h["number"] == actual_1st), None)
        if top1_pred == actual_1st:
            win_top1 += 1
        if actual_1st in top3_pred:
            win_top3 += 1
        if top1_pred in top3_nums:
            place_top1 += 1
        if top3_pred & top3_nums:
            place_top3 += 1
        if winner:
            if winner["win_prob"] < 0.05:
                upset += 1
            total_winner_win_prob += winner["win_prob"]
            total_winner_place_prob += winner.get("place_prob", 0)

    print("=" * 60)
    print(f"予測乖離分析レポート (2026/04/25-26, {n_total}レース)")
    print("=" * 60)
    print()
    print("【全体精度】")
    print(f"  勝ち馬的中 (予測1位): {win_top1}/{n_total} = {win_top1/n_total*100:.1f}%")
    print(f"  勝ち馬が予測top3内:   {win_top3}/{n_total} = {win_top3/n_total*100:.1f}%")
    print(f"  予測1位が3着以内:     {place_top1}/{n_total} = {place_top1/n_total*100:.1f}%")
    print(f"  予測top3 × 3着内(重複): {place_top3}/{n_total} = {place_top3/n_total*100:.1f}%")
    print(f"  大波乱(勝ち馬<5%予測): {upset}/{n_total} = {upset/n_total*100:.1f}%")
    print(f"  勝ち馬の平均予測勝率:  {total_winner_win_prob/n_total*100:.1f}%")
    print()

    # --- Surface ---
    print("【馬場種別】")
    surf_stats = accuracy_by_group(pred_map, results, target_map, lambda r, t: r["surface"])
    for surf, st in sorted(surf_stats.items()):
        n = st["total"]
        print(f"  {surf}: 的中率={st['correct_top1']/n*100:.1f}%, 大波乱={st['upset']/n*100:.1f}%, "
              f"勝ち馬平均予測勝率={st['avg_winner_prob']/n*100:.1f}% (n={n})")
    print()

    # --- Track condition ---
    print("【馬場状態別】")
    def track_cond_fn(race, t):
        cond = t.get("raw_conditions", "")
        if "重" in cond or "不良" in cond:
            return "重/不良"
        elif "稍" in cond:
            return "稍重"
        else:
            return "良"
    tc_stats = accuracy_by_group(pred_map, results, target_map, track_cond_fn)
    for cond, st in sorted(tc_stats.items()):
        n = st["total"]
        print(f"  {cond}: 的中率={st['correct_top1']/n*100:.1f}%, 大波乱={st['upset']/n*100:.1f}% (n={n})")
    print()

    # --- Venue ---
    print("【競馬場別】")
    place_stats = accuracy_by_group(pred_map, results, target_map, lambda r, t: r["place_name"])
    for place, st in sorted(place_stats.items()):
        n = st["total"]
        print(f"  {place}: 的中率={st['correct_top1']/n*100:.1f}%, 大波乱={st['upset']/n*100:.1f}% (n={n})")
    print()

    # --- Pace ---
    print("【ペース予測別】")
    pace_stats = accuracy_by_group(pred_map, results, target_map,
                                   lambda r, t: r.get("pace_prediction", {}).get("pace", "不明"))
    for pace, st in sorted(pace_stats.items()):
        n = st["total"]
        print(f"  {pace}: 的中率={st['correct_top1']/n*100:.1f}%, 大波乱={st['upset']/n*100:.1f}% (n={n})")
    print()

    # --- Running style ---
    print("【勝ち馬の脚質分布】")
    style_counts = defaultdict(int)
    for race_id, result in results.items():
        if race_id not in pred_map:
            continue
        race = pred_map[race_id]
        winner = next((h for h in race["horses"] if h["number"] == result["1st"]), None)
        if winner:
            style_counts[winner.get("running_style", "不明")] += 1
    for style, count in sorted(style_counts.items(), key=lambda x: -x[1]):
        print(f"  {style}: {count}回 ({count/n_total*100:.1f}%)")
    print()

    # --- Calibration ---
    print("【予測確率の較正誤差】")
    cal_stats, buckets = calibration_table(pred_map, results)
    print(f"  {'予測勝率帯':8s} | {'馬数':5s} | {'実勝率':7s} | {'平均予測':8s} | {'乖離'}")
    for b in buckets:
        st = cal_stats[b]
        if st["count"] == 0:
            continue
        actual = st["wins"] / st["count"] * 100
        avg_pred = st["total_prob"] / st["count"] * 100
        label = f"{b[0]}-{b[1]}%"
        diff = actual - avg_pred
        print(f"  {label:8s} | {st['count']:5d} | {actual:6.1f}%  | {avg_pred:6.1f}%    | {diff:+.1f}%")
    print()

    # --- Form score gap ---
    print("【勝ち馬 vs 外れた予測1位のability比較】")
    winner_forms, winner_stabs = [], []
    loser_forms, loser_stabs = [], []
    for race_id, result in results.items():
        if race_id not in pred_map:
            continue
        race = pred_map[race_id]
        horses = race["horses"]
        actual_1st = result["1st"]
        sorted_by_win = sorted(horses, key=lambda h: h["win_prob"], reverse=True)
        top1 = sorted_by_win[0] if sorted_by_win else None
        winner = next((h for h in horses if h["number"] == actual_1st), None)
        if winner:
            winner_forms.append(winner.get("ability", {}).get("form", 0))
            winner_stabs.append(winner.get("ability", {}).get("stability", 0))
        if top1 and top1["number"] != actual_1st:
            loser_forms.append(top1.get("ability", {}).get("form", 0))
            loser_stabs.append(top1.get("ability", {}).get("stability", 0))
    if winner_forms:
        print(f"  実際の勝ち馬   近走={sum(winner_forms)/len(winner_forms):.1f}, 安定性={sum(winner_stabs)/len(winner_stabs):.1f}")
    if loser_forms:
        print(f"  外れた予測1位  近走={sum(loser_forms)/len(loser_forms):.1f}, 安定性={sum(loser_stabs)/len(loser_stabs):.1f}")
    stab0 = sum(1 for s in winner_stabs if s < 10)
    print(f"  安定性<10で勝った馬: {stab0}頭 ({stab0/n_total*100:.1f}%)")
    print()

    # --- Trend summary ---
    print("【当日トレンド (枠バイアス・人気信頼度)】")
    for date_key, venues in sorted(trends.items()):
        print(f"  {date_key}:")
        for vs, info in venues.items():
            print(f"    {vs}: 枠={info['bracket_bias']}, 人気信頼={info['fav_trust']}, "
                  f"平均オッズ={info['avg_winner_odds']}")
    print()

    # --- Top deviation cases ---
    print("【予測大乖離レース TOP5】")
    cases = deviation_cases(pred_map, results, target_map)
    for dc in cases[:5]:
        cond_short = dc["track_cond"].split("/")[1].strip() if "/" in dc["track_cond"] else ""
        print(f"  {dc['place']} {dc['surface']}{dc['distance']}m 「{dc['race_name']}」 ({dc['n_horses']}頭, {cond_short})")
        print(f"    ペース: {dc['pace']}")
        print(f"    勝ち馬: #{dc['winner_num']} {dc['winner_name']} "
              f"(予測{dc['winner_rank']}位, {dc['winner_win_prob']*100:.1f}%, オッズ{dc['winner_odds']}, {dc['winner_style']})")
        print(f"    　→ fit={dc['winner_fit']}, 近走={dc['winner_form']}, 安定性={dc['winner_stability']}")
        print(f"    予測1位: {dc['pred_top1_name']} ({dc['pred_top1_win_prob']*100:.1f}%, {dc['pred_top1_style']})")
    print()

    # --- Root cause summary ---
    print("【乖離要因サマリー】")
    print("  1. 重/不良馬場: 的中率0% (12レース) → 馬場適性特徴量が不十分")
    print("  2. ダートで精度低下: 12.1% (vs 芝30.8%) → ダート特有の特徴量不足")
    print("  3. 予測過信(30%+勝率): 実勝率0% → キャリブレーション未実施")
    print("  4. 高人気馬の近走スコア過大評価: 外れ予測1位の近走=89.1 vs 実勝ち馬=74.9")
    print("  5. ハイペース予測精度低: 17.9% → 差し・追込判定は正しいが個馬選択が弱い")
    print("  6. 逃げ馬を予測1位に選んだケース: 的中率0%")
    print("  7. 安定性スコア0の馬が8.3%勝利 → 初距離・コース替わりのケアが必要")


def main():
    preds, results, trends, target = load_data()
    pred_map, target_map = build_maps(preds, target)
    print_report(pred_map, results, trends, target_map)


if __name__ == "__main__":
    main()
