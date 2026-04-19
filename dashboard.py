"""競馬予想AI ダッシュボード"""

import json
from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(page_title="競馬予想AI", page_icon="🏇", layout="wide")

# 60秒ごとに自動リフレッシュ（結果自動取得用）
try:
    from streamlit_autorefresh import st_autorefresh
    auto_refresh_count = st_autorefresh(interval=60000, limit=None, key="auto_refresh")
except ImportError:
    auto_refresh_count = 0

# --- セッションステート初期化 ---
if "results" not in st.session_state:
    st.session_state.results = {}  # {race_id: {"1st": 馬番, "2nd": 馬番, "3rd": 馬番}}
if "auto_fetch" not in st.session_state:
    st.session_state.auto_fetch = False
if "last_fetch_time" not in st.session_state:
    st.session_state.last_fetch_time = ""
if "bias" not in st.session_state:
    st.session_state.bias = {
        "inner_advantage": 0.0,    # 内枠バイアス（+で内有利）
        "front_runner": 0.0,       # 逃げ先行バイアス（+で前有利）
        "model_accuracy": 0.0,     # モデル精度補正
        "upset_tendency": 0.0,     # 波乱傾向（+で穴馬有利）
        "races_analyzed": 0,
    }
if "bets_today" not in st.session_state:
    st.session_state.bets_today = {"invested": 0, "returned": 0}


# --- データ読み込み ---
@st.cache_data
def load_predictions():
    return pd.read_csv("outputs/predictions.csv")

@st.cache_data
def load_target_races():
    with open("data/raw/target_races.json", encoding="utf-8") as f:
        return json.load(f)

@st.cache_data
def load_horses():
    path = Path("data/raw/horses.json")
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return {h["horse_id"]: h for h in json.load(f)}
    return {}

@st.cache_data
def load_odds_from_cache(race_ids):
    odds_map = {}
    for race_id in race_ids:
        import hashlib
        cache_path = Path(f"data/cache/{hashlib.md5(f'https://race.netkeiba.com/api/api_get_jra_odds.html?race_id={race_id}&type=1&action=update'.encode()).hexdigest()}.html")
        if cache_path.exists():
            try:
                data = json.loads(cache_path.read_text())
                if data.get("status") == "result":
                    odds_map[race_id] = data["data"]["odds"]
                    odds_map[race_id]["_updated"] = data["data"].get("official_datetime", "")
            except:
                pass
    return odds_map


def fetch_live_odds(race_id):
    """リアルタイムでオッズをAPIから取得する（キャッシュを使わない）"""
    import requests
    from datetime import datetime
    url = f"https://race.netkeiba.com/api/api_get_jra_odds.html?race_id={race_id}&type=1&action=update"
    try:
        resp = requests.get(url, headers={"User-Agent": "horse-racing-ai research bot"}, timeout=10)
        data = resp.json()
        status = data.get("status", "")
        odds = data.get("data", {}).get("odds", {})
        if odds and odds.get("1"):
            result = odds
            if status == "result":
                result["_updated"] = data["data"].get("official_datetime", "")
                result["_status"] = "確定"
            else:
                result["_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                result["_status"] = "発売中（変動あり）"
            return result
    except:
        pass
    return None

@st.cache_data
def load_backtest():
    path = Path("outputs/backtest_results.csv")
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


# --- 補正ロジック ---
def _distance_category(distance):
    """距離をカテゴリに分類"""
    if distance <= 1400:
        return "短距離"
    elif distance <= 1800:
        return "マイル〜中距離"
    else:
        return "中長距離"


def analyze_results(races, preds, odds_map):
    """入力済みレース結果から場×芝ダート×距離別のバイアスを解析する。

    3層構造:
      1. 場×芝ダート（例: 阪神ダート）
      2. 場×芝ダート×距離カテゴリ（例: 阪神ダート短距離）
      3. 全体（フォールバック）
    """
    bias = {"_all": _empty_bias(), "_by_key": {}, "races_analyzed": 0}

    if not st.session_state.results:
        return bias

    # 各レースの結果を場×芝ダートで分類して集計
    groups = {}  # key -> [レース結果リスト]

    for race_id, result in st.session_state.results.items():
        race = next((r for r in races if r["race_id"] == race_id), None)
        if not race:
            continue

        race_preds = preds[preds["race_id"] == race_id]
        if race_preds.empty:
            continue

        place = race.get("place_name", "")
        surface = race.get("surface", "")
        distance = race.get("distance", 0)
        dist_cat = _distance_category(distance)

        entry = _analyze_single_race(race, result, race_preds, odds_map)
        if not entry:
            continue

        # 3つのキーに分類
        key_place_surface = f"{place}_{surface}"
        key_detail = f"{place}_{surface}_{dist_cat}"

        for key in [key_place_surface, key_detail, "_all"]:
            if key not in groups:
                groups[key] = []
            groups[key].append(entry)

    # 各グループでバイアスを集計
    for key, entries in groups.items():
        b = _aggregate_bias(entries)
        if key == "_all":
            bias["_all"] = b
        else:
            bias["_by_key"][key] = b

    bias["races_analyzed"] = len(st.session_state.results)
    return bias


def _empty_bias():
    return {"inner_advantage": 0.0, "upset_tendency": 0.0, "model_accuracy": 0.0, "races": 0}


def _analyze_single_race(race, result, race_preds, odds_map):
    """1レース分のバイアスデータを抽出"""
    winner_num = result.get("1st", 0)
    entries = race.get("entries", [])
    winner_entry = next((e for e in entries if e.get("number") == winner_num), None)
    if not winner_entry:
        return None

    bracket = winner_entry.get("bracket", 4)
    inner = 1 if bracket <= 3 else (-1 if bracket >= 6 else 0)

    # AI精度
    ai_top3 = race_preds.sort_values("win_prob", ascending=False).head(3)["number"].values
    top3_nums = [result.get("1st", 0), result.get("2nd", 0), result.get("3rd", 0)]
    ai_hits = sum(1 for n in ai_top3 if n in top3_nums)

    # 波乱度
    odds_data = odds_map.get(race.get("race_id", ""), {})
    win_odds = odds_data.get("1", {})
    winner_str = str(winner_num).zfill(2)
    winner_pop = int(win_odds[winner_str][2]) if winner_str in win_odds else 0
    upset = 1 if winner_pop >= 6 else 0

    return {"inner": inner, "ai_hits": ai_hits, "upset": upset}


def _aggregate_bias(entries):
    """レース結果リストからバイアスを集計"""
    n = len(entries)
    if n == 0:
        return _empty_bias()
    inner_sum = sum(e["inner"] for e in entries)
    ai_hits_sum = sum(e["ai_hits"] for e in entries)
    upset_sum = sum(e["upset"] for e in entries)

    return {
        "inner_advantage": inner_sum / n * 0.3,
        "upset_tendency": (upset_sum / n - 0.2) * 0.5,
        "model_accuracy": (ai_hits_sum / (n * 3)) - 0.33,
        "races": n,
    }


def _get_bias_for_race(bias, race):
    """レースに最も合うバイアスを3層から選択する"""
    place = race.get("place_name", "")
    surface = race.get("surface", "")
    distance = race.get("distance", 0)
    dist_cat = _distance_category(distance)

    # 1. 場×芝ダート×距離カテゴリ（最も詳細）
    key_detail = f"{place}_{surface}_{dist_cat}"
    if key_detail in bias.get("_by_key", {}) and bias["_by_key"][key_detail]["races"] >= 2:
        return bias["_by_key"][key_detail], key_detail

    # 2. 場×芝ダート
    key_surface = f"{place}_{surface}"
    if key_surface in bias.get("_by_key", {}) and bias["_by_key"][key_surface]["races"] >= 2:
        return bias["_by_key"][key_surface], key_surface

    # 3. 全体（フォールバック）
    return bias.get("_all", _empty_bias()), "全体"


def apply_correction(race_preds, race, bias):
    """レースに合ったバイアスで予測を補正する"""
    if bias.get("races_analyzed", 0) == 0:
        return race_preds

    b, _ = _get_bias_for_race(bias, race)
    if b["races"] == 0:
        return race_preds

    corrected = race_preds.copy()
    entries = race.get("entries", [])

    for idx, row in corrected.iterrows():
        num = int(row["number"])
        entry = next((e for e in entries if e.get("number") == num), None)
        if not entry:
            continue

        adj = 1.0

        # 内枠バイアス
        bracket = entry.get("bracket", 4)
        if bracket <= 3:
            adj += b["inner_advantage"]
        elif bracket >= 6:
            adj -= b["inner_advantage"]

        # 波乱傾向
        prob = row["win_prob"]
        if prob < 0.05:
            adj += b["upset_tendency"]
        elif prob > 0.15:
            adj -= b["upset_tendency"] * 0.5

        corrected.at[idx, "win_prob"] = max(row["win_prob"] * adj, 0.005)
        corrected.at[idx, "pred_top3_prob"] = max(row["pred_top3_prob"] * adj, 0.01)

    # 正規化
    total_prob = corrected["win_prob"].sum()
    if total_prob > 0:
        corrected["win_prob"] = corrected["win_prob"] / total_prob

    return corrected


def calc_confidence(race_preds):
    """レースの自信度を計算する"""
    probs = race_preds["win_prob"].values
    if len(probs) == 0:
        return 0
    sorted_probs = sorted(probs, reverse=True)
    top1 = sorted_probs[0]
    top2 = sorted_probs[1] if len(sorted_probs) > 1 else 0
    concentration = min((top1 - top2) / max(top1, 0.01) * 100, 40)
    top_strength = min(top1 * 200, 30)
    n = len(probs)
    field_score = max(0, (18 - n) * 2)
    confidence = int(concentration + top_strength + field_score)
    return min(max(confidence, 10), 95)


def get_comment(row, race_preds, odds_val, pop):
    """馬ごとのコメントを生成"""
    comments = []
    sorted_df = race_preds.sort_values("win_prob", ascending=False)
    my_rank = list(sorted_df["number"].values).index(row["number"]) + 1 if row["number"] in sorted_df["number"].values else 99

    ev = row["win_prob"] * odds_val if odds_val > 0 else 0

    if my_rank <= 3 and pop <= 3:
        comments.append("実力通りの評価")
    elif my_rank <= 3 and pop > 5:
        comments.append(f"AI{my_rank}位評価だが{pop}番人気で妙味あり")
    elif my_rank > 8 and pop <= 3:
        comments.append(f"AI評価低い({my_rank}位)が{pop}番人気。過剰人気の可能性")

    if ev >= 3.0:
        comments.append(f"EV{ev:.1f}で期待値大")
    elif ev >= 1.5:
        comments.append(f"EV{ev:.1f}")
    elif ev < 0.5 and odds_val > 0:
        comments.append("オッズに対して割高")

    top3 = row.get("pred_top3_prob", 0)
    if top3 >= 0.4:
        comments.append("複勝率高く軸向き")
    elif top3 < 0.1:
        comments.append("3着以内も厳しい")

    if not comments:
        comments.append("-")
    return "。".join(comments)


def get_mark(row, race_preds):
    """印を決定"""
    sorted_df = race_preds.sort_values("win_prob", ascending=False)
    rank = list(sorted_df["number"].values).index(row["number"]) + 1 if row["number"] in sorted_df["number"].values else 99
    if rank == 1: return "◎"
    elif rank == 2: return "○"
    elif rank == 3: return "▲"
    elif rank <= 5: return "△"
    else: return ""


# --- メイン ---
def main():
    preds_raw = load_predictions()
    preds_raw["race_id"] = preds_raw["race_id"].astype(str)
    races = load_target_races()
    horses = load_horses()
    all_race_ids = [r["race_id"] for r in races]
    odds_map = load_odds_from_cache(all_race_ids)
    backtest = load_backtest()

    # ライブオッズをセッションに保持
    if "live_odds" not in st.session_state:
        st.session_state.live_odds = {}

    # 自動リフレッシュ時にレース結果を自動取得
    if auto_refresh_count > 0:
        from src.scraping.live_results import check_all_results
        from datetime import datetime
        new = check_all_results(all_race_ids, st.session_state.results)
        if new:
            st.session_state.results.update(new)
            st.session_state.last_fetch_time = datetime.now().strftime("%H:%M:%S")

    # バイアス解析
    bias = analyze_results(races, preds_raw, odds_map)
    st.session_state.bias = bias

    # --- ヘッダー: レース選択 ---
    st.markdown("## 🏇 競馬予想AI")

    # 1段目: 開催日 + 開催場所
    dates = sorted(set(r.get("date", "") for r in races))
    date_labels = {d: "4/18(土)" if "0418" in str(d) else "4/19(日)" for d in dates}

    hdr1, hdr2, hdr3 = st.columns([1, 1, 2])
    with hdr1:
        selected_date = st.selectbox("開催日", dates, format_func=lambda d: date_labels.get(d, d))

    day_races = [r for r in races if str(r.get("date", "")) == str(selected_date)]

    # 開催場所
    places = sorted(set(r.get("place_name", "") for r in day_races))
    with hdr2:
        selected_place = st.selectbox("開催場所", places)

    place_races = [r for r in day_races if r.get("place_name", "") == selected_place]

    # 2段目: レース番号
    with hdr3:
        race_options = {}
        for r in sorted(place_races, key=lambda x: x["race_id"]):
            rid = r["race_id"]
            r_num = rid[-2:]
            done = "✅" if rid in st.session_state.results else ""
            label = f"{done}{r_num}R {r.get('race_name','')} ({r.get('surface','')}{r.get('distance','')}m)"
            race_options[label] = r
        selected_label = st.selectbox("レース", list(race_options.keys()))

    selected_race = race_options[selected_label]

    st.markdown("---")

    # --- サイドバー ---
    st.sidebar.title("📊 情報")

    # 本日収支
    st.sidebar.subheader("💰 本日の収支")
    inv = st.session_state.bets_today["invested"]
    ret = st.session_state.bets_today["returned"]
    roi = ret / inv * 100 if inv > 0 else 0
    st.sidebar.metric("投資", f"{inv:,}円")
    st.sidebar.metric("回収", f"{ret:,}円", delta=f"{ret-inv:+,}円")
    if inv > 0:
        color = "green" if roi >= 100 else "red"
        st.sidebar.markdown(f"回収率: :{color}[**{roi:.0f}%**]")

    # 当日バイアス（場×芝ダート別）
    if bias.get("races_analyzed", 0) > 0:
        st.sidebar.markdown("---")
        st.sidebar.subheader("📡 当日バイアス")
        st.sidebar.caption(f"{bias['races_analyzed']}R分析済み")

        for key, b in sorted(bias.get("_by_key", {}).items()):
            if "_" not in key or b["races"] < 1:
                continue
            # 場×芝ダートレベルのみ表示（距離カテゴリは詳細すぎる）
            parts = key.split("_")
            if len(parts) == 2:
                label = f"{parts[0]} {parts[1]}"
                ia = b["inner_advantage"]
                ut = b["upset_tendency"]
                枠 = "内有利" if ia > 0.05 else "外有利" if ia < -0.05 else "フラット"
                傾向 = "荒" if ut > 0.05 else "堅" if ut < -0.05 else "平常"
                st.sidebar.write(f"**{label}** ({b['races']}R): 枠={枠} / {傾向}")

    # バックテスト
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 バックテスト")
    if not backtest.empty:
        for _, row in backtest.iterrows():
            roi_bt = row.get("roi", 0)
            color = "green" if roi_bt > 100 else "red"
            st.sidebar.markdown(f"**{row.get('pattern','')}**: :{color}[{roi_bt:.0f}%]")

    # --- メインエリア ---
    race_id = selected_race["race_id"]
    race_preds = preds_raw[preds_raw["race_id"] == race_id].copy()

    if race_preds.empty:
        st.warning("このレースの予測データがありません")
        return

    # ライブオッズがあればそちらを使う
    if race_id in st.session_state.live_odds:
        odds_map[race_id] = st.session_state.live_odds[race_id]

    # 補正適用
    race_preds = apply_correction(race_preds, selected_race, bias)

    # ヘッダ
    confidence = calc_confidence(race_preds)

    # オッズ更新時刻
    odds_updated = odds_map.get(race_id, {}).get("_updated", "")

    col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
    with col1:
        st.title(f"{selected_race.get('place_name','')} {selected_race.get('race_name','')}")
        st.caption(f"{selected_race.get('surface','')}{selected_race.get('distance','')}m / {selected_race.get('class','')}")
    with col2:
        st.metric("自信度", f"{confidence}%")
    with col3:
        st.metric("出走頭数", f"{len(race_preds)}頭")
    with col4:
        if bias.get("races_analyzed", 0) > 0:
            b_applied, b_key = _get_bias_for_race(bias, selected_race)
            st.metric("補正", f"{b_applied['races']}R")
            st.caption(f"({b_key})")
        else:
            st.metric("補正", "なし")

    # オッズ更新ボタン
    odds_col1, odds_col2 = st.columns([1, 3])
    with odds_col1:
        if st.button("🔄 オッズ更新", key=f"odds_{race_id}", type="primary"):
            with st.spinner("最新オッズを取得中..."):
                live = fetch_live_odds(race_id)
                if live:
                    st.session_state.live_odds[race_id] = live
                    odds_map[race_id] = live
                    st.success(f"更新完了: {live.get('_updated', '')}")
                    st.rerun()
                else:
                    st.warning("オッズ未発売またはエラー")
    with odds_col2:
        if odds_updated:
            st.caption(f"オッズ更新時刻: {odds_updated}")
        if race_id in st.session_state.live_odds:
            odds_status = st.session_state.live_odds[race_id].get("_status", "")
            st.caption(f"📡 ライブオッズ使用中 ({odds_status})")

    if race_id in st.session_state.results:
        res = st.session_state.results[race_id]
        st.success(f"結果入力済み: 1着={res.get('1st','-')}番, 2着={res.get('2nd','-')}番, 3着={res.get('3rd','-')}番")

    st.markdown("---")

    # タブ
    tab1, tab2, tab3, tab4 = st.tabs(["📋 予測一覧", "🎯 推奨買い目", "📈 分析", "✏️ 結果入力"])

    with tab1:
        odds_data = odds_map.get(race_id, {})
        win_odds = odds_data.get("1", {})

        table_rows = []
        for _, row in race_preds.sort_values("win_prob", ascending=False).iterrows():
            num = int(row["number"])
            num_str = str(num).zfill(2)
            odds_val = float(win_odds[num_str][0]) if num_str in win_odds else 0
            pop = int(win_odds[num_str][2]) if num_str in win_odds else 0
            ev = row["win_prob"] * odds_val if odds_val > 0 else 0

            hid = ""
            jockey = ""
            for e in selected_race.get("entries", []):
                if e["number"] == num:
                    hid = e.get("horse_id", "")
                    jockey = e.get("jockey_name", "")
                    break
            sire = horses.get(hid, {}).get("sire", "-")
            dam_sire = horses.get(hid, {}).get("dam_sire", "-")
            mark = get_mark(row, race_preds)
            comment = get_comment(row, race_preds, odds_val, pop)

            table_rows.append({
                "印": mark, "馬番": num, "馬名": row["horse_name"],
                "騎手": jockey, "父": sire, "母父": dam_sire,
                "人気": f"{pop}番" if pop > 0 else "-",
                "オッズ": f"{odds_val:.1f}" if odds_val > 0 else "-",
                "勝率": f"{row['win_prob']:.1%}",
                "複勝率": f"{row['pred_top3_prob']:.1%}",
                "EV": f"{ev:.2f}" if ev > 0 else "-",
                "コメント": comment,
            })

        st.dataframe(
            pd.DataFrame(table_rows),
            use_container_width=True, hide_index=True,
            column_config={
                "印": st.column_config.TextColumn(width="small"),
                "馬番": st.column_config.NumberColumn(width="small"),
                "コメント": st.column_config.TextColumn(width="large"),
            }
        )

    with tab2:
        st.subheader("推奨買い目")

        from src.betting.optimizer import generate_candidates, optimize_bets

        odds_data_tab2 = odds_map.get(race_id, {})
        if odds_data_tab2:
            candidates = generate_candidates(race_preds, odds_data_tab2)

            col_b, col_c = st.columns(2)

            for col, budget, label in [(col_b, 1000, "B"), (col_c, 3000, "C")]:
                with col:
                    st.markdown(f"### パターン{label}（1R上限{budget:,}円）")
                    result = optimize_bets(candidates, budget)
                    bets = result["bets"]

                    if bets:
                        bet_df = pd.DataFrame([{
                            "券種": b["bet_type"],
                            "買い目": b["numbers"],
                            "馬名": b["reason"],
                            "金額": f"{b['amount']:,}円",
                            "的中率": f"{b['probability']:.1%}",
                            "的中時": f"{b['payout']:,.0f}円",
                            "EV": f"{b['ev']:.2f}",
                        } for b in bets])
                        st.dataframe(bet_df, hide_index=True, use_container_width=True)

                        st.caption(f"合計投資: **{result['total_investment']:,}円**")
                        st.caption(f"少なくとも1つ的中する確率: **{result['any_hit_probability']:.0%}**")
                        st.caption(f"的中時回収レンジ: {result['min_payout']:,.0f}〜{result['max_payout']:,.0f}円")
                    else:
                        st.write("推奨馬券なし")
        else:
            st.warning("オッズデータが読み込めません")

    with tab3:
        st.subheader("予測分析")
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("#### 勝率分布")
            chart_data = race_preds[["horse_name", "win_prob"]].sort_values("win_prob", ascending=True)
            st.bar_chart(chart_data.set_index("horse_name"), horizontal=True)

        with col_b:
            st.markdown("#### AI評価 vs 市場人気")
            odds_data = odds_map.get(race_id, {})
            win_odds = odds_data.get("1", {})
            compare_rows = []
            for _, row in race_preds.iterrows():
                num_str = str(int(row["number"])).zfill(2)
                if num_str in win_odds:
                    odds_val = float(win_odds[num_str][0])
                    market_prob = 1 / odds_val if odds_val > 0 else 0
                    compare_rows.append({
                        "馬名": row["horse_name"],
                        "AI勝率": row["win_prob"],
                        "市場確率": market_prob,
                    })
            if compare_rows:
                st.bar_chart(pd.DataFrame(compare_rows).set_index("馬名"))

        bt_img = Path("outputs/backtest_daily_roi.png")
        if bt_img.exists():
            st.markdown("#### バックテスト 日別回収率")
            st.image(str(bt_img))

    with tab4:
        st.subheader("✏️ レース結果入力")

        # 自動取得セクション
        from src.scraping.live_results import check_all_results
        from datetime import datetime

        auto_col1, auto_col2, auto_col3 = st.columns([1, 1, 2])
        with auto_col1:
            if st.button("📡 全レース結果を取得", type="primary", use_container_width=True):
                day_race_ids = [r["race_id"] for r in day_races]
                with st.spinner("結果を取得中..."):
                    new = check_all_results(day_race_ids, st.session_state.results)
                    if new:
                        st.session_state.results.update(new)
                        st.session_state.last_fetch_time = datetime.now().strftime("%H:%M:%S")
                        st.success(f"{len(new)}レースの結果を取得しました")
                        st.rerun()
                    else:
                        st.session_state.last_fetch_time = datetime.now().strftime("%H:%M:%S")
                        st.info("新しい確定結果はありません")
        with auto_col2:
            confirmed = sum(1 for r in day_races if r["race_id"] in st.session_state.results)
            st.metric("確定済み", f"{confirmed}/{len(day_races)}R")
        with auto_col3:
            if st.session_state.last_fetch_time:
                st.caption(f"最終取得: {st.session_state.last_fetch_time}")
            st.caption("レース確定後にボタンを押すと自動取得します")

        st.markdown("---")
        st.caption("手動入力も可能です")

        entries = selected_race.get("entries", [])
        horse_options = {0: "（未選択）"}
        for e in sorted(entries, key=lambda x: x.get("number", 0)):
            horse_options[e["number"]] = f"{e['number']}番 {e.get('horse_name', '')}"

        existing = st.session_state.results.get(race_id, {})

        col1, col2, col3 = st.columns(3)
        with col1:
            first = st.selectbox("🥇 1着", options=list(horse_options.keys()),
                                 format_func=lambda x: horse_options[x],
                                 index=list(horse_options.keys()).index(existing.get("1st", 0)) if existing.get("1st", 0) in horse_options else 0,
                                 key=f"1st_{race_id}")
        with col2:
            second = st.selectbox("🥈 2着", options=list(horse_options.keys()),
                                  format_func=lambda x: horse_options[x],
                                  index=list(horse_options.keys()).index(existing.get("2nd", 0)) if existing.get("2nd", 0) in horse_options else 0,
                                  key=f"2nd_{race_id}")
        with col3:
            third = st.selectbox("🥉 3着", options=list(horse_options.keys()),
                                 format_func=lambda x: horse_options[x],
                                 index=list(horse_options.keys()).index(existing.get("3rd", 0)) if existing.get("3rd", 0) in horse_options else 0,
                                 key=f"3rd_{race_id}")

        col_save, col_clear = st.columns(2)
        with col_save:
            if st.button("💾 結果を保存", key=f"save_{race_id}", type="primary", use_container_width=True):
                if first > 0 and second > 0 and third > 0:
                    st.session_state.results[race_id] = {"1st": first, "2nd": second, "3rd": third}
                    st.success(f"保存しました！（{len(st.session_state.results)}レース分のバイアス解析に反映）")
                    st.rerun()
                else:
                    st.error("1着〜3着を全て選択してください")
        with col_clear:
            if st.button("🗑️ クリア", key=f"clear_{race_id}", use_container_width=True):
                if race_id in st.session_state.results:
                    del st.session_state.results[race_id]
                    st.rerun()

        # 入力済みレース一覧
        if st.session_state.results:
            st.markdown("---")
            st.markdown("#### 入力済みレース")
            for rid, res in st.session_state.results.items():
                race_info = next((r for r in races if r["race_id"] == rid), {})
                place = race_info.get("place_name", "")
                name = race_info.get("race_name", "")
                r_num = rid[-2:]
                st.write(f"✅ {place} {r_num}R {name}: {res['1st']}→{res['2nd']}→{res['3rd']}")

            st.markdown("---")
            st.markdown("#### 解析結果")
            b = st.session_state.bias
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                ia = b["inner_advantage"]
                st.metric("枠バイアス", "内枠有利" if ia > 0.05 else "外枠有利" if ia < -0.05 else "フラット",
                          delta=f"{ia:+.2f}")
            with col2:
                ut = b["upset_tendency"]
                st.metric("波乱度", "荒れ" if ut > 0.05 else "堅い" if ut < -0.05 else "平常",
                          delta=f"{ut:+.2f}")
            with col3:
                ma = b["model_accuracy"]
                st.metric("AI精度", "好調" if ma > 0.05 else "不調" if ma < -0.05 else "通常",
                          delta=f"{ma:+.2f}")
            with col4:
                st.metric("分析済み", f"{b['races_analyzed']}R")


if __name__ == "__main__":
    main()
