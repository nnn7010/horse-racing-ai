"""競馬予想AI ダッシュボード"""

import json
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(page_title="競馬予想AI", page_icon="🏇", layout="wide")

try:
    from streamlit_autorefresh import st_autorefresh
    auto_refresh_count = st_autorefresh(interval=60000, limit=None, key="auto_refresh")
except ImportError:
    auto_refresh_count = 0

# --- セッションステート初期化 ---
if "results" not in st.session_state:
    st.session_state.results = {}
if "last_fetch_time" not in st.session_state:
    st.session_state.last_fetch_time = ""
if "live_odds" not in st.session_state:
    st.session_state.live_odds = {}
if "selected_race_id" not in st.session_state:
    st.session_state.selected_race_id = None
if "bets_today" not in st.session_state:
    st.session_state.bets_today = {"invested": 0, "returned": 0}


# --- データ読み込み ---
@st.cache_data(ttl=30)
def load_today_predictions():
    """today_predictions.json から予測を読み込む"""
    path = Path("data/raw/today_predictions.json")
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def predictions_to_df(pred_data):
    """today_predictions.json の horses リストを DataFrame に変換"""
    rows = []
    for race in pred_data.get("races", []):
        rid = race["race_id"]
        for h in race.get("horses", []):
            rows.append({
                "race_id": rid,
                "number": h["number"],
                "horse_name": h["horse_name"],
                "jockey_name": h.get("jockey_name", ""),
                "impost": h.get("impost", 0),
                "win_odds": h.get("win_odds", 0),
                "win_prob": h["win_prob"],
                "pred_top3_prob": h.get("place_prob", 0),
            })
    return pd.DataFrame(rows)


def fetch_live_odds(race_id):
    import requests
    url = f"https://race.netkeiba.com/api/api_get_jra_odds.html?race_id={race_id}&type=1&action=update"
    try:
        resp = requests.get(url, headers={"User-Agent": "horse-racing-ai research bot"}, timeout=10)
        data = resp.json()
        odds = data.get("data", {}).get("odds", {})
        if odds and odds.get("1"):
            result = odds
            if data.get("status") == "result":
                result["_updated"] = data["data"].get("official_datetime", "")
                result["_status"] = "確定"
            else:
                result["_updated"] = datetime.now().strftime("%H:%M:%S")
                result["_status"] = "発売中"
            return result
    except:
        pass
    return None


def get_mark(rank):
    return {1: "◎", 2: "○", 3: "▲", 4: "△", 5: "△"}.get(rank, "")


def calc_upset_score(horses):
    if not horses: return 50
    probs = [h["win_prob"] for h in horses]
    n = len(probs)
    top1 = probs[0]
    top2 = probs[1] if n > 1 else 0
    score = 0
    score += max(0, min(30, (0.20 - top1) * 200))
    score += max(0, min(25, (0.05 - (top1 - top2)) * 500))
    score += max(0, min(20, (n - 8) * 2))
    return min(max(int(score), 5), 95)


# --- メイン ---
def main():
    pred_data = load_today_predictions()

    st.markdown("## 🏇 競馬予想AI - 4/25(土)")

    if pred_data is None:
        st.warning("⏳ 予測データを生成中... しばらくお待ちください")
        if st.button("🔄 再読み込み"):
            st.cache_data.clear()
            st.rerun()
        return

    races = pred_data.get("races", [])
    preds_df = predictions_to_df(pred_data)

    # 自動リフレッシュ時に結果取得
    if auto_refresh_count > 0:
        try:
            from src.scraping.live_results import check_all_results
            race_ids = [r["race_id"] for r in races]
            new = check_all_results(race_ids, st.session_state.results)
            if new:
                st.session_state.results.update(new)
                st.session_state.last_fetch_time = datetime.now().strftime("%H:%M:%S")
        except:
            pass

    # --- 会場タブ ---
    venues = []
    seen = set()
    for r in races:
        p = r["place_name"]
        if p not in seen:
            venues.append(p)
            seen.add(p)

    venue_tabs = st.tabs(["📋 全レース一覧"] + [f"🏟 {v}" for v in venues])

    # ===== タブ0: 全レース一覧 =====
    with venue_tabs[0]:
        st.markdown("### 本日の全レース予想")

        cols = st.columns(len(venues))
        for col, venue in zip(cols, venues):
            with col:
                st.markdown(f"#### {venue}")
                venue_races = [r for r in races if r["place_name"] == venue]
                for race in sorted(venue_races, key=lambda x: x["race_id"]):
                    rid = race["race_id"]
                    rnum = int(rid[-2:])
                    done = "✅" if rid in st.session_state.results else ""
                    top3 = race["horses"][:3]
                    upset = calc_upset_score(race["horses"])
                    upset_icon = "🔥" if upset >= 70 else "⚠️" if upset >= 50 else ""

                    with st.expander(
                        f"{done}{rnum}R {race['race_name']} {race['surface']}{race['distance']}m "
                        f"({race['track_condition']}) {upset_icon}",
                        expanded=False
                    ):
                        # 本命・対抗・三番手
                        for i, h in enumerate(top3, 1):
                            mark = get_mark(i)
                            col_a, col_b, col_c = st.columns([1, 3, 2])
                            with col_a:
                                st.markdown(f"**{mark} {h['number']}番**")
                            with col_b:
                                st.write(h["horse_name"])
                                st.caption(h["jockey_name"])
                            with col_c:
                                st.write(f"{h['win_prob']:.0%} / {h['place_prob']:.0%}")
                                if h["win_odds"] > 0:
                                    ev = h["win_prob"] * h["win_odds"]
                                    color = "🟢" if ev >= 1.10 else "🔴"
                                    st.caption(f"{color} EV {ev:.2f} ({h['win_odds']}倍)")

                        # 三連複本命
                        if race.get("trio_top5"):
                            t = race["trio_top5"][0]
                            nums = t["combo"]
                            st.caption(f"三連複本命: {nums[0]}-{nums[1]}-{nums[2]} ({t['prob']:.2%})")

                        if st.button("詳細を見る", key=f"detail_{rid}", use_container_width=True):
                            st.session_state.selected_race_id = rid
                            st.rerun()

    # ===== タブ1〜3: 会場別詳細 =====
    for tab_idx, (venue_tab, venue) in enumerate(zip(venue_tabs[1:], venues), 1):
        with venue_tab:
            venue_races = sorted([r for r in races if r["place_name"] == venue], key=lambda x: x["race_id"])

            # レース選択ボタン
            race_cols = st.columns(min(len(venue_races), 6))
            for i, race in enumerate(venue_races):
                rid = race["race_id"]
                rnum = int(rid[-2:])
                done = "✅" if rid in st.session_state.results else ""
                with race_cols[i % 6]:
                    is_sel = (st.session_state.selected_race_id == rid)
                    if st.button(
                        f"{done}{rnum}R\n{race['race_name'][:6]}",
                        key=f"btn_{rid}",
                        use_container_width=True,
                        type="primary" if is_sel else "secondary",
                    ):
                        st.session_state.selected_race_id = rid
                        st.rerun()

            # 選択されたレースがこの会場のものなら詳細表示
            sel_id = st.session_state.selected_race_id
            sel_race = next((r for r in venue_races if r["race_id"] == sel_id), None)
            if sel_race is None:
                sel_race = venue_races[0] if venue_races else None
                if sel_race:
                    st.session_state.selected_race_id = sel_race["race_id"]

            if not sel_race:
                continue

            _show_race_detail(sel_race, preds_df, races)


def _show_race_detail(race, preds_df, all_races):
    """レース詳細表示"""
    rid = race["race_id"]
    rnum = int(rid[-2:])
    horses = race["horses"]
    upset = calc_upset_score(horses)
    upset_label = "大荒れ" if upset >= 70 else "荒れ" if upset >= 50 else "やや荒れ" if upset >= 30 else "堅い"

    # ライブオッズ取得
    odds_data = st.session_state.live_odds.get(rid, {})
    win_odds_api = odds_data.get("1", {})

    st.markdown("---")
    h1, h2, h3, h4 = st.columns([3, 1, 1, 1])
    with h1:
        st.subheader(f"{race['place_name']} {rnum}R {race['race_name']}")
        st.caption(f"{race['surface']}{race['distance']}m / 馬場:{race['track_condition']} / {race.get('start_time','')}発走 / {race['n_horses']}頭")
    with h2:
        st.metric("荒れ度", f"{upset}%", upset_label)
    with h3:
        o_col1, o_col2 = st.columns(2)
        with o_col1:
            if st.button("🔄 オッズ", key=f"odds_{rid}"):
                with st.spinner("取得中..."):
                    live = fetch_live_odds(rid)
                    if live:
                        st.session_state.live_odds[rid] = live
                        st.rerun()
                    else:
                        st.warning("未発売")
        with o_col2:
            if odds_data.get("_status"):
                st.caption(odds_data["_status"])
    with h4:
        if rid in st.session_state.results:
            res = st.session_state.results[rid]
            st.success(f"✅ {res.get('1st','-')}→{res.get('2nd','-')}→{res.get('3rd','-')}")

    # タブ
    t1, t2, t3, t4 = st.tabs(["📋 予測", "🎯 買い目", "🏆 三連系", "✏️ 結果入力"])

    with t1:
        rows = []
        for i, h in enumerate(horses, 1):
            mark = get_mark(i)
            num_str = str(h["number"]).zfill(2)
            live_odds_val = float(win_odds_api[num_str][0]) if num_str in win_odds_api else h["win_odds"]
            pop = int(win_odds_api[num_str][2]) if num_str in win_odds_api else 0
            ev = h["win_prob"] * live_odds_val if live_odds_val > 0 else 0
            min_odds = (1.0 / h["win_prob"]) * 1.1 if h["win_prob"] > 0 else 999

            buy = ""
            if live_odds_val > 0:
                buy = "✅ 買い" if ev >= 1.10 else "❌ 見送"
            elif h["win_prob"] > 0:
                buy = f"{min_odds:.1f}倍以上"

            rows.append({
                "印": mark, "馬番": h["number"], "馬名": h["horse_name"],
                "騎手": h["jockey_name"], "斤量": h["impost"],
                "人気": f"{pop}" if pop > 0 else "-",
                "単勝": f"{live_odds_val:.1f}" if live_odds_val > 0 else "-",
                "勝率": f"{h['win_prob']:.1%}",
                "複勝率": f"{h['place_prob']:.1%}",
                "EV": f"{ev:.2f}" if ev > 0 else "-",
                "判定": buy,
            })

        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True, hide_index=True,
            column_config={
                "印": st.column_config.TextColumn(width="small"),
                "馬番": st.column_config.NumberColumn(width="small"),
                "判定": st.column_config.TextColumn(width="medium"),
            }
        )

    with t2:
        race_preds = preds_df[preds_df["race_id"] == rid].copy()
        if not race_preds.empty and win_odds_api:
            from src.betting.optimizer import build_recommendations
            all_odds = {"win": {k: float(v[0]) for k, v in win_odds_api.items() if v[0]}}
            result = build_recommendations(race_preds, all_odds, budget=3000)
            groups = result.get("ticket_groups", [])
            if groups:
                for group in groups:
                    bt = group["bet_type"]
                    n_bets = group.get("n_bets", 0)
                    total_prob = group["total_prob"]
                    min_c_odds = group.get("min_composite_odds", 0)
                    with st.container(border=True):
                        st.markdown(f"**🎯 {bt}** {n_bets}点  的中率:{total_prob:.1%}  合成オッズ:{min_c_odds:.1f}倍以上")
                        for p in group.get("picks", [])[:5]:
                            c1, c2, c3 = st.columns([2, 2, 2])
                            with c1: st.write(f"**{p['numbers']}**")
                            with c2: st.caption(p.get("names",""))
                            with c3:
                                cur = p.get("odds", 0)
                                min_i = (1.0/p["prob"])*1.1 if p.get("prob",0) > 0 else 999
                                if cur > 0:
                                    if cur >= min_i: st.success(f"{cur:.1f}倍 → 買い")
                                    else: st.error(f"{cur:.1f}倍 → 見送")
                                else:
                                    st.info(f"{min_i:.1f}倍以上なら買い")
                st.caption(f"いずれか的中確率: {result['any_hit_probability']:.0%}")
            else:
                st.info("推奨買い目なし（オッズが割に合わない）")
        elif not win_odds_api:
            # オッズなしでも確率ベースで表示
            st.info("💡 「🔄 オッズ」ボタンでオッズを取得すると、EV判定が可能になります")
            st.markdown("**確率ベースの推奨（EV計算なし）**")
            top3 = horses[:3]
            for i, h in enumerate(top3, 1):
                mark = get_mark(i)
                min_odds = (1.0/h["win_prob"])*1.1 if h["win_prob"] > 0 else 999
                st.write(f"{mark} **{h['number']}番 {h['horse_name']}**: 勝率{h['win_prob']:.1%} → {min_odds:.1f}倍以上なら単勝買い")

            # 三連複
            if race.get("trio_top5"):
                st.markdown("---")
                st.markdown("**三連複本命**")
                for t in race["trio_top5"][:3]:
                    nums = t["combo"]
                    names = [next((h["horse_name"] for h in horses if h["number"]==n), str(n)) for n in nums]
                    st.write(f"  {nums[0]}-{nums[1]}-{nums[2]}  ({names[0]}/{names[1]}/{names[2]})  {t['prob']:.2%}")

    with t3:
        st.markdown("#### 三連複 上位5組")
        if race.get("trio_top5"):
            rows3 = []
            for t in race["trio_top5"]:
                nums = t["combo"]
                names = "/".join(next((h["horse_name"] for h in horses if h["number"]==n), str(n)) for n in nums)
                rows3.append({"組み合わせ": f"{nums[0]}-{nums[1]}-{nums[2]}", "馬名": names, "確率": f"{t['prob']:.3%}"})
            st.dataframe(pd.DataFrame(rows3), hide_index=True, use_container_width=True)

        st.markdown("#### 三連単 上位5組")
        if race.get("trifecta_top5"):
            rows4 = []
            for t in race["trifecta_top5"]:
                nums = t["combo"]
                names = "→".join(next((h["horse_name"] for h in horses if h["number"]==n), str(n)) for n in nums)
                rows4.append({"組み合わせ": f"{nums[0]}→{nums[1]}→{nums[2]}", "馬名": names, "確率": f"{t['prob']:.3%}"})
            st.dataframe(pd.DataFrame(rows4), hide_index=True, use_container_width=True)

    with t4:
        auto_c1, auto_c2 = st.columns([1, 2])
        with auto_c1:
            if st.button("📡 結果を自動取得", type="primary", use_container_width=True, key=f"auto_{rid}"):
                try:
                    from src.scraping.live_results import check_all_results
                    new = check_all_results([rid], st.session_state.results)
                    if new:
                        st.session_state.results.update(new)
                        st.success("取得完了")
                        st.rerun()
                    else:
                        st.info("結果未確定")
                except Exception as e:
                    st.error(str(e))
        with auto_c2:
            if st.session_state.last_fetch_time:
                st.caption(f"最終取得: {st.session_state.last_fetch_time}")

        horse_options = {0: "（未選択）"}
        for h in sorted(horses, key=lambda x: x["number"]):
            horse_options[h["number"]] = f"{h['number']}番 {h['horse_name']}"
        existing = st.session_state.results.get(rid, {})

        c1, c2, c3 = st.columns(3)
        with c1:
            first = st.selectbox("🥇 1着", list(horse_options.keys()),
                                 format_func=lambda x: horse_options[x],
                                 index=list(horse_options.keys()).index(existing.get("1st",0)) if existing.get("1st",0) in horse_options else 0,
                                 key=f"1st_{rid}")
        with c2:
            second = st.selectbox("🥈 2着", list(horse_options.keys()),
                                  format_func=lambda x: horse_options[x],
                                  index=list(horse_options.keys()).index(existing.get("2nd",0)) if existing.get("2nd",0) in horse_options else 0,
                                  key=f"2nd_{rid}")
        with c3:
            third = st.selectbox("🥉 3着", list(horse_options.keys()),
                                 format_func=lambda x: horse_options[x],
                                 index=list(horse_options.keys()).index(existing.get("3rd",0)) if existing.get("3rd",0) in horse_options else 0,
                                 key=f"3rd_{rid}")

        sc1, sc2 = st.columns(2)
        with sc1:
            if st.button("💾 保存", key=f"save_{rid}", type="primary", use_container_width=True):
                if first > 0 and second > 0 and third > 0:
                    st.session_state.results[rid] = {"1st": first, "2nd": second, "3rd": third}
                    # 的中チェック
                    pred_1st = horses[0]["number"] if horses else 0
                    hit = "★的中" if first == pred_1st else ""
                    st.success(f"保存しました {hit}")
                    st.rerun()
                else:
                    st.error("1〜3着を全て選択してください")
        with sc2:
            if st.button("🗑️ クリア", key=f"clear_{rid}", use_container_width=True):
                st.session_state.results.pop(rid, None)
                st.rerun()

        # --- 収支サマリー（結果入力済みレース）---
        if st.session_state.results:
            st.markdown("---")
            st.markdown("#### 本日の結果サマリー")
            hit_count = 0
            for r_id, res in st.session_state.results.items():
                r_info = next((r for r in all_races if r["race_id"] == r_id), None)
                if not r_info: continue
                pred_1 = r_info["horses"][0]["number"] if r_info.get("horses") else 0
                actual_1 = res.get("1st", 0)
                is_hit = (pred_1 == actual_1)
                if is_hit: hit_count += 1
                rn = int(r_id[-2:])
                icon = "✅" if is_hit else "❌"
                st.write(f"{icon} {r_info['place_name']}{rn}R: 予測{pred_1}番 → 実際{actual_1}番")
            total = len(st.session_state.results)
            st.metric("単勝的中率", f"{hit_count}/{total} = {hit_count/total*100:.0f}%" if total > 0 else "0/0")


if __name__ == "__main__":
    main()
