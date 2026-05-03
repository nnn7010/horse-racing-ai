#!/usr/bin/env python3
"""Generate static HTML from today_predictions.json for GitHub Pages."""

import csv
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
PRED_FILE   = ROOT / "data/raw/today_predictions.json"
TRENDS_FILE = ROOT / "data/raw/today_trends.json"
RESULTS_FILE= ROOT / "data/raw/today_results.json"
OUT_FILE    = ROOT / "docs/index.html"

WIN_CAL_FILE  = ROOT / "outputs/win_calibration.csv"
TOP3_CAL_FILE = ROOT / "outputs/top3_calibration.csv"

MODEL_INFO = {
    "name": "LightGBM 共通モデル",
    "features": 88,
    "train_end": "2026-02-28",
    "valid_period": "2026-03-01 〜 2026-04-24",
    "valid_races": 420,
    "auc_valid": 0.7945,
    "auc_oos": 0.7235,
    "rank1_win_rate": 0.305,
}

TIER_COLOR = {"S": "#e53935", "A": "#fb8c00", "B": "#fdd835", "C": "#66bb6a", "D": "#78909c"}

# キャリブレーション実績（win_calibration.csv / top3_calibration.csv 帯に合わせた境界）
TIER_CALIB = {
    "S":  "実績39%勝",
    "A":  "実績27%勝",
    "B":  "実績12-17%勝",
    "C":  "実績8%勝",
    "D":  "実績2%勝",
}


def _win_tier(wp):
    """単勝確率Tier。境界はwin_calibration.csvの帯に一致。"""
    if wp >= 0.30: return "S"   # 30%+帯
    if wp >= 0.20: return "A"   # 20-30%帯
    if wp >= 0.10: return "B"   # 10-20%帯（10-15% + 15-20%を統合）
    if wp >= 0.05: return "C"   # 5-10%帯
    return "D"                   # <5%帯


def _top3_tier(tp):
    """3着以内確率Tier。境界はtop3_calibration.csvの帯に一致。"""
    if tp >= 0.60: return "S"   # 60-70%+ 帯
    if tp >= 0.40: return "A"   # 40-60%帯（40-50% + 50-60%を統合）
    if tp >= 0.30: return "B"   # 30-40%帯
    if tp >= 0.20: return "C"   # 20-30%帯
    return "D"                   # <20%帯


def _race_confidence(horses):
    """トップ馬win_probからレース全体の信頼度ラベルを返す (label, color, ref_text)。"""
    if not horses:
        return "混戦", "#546e7a", ""
    top_wp = max(h["win_prob"] for h in horses)
    if top_wp >= 0.35:
        return "確信", "#c62828", "実績62%的中"
    elif top_wp >= 0.25:
        return "本命", "#e65100", "実績33%的中"
    elif top_wp >= 0.15:
        return "注目", "#f9a825", "実績25%的中"
    else:
        return "混戦", "#546e7a", ""


def _load_csv(path):
    if not path.exists():
        return []
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def render_model_info():
    m = MODEL_INFO
    auc_color = "#81c784" if m["auc_oos"] >= 0.75 else "#ffb74d"
    return f"""<div class="model-info">
  <div class="mi-title">📐 採用モデル: {m["name"]} ({m["features"]}特徴量)</div>
  <div class="mi-body">
    <div class="mi-row"><span class="mi-label">学習〜</span><span class="mi-val">{m["train_end"]}</span></div>
    <div class="mi-row"><span class="mi-label">検証期間</span><span class="mi-val">{m["valid_period"]} ({m["valid_races"]}R)</span></div>
    <div class="mi-row"><span class="mi-label">検証AUC</span><span class="mi-val mi-good">{m["auc_valid"]:.4f}</span></div>
    <div class="mi-row"><span class="mi-label">直近OOS AUC</span><span class="mi-val" style="color:{auc_color}">{m["auc_oos"]:.4f} <small>(4/25-26)</small></span></div>
    <div class="mi-row"><span class="mi-label">本命的中率</span><span class="mi-val mi-good">{m["rank1_win_rate"]:.1%} <small>(win_rank=1)</small></span></div>
  </div>
  <div class="mi-tier-legend">
    <span class="mi-tier-title">Tier:</span>
    <span class="mi-tier-title">単勝Tier:</span>
    <span class="tier-badge" style="background:{TIER_COLOR['S']}">S ≥30%</span><span class="mi-calib">{TIER_CALIB['S']}</span>
    <span class="tier-badge" style="background:{TIER_COLOR['A']}">A 20-30%</span><span class="mi-calib">{TIER_CALIB['A']}</span>
    <span class="tier-badge" style="background:{TIER_COLOR['B']};color:#000">B 10-20%</span><span class="mi-calib">{TIER_CALIB['B']}</span>
    <span class="tier-badge" style="background:{TIER_COLOR['C']};color:#000">C 5-10%</span><span class="mi-calib">{TIER_CALIB['C']}</span>
    <span class="tier-badge" style="background:{TIER_COLOR['D']}">D &lt;5%</span><span class="mi-calib">{TIER_CALIB['D']}</span>
    <br><span class="mi-tier-title">3着Tier:</span>
    <span class="tier-badge" style="background:{TIER_COLOR['S']}">S ≥60%</span>
    <span class="tier-badge" style="background:{TIER_COLOR['A']}">A 40-60%</span>
    <span class="tier-badge" style="background:{TIER_COLOR['B']};color:#000">B 30-40%</span>
    <span class="tier-badge" style="background:{TIER_COLOR['C']};color:#000">C 20-30%</span>
    <span class="tier-badge" style="background:{TIER_COLOR['D']}">D &lt;20%</span>
  </div>
</div>"""


def render_calibration_section():
    win_rows  = _load_csv(WIN_CAL_FILE)
    top3_rows = _load_csv(TOP3_CAL_FILE)

    def _dev_color(dev_str):
        try:
            d = float(dev_str)
            if d > 0.03:  return "#81c784"
            if d < -0.03: return "#ef5350"
            return "#aaa"
        except: return "#aaa"

    def _tbl(rows, pred_col, actual_col, band_col):
        if not rows:
            return "<p style='color:#666;font-size:.8em'>データなし</p>"
        hdr = "<tr><th>予測帯</th><th>頭数</th><th>予測平均</th><th>実際</th><th>乖離</th></tr>"
        body = ""
        for r in rows:
            dev = r.get("deviation") or r.get("error", "0")
            dc  = _dev_color(dev)
            pp  = f"{float(r.get(pred_col,'0'))*100:.1f}%"
            ap  = f"{float(r.get(actual_col,'0'))*100:.1f}%"
            try: dp = f"{float(dev)*100:+.1f}%"
            except: dp = dev
            cnt = r.get('count') or r.get('n', '')
            body += (f"<tr><td class='cal-band'>{r.get(band_col,'')}</td>"
                     f"<td>{cnt}</td><td>{pp}</td>"
                     f"<td style='font-weight:bold;color:#fff'>{ap}</td>"
                     f"<td style='color:{dc};font-weight:bold'>{dp}</td></tr>")
        return f"<table class='cal-tbl'><thead>{hdr}</thead><tbody>{body}</tbody></table>"

    win_tbl  = _tbl(win_rows,  "avg_pred",  "actual_win_rate", "win_band")
    top3_tbl = _tbl(top3_rows, "pred_avg",       "actual",          "band3")

    return f"""<div class="cal-section">
  <div class="cal-title">📊 モデル精度キャリブレーション</div>
  <div class="cal-note">乖離 緑=モデルが控えめ（実際はより高い） 赤=過大評価 ／ 検証期間 2026-03-01〜04-24（420レース）</div>
  <div class="cal-tabs">
    <button class="cal-tab active" onclick="switchCal(this,'win')">単勝確率</button>
    <button class="cal-tab" onclick="switchCal(this,'top3')">3着以内確率</button>
  </div>
  <div class="cal-pane" id="cal-win">{win_tbl}</div>
  <div class="cal-pane" id="cal-top3" style="display:none">{top3_tbl}</div>
</div>"""


STYLE_BADGE = {
    "逃げ": '<span class="sb sb-hana">逃</span>',
    "先行": '<span class="sb sb-sen">先</span>',
    "差し": '<span class="sb sb-sashi">差</span>',
    "追込": '<span class="sb sb-oikomi">追</span>',
}

# 推奨買い目の閾値
WIN_PROB_MIN = 0.20      # 軸: 単勝確率
WIN_ODDS_MIN = 3.0       # 軸: 単勝オッズ
TOP3_MIN = 0.40          # 2-3着候補: top3確率
TOP3_TOP_N = 4           # 上位N頭から選ぶ


def build_recommendations(race):
    """レースから推奨買い目を生成する。
    軸: win_prob≥20% & 単勝オッズ≥3
    三連単フォーメーション: 軸固定 → top3上位4頭∩top3≥40% を2-3着に流し
    候補<2の場合は単勝のみ。
    """
    horses = race["horses"]
    win_axes = []
    for h in horses:
        if h["win_prob"] >= WIN_PROB_MIN and h.get("win_odds", 0) >= WIN_ODDS_MIN:
            win_axes.append(h)
    if not win_axes:
        return None

    # top3確率でソート
    sorted_top3 = sorted(horses, key=lambda h: -h["place_prob"])

    recs = []
    for axis in win_axes:
        axis_num = axis["number"]
        # top4 (軸除外) ∩ top3≥40%
        top4 = [h for h in sorted_top3 if h["number"] != axis_num][:TOP3_TOP_N]
        cands = [h for h in top4 if h["place_prob"] >= TOP3_MIN]

        rec = {
            "axis": axis,
            "win_only": len(cands) < 2,
            "candidates": cands,
            "n_tickets": 0,
        }
        if not rec["win_only"]:
            rec["n_tickets"] = len(cands) * (len(cands) - 1)
        recs.append(rec)
    return recs


def render_recommendations(recs):
    if not recs:
        return ""
    blocks = ""
    for rec in recs:
        axis = rec["axis"]
        axis_html = (
            f'<span class="rec-axis-num">{axis["number"]}</span>'
            f'<span class="rec-axis-name">{axis["horse_name"]}</span>'
            f'<span class="rec-axis-meta">単{axis.get("win_odds",0):.1f}倍 / 1着{axis["win_prob"]*100:.0f}%</span>'
        )
        if rec["win_only"]:
            blocks += (
                f'<div class="rec-card">'
                f'<div class="rec-head"><span class="rec-tag rec-tag-win">単勝のみ</span> 2-3着候補不足</div>'
                f'<div class="rec-axis">軸: {axis_html}</div>'
                f'</div>'
            )
        else:
            cand_html = ""
            for c in rec["candidates"]:
                cand_html += (
                    f'<span class="rec-cand">'
                    f'<b>{c["number"]}</b> {c["horse_name"]} '
                    f'<small>({c["place_prob"]*100:.0f}%/{c.get("win_odds",0):.1f}倍)</small>'
                    f'</span>'
                )
            blocks += (
                f'<div class="rec-card">'
                f'<div class="rec-head">'
                f'<span class="rec-tag rec-tag-tri">単勝＋三連単フォーメーション</span> '
                f'<span class="rec-tickets">{rec["n_tickets"]}点</span>'
                f'</div>'
                f'<div class="rec-axis">軸(1着): {axis_html}</div>'
                f'<div class="rec-cands"><span class="rec-label">2-3着候補:</span>{cand_html}</div>'
                f'</div>'
            )
    return f'<div class="rec-section"><div class="rec-title">💡 推奨買い目</div>{blocks}</div>'


def render_race(race, race_num):
    surface_icon = "🌿" if race["surface"] == "芝" else "🟤"
    horses_sorted = sorted(race["horses"], key=lambda h: -h["win_prob"])

    rows = ""
    for i, h in enumerate(horses_sorted):
        win_pct   = h["win_prob"] * 100
        place_pct = h["place_prob"] * 100
        odds      = h.get("win_odds", 0)
        odds_str  = f"{odds:.1f}" if odds > 0 else "-"
        style     = h.get("running_style", "")
        badge     = STYLE_BADGE.get(style, "")
        comment   = h.get("comment", "")
        tier      = _win_tier(h["win_prob"])
        tc        = TIER_COLOR[tier]
        txt_color = "#000" if tier in ("B", "C") else "#fff"
        tier_html = f'<span class="tier-badge-sm" style="background:{tc};color:{txt_color}">{tier}</span><span class="win-rank">#{i+1}</span>'
        t3tier     = _top3_tier(h["place_prob"])
        t3tc       = TIER_COLOR[t3tier]
        t3txt      = "#000" if t3tier in ("B", "C") else "#fff"
        t3tier_html = f'<span class="tier-badge-sm" style="background:{t3tc};color:{t3txt}">{t3tier}</span>'
        cls = ' class="top"' if i < 3 else ""
        comment_row = f'<tr class="comment-row"><td></td><td colspan="6" class="comment">{comment}</td></tr>' if comment else ""
        rows += (f'<tr{cls}><td>{h["number"]}</td><td>{badge}{h["horse_name"]}</td>'
                 f'<td>{h["jockey_name"]}</td>'
                 f'<td class="odds-cell" data-rid="{race["race_id"]}" data-num="{h["number"]}">{odds_str}</td>'
                 f'<td>{tier_html}</td>'
                 f'<td>{win_pct:.1f}%</td><td>{t3tier_html}{place_pct:.1f}%</td></tr>{comment_row}')

    ability_section = ""

    meta = f"{surface_icon}{race['surface']}{race['distance']}m"
    if race.get("track_condition"):
        meta += f" {race['track_condition']}"
    meta += f" {race['n_horses']}頭"
    time_str = race.get("start_time", "")

    # 推奨買い目
    recs = build_recommendations(race)
    rec_html = render_recommendations(recs) if recs else ""

    # 展開予測
    pp = race.get("pace_prediction", {})
    if pp:
        pace = pp.get("pace", "")
        note = pp.get("note", "")
        front = pp.get("front", 0)
        closers = pp.get("closers", 0)
        pace_color = "#ef5350" if pace == "ハイペース" else "#64b5f6" if pace == "スローペース" else "#ffb74d"
        pace_html = (
            f'<div class="pace">'
            f'<span class="pace-label">展開予測</span>'
            f'<span class="pace-val" style="color:{pace_color}">{pace}</span>'
            f'<span class="pace-note">{note}</span>'
            f'<span class="pace-dist">先行系{front}頭 差追{closers}頭</span>'
            f'</div>'
        )
    else:
        pace_html = ""

    # サマリー: 推奨買い目があるレースに🎯マーク
    rec_indicator = '<span class="rec-indicator">🎯</span>' if recs else ''

    # 信頼度バッジ
    conf_label, conf_color, conf_ref = _race_confidence(race["horses"])
    conf_ref_html = f'<span class="conf-ref">{conf_ref}</span>' if conf_ref else ''
    conf_badge = (
        f'<span class="conf-badge" style="background:{conf_color}">{conf_label}</span>'
        f'{conf_ref_html}'
    )

    return (
        f'<details><summary>'
        f'<b>{race_num}R</b> {race["race_name"]}{rec_indicator} {conf_badge}'
        f'<small> {meta}{" " + time_str if time_str else ""}</small>'
        f'</summary>'
        f'<div class="d">'
        f'{rec_html}'
        f'{pace_html}'
        f'<div class="race-toolbar">'
        f'<button class="btn-update-odds" data-rid="{race["race_id"]}" onclick="updateRaceOdds(this)">🔄 オッズ更新</button>'
        f'<span class="race-odds-status" data-rid="{race["race_id"]}"></span>'
        f'</div>'
        f'<table><thead><tr><th>馬番</th><th>馬名</th><th>騎手</th><th>単勝</th><th>Tier/#</th><th>1着%</th><th>3着%</th></tr></thead>'
        f'<tbody>{rows}</tbody></table>'
        f'{ability_section}'
        f'</div></details>'
    )


def render_trends(trends: dict, results: dict) -> str:
    """コース傾向セクションのHTMLを生成する。"""
    if not trends:
        return ""

    cards = ""
    for venue, t in trends.items():
        n = t["completed"]
        if n == 0:
            continue

        pw = {(int(k) if str(k).isdigit() else k): int(v) for k, v in t["pop_wins"].items()}
        fav_color = "#00e676" if t["fav_trust"] == "高" else "#ffeb3b" if t["fav_trust"] == "中" else "#ef5350"
        bracket_color = "#64b5f6" if "内枠" in t["bracket_bias"] else "#ffb74d" if "外枠" in t["bracket_bias"] else "#aaa"

        # 直近結果リスト
        recent = ""
        for r in reversed(t["races"][-5:]):
            pop_str = f"{r['winner_pop']}番人気" if r["winner_pop"] <= 18 else "?"
            recent += (
                f'<tr><td>{r["race_name"]}</td>'
                f'<td>{r["winner_num"]}番 {r["winner_name"]}</td>'
                f'<td>{pop_str}</td>'
                f'<td>{r["winner_odds"]:.1f}倍</td></tr>'
            )

        surface_icon = "🌿" if "芝" in venue else "🟤"
        cards += f"""
<div class="tc">
  <div class="tc-h">{surface_icon} {venue} <span class="tc-n">{n}R完了</span></div>
  <div class="tc-row">
    <div class="tc-item">
      <div class="tc-label">枠順傾向</div>
      <div class="tc-val" style="color:{bracket_color}">{t["bracket_bias"]}</div>
      <div class="tc-sub">内{t["inner_wins"]}勝 外{t["outer_wins"]}勝</div>
    </div>
    <div class="tc-item">
      <div class="tc-label">1番人気信頼度</div>
      <div class="tc-val" style="color:{fav_color}">{t["fav_trust"]}</div>
      <div class="tc-sub">{pw.get(1,0)}勝/{n}R ({pw.get(1,0)*100//n if n else 0}%)</div>
    </div>
    <div class="tc-item">
      <div class="tc-label">平均勝ちオッズ</div>
      <div class="tc-val">{t["avg_winner_odds"]}倍</div>
      <div class="tc-sub">2番人気{pw.get(2,0)}勝 3番人気{pw.get(3,0)}勝</div>
    </div>
  </div>
  <table class="tc-tbl"><thead><tr><th>レース</th><th>勝ち馬</th><th>人気</th><th>オッズ</th></tr></thead>
  <tbody>{recent}</tbody></table>
</div>"""

    return f'<div class="trends"><h2>📊 本日の傾向</h2>{cards}</div>'


def generate():
    with open(PRED_FILE, encoding="utf-8") as f:
        data = json.load(f)

    trends = {}
    if TRENDS_FILE.exists():
        with open(TRENDS_FILE, encoding="utf-8") as f:
            trends = json.load(f)

    results = {}
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE, encoding="utf-8") as f:
            results = json.load(f)

    import datetime
    today = datetime.date.today()
    date = today.strftime("%Y/%m/%d")
    today_str = today.strftime("%Y%m%d")

    # 同週末の2日(土日)を対象 — 開催が今日のみなら今日のみ
    weekday = today.weekday()  # 0=Mon, 5=Sat, 6=Sun
    if weekday == 5:  # 土
        weekend_dates = [today.strftime("%Y%m%d"), (today + datetime.timedelta(days=1)).strftime("%Y%m%d")]
    elif weekday == 6:  # 日
        weekend_dates = [(today - datetime.timedelta(days=1)).strftime("%Y%m%d"), today.strftime("%Y%m%d")]
    else:
        # 平日: 直近の土日
        days_to_sat = (5 - weekday) % 7
        sat = today + datetime.timedelta(days=days_to_sat)
        weekend_dates = [sat.strftime("%Y%m%d"), (sat + datetime.timedelta(days=1)).strftime("%Y%m%d")]

    # target_races.json から日付マップを作成
    TARGET_FILE = ROOT / "data/raw/target_races.json"
    date_map = {}
    if TARGET_FILE.exists():
        with open(TARGET_FILE, encoding="utf-8") as f:
            for r in json.load(f):
                date_map[r["race_id"]] = r.get("date", "")

    races = data["races"]
    if date_map:
        wknd_races = [r for r in races if date_map.get(r["race_id"]) in weekend_dates]
        if wknd_races:
            races = wknd_races

    # 日付別×場別にグルーピング
    by_date_venue = {}
    for race in races:
        d = date_map.get(race["race_id"], today_str)
        by_date_venue.setdefault(d, {}).setdefault(race["place_name"], []).append(race)

    # トレンドが日別構造({date: {venue: {...}}})か旧構造({venue: {...}})か検出
    is_dated_trends = trends and isinstance(next(iter(trends.values()), {}), dict) and any(
        isinstance(v, dict) and "completed" in v
        for inner in trends.values() if isinstance(inner, dict)
        for v in inner.values() if isinstance(v, dict)
    )

    # 日付ラベル
    weekday_jp = ["月", "火", "水", "木", "金", "土", "日"]
    def fmt_date(s):
        d = datetime.date(int(s[:4]), int(s[4:6]), int(s[6:]))
        is_today = (d == today)
        suffix = " (本日)" if is_today else ""
        return f"{d.month}/{d.day}({weekday_jp[d.weekday()]}){suffix}"

    sorted_dates = sorted(by_date_venue.keys())
    # タブ生成: 本日があればデフォルト本日、なければ最新
    default_date = today_str if today_str in sorted_dates else (sorted_dates[-1] if sorted_dates else today_str)

    # 週末合計トレンド計算（複数日のレース結果を統合）
    weekend_trends = {}
    if is_dated_trends and len(trends) >= 1:
        # venue別に複数日の races を統合して再集計
        venue_races = {}
        for date_key, day_t in trends.items():
            for venue, vt in day_t.items():
                venue_races.setdefault(venue, []).extend(vt.get("races", []))

        for venue, vraces in venue_races.items():
            n = len(vraces)
            if n == 0: continue
            inner_wins = sum(1 for r in vraces if r.get("winner_bracket", 0) in range(1, 5))
            outer_wins = sum(1 for r in vraces if r.get("winner_bracket", 0) in range(5, 9))
            pop_wins = {1: 0, 2: 0, 3: 0, "other": 0}
            for r in vraces:
                p = r.get("winner_pop", 99)
                if p in (1, 2, 3): pop_wins[p] += 1
                else: pop_wins["other"] += 1
            avg_winner_odds = sum(r.get("winner_odds", 0) for r in vraces if r.get("winner_odds", 0) > 0) / max(n, 1)
            fav_trust = "高" if pop_wins[1]/n >= 0.4 else "中" if pop_wins[1]/n >= 0.25 else "低"
            if n >= 3:
                if inner_wins/n >= 0.6: bracket_bias = "内枠有利"
                elif outer_wins/n >= 0.6: bracket_bias = "外枠有利"
                else: bracket_bias = "枠差なし"
            else:
                bracket_bias = "データ不足"
            weekend_trends[venue] = {
                "completed": n, "bracket_bias": bracket_bias,
                "inner_wins": inner_wins, "outer_wins": outer_wins,
                "pop_wins": pop_wins, "fav_trust": fav_trust,
                "avg_winner_odds": round(avg_winner_odds, 1),
                "races": vraces,
            }

    tabs_html = ""
    sections = ""
    if len(sorted_dates) >= 2:
        tabs = ""
        for d in sorted_dates:
            d_label = fmt_date(d)
            active = " active" if d == default_date else ""
            tabs += f'<button class="date-tab{active}" data-date="{d}" onclick="switchDate(this)">{d_label}</button>'
        # 週末合計タブ（トレンドあるなら）
        if weekend_trends:
            tabs += f'<button class="date-tab" data-date="weekend" onclick="switchDate(this)">📊 週末合計</button>'
        tabs_html = f'<div class="date-tabs">{tabs}</div>'

    for d in sorted_dates:
        venues = by_date_venue[d]
        d_label = fmt_date(d)
        is_today = (d == today_str)
        date_cls = " date-today" if is_today else ""
        hidden = ' style="display:none"' if (len(sorted_dates) >= 2 and d != default_date) else ""
        sections += f'<div class="date-section{date_cls}" data-date="{d}"{hidden}>'
        if len(sorted_dates) < 2:
            sections += f'<h2 class="date-h">📅 {d_label}</h2>'

        # 当該日付のトレンド表示
        if is_dated_trends:
            day_trends = trends.get(d, {})
        else:
            day_trends = trends if d == sorted_dates[0] else {}  # 旧構造は最初の日付に
        if day_trends:
            sections += render_trends(day_trends, results)
        elif not is_today:
            pass  # 過去日付でトレンドなしはスキップ
        else:
            sections += '<div class="trends-empty">📊 本日のレース未開催（傾向はレース終了後に表示）</div>'

        for venue, vraces in venues.items():
            blocks = "".join(render_race(r, i + 1) for i, r in enumerate(vraces))
            sections += f'<h3 class="venue-h">🏟 {venue}</h3>{blocks}'
        sections += '</div>'

    # 週末合計セクション（トレンドのみ表示）
    if weekend_trends:
        sections += f'<div class="date-section weekend-section" data-date="weekend" style="display:none">'
        sections += f'<h2 class="date-h">📊 週末合計（{fmt_date(sorted_dates[0])} 〜 {fmt_date(sorted_dates[-1])}）</h2>'
        sections += render_trends(weekend_trends, results)
        sections += '</div>'

    sections = tabs_html + sections

    model_info_html = render_model_info()
    calibration_html = render_calibration_section()

    updated = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>競馬予想 {date}</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#121212;color:#e0e0e0;font-family:-apple-system,sans-serif;font-size:14px;padding:10px}}
h1{{color:#90caf9;font-size:1.1em;margin-bottom:10px;border-bottom:1px solid #333;padding-bottom:6px}}
h2{{color:#ce93d8;font-size:.95em;margin:14px 0 6px}}
details{{background:#1e1e1e;border:1px solid #333;border-radius:6px;margin-bottom:5px;overflow:hidden}}
summary{{cursor:pointer;padding:9px 11px;list-style:none;display:flex;align-items:center;gap:8px}}
summary::-webkit-details-marker{{display:none}}
details[open]>summary{{border-bottom:1px solid #333;background:#252525}}
summary b{{background:#1565c0;color:#fff;border-radius:4px;padding:2px 6px;font-size:.82em;white-space:nowrap}}
summary small{{color:#aaa;font-size:.78em;margin-left:auto}}
.d{{padding:8px 10px}}
table{{width:100%;border-collapse:collapse;margin-bottom:6px;font-size:.82em}}
th{{background:#2a2a2a;color:#aaa;padding:4px 5px;text-align:right;white-space:nowrap}}
th:nth-child(2),th:nth-child(3){{text-align:left}}
td{{padding:4px 5px;border-bottom:1px solid #222;text-align:right;white-space:nowrap}}
td:nth-child(2),td:nth-child(3){{text-align:left}}
tr:last-child td{{border-bottom:none}}
tr.top td{{color:#fff;font-weight:bold}}
tr.top td:nth-child(5){{color:#64b5f6}}
tr.top td:nth-child(6){{color:#81c784}}
td:nth-child(5){{color:#546e7a}}
td:nth-child(6){{color:#4caf50}}
.bt{{display:grid;grid-template-columns:1fr 1fr;gap:10px}}
@media(max-width:420px){{.bt{{grid-template-columns:1fr}}}}
.upd{{color:#555;font-size:.72em;text-align:right;margin-top:12px}}
.trends{{margin-bottom:14px}}
.tc{{background:#1a1a2e;border:1px solid #334;border-radius:8px;padding:10px;margin-bottom:8px}}
.tc-h{{color:#90caf9;font-weight:bold;font-size:.9em;margin-bottom:8px}}
.tc-n{{color:#666;font-size:.8em;font-weight:normal;margin-left:6px}}
.tc-row{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-bottom:8px}}
.tc-item{{background:#111;border-radius:6px;padding:6px 8px;text-align:center}}
.tc-label{{color:#888;font-size:.72em;margin-bottom:2px}}
.tc-val{{font-size:1em;font-weight:bold}}
.tc-sub{{color:#666;font-size:.7em;margin-top:2px}}
.tc-tbl td,.tc-tbl th{{font-size:.75em;padding:3px 4px}}
.tc-tbl th:nth-child(2){{text-align:left}}
.tc-tbl td:nth-child(2){{text-align:left}}
.sb{{display:inline-block;font-size:.68em;font-weight:bold;border-radius:3px;padding:1px 4px;margin-right:3px;vertical-align:middle}}
.sb-hana{{background:#ef5350;color:#fff}}
.sb-sen{{background:#42a5f5;color:#fff}}
.sb-sashi{{background:#66bb6a;color:#fff}}
.sb-oikomi{{background:#ab47bc;color:#fff}}
.comment-row td{{padding:1px 5px 5px;border-bottom:none}}
.comment{{color:#9e9e9e;font-size:.73em;font-style:italic;text-align:left!important}}
.pace{{background:#1a1a2e;border-radius:5px;padding:5px 10px;margin-bottom:7px;display:flex;align-items:center;gap:8px;flex-wrap:wrap}}
.pace-label{{color:#666;font-size:.72em}}
.pace-val{{font-weight:bold;font-size:.88em}}
.pace-note{{color:#aaa;font-size:.75em}}
.pace-dist{{color:#666;font-size:.72em;margin-left:auto}}
.rec-indicator{{margin-left:6px}}
.rec-section{{background:linear-gradient(135deg,#1a2e1a 0%,#1e3a1e 100%);border:1px solid #2e5a2e;border-radius:8px;padding:9px 11px;margin-bottom:9px}}
.rec-title{{color:#a5d6a7;font-size:.82em;font-weight:bold;margin-bottom:7px}}
.rec-card{{background:#0d1f0d;border:1px solid #2a4a2a;border-radius:6px;padding:7px 9px;margin-bottom:6px}}
.rec-card:last-child{{margin-bottom:0}}
.rec-head{{display:flex;align-items:center;gap:8px;margin-bottom:5px;font-size:.78em}}
.rec-tag{{display:inline-block;padding:2px 7px;border-radius:3px;font-weight:bold;font-size:.92em}}
.rec-tag-tri{{background:#2e7d32;color:#fff}}
.rec-tag-win{{background:#f57c00;color:#fff}}
.rec-tickets{{color:#aaa;margin-left:auto;font-size:.92em}}
.rec-axis{{font-size:.85em;color:#e0e0e0;margin-bottom:5px}}
.rec-axis-num{{display:inline-block;background:#1565c0;color:#fff;border-radius:3px;padding:1px 6px;margin-right:5px;font-weight:bold}}
.rec-axis-name{{font-weight:bold}}
.rec-axis-meta{{color:#999;font-size:.88em;margin-left:6px}}
.rec-cands{{font-size:.78em;line-height:1.7}}
.rec-label{{color:#888;margin-right:6px}}
.rec-cand{{display:inline-block;background:#1a3a1a;border:1px solid #2e5a2e;border-radius:3px;padding:1px 6px;margin:0 3px 2px 0;color:#e0e0e0}}
.rec-cand b{{color:#a5d6a7}}
.rec-cand small{{color:#888;font-size:.85em}}
.date-section{{margin-bottom:20px;padding-bottom:8px}}
.date-h{{color:#fff;font-size:1.05em;margin:8px 0 10px;padding:6px 10px;background:#263238;border-radius:5px}}
.date-today .date-h{{background:linear-gradient(90deg,#1565c0 0%,#263238 60%);color:#fff}}
.venue-h{{color:#ce93d8;font-size:.95em;margin:14px 0 6px}}
.date-tabs{{display:flex;gap:6px;margin-bottom:14px;border-bottom:2px solid #333;padding-bottom:0}}
.date-tab{{flex:1;background:#1e1e1e;color:#aaa;border:none;border-radius:6px 6px 0 0;padding:10px 14px;font-size:.92em;font-weight:bold;cursor:pointer;transition:all .15s;border-bottom:3px solid transparent}}
.date-tab:hover{{background:#252525;color:#e0e0e0}}
.date-tab.active{{background:#263238;color:#fff;border-bottom:3px solid #1565c0}}
.date-tab.active::before{{content:"📅 "}}
.trends-empty{{background:#1a1a2e;border:1px dashed #334;border-radius:6px;padding:16px;text-align:center;color:#888;font-size:.85em;margin-bottom:14px}}
.odds-status{{font-size:.65em;margin-left:8px;color:#aaa;font-weight:normal;vertical-align:middle}}
.odds-flash{{animation:flash 1.2s ease-out}}
@keyframes flash{{
  0%{{background:#ffeb3b;color:#000}}
  100%{{background:transparent}}
}}
.odds-cell{{transition:background .3s}}
.race-toolbar{{display:flex;align-items:center;gap:8px;margin-bottom:7px;font-size:.78em}}
.btn-update-odds{{background:#1565c0;color:#fff;border:none;border-radius:4px;padding:4px 10px;font-size:.92em;cursor:pointer;transition:background .15s}}
.btn-update-odds:hover{{background:#1976d2}}
.btn-update-odds:active{{background:#0d47a1}}
.btn-update-odds:disabled{{background:#555;cursor:wait}}
.race-odds-status{{color:#888;font-size:.92em}}
.race-odds-status.ok{{color:#81c784}}
.race-odds-status.err{{color:#ef5350}}
.tier-badge{{display:inline-block;padding:2px 7px;border-radius:4px;font-weight:bold;font-size:.78em;color:#fff;margin:0 2px}}
.tier-badge-sm{{display:inline-block;padding:1px 5px;border-radius:3px;font-weight:bold;font-size:.75em;vertical-align:middle;margin-right:3px}}
.win-rank{{color:#aaa;font-size:.72em;vertical-align:middle}}
.conf-badge{{display:inline-block;padding:1px 7px;border-radius:10px;font-size:.72em;font-weight:bold;color:#fff;vertical-align:middle;margin-left:4px}}
.conf-ref{{font-size:.65em;color:#bbb;margin-left:4px;vertical-align:middle}}
.mi-calib{{font-size:.72em;color:#888;margin-right:8px;vertical-align:middle}}
.model-info{{background:#1a2030;border:1px solid #334;border-radius:8px;padding:10px 13px;margin-bottom:12px;font-size:.83em}}
.mi-title{{color:#90caf9;font-weight:bold;margin-bottom:7px}}
.mi-body{{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:4px 16px;margin-bottom:8px}}
.mi-row{{display:flex;gap:6px;align-items:center}}
.mi-label{{color:#888;min-width:80px;flex-shrink:0}}
.mi-val{{color:#e0e0e0;font-weight:bold}}
.mi-good{{color:#81c784}}
.mi-tier-legend{{display:flex;flex-wrap:wrap;gap:2px 6px;align-items:center;margin-top:6px;padding-top:6px;border-top:1px solid #333}}
.mi-tier-title{{color:#888;font-size:.85em;margin-right:4px}}
.cal-section{{background:#1a2a1a;border:1px solid #2a4a2a;border-radius:8px;padding:10px 13px;margin-bottom:12px;font-size:.83em}}
.cal-title{{color:#a5d6a7;font-weight:bold;margin-bottom:5px}}
.cal-note{{color:#777;font-size:.78em;margin-bottom:8px}}
.cal-tabs{{display:flex;gap:6px;margin-bottom:8px}}
.cal-tab{{background:#1e2e1e;color:#aaa;border:1px solid #334;border-radius:4px;padding:4px 12px;cursor:pointer;font-size:.85em}}
.cal-tab.active{{background:#2e5a2e;color:#fff;border-color:#4a8a4a}}
.cal-tbl{{width:100%;border-collapse:collapse;font-size:.82em}}
.cal-tbl th,.cal-tbl td{{padding:4px 8px;text-align:right;border-bottom:1px solid #222}}
.cal-tbl th{{background:#1e2e1e;color:#aaa;text-align:right}}
.cal-band{{text-align:left!important;color:#ccc;font-weight:bold}}
</style>
</head>
<body>
<h1>🏇 競馬予想 {date} <span id="odds-status" class="odds-status">⏳</span></h1>
{model_info_html}
{calibration_html}
{sections}
<p class="upd">更新: {updated} / オッズ: <span id="odds-updated">-</span></p>
<script>
async function refreshOdds() {{
  try {{
    const resp = await fetch('odds.json?t=' + Date.now());
    if (!resp.ok) return;
    const data = await resp.json();
    const status = document.getElementById('odds-status');
    const upd = document.getElementById('odds-updated');
    let n = 0;
    document.querySelectorAll('.odds-cell').forEach(td => {{
      const rid = td.dataset.rid;
      const num = td.dataset.num;
      const o = data.odds && data.odds[rid] && data.odds[rid][num];
      if (o && o.win > 0) {{
        const oldVal = td.textContent.trim();
        const newVal = o.win.toFixed(1);
        if (oldVal !== newVal && oldVal !== '-') {{
          td.classList.add('odds-flash');
          setTimeout(() => td.classList.remove('odds-flash'), 1200);
        }}
        td.textContent = newVal;
        n++;
      }}
    }});
    upd.textContent = data.updated || '-';
    status.textContent = '🟢 ' + n + '頭';
    status.title = 'オッズ更新中（60秒ごと）';
  }} catch (e) {{
    document.getElementById('odds-status').textContent = '🔴';
  }}
}}
refreshOdds();
setInterval(refreshOdds, 30000);

// キャリブレーションタブ切替
function switchCal(btn, pane) {{
  document.querySelectorAll('.cal-tab').forEach(t => t.classList.remove('active'));
  btn.classList.add('active');
  document.querySelectorAll('.cal-pane').forEach(p => p.style.display = 'none');
  var el = document.getElementById('cal-' + pane);
  if (el) el.style.display = '';
}}

// 日付タブ切替
function switchDate(btn) {{
  const target = btn.dataset.date;
  document.querySelectorAll('.date-tab').forEach(t => t.classList.toggle('active', t.dataset.date === target));
  document.querySelectorAll('.date-section').forEach(s => {{
    s.style.display = (s.dataset.date === target) ? '' : 'none';
  }});
  // タブ切替時にスクロール上端へ
  window.scrollTo({{top: 0, behavior: 'smooth'}});
}}

// レース別オッズ更新ボタン
// 1) サーバー側 odds.json を即時再取得（必ず成功）
// 2) 並行して netkeiba 直接フェッチを試みる（成功すれば上書き）
async function fetchWithTimeout(url, ms = 5000) {{
  const ctrl = new AbortController();
  const id = setTimeout(() => ctrl.abort(), ms);
  try {{
    const r = await fetch(url, {{signal: ctrl.signal, cache: 'no-store'}});
    return r;
  }} finally {{
    clearTimeout(id);
  }}
}}

async function fetchProxyOdds(raceId) {{
  const target = `https://race.netkeiba.com/api/api_get_jra_odds.html?race_id=${{raceId}}&type=1&action=update`;
  const proxies = [
    'https://api.allorigins.win/raw?url=' + encodeURIComponent(target),
    'https://corsproxy.io/?' + encodeURIComponent(target),
  ];
  for (const p of proxies) {{
    try {{
      const r = await fetchWithTimeout(p, 4000);
      if (!r.ok) continue;
      const data = await r.json();
      if (data && data.data && data.data.odds && data.data.odds['1']) {{
        const out = {{}};
        for (const [k, v] of Object.entries(data.data.odds['1'])) {{
          out[parseInt(k, 10).toString()] = parseFloat(Array.isArray(v) ? v[0] : v);
        }}
        return out;
      }}
    }} catch (e) {{
      console.warn('proxy', p.slice(0, 30), 'failed:', e.message);
    }}
  }}
  return null;
}}

async function fetchServerOddsForRace(raceId) {{
  try {{
    const r = await fetchWithTimeout('odds.json?t=' + Date.now(), 3000);
    if (!r.ok) return null;
    const data = await r.json();
    return {{
      odds: (data.odds && data.odds[raceId]) || null,
      updated: data.updated || null,
    }};
  }} catch (e) {{
    return null;
  }}
}}

function applyOddsToRace(rid, oddsMap) {{
  let n = 0;
  document.querySelectorAll(`.odds-cell[data-rid="${{rid}}"]`).forEach(td => {{
    const num = td.dataset.num;
    let v = oddsMap[num];
    // server data形式 ({{win, place}}) に対応
    if (v && typeof v === 'object') v = v.win;
    if (v && v > 0) {{
      const oldVal = td.textContent.trim();
      const newVal = parseFloat(v).toFixed(1);
      if (oldVal !== newVal) {{
        td.classList.add('odds-flash');
        setTimeout(() => td.classList.remove('odds-flash'), 1200);
      }}
      td.textContent = newVal;
      n++;
    }}
  }});
  return n;
}}

async function updateRaceOdds(btn) {{
  const rid = btn.dataset.rid;
  const status = document.querySelector(`.race-odds-status[data-rid="${{rid}}"]`);
  btn.disabled = true;
  btn.textContent = '⏳';
  status.className = 'race-odds-status';
  status.textContent = '取得中...';

  // まずサーバー版を反映（必ず成功するよう）
  const server = await fetchServerOddsForRace(rid);
  if (server && server.odds) {{
    const n = applyOddsToRace(rid, server.odds);
    status.className = 'race-odds-status ok';
    status.textContent = `サーバー版 ${{n}}頭 (${{server.updated || '?'}})`;
  }} else {{
    status.className = 'race-odds-status err';
    status.textContent = 'サーバー版取得失敗';
  }}

  // 並行してプロキシ経由のライブ取得を試みる
  const live = await fetchProxyOdds(rid);
  if (live) {{
    const n = applyOddsToRace(rid, live);
    status.className = 'race-odds-status ok';
    const t = new Date().toLocaleTimeString('ja-JP', {{hour: '2-digit', minute: '2-digit', second: '2-digit'}});
    status.textContent = `🟢 ライブ ${{n}}頭 ${{t}}`;
  }}

  btn.disabled = false;
  btn.textContent = '🔄 更新';
}}
</script>
</body>
</html>"""

    OUT_FILE.parent.mkdir(exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Generated: {OUT_FILE} ({len(html):,} bytes)")
    print(f"Races: {len(races)} across {len(by_date_venue)} day(s)")


if __name__ == "__main__":
    generate()
