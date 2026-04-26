#!/usr/bin/env python3
"""Generate static HTML from today_predictions.json for GitHub Pages."""

import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
PRED_FILE = ROOT / "data/raw/today_predictions.json"
TRENDS_FILE = ROOT / "data/raw/today_trends.json"
RESULTS_FILE = ROOT / "data/raw/today_results.json"
OUT_FILE = ROOT / "docs/index.html"




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
        win_pct = h["win_prob"] * 100
        place_pct = h["place_prob"] * 100
        odds = h.get("win_odds", 0)
        odds_str = f"{odds:.1f}" if odds > 0 else "-"
        style = h.get("running_style", "")
        badge = STYLE_BADGE.get(style, "")
        comment = h.get("comment", "")
        cls = ' class="top"' if i < 3 else ""
        comment_row = f'<tr class="comment-row"><td></td><td colspan="5" class="comment">{comment}</td></tr>' if comment else ""
        rows += f'<tr{cls}><td>{h["number"]}</td><td>{badge}{h["horse_name"]}</td><td>{h["jockey_name"]}</td><td class="odds-cell" data-rid="{race["race_id"]}" data-num="{h["number"]}">{odds_str}</td><td>{win_pct:.1f}%</td><td>{place_pct:.1f}%</td></tr>{comment_row}'

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

    return (
        f'<details><summary>'
        f'<b>{race_num}R</b> {race["race_name"]}{rec_indicator}'
        f'<small> {meta}{" " + time_str if time_str else ""}</small>'
        f'</summary>'
        f'<div class="d">'
        f'{rec_html}'
        f'{pace_html}'
        f'<div class="race-toolbar">'
        f'<button class="btn-update-odds" data-rid="{race["race_id"]}" onclick="updateRaceOdds(this)">🔄 オッズ更新</button>'
        f'<span class="race-odds-status" data-rid="{race["race_id"]}"></span>'
        f'</div>'
        f'<table><thead><tr><th>馬番</th><th>馬名</th><th>騎手</th><th>単勝</th><th>1着%</th><th>3着%</th></tr></thead>'
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

    tabs_html = ""
    sections = ""
    if len(sorted_dates) >= 2:
        tabs = ""
        for d in sorted_dates:
            d_label = fmt_date(d)
            active = " active" if d == default_date else ""
            tabs += f'<button class="date-tab{active}" data-date="{d}" onclick="switchDate(this)">{d_label}</button>'
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

    sections = tabs_html + sections

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
</style>
</head>
<body>
<h1>🏇 競馬予想 {date} <span id="odds-status" class="odds-status">⏳</span></h1>
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
