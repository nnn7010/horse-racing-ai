#!/usr/bin/env python3
"""Generate static HTML from today_predictions.json for GitHub Pages."""

import json
import os
from pathlib import Path

ROOT = Path(__file__).parent.parent
PRED_FILE = ROOT / "data/raw/today_predictions.json"
OUT_FILE = ROOT / "docs/index.html"


def ev_color(ev):
    if ev >= 1.3:
        return "#00e676"
    if ev >= 1.1:
        return "#ffeb3b"
    return "#aaa"


def render_race(race, race_num):
    surface_icon = "🌿" if race["surface"] == "芝" else "🟤"
    horses_sorted = sorted(race["horses"], key=lambda h: -h["win_prob"])

    rows = ""
    for h in horses_sorted:
        win_pct = h["win_prob"] * 100
        place_pct = h["place_prob"] * 100
        odds = h.get("win_odds", 0)
        odds_str = f"{odds:.1f}" if odds > 0 else "-"
        bold_open = "<strong>" if horses_sorted.index(h) < 3 else ""
        bold_close = "</strong>" if horses_sorted.index(h) < 3 else ""
        rows += f"""
        <tr>
          <td>{h['number']}</td>
          <td>{bold_open}{h['horse_name']}{bold_close}</td>
          <td>{h['jockey_name']}</td>
          <td>{odds_str}</td>
          <td style="color:#64b5f6">{win_pct:.1f}%</td>
          <td style="color:#81c784">{place_pct:.1f}%</td>
        </tr>"""

    trio_rows = ""
    for i, t in enumerate(race["trio_top5"], 1):
        combo = "-".join(map(str, t["combo"]))
        prob = t["prob"] * 100
        ev = t.get("ev", 0)
        ev_str = f'{ev:.2f}' if ev > 0 else '-'
        color = ev_color(ev) if ev > 0 else "#aaa"
        trio_rows += f"""
        <tr>
          <td>{i}</td>
          <td><strong>{combo}</strong></td>
          <td>{prob:.2f}%</td>
          <td style="color:{color}">{ev_str}</td>
        </tr>"""

    trifecta_rows = ""
    for i, t in enumerate(race["trifecta_top5"], 1):
        combo = "-".join(map(str, t["combo"]))
        prob = t["prob"] * 100
        ev = t.get("ev", 0)
        ev_str = f'{ev:.2f}' if ev > 0 else '-'
        color = ev_color(ev) if ev > 0 else "#aaa"
        trifecta_rows += f"""
        <tr>
          <td>{i}</td>
          <td><strong>{combo}</strong></td>
          <td>{prob:.3f}%</td>
          <td style="color:{color}">{ev_str}</td>
        </tr>"""

    return f"""
  <details class="race-block">
    <summary>
      <span class="race-num">{race_num}R</span>
      <span class="race-name">{race['race_name']}</span>
      <span class="race-meta">{surface_icon}{race['surface']}{race['distance']}m {race['track_condition']} {race['n_horses']}頭</span>
      <span class="race-time">{race.get('start_time','')}</span>
    </summary>
    <div class="race-detail">
      <h4>出走馬</h4>
      <div class="table-wrap">
      <table>
        <thead><tr><th>馬番</th><th>馬名</th><th>騎手</th><th>単勝</th><th>1着%</th><th>3着%</th></tr></thead>
        <tbody>{rows}</tbody>
      </table>
      </div>

      <div class="bet-tables">
        <div>
          <h4>三連複 TOP5</h4>
          <div class="table-wrap">
          <table>
            <thead><tr><th>#</th><th>組合</th><th>確率</th><th>EV</th></tr></thead>
            <tbody>{trio_rows}</tbody>
          </table>
          </div>
        </div>
        <div>
          <h4>三連単 TOP5</h4>
          <div class="table-wrap">
          <table>
            <thead><tr><th>#</th><th>順番</th><th>確率</th><th>EV</th></tr></thead>
            <tbody>{trifecta_rows}</tbody>
          </table>
          </div>
        </div>
      </div>
    </div>
  </details>"""


def generate():
    with open(PRED_FILE, encoding="utf-8") as f:
        data = json.load(f)

    date = data.get("date", "")
    races = data["races"]

    # Group by venue
    venues = {}
    for race in races:
        v = race["place_name"]
        venues.setdefault(v, []).append(race)

    venue_sections = ""
    for venue, vraces in venues.items():
        race_blocks = ""
        for i, race in enumerate(vraces, 1):
            race_blocks += render_race(race, i)
        venue_sections += f"""
  <section class="venue">
    <h2>🏟 {venue}</h2>
    {race_blocks}
  </section>"""

    html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>競馬予想 {date}</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ background: #121212; color: #e0e0e0; font-family: -apple-system, sans-serif; font-size: 14px; padding: 12px; }}
    h1 {{ color: #90caf9; font-size: 1.2em; margin-bottom: 12px; padding-bottom: 8px; border-bottom: 1px solid #333; }}
    h2 {{ color: #ce93d8; font-size: 1em; margin: 16px 0 8px; }}
    h4 {{ color: #90caf9; font-size: 0.85em; margin: 10px 0 4px; }}
    .venue {{ margin-bottom: 20px; }}
    .race-block {{ background: #1e1e1e; border: 1px solid #333; border-radius: 8px; margin-bottom: 6px; overflow: hidden; }}
    .race-block > summary {{ cursor: pointer; padding: 10px 12px; display: flex; align-items: center; gap: 8px; list-style: none; user-select: none; }}
    .race-block > summary::-webkit-details-marker {{ display: none; }}
    .race-block[open] > summary {{ border-bottom: 1px solid #333; background: #252525; }}
    .race-num {{ background: #1565c0; color: #fff; border-radius: 4px; padding: 2px 6px; font-weight: bold; font-size: 0.85em; min-width: 28px; text-align: center; }}
    .race-name {{ font-weight: bold; flex: 1; }}
    .race-meta {{ color: #aaa; font-size: 0.8em; }}
    .race-time {{ color: #ffcc80; font-size: 0.8em; }}
    .race-detail {{ padding: 10px 12px; }}
    .table-wrap {{ overflow-x: auto; }}
    table {{ width: 100%; border-collapse: collapse; margin-bottom: 8px; font-size: 0.85em; }}
    th {{ background: #2a2a2a; color: #aaa; padding: 5px 6px; text-align: right; white-space: nowrap; }}
    th:nth-child(2), th:nth-child(3) {{ text-align: left; }}
    td {{ padding: 5px 6px; border-bottom: 1px solid #2a2a2a; text-align: right; white-space: nowrap; }}
    td:nth-child(2), td:nth-child(3) {{ text-align: left; }}
    tr:last-child td {{ border-bottom: none; }}
    .bet-tables {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 8px; }}
    @media (max-width: 480px) {{ .bet-tables {{ grid-template-columns: 1fr; }} }}
    .updated {{ color: #666; font-size: 0.75em; margin-top: 16px; text-align: right; }}
  </style>
</head>
<body>
  <h1>🏇 競馬予想 {date}</h1>
  {venue_sections}
  <p class="updated">Generated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
</body>
</html>"""

    OUT_FILE.parent.mkdir(exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Generated: {OUT_FILE}")
    print(f"Races: {len(races)} ({', '.join(f'{v}:{len(r)}R' for v, r in venues.items())})")


if __name__ == "__main__":
    generate()
