"""今週・今開催の出馬表（shutuba）パーサ。

URL: https://nar.netkeiba.com/race/shutuba.html?race_id={race_id}
レース当日の予測対象として使う。確定オッズは後で
update_odds.py のような別スクリプトで更新する想定。
"""

import re

from bs4 import BeautifulSoup

from src.oi.scraping.http import fetch
from src.oi.scraping.race import parse_race_id
from src.utils.logger import get_logger

logger = get_logger(__name__)


def fetch_shutuba(race_id: str) -> dict:
    """指定レースの出馬表を取得する。"""
    url = f"https://nar.netkeiba.com/race/shutuba.html?race_id={race_id}"
    html = fetch(url, encoding="utf-8")
    soup = BeautifulSoup(html, "lxml")

    meta = parse_race_id(race_id)
    result: dict = {
        "race_id": race_id,
        "date": meta["date"].strftime("%Y%m%d"),
        "race_no": meta["race_no"],
        "entries": [],
    }

    # レース名・条件（result.html とほぼ同構造）
    race_name_tag = soup.select_one(".RaceName")
    result["race_name"] = race_name_tag.get_text(strip=True) if race_name_tag else ""

    race_data_box = soup.select_one(".RaceData01, .RaceList_Item02")
    if race_data_box:
        info_text = race_data_box.get_text(" ", strip=True)
        result["course_info"] = info_text
        if "芝" in info_text:
            result["surface"] = "芝"
        else:
            result["surface"] = "ダート"
        dist_m = re.search(r"(\d{3,4})m", info_text)
        result["distance"] = int(dist_m.group(1)) if dist_m else 0
        cond_m = re.search(r"馬場\s*[:：]?\s*(良|稍重|重|不良)", info_text)
        result["track_condition"] = cond_m.group(1) if cond_m else ""
        weather_m = re.search(r"天候\s*[:：]?\s*(晴|曇|雨|小雨|雪)", info_text)
        result["weather"] = weather_m.group(1) if weather_m else ""

    # 出走馬テーブル
    shutuba_table = soup.select_one("table.Shutuba_Table, table.RaceTable01")
    if shutuba_table:
        for tr in shutuba_table.select("tbody tr"):
            entry = _parse_shutuba_row(tr, race_id)
            if entry:
                result["entries"].append(entry)

    result["num_runners"] = len(result["entries"])
    return result


def _parse_shutuba_row(tr, race_id: str) -> dict | None:
    tds = tr.select("td")
    if len(tds) < 6:
        return None

    def _txt(idx: int) -> str:
        return tds[idx].get_text(strip=True) if idx < len(tds) else ""

    entry: dict = {"race_id": race_id}

    # 枠番・馬番
    entry["bracket"] = int(_txt(0)) if _txt(0).isdigit() else 0
    entry["number"] = int(_txt(1)) if _txt(1).isdigit() else 0

    horse_link = tr.select_one("a[href*='/horse/']")
    if horse_link:
        entry["horse_name"] = horse_link.get_text(strip=True)
        href = horse_link.get("href", "")
        hid = re.search(r"/horse/(\w+)", href)
        entry["horse_id"] = hid.group(1) if hid else ""
    else:
        entry["horse_name"] = ""
        entry["horse_id"] = ""

    # 性齢
    sex_age = ""
    for td in tds:
        t = td.get_text(strip=True)
        if re.fullmatch(r"[牡牝セ]\d{1,2}", t):
            sex_age = t
            break
    entry["sex_age"] = sex_age

    # 斤量
    impost = 0.0
    for td in tds:
        t = td.get_text(strip=True)
        m = re.fullmatch(r"\d{2}\.\d", t)
        if m:
            try:
                v = float(t)
                if 40 <= v <= 65:
                    impost = v
                    break
            except ValueError:
                pass
    entry["impost"] = impost

    # 騎手
    jockey_link = tr.select_one("a[href*='/jockey/']")
    if jockey_link:
        entry["jockey_name"] = jockey_link.get_text(strip=True)
        href = jockey_link.get("href", "")
        jid = re.search(r"/jockey/(?:result/)?(\w+)", href)
        entry["jockey_id"] = jid.group(1) if jid else ""
    else:
        entry["jockey_name"] = ""
        entry["jockey_id"] = ""

    # 調教師
    trainer_link = tr.select_one("a[href*='/trainer/']")
    if trainer_link:
        entry["trainer_name"] = trainer_link.get_text(strip=True)
        href = trainer_link.get("href", "")
        tid = re.search(r"/trainer/(?:result/)?(\w+)", href)
        entry["trainer_id"] = tid.group(1) if tid else ""
    else:
        entry["trainer_name"] = ""
        entry["trainer_id"] = ""

    # 馬体重・増減
    horse_weight = 0
    weight_change = 0
    for td in tds:
        t = td.get_text(strip=True)
        m = re.search(r"(\d{3,4})\(([+-]?\d+)\)", t)
        if m:
            horse_weight = int(m.group(1))
            weight_change = int(m.group(2))
            break
    entry["horse_weight"] = horse_weight
    entry["weight_change"] = weight_change

    # 想定オッズ・人気（あれば）
    odds = 0.0
    pop = 0
    for td in tds:
        t = td.get_text(strip=True)
        if not odds and re.fullmatch(r"\d{1,4}\.\d", t):
            odds = float(t)
        if not pop and re.fullmatch(r"\d{1,2}", t):
            v = int(t)
            if 1 <= v <= 18 and v != entry["number"] and v != entry["bracket"]:
                pop = v
    entry["win_odds"] = odds
    entry["popularity"] = pop

    return entry
