"""対象日のレース一覧を取得する。

netkeiba の開催一覧ページからレースIDを抽出し、
各レースの基本情報（競馬場・レース番号・距離・芝ダート等）を返す。
"""

import re
from datetime import date

from bs4 import BeautifulSoup

from src.utils.http import fetch
from src.utils.logger import get_logger

logger = get_logger(__name__)

# JRA競馬場コード
COURSE_CODES = {
    "01": "札幌", "02": "函館", "03": "福島", "04": "新潟",
    "05": "東京", "06": "中山", "07": "中京", "08": "京都",
    "09": "阪神", "10": "小倉",
}


def fetch_race_list(target_date: date) -> list[dict]:
    """指定日のJRA全レースIDと基本情報を取得する。"""
    dt_str = target_date.strftime("%Y%m%d")
    # race_list_sub.html はAJAX内部エンドポイントでレースIDを含む
    url = f"https://race.netkeiba.com/top/race_list_sub.html?kaisai_date={dt_str}"
    html = fetch(url, encoding="utf-8")
    soup = BeautifulSoup(html, "lxml")

    races = []
    # レースリストのリンクからrace_idを抽出
    for a_tag in soup.select("a[href*='race_id=']"):
        href = a_tag.get("href", "")
        m = re.search(r"race_id=(\d{12})", href)
        if not m:
            continue
        race_id = m.group(1)
        # JRAのみ（場コード01-10）
        place_code = race_id[4:6]
        if place_code not in COURSE_CODES:
            continue
        # 重複回避
        if any(r["race_id"] == race_id for r in races):
            continue
        races.append({
            "race_id": race_id,
            "date": dt_str,
            "place_code": place_code,
            "place_name": COURSE_CODES.get(place_code, "不明"),
        })

    logger.info(f"{target_date}: {len(races)} races found")
    return races


def fetch_race_detail(race_id: str) -> dict:
    """レース詳細ページから距離・芝ダート・クラス等の条件を取得する。"""
    url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
    html = fetch(url, encoding="euc-jp")
    soup = BeautifulSoup(html, "lxml")

    info = {"race_id": race_id}

    # レース名
    race_name_tag = soup.select_one(".RaceName")
    info["race_name"] = race_name_tag.get_text(strip=True) if race_name_tag else ""

    # コース情報（例: "芝2000m"）
    race_data_tag = soup.select_one(".RaceData01")
    if race_data_tag:
        text = race_data_tag.get_text(strip=True)
        # 芝/ダート
        if "芝" in text:
            info["surface"] = "芝"
        elif "ダ" in text or "ダート" in text:
            info["surface"] = "ダート"
        else:
            info["surface"] = "不明"
        # 距離
        dist_m = re.search(r"(\d{4})m", text)
        info["distance"] = int(dist_m.group(1)) if dist_m else 0
        # 天候・馬場状態
        info["raw_conditions"] = text
    else:
        info["surface"] = "不明"
        info["distance"] = 0
        info["raw_conditions"] = ""

    # クラス情報
    race_data2 = soup.select_one(".RaceData02")
    if race_data2:
        spans = race_data2.select("span")
        info["class"] = spans[0].get_text(strip=True) if spans else ""
        info["race_detail"] = race_data2.get_text(strip=True)
    else:
        info["class"] = ""
        info["race_detail"] = ""

    # 新馬・障害チェック
    full_text = info.get("race_name", "") + info.get("race_detail", "")
    info["is_debut"] = "新馬" in full_text
    info["is_hurdle"] = "障害" in full_text

    # 出走馬情報
    entries = []
    table = soup.select_one(".Shutuba_Table, .ShutubaTable")
    if table:
        for tr in table.select("tr.HorseList"):
            entry = _parse_entry_row(tr)
            if entry:
                entries.append(entry)
    info["entries"] = entries

    logger.info(f"Race {race_id}: {info.get('race_name', '')} {info.get('surface', '')}{info.get('distance', '')}m, {len(entries)} entries")
    return info


def _parse_entry_row(tr) -> dict | None:
    """出馬表の1行をパースする。"""
    tds = tr.select("td")
    if len(tds) < 4:
        return None

    entry = {}
    # 枠番
    waku = tds[0].get_text(strip=True)
    entry["bracket"] = int(waku) if waku.isdigit() else 0
    # 馬番
    umaban = tds[1].get_text(strip=True)
    entry["number"] = int(umaban) if umaban.isdigit() else 0
    # 馬名・馬ID
    horse_link = tr.select_one("a[href*='/horse/']")
    if horse_link:
        entry["horse_name"] = horse_link.get_text(strip=True)
        href = horse_link.get("href", "")
        hid = re.search(r"/horse/(\w+)", href)
        entry["horse_id"] = hid.group(1) if hid else ""
    else:
        entry["horse_name"] = ""
        entry["horse_id"] = ""
    # 騎手
    jockey_link = tr.select_one("a[href*='/jockey/']")
    if jockey_link:
        entry["jockey_name"] = jockey_link.get_text(strip=True)
        href = jockey_link.get("href", "")
        jid = re.search(r"/jockey/(?:result/recent/)?(\w+)", href)
        entry["jockey_id"] = jid.group(1) if jid else ""
    else:
        entry["jockey_name"] = ""
        entry["jockey_id"] = ""
    # 斤量
    for td in tds:
        text = td.get_text(strip=True)
        try:
            w = float(text)
            if 45 <= w <= 65:
                entry["weight"] = w
                break
        except ValueError:
            continue
    if "weight" not in entry:
        entry["weight"] = 0.0
    # 調教師
    trainer_link = tr.select_one("a[href*='/trainer/']")
    if trainer_link:
        entry["trainer_name"] = trainer_link.get_text(strip=True)
        href = trainer_link.get("href", "")
        tid = re.search(r"/trainer/(?:result/recent/)?(\w+)", href)
        entry["trainer_id"] = tid.group(1) if tid else ""
    else:
        entry["trainer_name"] = ""
        entry["trainer_id"] = ""

    return entry
