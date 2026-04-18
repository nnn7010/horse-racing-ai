"""馬の詳細情報（血統・過去成績）のスクレイピング。"""

import re

from bs4 import BeautifulSoup

from src.utils.http import fetch
from src.utils.logger import get_logger

logger = get_logger(__name__)


def fetch_horse_info(horse_id: str) -> dict:
    """馬の血統情報と基本情報を取得する。"""
    # 馬プロフィールページ
    url = f"https://db.netkeiba.com/horse/{horse_id}/"
    html = fetch(url, encoding="euc-jp")
    soup = BeautifulSoup(html, "lxml")

    info = {"horse_id": horse_id}

    # 馬名
    name_tag = soup.select_one(".horse_title h1, .horse_name h1")
    info["horse_name"] = name_tag.get_text(strip=True) if name_tag else ""

    # 血統テーブル（専用ページから取得）
    ped_url = f"https://db.netkeiba.com/horse/ped/{horse_id}/"
    ped_html = fetch(ped_url, encoding="euc-jp")
    ped_soup = BeautifulSoup(ped_html, "lxml")
    pedigree = _parse_pedigree(ped_soup)
    info.update(pedigree)

    # 過去成績
    info["past_results"] = _parse_past_results(soup)

    return info


def _parse_pedigree(soup) -> dict:
    """血統ページ(/horse/ped/)から父・母父・母母父を抽出する。

    5代血統表の構造（32行×62セル）:
    - row0  cell0 (rowspan=16) = 父
    - row16 cell0 (rowspan=16) = 母
    - row16 cell1 (rowspan=8)  = 母父（BMS）
    - row24 cell1 (rowspan=4)  = 母母父
    """
    result = {
        "sire": "",
        "sire_id": "",
        "dam_sire": "",
        "dam_sire_id": "",
        "dam_dam_sire": "",
        "dam_dam_sire_id": "",
    }

    ped_table = soup.select_one(".blood_table")
    if not ped_table:
        logger.warning("Pedigree table not found")
        return result

    rows = ped_table.select("tr")
    if len(rows) < 25:
        logger.warning(f"Pedigree table has only {len(rows)} rows")
        return result

    def _get_cell_info(row_idx, cell_idx):
        """指定行・セルから馬名とIDを取得する。"""
        if row_idx >= len(rows):
            return "", ""
        cells = rows[row_idx].select("td")
        if cell_idx >= len(cells):
            return "", ""
        td = cells[cell_idx]
        a = td.select_one("a")
        if a:
            name = a.get_text(strip=True)
            href = a.get("href", "")
            hid_m = re.search(r"/horse/(?:ped_sire/)?(\w+)", href)
            hid = hid_m.group(1) if hid_m else ""
            return name, hid
        return td.get_text(strip=True), ""

    # 父: row0, cell0
    result["sire"], result["sire_id"] = _get_cell_info(0, 0)
    # 母父: row16, cell1
    result["dam_sire"], result["dam_sire_id"] = _get_cell_info(16, 1)
    # 母母父: row24, cell1
    result["dam_dam_sire"], result["dam_dam_sire_id"] = _get_cell_info(24, 1)

    # 国名表記を除去（例: "エンパイアメーカーEmpire Maker(米)" → "エンパイアメーカー"）
    for key in ["sire", "dam_sire", "dam_dam_sire"]:
        name = result[key]
        # 日本語名があれば英語部分を除去
        import unicodedata
        jp_part = ""
        for ch in name:
            if unicodedata.category(ch).startswith('L') and ord(ch) > 0x2FFF:
                jp_part += ch
            elif jp_part and (ch in "ーァ-ヶ" or unicodedata.category(ch).startswith('L') and ord(ch) > 0x2FFF):
                jp_part += ch
            elif jp_part:
                break
        # カタカナが含まれる場合はカタカナ部分を使用
        katakana = re.findall(r'[ァ-ヶー]+', name)
        if katakana:
            result[key] = max(katakana, key=len)
        # そうでなければそのまま

    logger.debug(f"Pedigree: sire={result['sire']}, dam_sire={result['dam_sire']}, dam_dam_sire={result['dam_dam_sire']}")
    return result


def _parse_past_results(soup) -> list[dict]:
    """過去成績テーブルをパースする。"""
    results = []

    table = soup.select_one(".db_h_race_results, table.nk_tb_common")
    if not table:
        return results

    for tr in table.select("tr")[1:]:  # ヘッダスキップ
        tds = tr.select("td")
        if len(tds) < 12:
            continue

        row = {}
        # 日付
        date_text = tds[0].get_text(strip=True)
        row["date"] = date_text.replace("/", "")

        # 競馬場
        row["place"] = tds[1].get_text(strip=True)

        # レース名
        race_link = tds[4].select_one("a")
        if race_link:
            row["race_name"] = race_link.get_text(strip=True)
            href = race_link.get("href", "")
            rid_m = re.search(r"/race/(\d+)", href)
            row["race_id"] = rid_m.group(1) if rid_m else ""
        else:
            row["race_name"] = tds[4].get_text(strip=True)
            row["race_id"] = ""

        # 頭数
        runners_text = tds[6].get_text(strip=True) if len(tds) > 6 else ""
        row["num_runners"] = int(runners_text) if runners_text.isdigit() else 0

        # 枠番
        bracket_text = tds[7].get_text(strip=True) if len(tds) > 7 else ""
        row["bracket"] = int(bracket_text) if bracket_text.isdigit() else 0

        # 馬番
        number_text = tds[8].get_text(strip=True) if len(tds) > 8 else ""
        row["number"] = int(number_text) if number_text.isdigit() else 0

        # 着順
        finish_text = tds[11].get_text(strip=True) if len(tds) > 11 else ""
        try:
            row["finish_position"] = int(finish_text)
        except ValueError:
            row["finish_position"] = 0

        # 斤量
        impost_text = tds[13].get_text(strip=True) if len(tds) > 13 else ""
        try:
            row["impost"] = float(impost_text)
        except ValueError:
            row["impost"] = 0.0

        # 距離・芝ダート
        dist_text = tds[14].get_text(strip=True) if len(tds) > 14 else ""
        if dist_text:
            if dist_text.startswith("芝"):
                row["surface"] = "芝"
            elif dist_text.startswith("ダ"):
                row["surface"] = "ダート"
            else:
                row["surface"] = "不明"
            dist_m = re.search(r"(\d{3,4})", dist_text)
            row["distance"] = int(dist_m.group(1)) if dist_m else 0
        else:
            row["surface"] = "不明"
            row["distance"] = 0

        # タイム
        time_text = tds[17].get_text(strip=True) if len(tds) > 17 else ""
        row["time_str"] = time_text
        if time_text and ":" in time_text:
            try:
                parts = time_text.split(":")
                row["time"] = float(parts[0]) * 60 + float(parts[1])
            except (ValueError, IndexError):
                row["time"] = 0.0
        else:
            row["time"] = 0.0

        # 上がり3F
        agari_text = tds[22].get_text(strip=True) if len(tds) > 22 else ""
        try:
            row["last_3f"] = float(agari_text)
        except ValueError:
            row["last_3f"] = 0.0

        # 馬体重
        weight_text = tds[23].get_text(strip=True) if len(tds) > 23 else ""
        weight_m = re.search(r"(\d{3,4})\(([+-]?\d+)\)", weight_text)
        if weight_m:
            row["horse_weight"] = int(weight_m.group(1))
            row["weight_change"] = int(weight_m.group(2))
        else:
            row["horse_weight"] = 0
            row["weight_change"] = 0

        # 単勝オッズ
        odds_text = tds[9].get_text(strip=True) if len(tds) > 9 else ""
        try:
            row["win_odds"] = float(odds_text)
        except ValueError:
            row["win_odds"] = 0.0

        # 騎手
        jockey_td = tds[12] if len(tds) > 12 else None
        if jockey_td:
            j_link = jockey_td.select_one("a")
            if j_link:
                row["jockey_name"] = j_link.get_text(strip=True)
                href = j_link.get("href", "")
                jid = re.search(r"/jockey/(?:result/recent/)?(\w+)", href)
                row["jockey_id"] = jid.group(1) if jid else ""
            else:
                row["jockey_name"] = jockey_td.get_text(strip=True)
                row["jockey_id"] = ""
        else:
            row["jockey_name"] = ""
            row["jockey_id"] = ""

        # 馬場状態
        cond_text = tds[15].get_text(strip=True) if len(tds) > 15 else ""
        row["track_condition"] = cond_text

        results.append(row)

    return results
