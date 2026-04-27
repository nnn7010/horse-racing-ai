"""nar.netkeiba の大井レース結果ページのパーサ。

URL: https://nar.netkeiba.com/race/result.html?race_id={race_id}
race_id 構造: YYYY(4) + 場コード(2) + MM(2) + DD(2) + RR(2)  例: 202544122909
"""

import re
from datetime import date

from bs4 import BeautifulSoup

from src.oi.scraping.http import fetch
from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_race_id(race_id: str) -> dict:
    """race_id から日付・場コード・レース番号を抽出する。"""
    if len(race_id) != 12 or not race_id.isdigit():
        raise ValueError(f"Invalid race_id: {race_id}")
    year = int(race_id[0:4])
    place = race_id[4:6]
    month = int(race_id[6:8])
    day = int(race_id[8:10])
    race_no = int(race_id[10:12])
    return {
        "race_id": race_id,
        "date": date(year, month, day),
        "place_code": place,
        "race_no": race_no,
    }


def fetch_race_result(race_id: str) -> dict:
    """大井レース結果ページから着順・タイム・払戻等を取得する。"""
    url = f"https://nar.netkeiba.com/race/result.html?race_id={race_id}"
    html = fetch(url, encoding="utf-8")
    soup = BeautifulSoup(html, "lxml")

    meta = parse_race_id(race_id)
    result = {
        "race_id": race_id,
        "date": meta["date"].strftime("%Y%m%d"),
        "race_no": meta["race_no"],
        "results": [],
    }

    # レース名・条件
    race_name_tag = soup.select_one(".RaceName")
    result["race_name"] = race_name_tag.get_text(strip=True) if race_name_tag else ""

    race_data_box = soup.select_one(".RaceData01, .RaceList_Item02")
    if race_data_box:
        info_text = race_data_box.get_text(" ", strip=True)
        result["course_info"] = info_text

        # 距離・コース種別（地方競馬は基本ダート、稀に芝）
        if "芝" in info_text:
            result["surface"] = "芝"
        elif "ダ" in info_text:
            result["surface"] = "ダート"
        else:
            result["surface"] = "ダート"  # 大井はダートのみだが念のため
        dist_m = re.search(r"(\d{3,4})m", info_text)
        result["distance"] = int(dist_m.group(1)) if dist_m else 0

        # 馬場状態
        cond_m = re.search(r"馬場\s*[:：]?\s*(良|稍重|重|不良)", info_text)
        result["track_condition"] = cond_m.group(1) if cond_m else ""

        # 天候
        weather_m = re.search(r"天候\s*[:：]?\s*(晴|曇|雨|小雨|雪)", info_text)
        result["weather"] = weather_m.group(1) if weather_m else ""

        # 発走時刻
        time_m = re.search(r"(\d{1,2}):(\d{2})", info_text)
        result["post_time"] = f"{time_m.group(1)}:{time_m.group(2)}" if time_m else ""

        # 内回り/外回り判定（大井は1周コースだが念のため記録）
        if "外" in info_text:
            result["course_type"] = "外"
        elif "内" in info_text:
            result["course_type"] = "内"
        else:
            result["course_type"] = ""

    # クラス・条件
    race_data02 = soup.select_one(".RaceData02")
    if race_data02:
        spans = race_data02.select("span")
        if len(spans) >= 5:
            result["race_class"] = spans[4].get_text(strip=True)
        else:
            result["race_class"] = ""
    else:
        result["race_class"] = ""

    # 障害・新馬は除外フラグ
    full_text = result.get("race_name", "") + result.get("race_class", "")
    result["is_hurdle"] = "障害" in full_text
    result["is_debut"] = "新馬" in full_text

    # 結果テーブル
    result_table = soup.select_one("table.RaceTable01, table.ResultsByRaceTable")
    if not result_table:
        # フォールバック: 一般的なtable検索
        for table in soup.select("table"):
            if table.select_one("a[href*='/horse/']"):
                result_table = table
                break

    num_runners = 0
    if result_table:
        for tr in result_table.select("tbody tr"):
            row = _parse_result_row(tr, race_id)
            if row:
                result["results"].append(row)
                num_runners += 1

    result["num_runners"] = num_runners

    # 払戻
    result["payouts"] = _parse_payouts(soup)

    return result


def _parse_result_row(tr, race_id: str) -> dict | None:
    """結果テーブル1行をパースする。"""
    tds = tr.select("td")
    if len(tds) < 8:
        return None

    row: dict = {"race_id": race_id}

    def _txt(idx: int) -> str:
        return tds[idx].get_text(strip=True) if idx < len(tds) else ""

    # 着順
    finish_text = _txt(0)
    try:
        row["finish_position"] = int(finish_text)
    except ValueError:
        row["finish_position"] = 0  # 中止・除外等

    # 枠番
    row["bracket"] = int(_txt(1)) if _txt(1).isdigit() else 0
    # 馬番
    row["number"] = int(_txt(2)) if _txt(2).isdigit() else 0

    # 馬名・馬ID
    horse_link = tr.select_one("a[href*='/horse/']")
    if horse_link:
        row["horse_name"] = horse_link.get_text(strip=True)
        href = horse_link.get("href", "")
        hid = re.search(r"/horse/(\w+)", href)
        row["horse_id"] = hid.group(1) if hid else ""
    else:
        row["horse_name"] = ""
        row["horse_id"] = ""

    # 性齢
    row["sex_age"] = _txt(4)

    # 斤量
    impost_text = _txt(5)
    try:
        row["impost"] = float(impost_text)
    except ValueError:
        row["impost"] = 0.0

    # 騎手
    jockey_link = tr.select_one("a[href*='/jockey/']")
    if jockey_link:
        row["jockey_name"] = jockey_link.get_text(strip=True)
        href = jockey_link.get("href", "")
        jid = re.search(r"/jockey/(?:result/)?(\w+)", href)
        row["jockey_id"] = jid.group(1) if jid else ""
    else:
        row["jockey_name"] = _txt(6)
        row["jockey_id"] = ""

    # タイム
    time_text = _txt(7)
    row["time_str"] = time_text
    row["time"] = _parse_time(time_text)

    # 着差
    row["margin"] = _txt(8)

    # 通過順（大井は4角まで）
    passing_idx = _find_col_by_pattern(tds, r"^\d+(-\d+)*$", start=10)
    row["passing"] = _txt(passing_idx) if passing_idx >= 0 else ""

    # 上がり3F
    last_3f_idx = _find_col_by_header_neighbor(tds, prefer=15)
    last_3f_text = _txt(last_3f_idx) if last_3f_idx >= 0 else ""
    try:
        row["last_3f"] = float(last_3f_text)
    except ValueError:
        row["last_3f"] = 0.0

    # 単勝オッズ・人気・馬体重・調教師
    # 列位置がレイアウトにより微妙に異なるため、tr全体から取得
    odds_text = ""
    pop_text = ""
    weight_text = ""
    for td in tds:
        t = td.get_text(strip=True)
        # 単勝オッズらしき値（小数点ありの数値）
        if not odds_text and re.fullmatch(r"\d{1,4}\.\d{1,2}", t):
            odds_text = t
        elif not pop_text and re.fullmatch(r"\d{1,2}", t) and 1 <= int(t) <= 18:
            # 人気らしき1-18の整数（但し馬番と被るので確実ではない）
            pop_text = t
        # 馬体重 例: 480(+2)
        wt_m = re.search(r"(\d{3,4})\(([+-]?\d+)\)", t)
        if wt_m and not weight_text:
            weight_text = t

    try:
        row["win_odds"] = float(odds_text) if odds_text else 0.0
    except ValueError:
        row["win_odds"] = 0.0

    if weight_text:
        wt_m = re.search(r"(\d{3,4})\(([+-]?\d+)\)", weight_text)
        if wt_m:
            row["horse_weight"] = int(wt_m.group(1))
            row["weight_change"] = int(wt_m.group(2))
        else:
            row["horse_weight"] = 0
            row["weight_change"] = 0
    else:
        row["horse_weight"] = 0
        row["weight_change"] = 0

    # 人気は popularity 列を本来 dedicated に取りたいが、地方版のレイアウトを実物確認後に微調整
    row["popularity"] = 0

    # 調教師
    trainer_link = tr.select_one("a[href*='/trainer/']")
    if trainer_link:
        row["trainer_name"] = trainer_link.get_text(strip=True)
        href = trainer_link.get("href", "")
        tid = re.search(r"/trainer/(?:result/)?(\w+)", href)
        row["trainer_id"] = tid.group(1) if tid else ""
    else:
        row["trainer_name"] = ""
        row["trainer_id"] = ""

    return row


def _find_col_by_pattern(tds, pattern: str, start: int = 0) -> int:
    rx = re.compile(pattern)
    for i in range(start, len(tds)):
        if rx.fullmatch(tds[i].get_text(strip=True)):
            return i
    return -1


def _find_col_by_header_neighbor(tds, prefer: int) -> int:
    """テーブルレイアウト揺れ吸収用。preferの位置がfloatっぽければそこを返す。"""
    if prefer < len(tds):
        t = tds[prefer].get_text(strip=True)
        if re.fullmatch(r"\d{1,3}\.\d", t):
            return prefer
    # 走査して上がり3F相当（30〜45秒台の小数1桁）を探す
    for i, td in enumerate(tds):
        t = td.get_text(strip=True)
        if re.fullmatch(r"\d{2}\.\d", t):
            try:
                v = float(t)
                if 30.0 <= v <= 50.0:
                    return i
            except ValueError:
                pass
    return -1


def _parse_time(time_str: str) -> float:
    """1:23.4 → 83.4"""
    if not time_str:
        return 0.0
    try:
        if ":" in time_str:
            parts = time_str.split(":")
            return float(parts[0]) * 60 + float(parts[1])
        return float(time_str)
    except (ValueError, IndexError):
        return 0.0


def _parse_payouts(soup) -> dict:
    """払戻情報を抽出。"""
    payouts: dict = {}

    pay_tables = soup.select("table.Payout_Detail_Table, table.pay_table_01")
    for table in pay_tables:
        for tr in table.select("tr"):
            ths = tr.select("th")
            tds = tr.select("td")
            if not ths or not tds:
                continue
            bet_type = ths[0].get_text(strip=True)
            key = _bet_type_key(bet_type)
            if not key:
                continue

            # 番号と払戻額
            nums_html = tds[0].decode_contents() if len(tds) > 0 else ""
            amt_html = tds[1].decode_contents() if len(tds) > 1 else ""

            nums_list = [s.strip() for s in re.split(r"<br\s*/?>", nums_html) if s.strip()]
            amt_list = [s.strip() for s in re.split(r"<br\s*/?>", amt_html) if s.strip()]
            nums_list = [re.sub(r"<[^>]+>", "", s).strip() for s in nums_list]
            amt_list = [re.sub(r"<[^>]+>", "", s).replace(",", "").strip() for s in amt_list]

            payouts.setdefault(key, [])
            for i, nums in enumerate(nums_list):
                amt_str = amt_list[i] if i < len(amt_list) else "0"
                amt_digits = re.sub(r"\D", "", amt_str)
                amount = int(amt_digits) if amt_digits else 0
                payouts[key].append({"numbers": nums, "amount": amount})

    return payouts


def _bet_type_key(bet_type: str) -> str | None:
    if "単勝" in bet_type:
        return "win"
    if "複勝" in bet_type:
        return "place"
    if "枠連" in bet_type:
        return "bracket_quinella"
    if "馬連" in bet_type:
        return "quinella"
    if "ワイド" in bet_type:
        return "wide"
    if "馬単" in bet_type:
        return "exacta"
    if "三連複" in bet_type:
        return "trio"
    if "三連単" in bet_type:
        return "trifecta"
    return None
