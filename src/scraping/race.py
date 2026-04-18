"""過去レース結果のスクレイピング。

netkeiba の race result ページから着順・タイム・オッズ等を取得する。
"""

import re
from datetime import date, datetime

from bs4 import BeautifulSoup

from src.utils.http import fetch
from src.utils.logger import get_logger

logger = get_logger(__name__)

# JRA競馬場コード
JRA_PLACE_CODES = {"01", "02", "03", "04", "05", "06", "07", "08", "09", "10"}


def fetch_race_ids_by_date(target_date: date) -> list[str]:
    """指定日のJRA全レースIDを取得する。"""
    dt_str = target_date.strftime("%Y%m%d")
    url = f"https://db.netkeiba.com/race/list/{dt_str}/"
    html = fetch(url, encoding="euc-jp")

    race_ids = []
    for m in re.finditer(r"/race/(\d{12})/", html):
        rid = m.group(1)
        place_code = rid[4:6]
        if place_code in JRA_PLACE_CODES and rid not in race_ids:
            race_ids.append(rid)
    return race_ids


def search_race_ids(
    place_code: str,
    surface: str,
    distance: int,
    start_date: date,
    end_date: date,
) -> list[str]:
    """指定条件（競馬場×芝ダート×距離）の過去レースIDを日付一覧から収集する。

    db.netkeiba.com/race/list/YYYYMMDD/ を月ごとに巡回し、
    結果ページで条件一致するレースのみを返す。
    """
    from datetime import timedelta

    # 対象期間の開催日を取得するため、月ごとにカレンダーを確認
    all_dates = set()
    current = date(start_date.year, start_date.month, 1)
    end_month = date(end_date.year, end_date.month, 1)
    while current <= end_month:
        url = f"https://race.netkeiba.com/top/calendar.html?year={current.year}&month={current.month}"
        html = fetch(url, encoding="utf-8")
        for m in re.finditer(r"kaisai_date=(\d{8})", html):
            dt_str = m.group(1)
            try:
                dt = date(int(dt_str[:4]), int(dt_str[4:6]), int(dt_str[6:8]))
                if start_date <= dt <= end_date:
                    all_dates.add(dt)
            except ValueError:
                pass
        # 次の月
        if current.month == 12:
            current = date(current.year + 1, 1, 1)
        else:
            current = date(current.year, current.month + 1, 1)

    logger.info(f"Found {len(all_dates)} race dates in {start_date}~{end_date}")

    # 各開催日のレースIDを取得
    all_race_ids = []
    for dt in sorted(all_dates):
        race_ids = fetch_race_ids_by_date(dt)
        # place_codeでフィルタ
        for rid in race_ids:
            rid_place = rid[4:6]
            if rid_place == place_code and rid not in all_race_ids:
                all_race_ids.append(rid)

    logger.info(
        f"Search: place={place_code} {surface}{distance}m "
        f"({start_date}~{end_date}): {len(all_race_ids)} candidates"
    )
    return all_race_ids


def fetch_race_result(race_id: str) -> dict:
    """レース結果ページから全馬の着順・タイム等を取得する。"""
    url = f"https://db.netkeiba.com/race/{race_id}/"
    html = fetch(url, encoding="euc-jp")
    soup = BeautifulSoup(html, "lxml")

    result = {"race_id": race_id, "results": []}

    # レース情報ヘッダ
    diary_snap = soup.select_one(".diary_snap, .data_intro")
    if diary_snap:
        header_text = diary_snap.get_text()
        # 日付
        date_m = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", header_text)
        if date_m:
            result["date"] = f"{date_m.group(1)}{int(date_m.group(2)):02d}{int(date_m.group(3)):02d}"
        # レース名
        name_tag = diary_snap.select_one(".racedata dt, h1")
        result["race_name"] = name_tag.get_text(strip=True) if name_tag else ""

    # コース情報
    race_info_span = soup.select_one(".racedata span, .smalltxt")
    if race_info_span:
        info_text = race_info_span.get_text(strip=True)
        result["course_info"] = info_text
        if "芝" in info_text:
            result["surface"] = "芝"
        elif "ダ" in info_text:
            result["surface"] = "ダート"
        else:
            result["surface"] = "不明"
        dist_m = re.search(r"(\d{4})m", info_text)
        result["distance"] = int(dist_m.group(1)) if dist_m else 0
        # 馬場状態
        ba_m = re.search(r"馬場:(\S+)", info_text)
        if not ba_m:
            ba_m = re.search(r"[：:]\s*(良|稍重|重|不良)", info_text)
        result["track_condition"] = ba_m.group(1) if ba_m else ""
        # 天候
        weather_m = re.search(r"天候:(\S+)", info_text)
        if not weather_m:
            weather_m = re.search(r"[：:]\s*(晴|曇|雨|小雨|雪)", info_text)
        result["weather"] = weather_m.group(1) if weather_m else ""
    else:
        result["surface"] = "不明"
        result["distance"] = 0
        result["track_condition"] = ""
        result["weather"] = ""

    # 新馬・障害チェック
    full_text = result.get("race_name", "") + result.get("course_info", "")
    result["is_debut"] = "新馬" in full_text
    result["is_hurdle"] = "障害" in full_text

    # 頭数
    num_runners = 0

    # 結果テーブル
    table = soup.select_one(".race_table_01, table.nk_tb_common")
    if table:
        for tr in table.select("tr")[1:]:  # ヘッダスキップ
            row = _parse_result_row(tr)
            if row:
                row["race_id"] = race_id
                result["results"].append(row)
                num_runners += 1

    result["num_runners"] = num_runners

    # 払い戻し情報
    result["payouts"] = _parse_payouts(soup)

    return result


def _parse_result_row(tr) -> dict | None:
    """結果テーブルの1行をパースする。"""
    tds = tr.select("td")
    if len(tds) < 8:
        return None

    row = {}

    # 着順
    finish_text = tds[0].get_text(strip=True)
    try:
        row["finish_position"] = int(finish_text)
    except ValueError:
        row["finish_position"] = 0  # 除外・中止等

    # 枠番
    bracket_text = tds[1].get_text(strip=True)
    row["bracket"] = int(bracket_text) if bracket_text.isdigit() else 0

    # 馬番
    number_text = tds[2].get_text(strip=True)
    row["number"] = int(number_text) if number_text.isdigit() else 0

    # 馬名・馬ID
    horse_link = tr.select_one("a[href*='/horse/']")
    if horse_link:
        row["horse_name"] = horse_link.get_text(strip=True)
        href = horse_link.get("href", "")
        hid = re.search(r"/horse/(\w+)", href)
        row["horse_id"] = hid.group(1) if hid else ""
    else:
        row["horse_name"] = tds[3].get_text(strip=True)
        row["horse_id"] = ""

    # 性齢
    row["sex_age"] = tds[4].get_text(strip=True) if len(tds) > 4 else ""

    # 斤量
    weight_text = tds[5].get_text(strip=True) if len(tds) > 5 else ""
    try:
        row["impost"] = float(weight_text)
    except ValueError:
        row["impost"] = 0.0

    # 騎手
    jockey_link = tr.select_one("a[href*='/jockey/']")
    if jockey_link:
        row["jockey_name"] = jockey_link.get_text(strip=True)
        href = jockey_link.get("href", "")
        jid = re.search(r"/jockey/(?:result/recent/)?(\w+)", href)
        row["jockey_id"] = jid.group(1) if jid else ""
    else:
        row["jockey_name"] = tds[6].get_text(strip=True) if len(tds) > 6 else ""
        row["jockey_id"] = ""

    # タイム
    time_text = tds[7].get_text(strip=True) if len(tds) > 7 else ""
    row["time"] = _parse_time(time_text)
    row["time_str"] = time_text

    # 着差
    row["margin"] = tds[8].get_text(strip=True) if len(tds) > 8 else ""

    # 単勝オッズ
    if len(tds) > 9:
        odds_text = tds[9].get_text(strip=True) if len(tds) > 9 else ""
        # 場合によっては12列目あたりにオッズ
        # 通過順の列の後にオッズがある
        pass

    # 通過順位
    if len(tds) > 10:
        row["passing"] = tds[10].get_text(strip=True)
    else:
        row["passing"] = ""

    # 上がり3F
    if len(tds) > 11:
        agari_text = tds[11].get_text(strip=True)
        try:
            row["last_3f"] = float(agari_text)
        except ValueError:
            row["last_3f"] = 0.0
    else:
        row["last_3f"] = 0.0

    # 単勝オッズ（テーブル構造により位置が変わる）
    for i in range(12, min(len(tds), 16)):
        text = tds[i].get_text(strip=True).replace(",", "")
        try:
            val = float(text)
            if 1.0 <= val <= 9999.0:
                row["win_odds"] = val
                break
        except ValueError:
            continue
    if "win_odds" not in row:
        row["win_odds"] = 0.0

    # 馬体重
    for i in range(len(tds) - 1, max(len(tds) - 5, 7), -1):
        text = tds[i].get_text(strip=True)
        weight_m = re.search(r"(\d{3,4})\(([+-]?\d+)\)", text)
        if weight_m:
            row["horse_weight"] = int(weight_m.group(1))
            row["weight_change"] = int(weight_m.group(2))
            break
    if "horse_weight" not in row:
        row["horse_weight"] = 0
        row["weight_change"] = 0

    # 調教師
    trainer_link = tr.select_one("a[href*='/trainer/']")
    if trainer_link:
        row["trainer_name"] = trainer_link.get_text(strip=True)
        href = trainer_link.get("href", "")
        tid = re.search(r"/trainer/(?:result/recent/)?(\w+)", href)
        row["trainer_id"] = tid.group(1) if tid else ""
    else:
        row["trainer_name"] = ""
        row["trainer_id"] = ""

    return row


def _parse_time(time_str: str) -> float:
    """タイム文字列を秒数に変換する。例: '1:23.4' -> 83.4"""
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
    """払い戻し情報をパースする。"""
    payouts = {}

    payout_tables = soup.select(".pay_table_01, table.pay_block")
    for table in payout_tables:
        for tr in table.select("tr"):
            tds = tr.select("td, th")
            if len(tds) < 3:
                continue
            bet_type = tds[0].get_text(strip=True)
            numbers = tds[1].get_text(strip=True)
            amount = tds[2].get_text(strip=True).replace(",", "").replace("円", "")

            if "単勝" in bet_type:
                key = "win"
            elif "複勝" in bet_type:
                key = "place"
            elif "三連複" in bet_type:
                key = "trio"
            elif "三連単" in bet_type:
                key = "trifecta"
            else:
                continue

            try:
                if key not in payouts:
                    payouts[key] = []
                payouts[key].append({
                    "numbers": numbers,
                    "amount": int(re.sub(r"\D", "", amount)) if amount else 0,
                })
            except ValueError:
                pass

    return payouts
