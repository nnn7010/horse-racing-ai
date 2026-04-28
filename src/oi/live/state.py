"""当日状態（結果入力・バイアス推定・予測キャッシュ）の永続化。

優先順:
  1. Supabase 環境変数(SUPABASE_URL, SUPABASE_KEY)があればそこに保存
  2. なければローカルSQLite (data/oi/live/state.db) に保存

Streamlit Cloud で運用する場合は Supabase Free を使う想定。
ローカル運用ならSQLiteで完結する。
"""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
LOCAL_DB = ROOT / "data" / "oi" / "live" / "state.db"
LOCAL_DB.parent.mkdir(parents=True, exist_ok=True)


def _has_supabase() -> bool:
    return bool(os.environ.get("SUPABASE_URL") and os.environ.get("SUPABASE_KEY"))


# ---- ローカルSQLite実装 ----

@contextmanager
def _conn():
    con = sqlite3.connect(LOCAL_DB)
    try:
        yield con
    finally:
        con.commit()
        con.close()


def _ensure_local_schema() -> None:
    with _conn() as con:
        con.executescript(
            """
            CREATE TABLE IF NOT EXISTS oi_live_results (
                race_date TEXT NOT NULL,
                race_no INTEGER NOT NULL,
                finish_order TEXT NOT NULL,
                source TEXT DEFAULT 'manual',
                inserted_at TEXT DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (race_date, race_no)
            );
            CREATE TABLE IF NOT EXISTS oi_live_bias (
                race_date TEXT PRIMARY KEY,
                bias_inner REAL,
                bias_front REAL,
                inner_top3_rate REAL,
                outer_top3_rate REAL,
                front_top3_rate REAL,
                late_top3_rate REAL,
                sample_size INTEGER,
                computed_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS oi_live_predictions (
                race_date TEXT NOT NULL,
                race_no INTEGER NOT NULL,
                payload TEXT NOT NULL,
                computed_at TEXT DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (race_date, race_no, computed_at)
            );
            """
        )


_ensure_local_schema()


# ---- Supabase実装 ----

def _supabase_client():
    from supabase import create_client
    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_KEY"]
    return create_client(url, key)


# ---- パブリックAPI ----

def upsert_live_result(race_date: date, race_no: int, finish_order: list[int], source: str = "manual") -> None:
    finish_str = ",".join(str(x) for x in finish_order)
    if _has_supabase():
        sb = _supabase_client()
        sb.table("oi_live_results").upsert({
            "race_date": race_date.isoformat(),
            "race_no": race_no,
            "finish_order": finish_str,
            "source": source,
        }).execute()
    else:
        with _conn() as con:
            con.execute(
                "INSERT OR REPLACE INTO oi_live_results (race_date, race_no, finish_order, source) VALUES (?, ?, ?, ?)",
                (race_date.isoformat(), race_no, finish_str, source),
            )


def get_live_results(race_date: date) -> list[dict]:
    if _has_supabase():
        sb = _supabase_client()
        res = sb.table("oi_live_results").select("*").eq("race_date", race_date.isoformat()).order("race_no").execute()
        return res.data or []
    with _conn() as con:
        cur = con.execute(
            "SELECT race_date, race_no, finish_order, source, inserted_at FROM oi_live_results WHERE race_date = ? ORDER BY race_no",
            (race_date.isoformat(),),
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


def upsert_live_bias(race_date: date, bias: dict[str, Any]) -> None:
    if _has_supabase():
        sb = _supabase_client()
        sb.table("oi_live_bias").upsert({"race_date": race_date.isoformat(), **bias}).execute()
    else:
        with _conn() as con:
            con.execute(
                """
                INSERT OR REPLACE INTO oi_live_bias
                (race_date, bias_inner, bias_front, inner_top3_rate, outer_top3_rate, front_top3_rate, late_top3_rate, sample_size)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    race_date.isoformat(),
                    bias.get("bias_inner"),
                    bias.get("bias_front"),
                    bias.get("inner_top3_rate"),
                    bias.get("outer_top3_rate"),
                    bias.get("front_top3_rate"),
                    bias.get("late_top3_rate"),
                    bias.get("sample_size"),
                ),
            )


def get_live_bias(race_date: date) -> dict | None:
    if _has_supabase():
        sb = _supabase_client()
        res = sb.table("oi_live_bias").select("*").eq("race_date", race_date.isoformat()).limit(1).execute()
        return res.data[0] if res.data else None
    with _conn() as con:
        cur = con.execute(
            "SELECT * FROM oi_live_bias WHERE race_date = ?",
            (race_date.isoformat(),),
        )
        cols = [d[0] for d in cur.description]
        row = cur.fetchone()
        return dict(zip(cols, row)) if row else None


def upsert_prediction(race_date: date, race_no: int, payload: dict) -> None:
    if _has_supabase():
        sb = _supabase_client()
        sb.table("oi_live_predictions").insert({
            "race_date": race_date.isoformat(),
            "race_no": race_no,
            "payload": payload,
        }).execute()
    else:
        with _conn() as con:
            con.execute(
                "INSERT INTO oi_live_predictions (race_date, race_no, payload) VALUES (?, ?, ?)",
                (race_date.isoformat(), race_no, json.dumps(payload, ensure_ascii=False)),
            )


def get_latest_prediction(race_date: date, race_no: int) -> dict | None:
    if _has_supabase():
        sb = _supabase_client()
        res = (
            sb.table("oi_live_predictions")
            .select("*")
            .eq("race_date", race_date.isoformat())
            .eq("race_no", race_no)
            .order("computed_at", desc=True)
            .limit(1)
            .execute()
        )
        return res.data[0] if res.data else None
    with _conn() as con:
        cur = con.execute(
            "SELECT payload FROM oi_live_predictions WHERE race_date = ? AND race_no = ? ORDER BY computed_at DESC LIMIT 1",
            (race_date.isoformat(), race_no),
        )
        row = cur.fetchone()
        return json.loads(row[0]) if row else None
