"""当日予測パイプライン: shutuba取得 → 特徴量化 → 予測 → JSON出力。

Usage:
  python scripts/oi/08_predict_today.py --date 2026-04-29

出力:
  outputs/oi/today_{YYYY-MM-DD}.json
    Streamlitアプリ(oi_app.py)が読み込む形式
"""

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.oi import load_config
from src.oi.bias.estimator import BiasEstimate
from src.oi.features.build import _build_runner_row
from src.oi.live import state as live_state
from src.oi.models.predictor import load_models, predict_race, race_betting_table
from src.oi.scraping.calendar import fetch_race_ids_for_date
from src.oi.scraping.horse import fetch_horse_info
from src.oi.scraping.shutuba import fetch_shutuba
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _shutuba_to_features(shutuba: dict, horse_cache: dict[str, dict], bias: dict | None,
                         jockey_stats: dict, trainer_stats: dict) -> pd.DataFrame:
    """shutubaから特徴量行を作る（_build_runner_rowを再利用）。"""
    rows = []
    race_meta = {
        "race_id": shutuba["race_id"],
        "date": shutuba["date"],
        "distance": shutuba.get("distance", 0),
        "surface": shutuba.get("surface", "ダート"),
        "track_condition": shutuba.get("track_condition", ""),
        "weather": shutuba.get("weather", ""),
        "num_runners": shutuba.get("num_runners", 0),
    }
    for entry in shutuba.get("entries", []):
        # entryをresult行と同じ形に合わせる（finish_position=0）
        runner_like = {**entry, "finish_position": 0, "passing": "", "last_3f": 0.0,
                        "time": 0.0, "time_str": "", "margin": ""}
        horse_info = horse_cache.get(entry.get("horse_id", ""))
        row = _build_runner_row(
            race_meta, runner_like, horse_info, bias,
            jockey_stats.get(entry.get("jockey_id", "")),
            trainer_stats.get(entry.get("trainer_id", "")),
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _load_jockey_trainer_stats(raw_dir: Path) -> tuple[dict, dict]:
    """学習時に集計した騎手・調教師成績を再計算（メモリ内、簡易版）。"""
    from collections import defaultdict
    jockey: dict[str, list[dict]] = defaultdict(list)
    trainer: dict[str, list[dict]] = defaultdict(list)
    for jp in (raw_dir / "results").glob("*.json"):
        with open(jp, encoding="utf-8") as f:
            race = json.load(f)
        for r in race.get("results", []):
            if r.get("jockey_id"):
                jockey[r["jockey_id"]].append(r)
            if r.get("trainer_id"):
                trainer[r["trainer_id"]].append(r)

    def _agg(rows: list[dict]) -> dict:
        valid = [r for r in rows if r.get("finish_position", 0) > 0]
        if not valid:
            return {"runs": 0, "top3_rate": 0.0}
        top3 = sum(1 for r in valid if 1 <= r.get("finish_position", 0) <= 3)
        return {"runs": len(valid), "top3_rate": top3 / len(valid)}

    return ({k: _agg(v) for k, v in jockey.items()},
            {k: _agg(v) for k, v in trainer.items()})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, required=True, help="YYYY-MM-DD")
    args = parser.parse_args()

    target_date = date.fromisoformat(args.date)
    cfg = load_config()
    raw_dir = ROOT / cfg["paths"]["raw"]
    out_dir = ROOT / cfg["paths"]["outputs"]
    out_dir.mkdir(parents=True, exist_ok=True)

    # 当日のレースIDリスト
    race_ids = fetch_race_ids_for_date(target_date)
    if not race_ids:
        logger.error(f"{target_date} に大井のレースが見つかりません")
        sys.exit(1)

    # shutuba取得
    shutubas = []
    horse_ids: set[str] = set()
    for rid in race_ids:
        try:
            s = fetch_shutuba(rid)
            shutubas.append(s)
            for e in s.get("entries", []):
                if e.get("horse_id"):
                    horse_ids.add(e["horse_id"])
        except Exception as e:
            logger.warning(f"shutuba {rid} 取得失敗: {e}")

    # 馬個体取得（キャッシュ済みは再フェッチしない）
    horses_dir = raw_dir / "horses"
    horses_dir.mkdir(parents=True, exist_ok=True)
    horse_cache: dict[str, dict] = {}
    for hid in horse_ids:
        cache_path = horses_dir / f"{hid}.json"
        if cache_path.exists():
            with open(cache_path, encoding="utf-8") as f:
                horse_cache[hid] = json.load(f)
        else:
            try:
                info = fetch_horse_info(hid)
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(info, f, ensure_ascii=False, indent=2)
                horse_cache[hid] = info
            except Exception as e:
                logger.warning(f"horse {hid} 取得失敗: {e}")

    # 騎手・調教師成績
    jockey_stats, trainer_stats = _load_jockey_trainer_stats(raw_dir)

    # 当日バイアス（live_stateにあれば使用、なければNone）
    bias = live_state.get_live_bias(target_date)

    # モデルロード
    models = load_models(ROOT / cfg["paths"]["models"])

    # 各レース予測
    output: dict[str, dict] = {}
    for shutuba in shutubas:
        df_feat = _shutuba_to_features(shutuba, horse_cache, bias, jockey_stats, trainer_stats)
        if len(df_feat) == 0:
            continue
        df_pred = predict_race(df_feat, models)
        report = race_betting_table(
            df_pred,
            win_ev_threshold=cfg["betting"]["win_ev_threshold"],
            partner_count=cfg["betting"]["top3_partner_count"],
        )
        # メタ情報を追加
        report.update({
            "race_id": shutuba["race_id"],
            "race_name": shutuba.get("race_name", ""),
            "distance": shutuba.get("distance", 0),
            "surface": shutuba.get("surface", ""),
            "num_runners": shutuba.get("num_runners", 0),
        })
        # JSONシリアライズ可能化（tupleキーをstr化）
        if "trifecta_axis_first" in report:
            report["trifecta_axis_first"] = {
                f"{k[0]}-{k[1]}-{k[2]}": v for k, v in report["trifecta_axis_first"].items()
            }
        output[str(shutuba["race_no"])] = report

        # Supabaseにキャッシュ
        try:
            live_state.upsert_prediction(target_date, shutuba["race_no"], report)
        except Exception as e:
            logger.warning(f"prediction永続化失敗: {e}")

    # JSON出力
    out_path = out_dir / f"today_{target_date.isoformat()}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)
    logger.info(f"保存: {out_path} ({len(output)}レース)")


if __name__ == "__main__":
    main()
