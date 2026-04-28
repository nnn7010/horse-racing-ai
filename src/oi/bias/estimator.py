"""トラックバイアスの事後推定。

過去レース日次バイアスの計算と、当日途中までの結果から
リアルタイムにバイアス推定を更新する関数。

返す指標:
  - bias_inner: 内枠(1-3) と 外枠(7+) の複勝率差
  - bias_front: 4角通過順 上位30% と 下位30% の複勝率差
  - sample_size: 推定に使ったレース数

これらは特徴量としてモデル入力に使い、
かつ Streamlit上で当日の傾向として可視化する。
"""

from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass
class BiasEstimate:
    date: str
    sample_size: int
    bias_inner: float        # +なら内枠有利
    bias_front: float        # +なら前々有利
    inner_top3_rate: float
    outer_top3_rate: float
    front_top3_rate: float
    late_top3_rate: float

    def to_dict(self) -> dict:
        return asdict(self)


def _passing_first(passing: str) -> int | None:
    """通過順 '5-3-2-1' から最後（4角）の順位を返す。"""
    if not passing:
        return None
    parts = [p for p in passing.split("-") if p.strip().isdigit()]
    if not parts:
        return None
    return int(parts[-1])


def _classify_bracket(bracket: int) -> str:
    if bracket <= 0:
        return "unknown"
    if bracket <= 3:
        return "inner"
    if bracket >= 7:
        return "outer"
    return "middle"


def _classify_position(corner_pos: int | None, num_runners: int) -> str:
    if corner_pos is None or num_runners <= 0:
        return "unknown"
    third = max(1, num_runners // 3)
    if corner_pos <= third:
        return "front"
    if corner_pos >= num_runners - third + 1:
        return "late"
    return "middle"


def estimate_bias(races: list[dict], target_date: str) -> BiasEstimate:
    """指定日のバイアスを推定する。

    races: 同一日のレース結果リスト（fetch_race_resultの戻り値の results 配列を含むdict）
    target_date: "YYYYMMDD"
    """
    inner_top3 = inner_total = 0
    outer_top3 = outer_total = 0
    front_top3 = front_total = 0
    late_top3 = late_total = 0
    total_runs = 0

    for race in races:
        results = race.get("results", [])
        num_runners = race.get("num_runners", len(results))
        for r in results:
            finish = r.get("finish_position", 0)
            if finish <= 0:
                continue  # 中止・除外
            total_runs += 1
            is_top3 = finish <= 3

            bracket_class = _classify_bracket(r.get("bracket", 0))
            if bracket_class == "inner":
                inner_total += 1
                if is_top3:
                    inner_top3 += 1
            elif bracket_class == "outer":
                outer_total += 1
                if is_top3:
                    outer_top3 += 1

            corner = _passing_first(r.get("passing", ""))
            pos_class = _classify_position(corner, num_runners)
            if pos_class == "front":
                front_total += 1
                if is_top3:
                    front_top3 += 1
            elif pos_class == "late":
                late_total += 1
                if is_top3:
                    late_top3 += 1

    inner_rate = inner_top3 / inner_total if inner_total else 0.0
    outer_rate = outer_top3 / outer_total if outer_total else 0.0
    front_rate = front_top3 / front_total if front_total else 0.0
    late_rate = late_top3 / late_total if late_total else 0.0

    return BiasEstimate(
        date=target_date,
        sample_size=len(races),
        bias_inner=inner_rate - outer_rate,
        bias_front=front_rate - late_rate,
        inner_top3_rate=inner_rate,
        outer_top3_rate=outer_rate,
        front_top3_rate=front_rate,
        late_top3_rate=late_rate,
    )


def smooth_bias(prior: BiasEstimate | None, current: BiasEstimate, alpha: float = 0.3) -> BiasEstimate:
    """前日までの傾向と当日推定を指数平滑する（当日サンプル少ない時の安定化）。"""
    if prior is None:
        return current
    return BiasEstimate(
        date=current.date,
        sample_size=current.sample_size,
        bias_inner=alpha * current.bias_inner + (1 - alpha) * prior.bias_inner,
        bias_front=alpha * current.bias_front + (1 - alpha) * prior.bias_front,
        inner_top3_rate=alpha * current.inner_top3_rate + (1 - alpha) * prior.inner_top3_rate,
        outer_top3_rate=alpha * current.outer_top3_rate + (1 - alpha) * prior.outer_top3_rate,
        front_top3_rate=alpha * current.front_top3_rate + (1 - alpha) * prior.front_top3_rate,
        late_top3_rate=alpha * current.late_top3_rate + (1 - alpha) * prior.late_top3_rate,
    )
