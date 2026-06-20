from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import pandas as pd

from src.country_profiles import EconomyProfile
from src.data_sources import configured_placeholders, fetch_economy_data, fetch_fred_series
from src.indicators import build_indicators
from src.policy_signal import build_policy_signal


@dataclass(frozen=True)
class BacktestRow:
    date: pd.Timestamp
    signal: str
    policy_score: float
    confidence: int
    data_coverage: str
    missing_data: str


@dataclass(frozen=True)
class ComparisonRow:
    date: pd.Timestamp
    model_signal: str
    actual_same_month: str
    hit: bool | None


@dataclass(frozen=True)
class ForwardHitRow:
    date: pd.Timestamp
    model_signal: str
    window_months: int
    actual_forward_move: str
    hit: bool | None


@dataclass(frozen=True)
class RegimeRow:
    date: pd.Timestamp
    model_signal: str
    actual_regime: str
    six_month_rate_change_bp: float | None
    hit: bool | None


@dataclass(frozen=True)
class LeadLagCycle:
    direction: str
    cycle_start: pd.Timestamp
    cycle_end: pd.Timestamp
    first_model_signal: pd.Timestamp | None
    lead_lag_months: int | None


@dataclass(frozen=True)
class BacktestMetrics:
    same_month_match_rate: float | None
    forward_hit_rates: dict[tuple[str, int], float | None]
    regime_match_rate: float | None
    average_easing_lead_lag: float | None
    average_tightening_lead_lag: float | None


def run_backtest(
    economy_code: str,
    profile: EconomyProfile,
    config: dict[str, Any],
    project_dir: Path,
    start_year: int = 2010,
) -> str:
    data, data_warnings = fetch_economy_data(config, economy_code)
    placeholders = configured_placeholders(config, economy_code)
    signal_rows = generate_signal_history(economy_code, profile, config, data, data_warnings, placeholders, start_year)

    policy_series, policy_warning = fetch_actual_policy_series(config, economy_code)
    comparison_rows = compare_to_actual_actions(signal_rows, policy_series, config, economy_code)
    forward_rows = evaluate_forward_hits(signal_rows, policy_series, config, economy_code)
    regime_rows = evaluate_regimes(signal_rows, policy_series)
    lead_lag_cycles = analyze_lead_lag_cycles(signal_rows, policy_series)
    metrics = calculate_metrics(comparison_rows, forward_rows, regime_rows, lead_lag_cycles)

    output_dir = project_dir / "outputs"
    output_dir.mkdir(exist_ok=True)
    save_signal_history(signal_rows, output_dir, economy_code)
    save_comparison_table(comparison_rows, output_dir, economy_code)
    save_forward_hit_results(forward_rows, output_dir, economy_code)
    save_regime_comparison(regime_rows, output_dir, economy_code)
    save_lead_lag_cycles(lead_lag_cycles, output_dir, economy_code)
    chart_warnings = save_charts(signal_rows, policy_series, regime_rows, forward_rows, output_dir, economy_code)

    warnings = list(data_warnings)
    if policy_warning:
        warnings.append(policy_warning)
    warnings.extend(chart_warnings)
    if placeholders:
        warnings.extend(f"TODO data source not integrated: {placeholder}." for placeholder in placeholders)

    return render_backtest_results(economy_code, start_year, signal_rows, metrics, warnings, output_dir)


def generate_signal_history(
    economy_code: str,
    profile: EconomyProfile,
    config: dict[str, Any],
    data: dict[str, pd.Series],
    data_warnings: list[str],
    placeholders: list[str],
    start_year: int,
) -> list[BacktestRow]:
    today = pd.Timestamp.today().normalize()
    start = pd.Timestamp(year=start_year, month=1, day=31)
    dates = pd.date_range(start=start, end=today, freq="ME")
    rows: list[BacktestRow] = []

    for date in dates:
        point_in_time_data = {
            name: series.loc[series.index <= date]
            for name, series in data.items()
            if not series.empty
        }
        indicators = build_indicators(economy_code, point_in_time_data)
        signal = build_policy_signal(indicators, profile, config, data_warnings, placeholders)
        blocks = {
            "inflation": signal.inflation,
            "labor": signal.labor,
            "growth": signal.growth,
            "financial": signal.financial,
            "currency": signal.currency,
        }
        available = sum(block.available for block in blocks.values())
        expected = sum(block.expected for block in blocks.values())
        rows.append(
            BacktestRow(
                date=date,
                signal=signal.signal,
                policy_score=signal.policy_score,
                confidence=signal.confidence,
                data_coverage=f"{available}/{expected}",
                missing_data=missing_data_summary(blocks),
            )
        )

    return rows


def fetch_actual_policy_series(config: dict[str, Any], economy_code: str) -> tuple[pd.Series, str | None]:
    economy_config = config.get("economies", {}).get(economy_code, {})
    actual_policy_series = economy_config.get("actual_policy_series", {})
    if not actual_policy_series:
        return pd.Series(dtype="float64"), "Actual policy-rate comparison data is not configured for this economy."

    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        return pd.Series(dtype="float64"), "Actual policy-rate data was not loaded because FRED_API_KEY is missing."

    first_name, first_series_id = next(iter(actual_policy_series.items()))
    series = fetch_fred_series(first_series_id, api_key, config)
    if series.empty:
        return series, f"Actual policy series {first_series_id} for {first_name} returned no data."
    return series, None


def compare_to_actual_actions(
    signal_rows: list[BacktestRow],
    policy_series: pd.Series,
    config: dict[str, Any],
    economy_code: str,
) -> list[ComparisonRow]:
    if policy_series.empty:
        manual_actions = config.get("economies", {}).get(economy_code, {}).get("policy_actions", [])
        return compare_to_manual_actions(signal_rows, manual_actions)

    monthly_policy = policy_series.resample("ME").last().dropna()
    comparison_rows: list[ComparisonRow] = []

    for row in signal_rows:
        current = latest_at_or_before(monthly_policy, row.date)
        previous = latest_before(monthly_policy, row.date)
        actual = actual_policy_move(previous, current)
        hit = directional_hit(row.signal, actual)
        comparison_rows.append(ComparisonRow(row.date, row.signal, actual, hit))

    return comparison_rows


def compare_to_manual_actions(signal_rows: list[BacktestRow], manual_actions: list[dict[str, str]]) -> list[ComparisonRow]:
    actions: list[tuple[pd.Timestamp, str]] = []
    for action in manual_actions:
        date = pd.to_datetime(action.get("date"), errors="coerce")
        direction = str(action.get("direction", "")).upper()
        if pd.notna(date) and direction in {"EASING", "TIGHTENING"}:
            actions.append((pd.Timestamp(date), direction))

    if not actions:
        return [ComparisonRow(row.date, row.signal, "UNKNOWN", None) for row in signal_rows]

    comparison_rows: list[ComparisonRow] = []
    for row in signal_rows:
        same_month_directions = [
            direction
            for date, direction in actions
            if date.to_period("M") == row.date.to_period("M")
        ]
        actual = same_month_directions[0] if same_month_directions else "HOLD"
        hit = directional_hit(row.signal, actual)
        comparison_rows.append(ComparisonRow(row.date, row.signal, actual, hit))

    return comparison_rows


def latest_at_or_before(series: pd.Series, date: pd.Timestamp) -> float | None:
    history = series[series.index <= date]
    if history.empty:
        return None
    return float(history.iloc[-1])


def latest_before(series: pd.Series, date: pd.Timestamp) -> float | None:
    history = series[series.index < date]
    if history.empty:
        return None
    return float(history.iloc[-1])


def actual_policy_move(previous: float | None, current: float | None) -> str:
    if previous is None or current is None:
        return "UNKNOWN"
    change = current - previous
    tolerance = 0.125
    if change <= -tolerance:
        return "EASING"
    if change >= tolerance:
        return "TIGHTENING"
    return "HOLD"


def actual_direction(current: float | None, future: pd.Series) -> str:
    if current is None or future.empty:
        return "UNKNOWN"
    min_future = float(future.min())
    max_future = float(future.max())
    tolerance = 0.125
    easing_move = current - min_future
    tightening_move = max_future - current
    if easing_move >= tolerance and easing_move >= tightening_move:
        return "EASING"
    if tightening_move >= tolerance:
        return "TIGHTENING"
    return "HOLD"


def directional_hit(model_signal: str, actual_signal: str) -> bool | None:
    if actual_signal == "UNKNOWN":
        return None
    if model_signal == "HOLD":
        return None
    return model_signal == actual_signal


def evaluate_forward_hits(
    signal_rows: list[BacktestRow],
    policy_series: pd.Series,
    config: dict[str, Any],
    economy_code: str,
) -> list[ForwardHitRow]:
    if policy_series.empty:
        manual_actions = config.get("economies", {}).get(economy_code, {}).get("policy_actions", [])
        return evaluate_manual_forward_hits(signal_rows, manual_actions)

    monthly_policy = policy_series.resample("ME").last().dropna()
    forward_rows: list[ForwardHitRow] = []
    for row in signal_rows:
        if row.signal not in {"EASING", "TIGHTENING"}:
            continue
        current = latest_at_or_before(monthly_policy, row.date)
        for window in (3, 6, 12):
            future = monthly_policy[
                (monthly_policy.index > row.date)
                & (monthly_policy.index <= row.date + pd.DateOffset(months=window))
            ]
            actual = actual_direction(current, future)
            forward_rows.append(ForwardHitRow(row.date, row.signal, window, actual, directional_hit(row.signal, actual)))
    return forward_rows


def evaluate_manual_forward_hits(signal_rows: list[BacktestRow], manual_actions: list[dict[str, str]]) -> list[ForwardHitRow]:
    actions: list[tuple[pd.Timestamp, str]] = []
    for action in manual_actions:
        date = pd.to_datetime(action.get("date"), errors="coerce")
        direction = str(action.get("direction", "")).upper()
        if pd.notna(date) and direction in {"EASING", "TIGHTENING"}:
            actions.append((pd.Timestamp(date), direction))

    forward_rows: list[ForwardHitRow] = []
    for row in signal_rows:
        if row.signal not in {"EASING", "TIGHTENING"}:
            continue
        for window in (3, 6, 12):
            future_directions = [
                direction
                for date, direction in actions
                if row.date < date <= row.date + pd.DateOffset(months=window)
            ]
            actual = future_directions[0] if future_directions else "HOLD"
            forward_rows.append(ForwardHitRow(row.date, row.signal, window, actual, directional_hit(row.signal, actual)))
    return forward_rows


def evaluate_regimes(signal_rows: list[BacktestRow], policy_series: pd.Series) -> list[RegimeRow]:
    if policy_series.empty:
        return [RegimeRow(row.date, row.signal, "UNKNOWN", None, None) for row in signal_rows]

    monthly_policy = policy_series.resample("ME").last().dropna()
    regime_rows: list[RegimeRow] = []
    for row in signal_rows:
        current = latest_at_or_before(monthly_policy, row.date)
        prior_date = row.date - pd.DateOffset(months=6)
        prior = latest_at_or_before(monthly_policy, prior_date)
        regime, change_bp = policy_regime(current, prior)
        hit = regime_hit(row.signal, regime)
        regime_rows.append(RegimeRow(row.date, row.signal, regime, change_bp, hit))
    return regime_rows


def policy_regime(current: float | None, prior: float | None) -> tuple[str, float | None]:
    if current is None or prior is None:
        return "UNKNOWN", None
    change_bp = (current - prior) * 100
    if change_bp <= -25:
        return "EASING", change_bp
    if change_bp >= 25:
        return "TIGHTENING", change_bp
    return "NEUTRAL", change_bp


def regime_hit(model_signal: str, actual_regime: str) -> bool | None:
    if model_signal not in {"EASING", "TIGHTENING"} or actual_regime == "UNKNOWN":
        return None
    return model_signal == actual_regime


def analyze_lead_lag_cycles(signal_rows: list[BacktestRow], policy_series: pd.Series) -> list[LeadLagCycle]:
    if policy_series.empty:
        return []
    first_backtest_date = signal_rows[0].date if signal_rows else None

    monthly_policy = policy_series.resample("ME").last().dropna()
    cycle_months: list[tuple[pd.Timestamp, str]] = []
    previous_value: float | None = None
    for date, value in monthly_policy.items():
        direction = actual_policy_move(previous_value, float(value))
        if direction in {"EASING", "TIGHTENING"}:
            cycle_months.append((date, direction))
        previous_value = float(value)

    cycles: list[tuple[str, pd.Timestamp, pd.Timestamp]] = []
    current_direction: str | None = None
    current_start: pd.Timestamp | None = None
    current_end: pd.Timestamp | None = None
    for date, direction in cycle_months:
        if current_direction == direction and current_end is not None and date <= current_end + pd.DateOffset(months=2):
            current_end = date
            continue
        if current_direction is not None and current_start is not None and current_end is not None:
            cycles.append((current_direction, current_start, current_end))
        current_direction = direction
        current_start = date
        current_end = date
    if current_direction is not None and current_start is not None and current_end is not None:
        cycles.append((current_direction, current_start, current_end))

    lead_lag_cycles: list[LeadLagCycle] = []
    for direction, start, end in cycles:
        if first_backtest_date is not None and end < first_backtest_date:
            continue
        first_signal = first_model_signal_for_cycle(signal_rows, direction, start, end)
        lead_lag = None if first_signal is None else months_between(first_signal, start)
        lead_lag_cycles.append(LeadLagCycle(direction, start, end, first_signal, lead_lag))
    return lead_lag_cycles


def first_model_signal_for_cycle(
    signal_rows: list[BacktestRow],
    direction: str,
    cycle_start: pd.Timestamp,
    cycle_end: pd.Timestamp,
) -> pd.Timestamp | None:
    lookback_start = cycle_start - pd.DateOffset(months=12)
    prior_signals = [
        row.date
        for row in signal_rows
        if row.signal == direction and lookback_start <= row.date < cycle_start
    ]
    if prior_signals:
        return min(prior_signals)

    lagging_signals = [
        row.date
        for row in signal_rows
        if row.signal == direction and cycle_start <= row.date <= cycle_end
    ]
    if lagging_signals:
        return min(lagging_signals)
    return None


def months_between(signal_date: pd.Timestamp, cycle_start: pd.Timestamp) -> int:
    return (cycle_start.year - signal_date.year) * 12 + (cycle_start.month - signal_date.month)


def calculate_metrics(
    comparison_rows: list[ComparisonRow],
    forward_rows: list[ForwardHitRow],
    regime_rows: list[RegimeRow],
    lead_lag_cycles: list[LeadLagCycle],
) -> BacktestMetrics:
    same_month_predictions = [row for row in comparison_rows if row.model_signal in {"EASING", "TIGHTENING"} and row.actual_same_month != "UNKNOWN"]
    same_month_hits = [row for row in same_month_predictions if row.hit is True]

    forward_hit_rates: dict[tuple[str, int], float | None] = {}
    for signal in ("EASING", "TIGHTENING"):
        for window in (3, 6, 12):
            rows = [
                row
                for row in forward_rows
                if row.model_signal == signal and row.window_months == window and row.actual_forward_move != "UNKNOWN"
            ]
            hits = [row for row in rows if row.hit is True]
            forward_hit_rates[(signal, window)] = ratio(len(hits), len(rows))

    regime_predictions = [row for row in regime_rows if row.hit is not None]
    regime_hits = [row for row in regime_predictions if row.hit is True]

    easing_leads = [cycle.lead_lag_months for cycle in lead_lag_cycles if cycle.direction == "EASING" and cycle.lead_lag_months is not None]
    tightening_leads = [
        cycle.lead_lag_months
        for cycle in lead_lag_cycles
        if cycle.direction == "TIGHTENING" and cycle.lead_lag_months is not None
    ]

    return BacktestMetrics(
        same_month_match_rate=ratio(len(same_month_hits), len(same_month_predictions)),
        forward_hit_rates=forward_hit_rates,
        regime_match_rate=ratio(len(regime_hits), len(regime_predictions)),
        average_easing_lead_lag=average(easing_leads),
        average_tightening_lead_lag=average(tightening_leads),
    )


def save_signal_history(signal_rows: list[BacktestRow], output_dir: Path, economy_code: str) -> None:
    frame = pd.DataFrame(
        {
            "Date": [row.date.strftime("%Y-%m") for row in signal_rows],
            "Signal": [row.signal for row in signal_rows],
            "Policy Score": [round(row.policy_score, 2) for row in signal_rows],
            "Confidence %": [row.confidence for row in signal_rows],
            "Data Coverage": [row.data_coverage for row in signal_rows],
            "Missing Data": [row.missing_data for row in signal_rows],
        }
    )
    frame.to_csv(output_dir / f"{economy_code.lower()}_signal_history.csv", index=False)


def save_comparison_table(comparison_rows: list[ComparisonRow], output_dir: Path, economy_code: str) -> None:
    frame = pd.DataFrame(
        {
            "Date": [row.date.strftime("%Y-%m") for row in comparison_rows],
            "Model Signal": [row.model_signal for row in comparison_rows],
            "Actual Same-Month Move": [row.actual_same_month for row in comparison_rows],
            "Hit": [row.hit for row in comparison_rows],
        }
    )
    frame.to_csv(output_dir / f"{economy_code.lower()}_policy_comparison.csv", index=False)


def save_forward_hit_results(forward_rows: list[ForwardHitRow], output_dir: Path, economy_code: str) -> None:
    frame = pd.DataFrame(
        {
            "Date": [row.date.strftime("%Y-%m") for row in forward_rows],
            "Model Signal": [row.model_signal for row in forward_rows],
            "Window Months": [row.window_months for row in forward_rows],
            "Actual Forward Move": [row.actual_forward_move for row in forward_rows],
            "Hit": [row.hit for row in forward_rows],
        }
    )
    frame.to_csv(output_dir / f"{economy_code.lower()}_forward_hit_results.csv", index=False)
    frame.to_csv(output_dir / "forward_hit_results.csv", index=False)


def save_regime_comparison(regime_rows: list[RegimeRow], output_dir: Path, economy_code: str) -> None:
    frame = pd.DataFrame(
        {
            "Date": [row.date.strftime("%Y-%m") for row in regime_rows],
            "Model Signal": [row.model_signal for row in regime_rows],
            "Actual Policy Regime": [row.actual_regime for row in regime_rows],
            "6M Rate Change BP": [
                None if row.six_month_rate_change_bp is None else round(row.six_month_rate_change_bp, 1)
                for row in regime_rows
            ],
            "Directional Regime Hit": [row.hit for row in regime_rows],
        }
    )
    frame.to_csv(output_dir / f"{economy_code.lower()}_regime_comparison.csv", index=False)
    frame.to_csv(output_dir / "regime_comparison.csv", index=False)


def save_lead_lag_cycles(lead_lag_cycles: list[LeadLagCycle], output_dir: Path, economy_code: str) -> None:
    frame = pd.DataFrame(
        {
            "Direction": [cycle.direction for cycle in lead_lag_cycles],
            "Cycle Start": [cycle.cycle_start.strftime("%Y-%m") for cycle in lead_lag_cycles],
            "Cycle End": [cycle.cycle_end.strftime("%Y-%m") for cycle in lead_lag_cycles],
            "First Model Signal": [
                "" if cycle.first_model_signal is None else cycle.first_model_signal.strftime("%Y-%m")
                for cycle in lead_lag_cycles
            ],
            "Lead/Lag Months": [cycle.lead_lag_months for cycle in lead_lag_cycles],
        }
    )
    frame.to_csv(output_dir / f"{economy_code.lower()}_lead_lag_cycles.csv", index=False)
    frame.to_csv(output_dir / "lead_lag_cycles.csv", index=False)


def save_charts(
    signal_rows: list[BacktestRow],
    policy_series: pd.Series,
    regime_rows: list[RegimeRow],
    forward_rows: list[ForwardHitRow],
    output_dir: Path,
    economy_code: str,
) -> list[str]:
    mpl_config_dir = output_dir / ".mplconfig"
    mpl_config_dir.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return ["matplotlib is not installed, so backtest charts were not created."]

    warnings: list[str] = []
    dates = [row.date for row in signal_rows]
    scores = [row.policy_score for row in signal_rows]
    signal_values = [signal_to_number(row.signal) for row in signal_rows]

    plt.figure(figsize=(10, 4))
    plt.plot(dates, scores, color="#1f77b4", linewidth=1.8)
    plt.axhline(0, color="#666666", linewidth=0.8)
    plt.title(f"{economy_code} Policy Score Through Time")
    plt.xlabel("Month")
    plt.ylabel("Policy Score")
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / f"{economy_code.lower()}_policy_score.png", dpi=140)
    plt.close()

    if policy_series.empty:
        warnings.append("Actual policy-rate chart was not created because policy data is unavailable.")
    else:
        monthly_policy = policy_series.resample("ME").last().dropna()
        plt.figure(figsize=(10, 4))
        plt.plot(monthly_policy.index, monthly_policy.values, color="#2ca02c", linewidth=1.8)
        plt.title(f"{economy_code} Actual Policy Rate")
        plt.xlabel("Month")
        plt.ylabel("Policy Rate")
        plt.grid(True, axis="y", alpha=0.25)
        plt.tight_layout()
        plt.savefig(output_dir / f"{economy_code.lower()}_actual_policy_rate.png", dpi=140)
        plt.close()

    plt.figure(figsize=(10, 3.5))
    plt.step(dates, signal_values, where="post", color="#d62728", linewidth=1.8)
    plt.yticks([-1, 0, 1], ["EASING", "HOLD", "TIGHTENING"])
    plt.ylim(-1.2, 1.2)
    plt.title(f"{economy_code} Signal Changes")
    plt.xlabel("Month")
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / f"{economy_code.lower()}_signal_changes.png", dpi=140)
    plt.close()

    if regime_rows:
        plt.figure(figsize=(10, 3.8))
        plt.step(dates, signal_values, where="post", color="#d62728", linewidth=1.8, label="Model Signal")
        regime_values = [regime_to_number(row.actual_regime) for row in regime_rows]
        plt.step([row.date for row in regime_rows], regime_values, where="post", color="#1f77b4", linewidth=1.5, label="Actual Regime")
        plt.yticks([-1, 0, 1], ["EASING", "NEUTRAL/HOLD", "TIGHTENING"])
        plt.ylim(-1.2, 1.2)
        plt.title(f"{economy_code} Model Signal vs Actual Policy Regime")
        plt.xlabel("Month")
        plt.grid(True, axis="y", alpha=0.25)
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(output_dir / f"{economy_code.lower()}_signal_vs_policy_regime.png", dpi=140)
        plt.close()

    if forward_rows:
        labels: list[str] = []
        values: list[float] = []
        for signal in ("EASING", "TIGHTENING"):
            for window in (3, 6, 12):
                rows = [
                    row
                    for row in forward_rows
                    if row.model_signal == signal and row.window_months == window and row.actual_forward_move != "UNKNOWN"
                ]
                hits = [row for row in rows if row.hit is True]
                labels.append(f"{signal.title()} {window}M")
                values.append(0 if not rows else ratio(len(hits), len(rows)) or 0)
        plt.figure(figsize=(9, 4))
        plt.bar(labels, [value * 100 for value in values], color=["#4c78a8", "#4c78a8", "#4c78a8", "#f58518", "#f58518", "#f58518"])
        plt.title(f"{economy_code} Forward Hit Rates")
        plt.ylabel("Hit Rate (%)")
        plt.ylim(0, 100)
        plt.xticks(rotation=25, ha="right")
        plt.grid(True, axis="y", alpha=0.25)
        plt.tight_layout()
        plt.savefig(output_dir / f"{economy_code.lower()}_forward_hit_rates.png", dpi=140)
        plt.close()

    return warnings


def render_backtest_results(
    economy_code: str,
    start_year: int,
    signal_rows: list[BacktestRow],
    metrics: BacktestMetrics,
    warnings: list[str],
    output_dir: Path,
) -> str:
    counts = signal_counts(signal_rows)
    parts = [
        "BACKTEST RESULTS",
        "=================================================",
        "",
        f"Economy: {economy_code}",
        f"Period: {start_year}-present",
        "",
        "Signal Counts:",
        f"EASING: {counts['EASING']}",
        f"HOLD: {counts['HOLD']}",
        f"TIGHTENING: {counts['TIGHTENING']}",
        "",
        "Same-Month Policy Move Match Rate:",
        format_pct(metrics.same_month_match_rate),
        "Warning: This is expected to be low because central banks do not change policy every month.",
        "",
        "Forward Hit Rates:",
        f"EASING 3M: {format_pct(metrics.forward_hit_rates.get(('EASING', 3)))}",
        f"EASING 6M: {format_pct(metrics.forward_hit_rates.get(('EASING', 6)))}",
        f"EASING 12M: {format_pct(metrics.forward_hit_rates.get(('EASING', 12)))}",
        f"TIGHTENING 3M: {format_pct(metrics.forward_hit_rates.get(('TIGHTENING', 3)))}",
        f"TIGHTENING 6M: {format_pct(metrics.forward_hit_rates.get(('TIGHTENING', 6)))}",
        f"TIGHTENING 12M: {format_pct(metrics.forward_hit_rates.get(('TIGHTENING', 12)))}",
        "",
        "Regime Match Rate:",
        format_pct(metrics.regime_match_rate),
        "",
        "Lead/Lag Analysis:",
        f"Average easing lead time: {format_months(metrics.average_easing_lead_lag)}",
        f"Average tightening lead time: {format_months(metrics.average_tightening_lead_lag)}",
        "",
        "Data Methodology:",
        "This backtest uses observation-date filtering based on currently available revised data.",
        "It is not a true real-time vintage-data backtest.",
        "",
        "Saved Outputs:",
        str(output_dir),
    ]
    if warnings:
        parts.extend(["", "Backtest Warnings:", *[f"- {warning}" for warning in dedupe(warnings)]])
    return "\n".join(parts)


def signal_counts(signal_rows: list[BacktestRow]) -> dict[str, int]:
    return {
        "EASING": sum(1 for row in signal_rows if row.signal == "EASING"),
        "HOLD": sum(1 for row in signal_rows if row.signal == "HOLD"),
        "TIGHTENING": sum(1 for row in signal_rows if row.signal == "TIGHTENING"),
    }


def signal_to_number(signal: str) -> int:
    if signal == "EASING":
        return -1
    if signal == "TIGHTENING":
        return 1
    return 0


def regime_to_number(regime: str) -> int:
    if regime == "EASING":
        return -1
    if regime == "TIGHTENING":
        return 1
    return 0


def missing_data_summary(blocks: dict[str, Any]) -> str:
    missing = [
        f"{name} {block.available}/{block.expected}"
        for name, block in blocks.items()
        if block.available < block.expected
    ]
    return "; ".join(missing) if missing else "None"


def ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def average(values: list[int]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def format_pct(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.0f}%"


def format_months(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.1f} months"


def dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            result.append(item)
            seen.add(item)
    return result
