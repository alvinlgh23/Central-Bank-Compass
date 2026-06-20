from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data_sources import configured_placeholders, fetch_economy_data, fetch_fred_series
from src.data_sources import fred, fx
from src.indicators import average_change, point_change, unemployment_gap_from_12m_low, yoy, yoy_change


CLASSES = ["EASING", "HOLD", "TIGHTENING"]
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASSES)}


@dataclass(frozen=True)
class ProbabilityFeature:
    name: str
    key: str
    block: str
    value: float | None
    value_kind: str
    percentile: float | None
    z_score: float | None
    trend: str
    pressure: str
    contribution: str


@dataclass(frozen=True)
class ProbabilitySignal:
    economy_code: str
    economy_name: str
    probabilities_3m: dict[str, float]
    probabilities_6m: dict[str, float]
    signal: str
    policy_bias: str
    confidence: int
    model_quality: str
    features: list[ProbabilityFeature]
    data_coverage: str
    block_coverage: dict[str, tuple[int, int]]
    market_view: str | None
    warnings: list[str]
    pipeline_failure: bool
    debug_diagnostics: list[str]


@dataclass(frozen=True)
class ProbabilityBacktestRow:
    date: pd.Timestamp
    signal: str
    probabilities: dict[str, float]
    actual_3m: str
    actual_6m: str
    confidence: int
    brier_3m: float | None
    brier_6m: float | None


def run_probability_policy_report(
    config: dict[str, Any],
    debug_data: bool = False,
    market_view: str | None = None,
) -> str:
    signal = build_us_probability_signal(config, debug_data=debug_data, market_view=market_view)
    return render_probability_report(signal)


def build_us_probability_signal(
    config: dict[str, Any],
    debug_data: bool = False,
    market_view: str | None = None,
) -> ProbabilitySignal:
    data, data_warnings = fetch_economy_data(config, "US", debug_data=debug_data)
    policy_series = fetch_policy_series(config)
    monthly_features = build_monthly_feature_frame(data)
    current_raw = latest_feature_values(data)
    normalized_history = normalize_feature_frame(monthly_features)
    current_features = normalize_current_features(current_raw, monthly_features)
    coverage = data_coverage(current_features)
    pipeline_failure = coverage[1] == 0 or coverage[0] / coverage[1] < 0.30
    if pipeline_failure:
        probs_3m = {label: 0.0 for label in CLASSES}
        probs_6m = {label: 0.0 for label in CLASSES}
        combined = probs_6m
        signal = "INSUFFICIENT DATA"
        policy_bias = "Insufficient Data"
        quality_3m = quality_6m = "Not estimated"
        warning_3m = warning_6m = None
    else:
        probs_3m, quality_3m, warning_3m = model_probabilities(normalized_history, policy_series, 3, current_features)
        probs_6m, quality_6m, warning_6m = model_probabilities(normalized_history, policy_series, 6, current_features)
        combined = average_probabilities(probs_3m, probs_6m)
        signal = max(combined, key=combined.get)
        policy_bias = probability_bias(combined)
    features = feature_explanations(current_features)

    warnings = list(data_warnings)
    warnings.extend(warning for warning in [warning_3m, warning_6m] if warning)
    placeholders = configured_placeholders(config, "US")
    warnings.extend(f"TODO data source not integrated: {placeholder}." for placeholder in placeholders)
    if coverage[1] == 0:
        warnings.append("No live US probability-model features were available.")
    if pipeline_failure:
        warnings.append("DATA PIPELINE FAILURE: overall feature coverage is below 30%; policy pressure was not estimated.")

    return ProbabilitySignal(
        economy_code="US",
        economy_name="United States",
        probabilities_3m=probs_3m,
        probabilities_6m=probs_6m,
        signal=signal,
        policy_bias=policy_bias,
        confidence=0 if pipeline_failure else probability_confidence(combined, coverage),
        model_quality=combine_quality(quality_3m, quality_6m),
        features=features,
        data_coverage=f"{coverage[0]}/{coverage[1]}",
        block_coverage=block_coverage(features),
        market_view=market_view,
        warnings=dedupe(warnings),
        pipeline_failure=pipeline_failure,
        debug_diagnostics=build_us_debug_diagnostics(config, data, current_raw) if debug_data else [],
    )


def build_monthly_feature_frame(data: dict[str, pd.Series]) -> pd.DataFrame:
    dates = monthly_dates(data)
    rows: list[dict[str, float | pd.Timestamp | None]] = []
    for date in dates:
        point = {name: series.loc[series.index <= date] for name, series in data.items() if not series.empty}
        row = latest_feature_values(point)
        row["date"] = date
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("date").sort_index()


def latest_feature_values(data: dict[str, pd.Series]) -> dict[str, float | None]:
    ten_year = latest(data.get("ten_year_yield"))
    two_year = latest(data.get("two_year_yield"))
    payrolls_3m = average_change(data.get("payrolls"), 3)
    return {
        "core_pce_yoy": yoy(data.get("core_pce"), 12),
        "headline_pce_yoy": yoy(data.get("pce"), 12),
        "cpi_yoy": yoy(data.get("cpi"), 12),
        "core_cpi_yoy": yoy(data.get("core_cpi"), 12),
        "inflation_momentum_3m": first_available(
            yoy_change(data.get("core_pce"), 12, 3),
            yoy_change(data.get("cpi"), 12, 3),
        ),
        "inflation_trend_6m_annualized": first_available(
            annualized_change(data.get("core_pce"), 6),
            annualized_change(data.get("cpi"), 6),
        ),
        "unemployment_rate": latest(data.get("unemployment")),
        "unemployment_gap": unemployment_gap_from_12m_low(data.get("unemployment")),
        "nonfarm_payrolls_change": point_change(data.get("payrolls"), 1),
        "payrolls_3m_average": payrolls_3m,
        "initial_claims_yoy": yoy(data.get("initial_claims"), 52),
        "continuing_claims_trend": yoy(data.get("continuing_claims"), 52),
        "real_gdp_yoy": yoy(data.get("real_gdp"), 4),
        "retail_sales_yoy": yoy(data.get("retail_sales"), 12),
        "industrial_production_yoy": yoy(data.get("industrial_production"), 12),
        "pmi_external_proxy": None,
        "ten_year_yield": ten_year,
        "two_year_yield": two_year,
        "yield_curve_10y_2y": subtract(ten_year, two_year),
        "vix": latest(data.get("vix")),
        "credit_spread_change": point_change(data.get("credit_spread"), 126),
        "broad_dollar_yoy": yoy(data.get("broad_dollar"), 260),
        "usd_momentum_proxy": point_change(data.get("broad_dollar"), 60),
    }


US_FETCH_FEATURES = {
    "core_pce": ["core_pce_yoy", "inflation_momentum_3m", "inflation_trend_6m_annualized"],
    "pce": ["headline_pce_yoy"],
    "cpi": ["cpi_yoy", "inflation_momentum_3m", "inflation_trend_6m_annualized"],
    "core_cpi": ["core_cpi_yoy"],
    "unemployment": ["unemployment_rate", "unemployment_gap"],
    "payrolls": ["nonfarm_payrolls_change", "payrolls_3m_average"],
    "initial_claims": ["initial_claims_yoy"],
    "continuing_claims": ["continuing_claims_trend"],
    "real_gdp": ["real_gdp_yoy"],
    "retail_sales": ["retail_sales_yoy"],
    "industrial_production": ["industrial_production_yoy"],
    "ten_year_yield": ["ten_year_yield", "yield_curve_10y_2y"],
    "two_year_yield": ["two_year_yield", "yield_curve_10y_2y"],
    "vix": ["vix"],
    "credit_spread": ["credit_spread_change"],
    "broad_dollar": ["broad_dollar_yoy", "usd_momentum_proxy"],
}


def build_us_debug_diagnostics(
    config: dict[str, Any],
    data: dict[str, pd.Series],
    transformed: dict[str, float | None],
) -> list[str]:
    project_env = Path(__file__).resolve().parents[1] / ".env"
    lines: list[str] = [
        f".env File Found: {'yes' if project_env.is_file() else 'no'} ({project_env})",
        f"FRED_API_KEY Exists After Environment Load: {'yes' if bool(os.getenv('FRED_API_KEY')) else 'no'}",
        "FRED_API_KEY Value: redacted",
        "",
    ]
    indicator_map = config.get("economies", {}).get("US", {}).get("indicator_map", {})
    for indicator_key, spec in indicator_map.items():
        source = spec.get("source", "fred") if isinstance(spec, dict) else "fred"
        configured_id = spec.get("series_id") if isinstance(spec, dict) else spec
        series_id = fx.fred_series_id(configured_id) if source == "fx" else configured_id
        diagnostic = fred.get_fetch_diagnostic(series_id) if series_id else {}
        series = data.get(indicator_key, pd.Series(dtype="float64"))
        feature_keys = US_FETCH_FEATURES.get(indicator_key, [])
        transformed_status = ", ".join(
            feature_debug_status(key, transformed.get(key), series) for key in feature_keys
        ) or "not used by the US probability report"
        label = spec.get("label", indicator_key) if isinstance(spec, dict) else indicator_key
        lines.extend(
            [
                f"Indicator: {label}",
                f"  Source: {diagnostic.get('source', source.upper())}",
                f"  Series ID / Endpoint: {series_id or 'N/A'} | {diagnostic.get('endpoint', 'N/A')}",
                f"  Request Status: {diagnostic.get('request_status', 'Not attempted')}",
                f"  api_key_in_params: {'yes' if diagnostic.get('api_key_in_params') else 'no'}",
                f"  api_key_length: {diagnostic.get('api_key_length', 0)}",
                f"  api_key_prefix/suffix: {diagnostic.get('api_key_prefix', 'N/A')}...{diagnostic.get('api_key_suffix', 'N/A')}",
                f"  Exception Type: {diagnostic.get('exception_type') or 'None'}",
                f"  Raw Latest Observation: {'available' if not series.empty else 'unavailable'}",
                f"  API Observation Count: {diagnostic.get('raw_observations', 0)}",
                f"  Parsed Non-Null Row Count: {diagnostic.get('parsed_row_count', len(series))}",
                f"  Latest Non-Null Value Date: {diagnostic.get('latest_non_null_date') or 'N/A'}",
                f"  Feature Transformation: {transformed_status}",
            ]
        )

    policy_diagnostic = fred.get_fetch_diagnostic("FEDFUNDS")
    lines.extend(
        [
            "Indicator: Fed Funds Rate (historical policy outcome)",
            "  Source: FRED",
            f"  Series ID / Endpoint: FEDFUNDS | {policy_diagnostic.get('endpoint', 'N/A')}",
            f"  Request Status: {policy_diagnostic.get('request_status', 'Not attempted')}",
            f"  Exception Type: {policy_diagnostic.get('exception_type') or 'None'}",
            f"  Raw Latest Observation: {'available' if policy_diagnostic.get('latest_observation_available') else 'unavailable'}",
            f"  API Observation Count: {policy_diagnostic.get('raw_observations', 0)}",
            f"  Parsed Non-Null Row Count: {policy_diagnostic.get('parsed_row_count', 0)}",
            f"  Latest Non-Null Value Date: {policy_diagnostic.get('latest_non_null_date') or 'N/A'}",
            f"  Transformed Feature Availability: {'available' if policy_diagnostic.get('latest_observation_available') else 'unavailable'}",
        ]
    )
    return lines


def feature_debug_status(key: str, value: float | None, source_series: pd.Series) -> str:
    if value is not None:
        return f"{key}=available"
    if source_series.empty:
        return f"{key}=None at source input (series has no parsed non-null rows)"
    return f"{key}=None during feature transformation (insufficient history or unavailable dependency)"


def normalize_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    normalized = pd.DataFrame(index=frame.index)
    for column in frame.columns:
        normalized[column] = expanding_zscore(frame[column])
    return normalized


def normalize_current_features(raw: dict[str, float | None], history: pd.DataFrame) -> dict[str, dict[str, float | None]]:
    result: dict[str, dict[str, float | None]] = {}
    for name, value in raw.items():
        values = clean(history[name]) if name in history else pd.Series(dtype="float64")
        result[name] = {
            "value": value,
            "z_score": z_score(value, values),
            "percentile": percentile(value, values),
        }
    return result


def model_probabilities(
    normalized_history: pd.DataFrame,
    policy_series: pd.Series,
    months: int,
    current_features: dict[str, dict[str, float | None]],
) -> tuple[dict[str, float], str, str | None]:
    fallback = heuristic_probabilities(current_features)
    if normalized_history.empty or policy_series.empty:
        return fallback, "Heuristic fallback", "Probability model fell back to rule-based probability pressure because historical data or policy outcomes were unavailable."

    labels = forward_outcomes(policy_series, normalized_history.index, months)
    training = normalized_history.copy()
    training["target"] = labels
    training = training.dropna(subset=["target"])
    feature_columns = [column for column in normalized_history.columns if training[column].notna().sum() >= 36]
    if len(feature_columns) < 5 or training.empty:
        return fallback, "Heuristic fallback", "Probability model fell back because too few normalized features were available."

    training = training.dropna(subset=feature_columns)
    classes_present = set(training["target"].astype(str))
    if len(training) < 48 or len(classes_present) < 2:
        return fallback, "Heuristic fallback", "Probability model fell back because historical policy outcomes were too sparse."

    x = training[feature_columns].to_numpy(dtype=float)
    y = np.array([CLASS_TO_INDEX[label] for label in training["target"].astype(str)], dtype=int)
    weights, intercept = fit_multinomial_logit(x, y)
    current = np.array([[current_features[column]["z_score"] for column in feature_columns]], dtype=float)
    if np.isnan(current).any():
        return fallback, "Heuristic fallback", "Probability model fell back because current normalized features were incomplete."
    probabilities = softmax(current @ weights + intercept)[0]
    return {label: float(probabilities[index]) for index, label in enumerate(CLASSES)}, "Multinomial logistic regression", None


def fit_multinomial_logit(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    means = x.mean(axis=0)
    stds = x.std(axis=0)
    stds[stds == 0] = 1
    x = (x - means) / stds
    n_samples, n_features = x.shape
    weights = np.zeros((n_features, len(CLASSES)))
    intercept = np.zeros(len(CLASSES))
    y_one_hot = np.zeros((n_samples, len(CLASSES)))
    y_one_hot[np.arange(n_samples), y] = 1
    learning_rate = 0.08
    regularization = 0.02
    for _ in range(700):
        probabilities = softmax(x @ weights + intercept)
        error = probabilities - y_one_hot
        weights -= learning_rate * ((x.T @ error / n_samples) + regularization * weights)
        intercept -= learning_rate * error.mean(axis=0)
    return weights / stds[:, None], intercept - (means / stds) @ weights


def heuristic_probabilities(current_features: dict[str, dict[str, float | None]]) -> dict[str, float]:
    easing = 0.0
    tightening = 0.0
    for name, values in current_features.items():
        z = values.get("z_score")
        if z is None:
            continue
        if name in {"core_pce_yoy", "headline_pce_yoy", "cpi_yoy", "core_cpi_yoy", "inflation_momentum_3m", "inflation_trend_6m_annualized"}:
            tightening += max(z, 0) * 0.22
            easing += max(-z, 0) * 0.08
        elif name in {"unemployment_rate", "unemployment_gap", "initial_claims_yoy", "continuing_claims_trend", "vix", "credit_spread_change"}:
            easing += max(z, 0) * 0.22
            tightening += max(-z, 0) * 0.06
        elif name in {"nonfarm_payrolls_change", "payrolls_3m_average", "real_gdp_yoy", "retail_sales_yoy", "industrial_production_yoy"}:
            tightening += max(z, 0) * 0.10
            easing += max(-z, 0) * 0.18
        elif name in {"ten_year_yield", "two_year_yield"}:
            tightening += max(z, 0) * 0.08
        elif name == "yield_curve_10y_2y":
            easing += max(-z, 0) * 0.10
        elif name in {"broad_dollar_yoy", "usd_momentum_proxy"}:
            tightening += max(z, 0) * 0.06
    logits = np.array([easing, 0.8, tightening], dtype=float)
    probabilities = softmax(logits.reshape(1, -1))[0]
    return {label: float(probabilities[index]) for index, label in enumerate(CLASSES)}


def forward_outcomes(policy_series: pd.Series, dates: pd.Index, months: int) -> pd.Series:
    monthly_policy = policy_series.resample("ME").last().dropna()
    labels: dict[pd.Timestamp, str | None] = {}
    for date in dates:
        current = latest_at_or_before(monthly_policy, date)
        future = monthly_policy[(monthly_policy.index > date) & (monthly_policy.index <= date + pd.DateOffset(months=months))]
        labels[pd.Timestamp(date)] = actual_direction(current, future)
    return pd.Series(labels)


def run_probability_backtest(
    config: dict[str, Any],
    project_dir: Path,
    start_year: int,
    debug_data: bool = False,
) -> str:
    data, data_warnings = fetch_economy_data(config, "US", debug_data=debug_data)
    policy_series = fetch_policy_series(config)
    monthly_features = build_monthly_feature_frame(data)
    normalized_history = normalize_feature_frame(monthly_features)
    start = pd.Timestamp(year=start_year, month=1, day=31)
    dates = [date for date in monthly_features.index if date >= start]
    rows: list[ProbabilityBacktestRow] = []
    for date in dates:
        training = normalized_history.loc[normalized_history.index <= date]
        raw_history = monthly_features.loc[monthly_features.index <= date]
        raw_current = raw_history.iloc[-1].to_dict()
        current_features = normalize_current_features(raw_current, raw_history)
        train_3m = normalized_history.loc[normalized_history.index <= date - pd.DateOffset(months=3)]
        train_6m = normalized_history.loc[normalized_history.index <= date - pd.DateOffset(months=6)]
        policy_history = date_filtered_series(policy_series, date)
        probs_3m, _, _ = model_probabilities(train_3m, policy_history, 3, current_features)
        probs_6m, _, _ = model_probabilities(train_6m, policy_history, 6, current_features)
        combined = average_probabilities(probs_3m, probs_6m)
        actual_3m = forward_outcomes(policy_series, pd.Index([date]), 3).iloc[0]
        actual_6m = forward_outcomes(policy_series, pd.Index([date]), 6).iloc[0]
        rows.append(
            ProbabilityBacktestRow(
                date=date,
                signal=max(combined, key=combined.get),
                probabilities=combined,
                actual_3m=actual_3m or "UNKNOWN",
                actual_6m=actual_6m or "UNKNOWN",
                confidence=probability_confidence(combined, data_coverage(current_features)),
                brier_3m=brier_score(probs_3m, actual_3m),
                brier_6m=brier_score(probs_6m, actual_6m),
            )
        )

    output_dir = project_dir / "outputs"
    output_dir.mkdir(exist_ok=True)
    save_probability_backtest(rows, output_dir)
    return render_probability_backtest(rows, start_year, data_warnings, output_dir)


def render_probability_summary(signal: ProbabilitySignal, energy_context: str) -> str:
    if signal.pipeline_failure:
        return "\n".join(
            [
                "MACRO NOISE SUMMARY",
                "=================================================",
                "DATA PIPELINE FAILURE",
                "",
                "Current Macro View: INSUFFICIENT DATA",
                "Policy Pressure Estimate: Not produced because overall data coverage is below 30%.",
                "The model did not run rule-based fallback probabilities.",
                "",
                *coverage_lines(signal),
                "",
                "Energy Context:",
                energy_context,
            ]
        )
    parts = [
        "MACRO NOISE SUMMARY",
        "=================================================",
        "Current Macro View:",
        model_view_text(signal),
        "",
        "Policy Pressure Estimate:",
        f"Easing Pressure: {format_pct(signal.probabilities_6m['EASING'])}",
        f"Hold Pressure: {format_pct(signal.probabilities_6m['HOLD'])}",
        f"Tightening Pressure: {format_pct(signal.probabilities_6m['TIGHTENING'])}",
        "",
        "Key Tension:",
        key_tension(signal),
        "",
        "Noise Filter:",
        summary_noise_filter(signal),
        "",
        "What Would Change the View:",
        "For cuts: clear labor deterioration, sustained disinflation, or meaningful credit stress.",
        "For hikes: persistent core inflation re-acceleration, resilient labor and growth, and low financial stress.",
        "",
        "Energy Context:",
        energy_context,
    ]
    if signal.market_view:
        gap, _ = narrative_gap(signal.market_view, model_view_text(signal))
        parts.extend(
            [
                "",
                f"Manual Market View: {signal.market_view.replace('_', ' ').title()}",
                f"Narrative Gap: {gap}",
            ]
        )
    return "\n".join(parts)


def render_policy_constraint_check(signal: ProbabilitySignal) -> str:
    if signal.pipeline_failure:
        return "\n".join(
            [
                "POLICY CONSTRAINT CHECK",
                "=================================================",
                "Not evaluated: the live data pipeline supplied less than 30% of required US features.",
            ]
        )
    inflation_features = [feature for feature in signal.features if feature.block == "Inflation" and feature.value is not None]
    elevated = any(feature.z_score is not None and feature.z_score >= 0.5 for feature in inflation_features)
    opening = "Inflation pressure remains elevated." if elevated else "Inflation pressure is not uniformly elevated."
    return "\n".join(
        [
            "POLICY CONSTRAINT CHECK",
            "=================================================",
            opening,
            "",
            "However, the model distinguishes between:",
            "1. inflation pressure",
            "2. actual policy-action pressure",
            "",
            "High inflation raises tightening pressure, but it does not automatically mean another hike is the most likely path when policy is already restrictive and broader macro conditions do not confirm an immediate move.",
        ]
    )


def render_probability_details(signal: ProbabilitySignal) -> str:
    parts = [
        "DATA COVERAGE",
        "=================================================",
        *coverage_lines(signal),
        "",
    ]
    if signal.debug_diagnostics:
        parts.extend(
            [
            "DATA FETCH DIAGNOSTICS",
            "=================================================",
            *signal.debug_diagnostics,
            "",
            ]
        )
    if signal.pipeline_failure:
        parts.extend(
            [
                "POLICY PRESSURE",
                "=================================================",
                "Not evaluated. No logistic or rule-based fallback probabilities were produced.",
            ]
        )
        if signal.warnings:
            parts.extend(["", "Data Warnings:", *[f"- {warning}" for warning in signal.warnings]])
        return "\n".join(parts)
    parts.extend(
        [
            "MARKET NARRATIVE FILTER",
            "=================================================",
            *market_narrative_lines(signal),
            "",
            "WHAT WOULD CHANGE THE VIEW",
            "=================================================",
            *what_would_change_lines(signal),
            "",
            "EXPLAINABILITY",
            "=================================================",
        ]
    )
    for feature in signal.features:
        percentile_label = "Historical Distribution Position" if feature.block == "Inflation" else "Historical Percentile"
        context_title, context_label, policy_effect, explanation = get_indicator_context(feature)
        parts.extend(
            [
                f"- {feature.name}: {format_value(feature.value, feature.value_kind)}",
                f"  {percentile_label}: {format_percentile(feature.percentile, include_word=feature.block == 'Inflation')}",
                f"  Z-Score: {format_z(feature.z_score)}",
                *([f"  Target Consistency: {inflation_target_consistency(feature)}"] if feature.block == "Inflation" else []),
                f"  {context_title}: {context_label}",
                "  Policy Effect:",
                f"  {policy_effect}",
                f"  Why: {explanation}",
            ]
        )
    reasoning = "Not evaluated because overall data coverage is below 30%." if signal.pipeline_failure else probability_reasoning(signal)
    parts.extend(["", "Reasoning:", reasoning])
    if signal.warnings:
        parts.extend(["", "Data Warnings:", *[f"- {warning}" for warning in signal.warnings]])
    return "\n".join(parts)


def render_probability_report(signal: ProbabilitySignal, energy_context: str = "Energy context unavailable.") -> str:
    return "\n\n".join(
        [
            render_probability_summary(signal, energy_context),
            render_policy_constraint_check(signal),
            render_probability_details(signal),
        ]
    )


def render_probability_backtest(
    rows: list[ProbabilityBacktestRow],
    start_year: int,
    warnings: list[str],
    output_dir: Path,
) -> str:
    counts = {label: sum(1 for row in rows if row.signal == label) for label in CLASSES}
    parts = [
        "POLICY PRESSURE BACKTEST RESULTS",
        "=================================================",
        "",
        "Economy: US",
        f"Period: {start_year}-present",
        "",
        "Signal Counts:",
        f"EASING: {counts['EASING']}",
        f"HOLD: {counts['HOLD']}",
        f"TIGHTENING: {counts['TIGHTENING']}",
        "",
        "Forward-Window Accuracy:",
        f"3M Directional Accuracy: {format_pct(direction_accuracy(rows, 'actual_3m'))}",
        f"6M Directional Accuracy: {format_pct(direction_accuracy(rows, 'actual_6m'))}",
        "",
        "Brier Score:",
        f"3M: {format_number(average([row.brier_3m for row in rows if row.brier_3m is not None]))}",
        f"6M: {format_number(average([row.brier_6m for row in rows if row.brier_6m is not None]))}",
        "",
        "Calibration Table:",
        *calibration_lines(rows),
        "",
        "Easing Pressure Calibration:",
        *pressure_calibration_lines(rows, "EASING"),
        "",
        "Regime Match Rate:",
        format_pct(direction_accuracy(rows, "actual_6m")),
        "",
        "Historical Scenario Table:",
        *historical_scenario_lines(rows),
        "",
        "Data Methodology:",
        "Features use expanding historical z-scores/percentiles available up to each month.",
        "Policy outcomes are forward-looking 3M and 6M Fed Funds moves of at least 12.5 bps.",
        "This is not a true vintage-data backtest; revised observations are filtered by observation date.",
        "",
        "Saved Outputs:",
        str(output_dir / "us_probability_backtest.csv"),
    ]
    if warnings:
        parts.extend(["", "Backtest Warnings:", *[f"- {warning}" for warning in dedupe(warnings)]])
    return "\n".join(parts)


def feature_explanations(features: dict[str, dict[str, float | None]]) -> list[ProbabilityFeature]:
    output: list[ProbabilityFeature] = []
    for name, values in features.items():
        value = values.get("value")
        z = values.get("z_score")
        percentile_value = values.get("percentile")
        output.append(
            ProbabilityFeature(
                name=display_name(name),
                key=name,
                block=feature_block(name),
                value=value,
                value_kind=value_kind(name),
                percentile=percentile_value,
                z_score=z,
                trend=trend_text(z),
                pressure=pressure_text(name, z),
                contribution=contribution_text(name, z),
            )
        )
    return output


def pressure_text(name: str, z: float | None) -> str:
    if z is None:
        return "missing; lowers model confidence"
    high = z >= 0.5
    low = z <= -0.5
    if name in {"core_pce_yoy", "headline_pce_yoy", "cpi_yoy", "core_cpi_yoy", "inflation_momentum_3m", "inflation_trend_6m_annualized"}:
        if high:
            return "Strongly raises tightening pressure and lowers easing pressure."
        if low:
            return "Raises easing/hold pressure and lowers tightening pressure."
    if name in {"unemployment_rate", "unemployment_gap", "initial_claims_yoy", "continuing_claims_trend", "vix", "credit_spread_change"}:
        if high:
            return "Raises easing pressure."
        if low:
            return "Lowers easing pressure."
    if name in {"nonfarm_payrolls_change", "payrolls_3m_average", "real_gdp_yoy", "retail_sales_yoy", "industrial_production_yoy"}:
        if high:
            return "Raises hold/tightening pressure by showing labor or growth resilience."
        if low:
            return "Raises easing pressure by showing weaker activity."
    if name == "yield_curve_10y_2y" and low:
        return "Raises future easing pressure through yield-curve recession risk."
    if name in {"broad_dollar_yoy", "usd_momentum_proxy"} and high:
        return "Raises imported-disinflation and global financial-condition pressure."
    return "Near historical middle; limited directional pressure."


def contribution_text(name: str, z: float | None) -> str:
    if z is None:
        return "No contribution because the feature is unavailable."
    if abs(z) < 0.5:
        return "Small contribution because the value is near its historical norm."
    if name in {"core_pce_yoy", "headline_pce_yoy", "cpi_yoy", "core_cpi_yoy"}:
        return "Inflation is away from its norm, shifting probabilities through the inflation channel."
    if name in {"inflation_momentum_3m", "inflation_trend_6m_annualized"}:
        return "Recent inflation momentum shifts probabilities through the inflation channel."
    if name in {"unemployment_rate", "unemployment_gap", "initial_claims_yoy", "continuing_claims_trend"}:
        return "Labor-market slack or stress shifts easing versus hold/tightening pressure."
    if name in {"nonfarm_payrolls_change", "payrolls_3m_average"}:
        return "Payroll momentum changes the model's view of labor-market resilience."
    if name in {"real_gdp_yoy", "retail_sales_yoy", "pmi_external_proxy", "industrial_production_yoy"}:
        return "Growth momentum changes the balance between easing risk and restrictive patience."
    if name in {"broad_dollar_yoy", "usd_momentum_proxy"}:
        return "Dollar pressure can affect imported inflation and global financial conditions."
    return "Financial conditions shift the balance between easing pressure and restrictive policy pressure."


def probability_reasoning(signal: ProbabilitySignal) -> str:
    top = [
        feature
        for feature in signal.features
        if feature.z_score is not None and abs(feature.z_score) >= 0.5
    ][:5]
    if not top:
        return "Most available indicators are near their historical middle, so the model leans toward HOLD unless forward policy outcomes strongly favor another class."
    drivers = "; ".join(f"{feature.name} ({feature.pressure})" for feature in top)
    return f"The largest probability pressures come from: {drivers}."


def key_tension(signal: ProbabilitySignal) -> str:
    inflation = [feature for feature in signal.features if feature.block == "Inflation" and feature.z_score is not None]
    labor = [feature for feature in signal.features if feature.block == "Labor" and feature.z_score is not None]
    inflation_firm = any(feature.z_score >= 0.5 for feature in inflation)
    labor_stress = any(feature.key in {"unemployment_gap", "initial_claims_yoy", "continuing_claims_trend"} and feature.z_score >= 0.5 for feature in labor)
    if inflation_firm and not labor_stress:
        return "Inflation remains elevated and recent momentum is firm, but labor stress is still limited and financial conditions are not deteriorating sharply."
    if labor_stress:
        return "Labor-market deterioration is increasing easing pressure, while inflation persistence still limits how quickly policy can turn dovish."
    return "Inflation, labor, growth, and financial conditions are not producing a single dominant policy direction."


def summary_noise_filter(signal: ProbabilitySignal) -> str:
    if signal.signal == "HOLD":
        return "A rapid Fed easing narrative is not yet strongly supported. A 'Fed must hike again immediately' narrative is also not the base case."
    if signal.signal == "EASING":
        return "The data support an easing bias, but this is not an exact next-meeting cut forecast."
    return "The data support a tightening bias, but this does not imply an immediate rate hike."


def inflation_target_consistency(feature: ProbabilityFeature) -> str:
    if feature.value is None:
        return "Unavailable"
    target_ranges = {
        "core_pce_yoy": (1.8, 2.3),
        "headline_pce_yoy": (1.5, 2.5),
        "cpi_yoy": (1.8, 2.7),
        "core_cpi_yoy": (1.8, 2.7),
        "inflation_momentum_3m": (-0.2, 0.2),
        "inflation_trend_6m_annualized": (1.5, 2.5),
    }
    low, high = target_ranges.get(feature.key, (1.5, 2.5))
    if feature.value < low:
        return "Below Fed target-consistent range"
    if feature.value <= high:
        return "Near Fed target-consistent range"
    return "Above Fed target-consistent range"


def inflation_policy_trend(feature: ProbabilityFeature) -> str:
    if feature.value is None:
        return "Unavailable"
    consistency = inflation_target_consistency(feature)
    if consistency.startswith("Above"):
        return "Firm versus policy target"
    if consistency.startswith("Below"):
        return "Soft versus policy target"
    return "Near policy-target-consistent range"


def display_policy_effect(feature: ProbabilityFeature) -> str:
    if feature.block == "Inflation" and inflation_target_consistency(feature).startswith("Above"):
        if feature.z_score is None or feature.z_score < 0.5:
            return "Still restricts the easing case, even if the reading is not historically extreme."
    return feature.pressure


def display_why(feature: ProbabilityFeature) -> str:
    if feature.block == "Inflation" and inflation_target_consistency(feature).startswith("Above"):
        if feature.z_score is None or feature.z_score < 0.5:
            return "Inflation remains above the Fed's target-consistent range, so historical normality does not make it policy-neutral."
    return feature.contribution


def get_indicator_context(feature: ProbabilityFeature) -> tuple[str, str, str, str]:
    if feature.value is None:
        title = context_title_for_block(feature.block)
        return title, "Unavailable", "No directional pressure because the indicator is unavailable.", "Missing data lowers confidence and is not replaced with a neutral value."
    if feature.block == "Inflation":
        return "Trend", inflation_policy_trend(feature), display_policy_effect(feature), display_why(feature)
    if feature.block == "Labor":
        return labor_indicator_context(feature)
    if feature.block == "Growth":
        context = percentile_context(feature.percentile)
        if feature.percentile is not None and feature.percentile >= 61:
            effect = "Supports resilience in growth and limits the urgency for easing."
            why = "Activity remains stronger than typical historical readings."
        elif feature.percentile is not None and feature.percentile <= 40:
            effect = "Raises easing pressure through weaker activity."
            why = "Activity is below typical historical readings and may signal softer demand."
        else:
            effect = "Limited directional policy pressure."
            why = "The activity reading is close to its historical middle."
        return "Trend", context, effect, why
    if feature.block == "Financial":
        return financial_indicator_context(feature)
    if feature.block == "Currency":
        context = percentile_context(feature.percentile)
        if feature.percentile is not None and feature.percentile >= 81:
            effect = "Adds restrictive dollar pressure and can reduce imported inflation."
            why = "A strong dollar tightens global financial conditions and restrains some US import-price pressure."
        elif feature.percentile is not None and feature.percentile <= 20:
            effect = "Reduces dollar-related restrictive pressure."
            why = "A weak dollar can loosen financial conditions and increase imported-price pressure."
        else:
            effect = "Limited directional currency pressure."
            why = "Dollar conditions are not historically extreme."
        return "Currency Conditions", context, effect, why
    return "Trend", percentile_context(feature.percentile), feature.pressure, feature.contribution


def labor_indicator_context(feature: ProbabilityFeature) -> tuple[str, str, str, str]:
    percentile_value = feature.percentile
    if feature.key == "unemployment_rate":
        if percentile_value is not None and percentile_value <= 40:
            return "Labor Condition", "Limited labor slack", "Lowers easing pressure.", "The unemployment rate remains low enough that the labor market does not yet show broad slack or recession-like deterioration."
        if percentile_value is not None and percentile_value >= 61:
            return "Labor Condition", "Weakening / rising labor slack", "Raises easing pressure.", "Unemployment is high relative to its history, indicating broader labor-market slack."
        return "Labor Condition", "Balanced / no clear labor slack signal", "Limited directional pressure.", "The unemployment rate is near the middle of its historical distribution."
    if feature.key == "unemployment_gap":
        if feature.value < 0.3:
            return "Labor Condition", "Limited labor slack", "Lowers easing pressure.", "Unemployment remains close to its 12-month low, so deterioration is limited."
        if feature.value >= 0.5:
            return "Labor Condition", "Weakening / rising labor slack", "Raises easing pressure.", "Unemployment has risen materially from its recent low."
        return "Labor Condition", "Moderate softening", "Adds some easing pressure.", "Unemployment has risen, but not yet to a broad recession-like signal."
    if feature.key in {"initial_claims_yoy", "continuing_claims_trend"}:
        if feature.value <= 0:
            return "Labor Condition", "Claims remain contained", "Does not support a strong easing case.", "Claims are not rising year-over-year, so labor stress remains limited."
        if feature.value > 15:
            return "Labor Condition", "Claims signal meaningful deterioration", "Raises easing pressure.", "Claims are rising materially, indicating increasing labor-market stress."
        return "Labor Condition", "Claims show mild softening", "Adds modest easing pressure.", "Claims are rising, but not yet at a severe pace."
    if feature.key in {"nonfarm_payrolls_change", "payrolls_3m_average"}:
        if feature.value >= 150:
            return "Labor Condition", "Labor demand remains broadly resilient", "Limits easing pressure.", "Payroll growth remains strong enough to argue against urgent easing."
        if feature.value < 50:
            return "Labor Condition", "Labor demand is weak", "Raises easing pressure.", "Payroll growth is weak enough to suggest material labor-market deterioration."
        return "Labor Condition", "Labor demand is moderating", "Adds modest easing pressure.", "Payroll growth is positive but softer than a resilient expansion pace."
    return "Labor Condition", percentile_context(percentile_value), feature.pressure, feature.contribution


def financial_indicator_context(feature: ProbabilityFeature) -> tuple[str, str, str, str]:
    context = percentile_context(feature.percentile)
    if feature.key in {"ten_year_yield", "two_year_yield"}:
        if feature.percentile is not None and feature.percentile >= 61:
            return "Financial Conditions", context, "Provides restrictive financial-condition pressure, reducing the need for additional policy tightening.", "Higher yields can already restrain borrowing, investment, and demand."
        return "Financial Conditions", context, "Limited incremental policy pressure.", "Yields are not high enough relative to history to create an unusually restrictive signal."
    if feature.key == "vix":
        if feature.percentile is not None and feature.percentile >= 81:
            return "Financial Stress", context, "Raises easing pressure through market stress.", "Volatility is unusually high and may tighten financial conditions abruptly."
        return "Financial Stress", context, "No meaningful financial-stress signal.", "Market volatility is not currently high enough to create a strong easing case."
    if feature.key == "credit_spread_change":
        if feature.value > 0.25:
            return "Financial Stress", context, "Raises easing pressure through widening credit stress.", "Material spread widening makes financing more restrictive."
        return "Financial Stress", context, "No meaningful credit-stress signal.", "Credit spreads are stable or narrowing rather than signaling funding stress."
    if feature.key == "yield_curve_10y_2y":
        if feature.value < 0:
            return "Financial Conditions", context, "Raises future easing pressure through curve inversion.", "An inverted curve can signal restrictive policy and future growth risk."
        return "Financial Conditions", context, "Limited yield-curve easing signal.", "The curve is not currently inverted."
    return "Financial Conditions", context, feature.pressure, feature.contribution


def percentile_context(value: float | None) -> str:
    if value is None:
        return "Unavailable"
    percentile_value = int(round(value))
    if percentile_value <= 20:
        return "Very weak versus history"
    if percentile_value <= 40:
        return "Below historical norm"
    if percentile_value <= 60:
        return "Near historical norm"
    if percentile_value <= 80:
        return "Above historical norm"
    return "Elevated versus history"


def context_title_for_block(block: str) -> str:
    return {
        "Labor": "Labor Condition",
        "Growth": "Trend",
        "Financial": "Financial Conditions",
        "Currency": "Currency Conditions",
        "Inflation": "Trend",
    }.get(block, "Context")


def save_probability_backtest(rows: list[ProbabilityBacktestRow], output_dir: Path) -> None:
    frame = pd.DataFrame(
        {
            "Date": [row.date.strftime("%Y-%m") for row in rows],
            "Signal": [row.signal for row in rows],
            "Easing Probability": [round(row.probabilities["EASING"], 4) for row in rows],
            "Hold Probability": [round(row.probabilities["HOLD"], 4) for row in rows],
            "Tightening Probability": [round(row.probabilities["TIGHTENING"], 4) for row in rows],
            "Actual 3M": [row.actual_3m for row in rows],
            "Actual 6M": [row.actual_6m for row in rows],
            "Confidence": [row.confidence for row in rows],
            "Brier 3M": [row.brier_3m for row in rows],
            "Brier 6M": [row.brier_6m for row in rows],
        }
    )
    frame.to_csv(output_dir / "us_probability_backtest.csv", index=False)


def coverage_lines(signal: ProbabilitySignal) -> list[str]:
    lines = []
    total_available = 0
    total_expected = 0
    for block in ["Inflation", "Labor", "Growth", "Financial", "Currency"]:
        available, expected = signal.block_coverage.get(block, (0, 0))
        total_available += available
        total_expected += expected
        lines.append(f"{block}: {available}/{expected}")
    overall = (total_available / total_expected * 100) if total_expected else 0
    lines.append(f"Overall: {overall:.0f}%")
    lines.append(f"Coverage Confidence Adjustment: {coverage_adjustment(total_available, total_expected)}")
    return lines


def market_narrative_lines(signal: ProbabilitySignal) -> list[str]:
    model_view = model_view_text(signal)
    if not signal.market_view:
        return [
            "Market Pricing: Unavailable",
            "Narrative Filter:",
            "The model can assess macro-policy pressure, but cannot currently compare it against live market pricing.",
            "Use --market-view dovish, neutral, hawkish, aggressive_easing, or aggressive_tightening for a manual narrative check.",
        ]
    market = signal.market_view.replace("_", " ").title()
    gap, explanation = narrative_gap(signal.market_view, model_view)
    return [
        f"Manual Market View: {market}",
        f"Model Policy Pressure: {model_view}",
        f"Narrative Gap: {gap}",
        "Noise Filter Interpretation:",
        explanation,
    ]


def what_would_change_lines(signal: ProbabilitySignal) -> list[str]:
    return [
        f"Current View: {model_view_text(signal)}",
        "",
        "Conditions that would increase easing pressure:",
        "- Unemployment gap rises above roughly 0.5 pp from its 12-month low.",
        "- Initial or continuing claims rise materially year-over-year.",
        "- Core PCE falls closer to or below 2.5% with cooling 3M momentum.",
        "- GDP, retail sales, or industrial production slow toward weak historical percentiles.",
        "- Credit spreads widen materially or VIX moves into a stress regime.",
        "",
        "Conditions that would increase tightening pressure:",
        "- Core PCE or CPI re-accelerates from the current trend.",
        "- 6M annualized inflation remains elevated for multiple months.",
        "- Payroll growth remains resilient and unemployment stays near recent lows.",
        "- Growth and retail spending remain above trend.",
        "- Financial stress remains low while inflation stays elevated.",
        "",
        "These are scenario triggers, not forecasts.",
    ]


def block_coverage(features: list[ProbabilityFeature]) -> dict[str, tuple[int, int]]:
    coverage: dict[str, list[int]] = {}
    for feature in features:
        current = coverage.setdefault(feature.block, [0, 0])
        current[1] += 1
        if feature.value is not None:
            current[0] += 1
    return {block: (values[0], values[1]) for block, values in coverage.items()}


def feature_block(name: str) -> str:
    if name in {"core_pce_yoy", "headline_pce_yoy", "cpi_yoy", "core_cpi_yoy", "inflation_momentum_3m", "inflation_trend_6m_annualized"}:
        return "Inflation"
    if name in {"unemployment_rate", "unemployment_gap", "nonfarm_payrolls_change", "payrolls_3m_average", "initial_claims_yoy", "continuing_claims_trend"}:
        return "Labor"
    if name in {"real_gdp_yoy", "retail_sales_yoy", "pmi_external_proxy", "industrial_production_yoy"}:
        return "Growth"
    if name in {"ten_year_yield", "two_year_yield", "yield_curve_10y_2y", "vix", "credit_spread_change"}:
        return "Financial"
    return "Currency"


def coverage_adjustment(available: int, expected: int) -> str:
    coverage = available / expected if expected else 0
    if coverage > 0.90:
        return "None"
    if coverage >= 0.75:
        return "Small"
    if coverage >= 0.50:
        return "Moderate"
    return "Large"


def stance_label(signal: ProbabilitySignal) -> str:
    if signal.signal == "EASING":
        return "EASING BIAS"
    if signal.signal == "TIGHTENING":
        return "TIGHTENING BIAS"
    return "HOLD"


def model_view_text(signal: ProbabilitySignal) -> str:
    if signal.signal == "HOLD" and signal.policy_bias == "Hawkish":
        return "Hawkish Hold"
    if signal.signal == "HOLD" and signal.policy_bias == "Dovish":
        return "Dovish Hold"
    if signal.signal == "EASING":
        return "Cautious Easing Bias"
    if signal.signal == "TIGHTENING":
        return "Tightening Bias"
    return "Neutral Hold"


def narrative_gap(market_view: str, model_view: str) -> tuple[str, str]:
    market_scale = {
        "aggressive_easing": -2,
        "dovish": -1,
        "neutral": 0,
        "hawkish": 1,
        "aggressive_tightening": 2,
    }
    model_scale = {
        "Cautious Easing Bias": -1,
        "Dovish Hold": -1,
        "Neutral Hold": 0,
        "Hawkish Hold": 1,
        "Tightening Bias": 1,
    }
    gap = market_scale.get(market_view, 0) - model_scale.get(model_view, 0)
    if abs(gap) >= 2:
        label = "High"
    elif abs(gap) == 1:
        label = "Moderate"
    else:
        label = "Low"
    if gap < 0:
        explanation = "The market narrative appears more dovish than current macro-policy pressure. Inflation and labor data may need to soften further to validate aggressive easing expectations."
    elif gap > 0:
        explanation = "The market narrative appears more hawkish than current macro-policy pressure. Inflation would likely need to re-accelerate or growth remain unusually firm to validate that stance."
    else:
        explanation = "The manual market narrative is broadly aligned with the model's current macro-policy pressure."
    return label, explanation


def trend_text(z: float | None) -> str:
    if z is None:
        return "Unavailable"
    if z >= 0.5:
        return "Elevated / firm versus history"
    if z <= -0.5:
        return "Soft versus history"
    return "Near historical norm"


def pressure_calibration_lines(rows: list[ProbabilityBacktestRow], label: str) -> list[str]:
    lines = ["Predicted Easing Pressure Range | Actual Easing Frequency"] if label == "EASING" else []
    for floor, ceiling in [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]:
        bucket = [row for row in rows if floor <= row.probabilities[label] < ceiling and row.actual_6m in CLASSES]
        if not bucket:
            lines.append(f"- {int(floor * 100)}-{int(min(ceiling, 1) * 100)}% | N/A")
            continue
        realized = sum(1 for row in bucket if row.actual_6m == label) / len(bucket)
        lines.append(f"- {int(floor * 100)}-{int(min(ceiling, 1) * 100)}% | {realized * 100:.0f}% ({len(bucket)} months)")
    return lines


def historical_scenario_lines(rows: list[ProbabilityBacktestRow]) -> list[str]:
    if not rows:
        return ["- No historical scenarios available."]
    current = rows[-1].probabilities
    similar = sorted(
        rows[:-1],
        key=lambda row: sum(abs(row.probabilities[label] - current[label]) for label in CLASSES),
    )[:3]
    if not similar:
        return ["- Not enough history to compare similar pressure profiles."]
    lines = ["Current pressure profile most closely resembles:"]
    for row in similar:
        quarter = (row.date.month - 1) // 3 + 1
        lines.append(
            f"- {row.date.year} Q{quarter}: "
            f"Easing {format_pct(row.probabilities['EASING'])}, Hold {format_pct(row.probabilities['HOLD'])}, "
            f"Tightening {format_pct(row.probabilities['TIGHTENING'])}; later 6M path: {row.actual_6m}"
        )
    lines.append("These are similarity references, not causal forecasts.")
    return lines


def fetch_policy_series(config: dict[str, Any]) -> pd.Series:
    import os

    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        return pd.Series(dtype="float64")
    return fetch_fred_series("FEDFUNDS", api_key, config)


def monthly_dates(data: dict[str, pd.Series]) -> pd.DatetimeIndex:
    dates = [series.index.max() for series in data.values() if not series.empty]
    if not dates:
        return pd.DatetimeIndex([])
    start = pd.Timestamp("2000-01-31")
    end = max(dates)
    return pd.date_range(start=start, end=end, freq="ME")


def expanding_zscore(series: pd.Series) -> pd.Series:
    output: list[float | None] = []
    for index in range(len(series)):
        history = clean(series.iloc[: index + 1])
        value = series.iloc[index]
        output.append(z_score(value, history))
    return pd.Series(output, index=series.index, dtype="float64")


def z_score(value: float | None, history: pd.Series) -> float | None:
    if value is None or pd.isna(value) or len(history) < 24:
        return None
    std = float(history.std())
    if std == 0 or pd.isna(std):
        return None
    return float((value - history.mean()) / std)


def percentile(value: float | None, history: pd.Series) -> float | None:
    if value is None or pd.isna(value) or len(history) < 24:
        return None
    return float((history <= value).mean() * 100)


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def actual_direction(current: float | None, future: pd.Series) -> str | None:
    if current is None or future.empty:
        return None
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


def brier_score(probabilities: dict[str, float], actual: str | None) -> float | None:
    if actual not in CLASS_TO_INDEX:
        return None
    return float(sum((probabilities[label] - (1.0 if label == actual else 0.0)) ** 2 for label in CLASSES))


def direction_accuracy(rows: list[ProbabilityBacktestRow], attr: str) -> float | None:
    usable = [row for row in rows if getattr(row, attr) in CLASSES]
    if not usable:
        return None
    hits = [row for row in usable if row.signal == getattr(row, attr)]
    return len(hits) / len(usable)


def calibration_lines(rows: list[ProbabilityBacktestRow]) -> list[str]:
    lines = []
    for floor, ceiling in [(0.0, 0.4), (0.4, 0.6), (0.6, 1.01)]:
        bucket = [
            row
            for row in rows
            if floor <= row.probabilities[row.signal] < ceiling and row.actual_6m in CLASSES
        ]
        if not bucket:
            lines.append(f"- {int(floor * 100)}-{int(min(ceiling, 1) * 100)}% predicted confidence: N/A")
            continue
        hits = sum(1 for row in bucket if row.signal == row.actual_6m)
        lines.append(f"- {int(floor * 100)}-{int(min(ceiling, 1) * 100)}% predicted confidence: {hits / len(bucket) * 100:.0f}% realized hit rate ({len(bucket)} months)")
    return lines


def probability_bias(probabilities: dict[str, float]) -> str:
    signal = max(probabilities, key=probabilities.get)
    if signal == "EASING":
        return "Dovish"
    if signal == "TIGHTENING":
        return "Hawkish"
    if probabilities["TIGHTENING"] - probabilities["EASING"] > 0.08:
        return "Hawkish"
    if probabilities["EASING"] - probabilities["TIGHTENING"] > 0.08:
        return "Dovish"
    return "Neutral"


def probability_confidence(probabilities: dict[str, float], coverage: tuple[int, int]) -> int:
    top = max(probabilities.values())
    second = sorted(probabilities.values())[-2]
    coverage_ratio = coverage[0] / coverage[1] if coverage[1] else 0
    confidence = 30 + (top - second) * 60 + coverage_ratio * 30
    return int(max(20, min(confidence, 80)))


def data_coverage(features: dict[str, dict[str, float | None]]) -> tuple[int, int]:
    expected = len(features)
    available = sum(1 for item in features.values() if item.get("value") is not None)
    return available, expected


def combine_quality(first: str, second: str) -> str:
    if first == second:
        return first
    if "Multinomial" in {first, second}:
        return "Mixed: multinomial logistic regression where available, heuristic fallback otherwise"
    return "Heuristic fallback"


def average_probabilities(first: dict[str, float], second: dict[str, float]) -> dict[str, float]:
    return {label: (first[label] + second[label]) / 2 for label in CLASSES}


def clean(series: pd.Series | None) -> pd.Series:
    if series is None or series.empty:
        return pd.Series(dtype="float64")
    return series.astype(float).replace([np.inf, -np.inf], np.nan).dropna()


def latest(series: pd.Series | None) -> float | None:
    values = clean(series)
    if values.empty:
        return None
    return float(values.iloc[-1])


def latest_at_or_before(series: pd.Series, date: pd.Timestamp) -> float | None:
    history = series[series.index <= date]
    if history.empty:
        return None
    return float(history.iloc[-1])


def date_filtered_series(series: pd.Series, date: pd.Timestamp) -> pd.Series:
    if series.empty or not isinstance(series.index, pd.DatetimeIndex):
        return pd.Series(dtype="float64")
    return series.loc[series.index <= date]


def first_available(*values: float | None) -> float | None:
    for value in values:
        if value is not None:
            return value
    return None


def subtract(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return float(left - right)


def annualized_change(series: pd.Series | None, periods: int) -> float | None:
    values = clean(series)
    if len(values) <= periods:
        return None
    current = values.iloc[-1]
    prior = values.iloc[-periods - 1]
    if prior == 0 or pd.isna(prior):
        return None
    return float(((current / prior) ** (12 / periods) - 1) * 100)


def display_name(name: str) -> str:
    names = {
        "core_pce_yoy": "Core PCE YoY",
        "headline_pce_yoy": "Headline PCE YoY",
        "cpi_yoy": "CPI YoY",
        "core_cpi_yoy": "Core CPI YoY",
        "inflation_momentum_3m": "3M Inflation Momentum",
        "inflation_trend_6m_annualized": "6M Annualized Inflation Trend",
        "unemployment_rate": "Unemployment Rate",
        "unemployment_gap": "Unemployment Gap from 12M Low",
        "nonfarm_payrolls_change": "Nonfarm Payrolls Monthly Change",
        "payrolls_3m_average": "Nonfarm Payrolls 3M Average",
        "initial_claims_yoy": "Initial Claims YoY",
        "continuing_claims_trend": "Continuing Claims YoY",
        "real_gdp_yoy": "Real GDP YoY",
        "retail_sales_yoy": "Retail Sales YoY",
        "industrial_production_yoy": "Industrial Production YoY",
        "pmi_external_proxy": "PMI / External Demand Proxy",
        "ten_year_yield": "10Y Yield",
        "two_year_yield": "2Y Yield",
        "yield_curve_10y_2y": "10Y-2Y Yield Curve",
        "vix": "VIX",
        "credit_spread_change": "Credit Spread Change",
        "broad_dollar_yoy": "Broad Trade Weighted Dollar Index YoY",
        "usd_momentum_proxy": "USD Momentum Proxy",
    }
    return names.get(name, name)


def value_kind(name: str) -> str:
    if name in {"unemployment_gap", "yield_curve_10y_2y", "credit_spread_change", "inflation_momentum_3m"}:
        return "pp"
    if name in {"nonfarm_payrolls_change", "payrolls_3m_average"}:
        return "number"
    if name == "vix":
        return "plain_number"
    return "pct"


def format_value(value: float | None, kind: str) -> str:
    if value is None:
        return "N/A"
    if kind == "pct":
        return f"{value:.1f}%"
    if kind == "pp":
        return f"{value:.2f} pp"
    if kind == "plain_number":
        return f"{value:.1f}"
    return f"{value:.0f}k"


def format_pct(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.0f}%"


def format_percentile(value: float | None, include_word: bool = False) -> str:
    if value is None:
        return "N/A"
    integer = int(round(value))
    suffix = "th" if 10 <= integer % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(integer % 10, "th")
    formatted = f"{integer}{suffix}"
    return f"{formatted} percentile" if include_word else formatted


def format_z(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:+.2f}"


def format_number(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.3f}"


def average(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            result.append(item)
            seen.add(item)
    return result
