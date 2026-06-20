from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.data_sources import configured_placeholders, fetch_economy_data
from src.indicators import latest, unemployment_gap_from_12m_low, yoy


@dataclass(frozen=True)
class JapanFeature:
    name: str
    value: float | None
    kind: str
    percentile: float | None
    pressure_effect: str
    why: str


@dataclass(frozen=True)
class JapanPressureSignal:
    easing: float
    hold: float
    normalization: float
    stance: str
    policy_bias: str
    confidence: int
    model_view: str
    inflation_character: str
    features: list[JapanFeature]
    coverage: dict[str, tuple[int, int]]
    warnings: list[str]
    market_view: str | None


def run_japan_policy_pressure_report(
    config: dict[str, Any],
    debug_data: bool = False,
    market_view: str | None = None,
) -> str:
    signal = build_japan_policy_pressure(config, debug_data=debug_data, market_view=market_view)
    return render_japan_pressure(signal)


def build_japan_policy_pressure(
    config: dict[str, Any],
    debug_data: bool = False,
    market_view: str | None = None,
) -> JapanPressureSignal:
    data, warnings = fetch_economy_data(config, "JP", debug_data=debug_data)
    placeholders = configured_placeholders(config, "JP")
    warnings.extend(f"TODO data source not integrated: {placeholder}." for placeholder in placeholders)
    raw = latest_values(data)
    features = explain_features(raw, data)
    easing, hold, normalization = pressure_estimates(raw)
    stance, bias = stance_and_bias(easing, hold, normalization)
    coverage = block_coverage(raw)
    inflation_character = inflation_driver_view(raw)
    return JapanPressureSignal(
        easing=easing,
        hold=hold,
        normalization=normalization,
        stance=stance,
        policy_bias=bias,
        confidence=confidence(coverage, easing, hold, normalization),
        model_view=model_view(stance, bias, raw, inflation_character),
        inflation_character=inflation_character,
        features=features,
        coverage=coverage,
        warnings=dedupe(warnings),
        market_view=market_view,
    )


def latest_values(data: dict[str, pd.Series]) -> dict[str, float | None]:
    return {
        "cpi_ex_fresh_food": yoy(data.get("cpi_ex_fresh_food"), 12),
        "cpi_ex_fresh_food_energy": None,
        "services_inflation": None,
        "wage_growth": latest(data.get("wage_growth")),
        "unemployment_rate": latest(data.get("unemployment")),
        "unemployment_gap": unemployment_gap_from_12m_low(data.get("unemployment")),
        "gdp_growth": yoy(data.get("real_gdp"), 4),
        "usd_jpy_yoy": yoy(data.get("usd_jpy"), 260),
        "jgb_10y": latest(data.get("ten_year_yield")),
        "boj_balance_sheet_trend": None,
        "inflation_expectations": latest(data.get("inflation_expectations")),
    }


def pressure_estimates(raw: dict[str, float | None]) -> tuple[float, float, float]:
    easing = 0.8
    normalization = 0.8
    cpi = raw.get("cpi_ex_fresh_food")
    gdp = raw.get("gdp_growth")
    unemployment_gap = raw.get("unemployment_gap")
    usd_jpy = raw.get("usd_jpy_yoy")
    jgb = raw.get("jgb_10y")
    wage = raw.get("wage_growth")
    expectations = raw.get("inflation_expectations")

    if cpi is not None:
        if cpi >= 2.0:
            normalization += 1.4
        elif cpi < 1.0:
            easing += 1.2
            normalization -= 0.5
        else:
            normalization += 0.4
    if wage is not None:
        if wage > 2.5:
            normalization += 1.2
        elif wage < 1.0:
            easing += 0.8
    if usd_jpy is not None:
        if usd_jpy >= 8:
            normalization += 0.9
        elif usd_jpy <= -5:
            easing += 0.4
    if gdp is not None:
        if gdp < 0:
            easing += 1.0
        elif gdp > 1:
            normalization += 0.4
    if unemployment_gap is not None:
        if unemployment_gap >= 0.5:
            easing += 0.6
        elif unemployment_gap < 0.3:
            normalization += 0.3
    if jgb is not None and jgb > 1.5:
        normalization += 0.4
    if expectations is not None and expectations >= 2:
        normalization += 0.6

    normalization = max(normalization, 0.1)
    logits = np.array([easing, 1.2, normalization], dtype=float)
    probs = softmax(logits)
    return float(probs[0]), float(probs[1]), float(probs[2])


def explain_features(raw: dict[str, float | None], data: dict[str, pd.Series]) -> list[JapanFeature]:
    definitions = [
        ("cpi_ex_fresh_food", "CPI ex Fresh Food", "pct"),
        ("cpi_ex_fresh_food_energy", "CPI ex Fresh Food and Energy", "pct"),
        ("services_inflation", "Services Inflation", "pct"),
        ("wage_growth", "Wage Growth / Shunto Proxy", "pct"),
        ("unemployment_rate", "Unemployment Rate", "pct"),
        ("unemployment_gap", "Unemployment Gap from 12M Low", "pp"),
        ("gdp_growth", "GDP Growth YoY", "pct"),
        ("usd_jpy_yoy", "USD/JPY YoY", "pct"),
        ("jgb_10y", "10Y JGB Yield", "pct"),
        ("boj_balance_sheet_trend", "BOJ Balance Sheet / JGB Purchase Proxy", "pct"),
        ("inflation_expectations", "Inflation Expectations", "pct"),
    ]
    return [
        JapanFeature(
            name=label,
            value=raw.get(key),
            kind=kind,
            percentile=historical_percentile(key, raw.get(key), data),
            pressure_effect=pressure_effect(key, raw.get(key)),
            why=why_it_matters(key),
        )
        for key, label, kind in definitions
    ]


def historical_percentile(key: str, value: float | None, data: dict[str, pd.Series]) -> float | None:
    if value is None:
        return None
    if key == "cpi_ex_fresh_food":
        history = data.get("cpi_ex_fresh_food")
        transformed = rolling_yoy(history, 12)
    elif key == "gdp_growth":
        transformed = rolling_yoy(data.get("real_gdp"), 4)
    elif key == "usd_jpy_yoy":
        transformed = rolling_yoy(data.get("usd_jpy"), 260)
    elif key == "jgb_10y":
        transformed = clean(data.get("ten_year_yield"))
    elif key == "unemployment_rate":
        transformed = clean(data.get("unemployment"))
    else:
        transformed = pd.Series(dtype="float64")
    if len(transformed) < 24:
        return None
    return float((transformed <= value).mean() * 100)


def pressure_effect(key: str, value: float | None) -> str:
    if value is None:
        return "Unavailable; lowers confidence and limits the BOJ normalization assessment."
    if key == "cpi_ex_fresh_food":
        if value >= 2:
            return "Raises normalization / hike pressure."
        if value < 1:
            return "Raises easing pressure."
    if key in {"wage_growth", "services_inflation", "inflation_expectations"}:
        return "Would raise normalization pressure if firm and rising."
    if key == "usd_jpy_yoy" and value >= 8:
        return "Raises normalization pressure through imported inflation risk."
    if key == "gdp_growth" and value < 0:
        return "Raises easing pressure because activity is contracting."
    if key == "unemployment_gap" and value >= 0.5:
        return "Raises easing pressure through labor-market softening."
    return "Supports holding unless confirmed by broader wage, services, or growth evidence."


def why_it_matters(key: str) -> str:
    reasons = {
        "cpi_ex_fresh_food": "BOJ normalization is more durable when underlying inflation is near or above 2%.",
        "cpi_ex_fresh_food_energy": "A broader core measure helps separate demand inflation from volatile import components.",
        "services_inflation": "Services inflation can indicate domestic demand and wage pass-through.",
        "wage_growth": "Sustained wage growth is central to Japan's virtuous wage-price cycle.",
        "unemployment_rate": "A resilient labor market gives BOJ more room to normalize.",
        "unemployment_gap": "Rising slack argues against aggressive normalization.",
        "gdp_growth": "Sharp contraction would argue for patience or easing.",
        "usd_jpy_yoy": "JPY weakness can lift imported inflation even when domestic demand is less strong.",
        "jgb_10y": "Higher JGB yields can reflect normalization pressure or bond-market stress.",
        "boj_balance_sheet_trend": "BOJ balance-sheet runoff or lower purchases would signal normalization.",
        "inflation_expectations": "Stable expectations near 2% support normalization more than one-off price shocks.",
    }
    return reasons[key]


def inflation_driver_view(raw: dict[str, float | None]) -> str:
    cpi = raw.get("cpi_ex_fresh_food")
    usd_jpy = raw.get("usd_jpy_yoy")
    wage = raw.get("wage_growth")
    services = raw.get("services_inflation")
    if cpi is None:
        return "Inflation breadth is unclear because key inflation data is incomplete."
    if (wage is not None and wage > 2.5) or (services is not None and services >= 2):
        return "Inflation looks more demand/wage-driven, which would strengthen the normalization case."
    if usd_jpy is not None and usd_jpy >= 8:
        return "Inflation pressure appears materially import/currency-driven because JPY weakness is significant."
    return "Inflation pressure is present, but wage and services confirmation is incomplete."


def render_japan_pressure(signal: JapanPressureSignal) -> str:
    parts = [
        "BANK OF JAPAN POLICY PRESSURE",
        "=================================================",
        f"Easing Pressure: {format_pct(signal.easing)}",
        f"Hold Pressure: {format_pct(signal.hold)}",
        f"Normalization / Hike Pressure: {format_pct(signal.normalization)}",
        "",
        f"Current Stance: {signal.stance}",
        f"Policy Bias: {signal.policy_bias}",
        f"Confidence: {signal.confidence}%",
        "",
        "Model View:",
        signal.model_view,
        "",
        "Inflation Character:",
        signal.inflation_character,
        "",
        "DATA COVERAGE",
        "=================================================",
        *coverage_lines(signal.coverage),
        "",
        "Noise Filter:",
        *noise_filter_lines(signal),
        "",
        "EXPLAINABILITY",
        "=================================================",
    ]
    for feature in signal.features:
        parts.extend(
            [
                f"- {feature.name}: {format_value(feature.value, feature.kind)}",
                f"  Historical Percentile: {format_percentile(feature.percentile)}",
                f"  Policy Effect: {feature.pressure_effect}",
                f"  Why: {feature.why}",
            ]
        )
    if signal.warnings:
        parts.extend(["", "Data Warnings:", *[f"- {warning}" for warning in signal.warnings]])
    return "\n".join(parts)


def model_view(stance: str, bias: str, raw: dict[str, float | None], inflation_character: str) -> str:
    if stance == "NORMALIZATION / HIKE BIAS":
        return f"BOJ has moderate normalization pressure, but the case depends on wage growth and services inflation. {inflation_character}"
    if stance == "EASING BIAS":
        return "BOJ easing pressure is elevated because macro data point to weaker inflation or growth conditions."
    if bias == "Hawkish":
        return f"BOJ has a hawkish hold: normalization pressure exists, but confirmation from wages, services inflation, and growth is still important. {inflation_character}"
    if bias == "Dovish":
        return "BOJ has a dovish hold because easing pressure is stronger than normalization pressure."
    return f"BOJ has a neutral hold. {inflation_character}"


def stance_and_bias(easing: float, hold: float, normalization: float) -> tuple[str, str]:
    if normalization > max(easing, hold) and normalization >= 0.42:
        return "NORMALIZATION / HIKE BIAS", "Hawkish"
    if easing > max(hold, normalization) and easing >= 0.42:
        return "EASING BIAS", "Dovish"
    if normalization - easing > 0.08:
        return "HOLD", "Hawkish"
    if easing - normalization > 0.08:
        return "HOLD", "Dovish"
    return "HOLD", "Neutral"


def noise_filter_lines(signal: JapanPressureSignal) -> list[str]:
    if not signal.market_view:
        return [
            "Market Pricing: Unavailable",
            "The model can assess BOJ normalization pressure, but cannot compare it to live market pricing without a stable free source.",
            "Use --market-view hawkish or --market-view aggressive_tightening to test a manual BOJ hike narrative.",
        ]
    wage_services_missing = any(
        feature.name in {"Wage Growth / Shunto Proxy", "Services Inflation"} and feature.value is None
        for feature in signal.features
    )
    if signal.market_view == "aggressive_tightening" and wage_services_missing:
        return [
            "Manual Market View: Aggressive Tightening",
            "Narrative Gap: Moderate to High",
            "If the market narrative says BOJ will aggressively hike, the model checks whether wage growth and services inflation support that story. Those confirmations are missing, so the aggressive narrative is not fully validated.",
        ]
    if signal.market_view in {"hawkish", "aggressive_tightening"} and signal.policy_bias != "Hawkish":
        return [
            f"Manual Market View: {signal.market_view.replace('_', ' ').title()}",
            "Narrative Gap: High",
            "If the market narrative says BOJ will aggressively hike, the model asks whether wages and services inflation support that story. Current data coverage is not strong enough to fully validate aggressive normalization.",
        ]
    return [
        f"Manual Market View: {signal.market_view.replace('_', ' ').title()}",
        "Narrative Gap: Low to Moderate",
        "The manual market narrative is not sharply inconsistent with the current BOJ pressure estimate.",
    ]


def block_coverage(raw: dict[str, float | None]) -> dict[str, tuple[int, int]]:
    blocks = {
        "Inflation": ["cpi_ex_fresh_food", "cpi_ex_fresh_food_energy", "services_inflation", "inflation_expectations"],
        "Labor/Wages": ["wage_growth", "unemployment_rate", "unemployment_gap"],
        "Growth": ["gdp_growth"],
        "Currency": ["usd_jpy_yoy"],
        "Financial/BOJ Operations": ["jgb_10y", "boj_balance_sheet_trend"],
    }
    return {
        block: (sum(1 for key in keys if raw.get(key) is not None), len(keys))
        for block, keys in blocks.items()
    }


def coverage_lines(coverage: dict[str, tuple[int, int]]) -> list[str]:
    lines = []
    available = 0
    expected = 0
    for block, (block_available, block_expected) in coverage.items():
        available += block_available
        expected += block_expected
        lines.append(f"{block}: {block_available}/{block_expected}")
    pct = available / expected * 100 if expected else 0
    lines.append(f"Overall: {pct:.0f}%")
    return lines


def confidence(coverage: dict[str, tuple[int, int]], easing: float, hold: float, normalization: float) -> int:
    available = sum(item[0] for item in coverage.values())
    expected = sum(item[1] for item in coverage.values())
    coverage_ratio = available / expected if expected else 0
    separation = max(easing, hold, normalization) - sorted([easing, hold, normalization])[-2]
    return int(max(20, min(75, 25 + coverage_ratio * 35 + separation * 60)))


def rolling_yoy(series: pd.Series | None, periods: int) -> pd.Series:
    values = clean(series)
    if len(values) <= periods:
        return pd.Series(dtype="float64")
    return ((values / values.shift(periods)) - 1).replace([np.inf, -np.inf], np.nan).dropna() * 100


def clean(series: pd.Series | None) -> pd.Series:
    if series is None or series.empty:
        return pd.Series(dtype="float64")
    return series.astype(float).replace([np.inf, -np.inf], np.nan).dropna()


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max()
    exp = np.exp(shifted)
    return exp / exp.sum()


def format_pct(value: float) -> str:
    return f"{value * 100:.0f}%"


def format_value(value: float | None, kind: str) -> str:
    if value is None:
        return "N/A"
    if kind == "pp":
        return f"{value:.2f} pp"
    return f"{value:.1f}%"


def format_percentile(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.0f}th"


def dedupe(items: list[str]) -> list[str]:
    result = []
    seen = set()
    for item in items:
        if item not in seen:
            result.append(item)
            seen.add(item)
    return result
