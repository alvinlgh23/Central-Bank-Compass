from dataclasses import dataclass, replace
import os
from pathlib import Path
from typing import Any

import pandas as pd

from src.country_profiles import get_profile
from src.data_sources import configured_placeholders, fetch_economy_data, fetch_fred_series
from src.data_sources import pbc
from src.indicators import build_indicators, yoy
from src.japan_pressure import build_japan_policy_pressure
from src.policy_signal import PolicySignal, build_policy_signal
from src.probability_model import build_us_probability_signal


@dataclass(frozen=True)
class LiquidityDriver:
    name: str
    value: float | str | None
    value_kind: str
    thresholds: list[str]
    classification: str
    score_effect: float
    explanation: str


@dataclass(frozen=True)
class EconomyLiquidity:
    code: str
    name: str
    weight: float
    score: float
    classification: str
    confidence: int
    available_drivers: int
    expected_drivers: int
    drivers: list[LiquidityDriver]
    warnings: list[str]
    policy_signal: str | None = None
    policy_bias: str | None = None
    inflation_label: str | None = None
    growth_label: str | None = None
    financial_label: str | None = None


@dataclass(frozen=True)
class GlobalLiquidityResult:
    score: float
    classification: str
    confidence: int
    economies: list[EconomyLiquidity]
    top_drivers: list[str]
    warnings: list[str]
    output_dir: Path


ECONOMY_NAMES = {
    "US": "United States",
    "CN": "China",
    "EZ": "Eurozone",
    "JP": "Japan",
    "SG": "Singapore",
}


def run_liquidity_compass(
    config: dict[str, Any],
    project_dir: Path,
    show_details: bool = False,
    debug_data: bool = False,
) -> str:
    output_dir = project_dir / "outputs"
    output_dir.mkdir(exist_ok=True)

    weights = config.get("liquidity", {}).get("weights", {})
    weight_total = sum(float(weight) for weight in weights.values())
    economies = [
        build_economy_liquidity(code, float(weight), config, debug_data=debug_data)
        for code, weight in weights.items()
    ]
    global_score = weighted_average([economy.score for economy in economies], [economy.weight for economy in economies])
    global_confidence = int(round(weighted_average([economy.confidence for economy in economies], [economy.weight for economy in economies])))
    result = GlobalLiquidityResult(
        score=global_score,
        classification=classify_liquidity(global_score),
        confidence=global_confidence,
        economies=economies,
        top_drivers=top_global_drivers(economies),
        warnings=liquidity_warnings(economies, weight_total),
        output_dir=output_dir,
    )
    save_liquidity_outputs(result)
    return render_liquidity_report(result, show_details)


def build_economy_liquidity(
    code: str,
    weight: float,
    config: dict[str, Any],
    debug_data: bool = False,
) -> EconomyLiquidity:
    if code == "CN":
        return build_china_liquidity(weight, config)

    signal, indicators, warnings = current_policy_signal(code, config, debug_data=debug_data)
    drivers = policy_drivers(code, signal)
    drivers.extend(market_liquidity_drivers(code, indicators, config))

    raw_score = 50 + sum(driver.score_effect for driver in drivers)
    score = clamp(raw_score, 0, 100)
    expected = len(drivers)
    available = sum(1 for driver in drivers if driver.value is not None)
    confidence = coverage_confidence(available, expected, warnings)

    return EconomyLiquidity(
        code=code,
        name=ECONOMY_NAMES[code],
        weight=weight,
        score=score,
        classification=classify_liquidity(score),
        confidence=confidence,
        available_drivers=available,
        expected_drivers=expected,
        drivers=drivers,
        warnings=warnings,
        policy_signal=signal.signal,
        policy_bias=signal.policy_bias,
        inflation_label=signal.inflation.label,
        growth_label=signal.growth.label,
        financial_label=signal.financial.label,
    )


def build_china_placeholder(weight: float, config: dict[str, Any]) -> EconomyLiquidity:
    placeholders = config.get("liquidity", {}).get("placeholders", {}).get("CN", [])
    drivers = [
        LiquidityDriver(
            name=placeholder,
            value=None,
            value_kind="number",
            thresholds=["Data source not integrated = no score effect"],
            classification="MISSING DATA",
            score_effect=0,
            explanation=f"{placeholder} is expected for China liquidity analysis, but no source is configured yet.",
        )
        for placeholder in placeholders
    ]
    warnings = [f"China liquidity data unavailable: {placeholder}." for placeholder in placeholders]
    return EconomyLiquidity(
        code="CN",
        name="China",
        weight=weight,
        score=50,
        classification="Missing Data / Placeholder",
        confidence=5,
        available_drivers=0,
        expected_drivers=len(drivers),
        drivers=drivers,
        warnings=warnings,
        policy_signal=None,
    )


def build_china_liquidity(weight: float, config: dict[str, Any]) -> EconomyLiquidity:
    drivers = [
        money_trend_driver("China M2 YoY", yoy(pbc.fetch_m2_growth(config), 12)),
        missing_driver("China Credit Impulse Proxy", ["Official TSF credit impulse adapter not integrated = no score effect"]),
        policy_rate_driver("China Policy Rate Proxy", pbc.fetch_lpr(config)),
        missing_driver("China RRR", ["PBC RRR adapter not integrated = no score effect"]),
        currency_driver("CNY Pressure, USD/CNY YoY", yoy(pbc.fetch_cny_pressure(config), 12), weak_currency_negative=True),
        property_stress_driver("China Real Residential Property Prices YoY", yoy(pbc.fetch_property_stress_proxy(config), 4)),
    ]
    raw_score = 50 + sum(driver.score_effect for driver in drivers)
    score = clamp(raw_score, 0, 100)
    expected = len(drivers)
    available = sum(1 for driver in drivers if driver.value is not None)
    warnings = [
        f"China liquidity data unavailable: {driver.name}."
        for driver in drivers
        if driver.value is None
    ]
    return EconomyLiquidity(
        code="CN",
        name="China",
        weight=weight,
        score=score,
        classification=classify_liquidity(score),
        confidence=coverage_confidence(available, expected, warnings),
        available_drivers=available,
        expected_drivers=expected,
        drivers=drivers,
        warnings=warnings,
        policy_signal=None,
    )


def current_policy_signal(
    code: str,
    config: dict[str, Any],
    debug_data: bool = False,
) -> tuple[PolicySignal, Any, list[str]]:
    profile = get_profile(code)
    data, data_warnings = fetch_economy_data(config, code, debug_data=debug_data)
    placeholders = configured_placeholders(config, code)
    indicators = build_indicators(code, data)
    signal = build_policy_signal(indicators, profile, config, data_warnings, placeholders)
    if code == "US":
        probability_signal = build_us_probability_signal(config, debug_data=debug_data)
        signal = replace(
            signal,
            signal=probability_signal.signal,
            policy_bias=probability_signal.policy_bias,
            confidence=probability_signal.confidence,
        )
        if probability_signal.pipeline_failure:
            data_warnings.append("US policy input unavailable: probability model data coverage is below 30%.")
    if code == "JP":
        japan_signal = build_japan_policy_pressure(config, debug_data=debug_data)
        signal = replace(
            signal,
            signal=japan_liquidity_stance(japan_signal.stance),
            policy_bias=japan_signal.policy_bias,
            confidence=japan_signal.confidence,
        )
    warnings = list(data_warnings)
    warnings.extend(f"{code} policy model placeholder: {placeholder}." for placeholder in placeholders)
    return signal, indicators, warnings


def japan_liquidity_stance(stance: str) -> str:
    if stance == "NORMALIZATION / HIKE BIAS":
        return "TIGHTENING"
    if stance == "EASING BIAS":
        return "EASING"
    return "HOLD"


def policy_drivers(code: str, signal: PolicySignal) -> list[LiquidityDriver]:
    if signal.signal not in {"EASING", "HOLD", "TIGHTENING"}:
        return [
            LiquidityDriver(
                name="Central Bank Policy Signal",
                value=None,
                value_kind="text",
                thresholds=["Requires EASING, HOLD, or TIGHTENING from a sufficiently covered policy model"],
                classification="MISSING DATA",
                score_effect=0,
                explanation="The policy model reported insufficient data, so no neutral or directional liquidity effect was assigned.",
            )
        ]
    effect = {"EASING": 18, "HOLD": 0, "TIGHTENING": -18}[signal.signal]
    if signal.signal == "HOLD" and signal.policy_bias == "Dovish":
        effect = 6
    elif signal.signal == "HOLD" and signal.policy_bias == "Hawkish":
        effect = -6
    if code == "SG":
        name = "MAS FX Policy Signal"
        thresholds = [
            "EASING = +18 via lower/flatter SGD NEER stance",
            "HOLD/Dovish = +6",
            "HOLD/Hawkish = -6",
            "TIGHTENING = -18 via steeper/stronger SGD NEER stance",
        ]
        explanation = (
            f"The Singapore policy model reads {signal.signal} with a {signal.policy_bias.lower()} bias. "
            f"Because MAS uses the SGD NEER policy band rather than a policy rate, liquidity gets {format_score(effect)} points through the FX-policy channel."
        )
    else:
        name = "Central Bank Policy Signal"
        thresholds = ["EASING = +18", "HOLD/Dovish = +6", "HOLD/Hawkish = -6", "TIGHTENING = -18"]
        explanation = f"The existing policy model reads {signal.signal} with a {signal.policy_bias.lower()} bias, so liquidity gets {format_score(effect)} points."
    return [
        LiquidityDriver(
            name=name,
            value=f"{signal.signal} / {signal.policy_bias}",
            value_kind="text",
            thresholds=thresholds,
            classification=f"{signal.signal} / {signal.policy_bias}",
            score_effect=effect,
            explanation=explanation,
        )
    ]


def market_liquidity_drivers(code: str, indicators: Any, config: dict[str, Any]) -> list[LiquidityDriver]:
    if code == "US":
        return us_liquidity_drivers(indicators, config)
    if code == "EZ":
        return eurozone_liquidity_drivers(indicators, config)
    if code == "JP":
        return japan_liquidity_drivers(indicators)
    if code == "SG":
        return singapore_liquidity_drivers(indicators, config)
    return []


def us_liquidity_drivers(indicators: Any, config: dict[str, Any]) -> list[LiquidityDriver]:
    ten_year = indicators.get("ten_year_yield")
    credit_spread_change = indicators.get("credit_spread_change")
    vix = indicators.get("vix")
    m2_trend = fred_yoy(config, "US", "m2", 12)
    balance_sheet_trend = fred_yoy(config, "US", "fed_balance_sheet", 52)
    return [
        yield_driver("10Y Treasury Yield", ten_year),
        spread_driver("Credit Spread Change, 6M", credit_spread_change),
        vix_driver(vix),
        money_trend_driver("M2 Money Supply YoY", m2_trend),
        money_trend_driver("Fed Balance Sheet YoY", balance_sheet_trend),
    ]


def eurozone_liquidity_drivers(indicators: Any, config: dict[str, Any]) -> list[LiquidityDriver]:
    eur_pressure = indicators.get("currency_change_yoy")
    money_trend = fred_yoy(config, "EZ", "money_supply", 12)
    return [
        stress_placeholder_driver("Credit Spread / Financial Stress"),
        money_trend_driver("Eurozone Money Supply YoY", money_trend),
        currency_driver("EUR Pressure", eur_pressure, weak_currency_negative=False),
    ]


def japan_liquidity_drivers(indicators: Any) -> list[LiquidityDriver]:
    return [
        currency_driver("JPY Trend, USD/JPY YoY", indicators.get("currency_change_yoy"), weak_currency_negative=True),
        yield_driver("10Y JGB Yield", indicators.get("ten_year_yield")),
        stress_placeholder_driver("Japan Financial Conditions"),
    ]


def singapore_liquidity_drivers(indicators: Any, config: dict[str, Any]) -> list[LiquidityDriver]:
    placeholders = config.get("liquidity", {}).get("placeholders", {}).get("SG", [])
    drivers = [
        currency_driver("SGD Pressure, USD/SGD YoY", indicators.get("currency_change_yoy"), weak_currency_negative=True),
    ]
    drivers.extend(
        LiquidityDriver(
            name=placeholder,
            value=None,
            value_kind="number",
            thresholds=["Data source not integrated = no score effect"],
            classification="MISSING DATA",
            score_effect=0,
            explanation=f"{placeholder} is relevant for Singapore liquidity, but no source is configured yet.",
        )
        for placeholder in placeholders
    )
    return drivers


def fred_yoy(config: dict[str, Any], economy_code: str, key: str, periods: int) -> float | None:
    api_key = os.getenv("FRED_API_KEY")
    series_id = config.get("liquidity", {}).get("fred_series", {}).get(economy_code, {}).get(key)
    if not api_key or not series_id:
        return None
    series = fetch_fred_series(series_id, api_key, config)
    if series.empty:
        return None
    return yoy(series, periods)


def yield_driver(name: str, value: float | None) -> LiquidityDriver:
    if value is None:
        return missing_driver(name, ["<3% = supportive", "3-5% = neutral", ">5% = restrictive"])
    if value > 5:
        return driver(name, value, "pct", ["<3% = supportive", "3-5% = neutral", ">5% = restrictive"], "RESTRICTIVE", -10, "High long-term yields tighten discount rates and financial conditions.")
    if value < 3:
        return driver(name, value, "pct", ["<3% = supportive", "3-5% = neutral", ">5% = restrictive"], "SUPPORTIVE", 8, "Lower long-term yields support liquidity-sensitive assets.")
    return driver(name, value, "pct", ["<3% = supportive", "3-5% = neutral", ">5% = restrictive"], "NEUTRAL", 0, "Long-term yields are not extreme enough to drive liquidity.")


def spread_driver(name: str, value: float | None) -> LiquidityDriver:
    if value is None:
        return missing_driver(name, ["<=0 pp = supportive", "0-0.25 pp = neutral", ">0.25 pp = restrictive"])
    if value > 0.25:
        return driver(name, value, "pp", ["<=0 pp = supportive", "0-0.25 pp = neutral", ">0.25 pp = restrictive"], "RESTRICTIVE", -10, "Widening credit spreads signal tighter private-sector liquidity.")
    if value <= 0:
        return driver(name, value, "pp", ["<=0 pp = supportive", "0-0.25 pp = neutral", ">0.25 pp = restrictive"], "SUPPORTIVE", 8, "Stable or narrowing spreads signal easier credit conditions.")
    return driver(name, value, "pp", ["<=0 pp = supportive", "0-0.25 pp = neutral", ">0.25 pp = restrictive"], "NEUTRAL", 0, "Credit spread widening is modest.")


def vix_driver(value: float | None) -> LiquidityDriver:
    if value is None:
        return missing_driver("VIX", ["<18 = supportive", "18-25 = neutral", ">25 = restrictive"])
    if value > 25:
        return driver("VIX", value, "number", ["<18 = supportive", "18-25 = neutral", ">25 = restrictive"], "RESTRICTIVE", -12, "High volatility usually tightens risk appetite and market liquidity.")
    if value < 18:
        return driver("VIX", value, "number", ["<18 = supportive", "18-25 = neutral", ">25 = restrictive"], "SUPPORTIVE", 8, "Low volatility supports risk appetite and market liquidity.")
    return driver("VIX", value, "number", ["<18 = supportive", "18-25 = neutral", ">25 = restrictive"], "NEUTRAL", 0, "Volatility is not high enough to be a major liquidity drag.")


def money_trend_driver(name: str, value: float | None) -> LiquidityDriver:
    if value is None:
        return missing_driver(name, ["<0% = contracting", "0-5% = neutral", ">5% = expanding"])
    if value > 5:
        return driver(name, value, "pct", ["<0% = contracting", "0-5% = neutral", ">5% = expanding"], "EXPANDING", 12, "Money or central-bank balance-sheet growth is positive for liquidity.")
    if value < 0:
        return driver(name, value, "pct", ["<0% = contracting", "0-5% = neutral", ">5% = expanding"], "CONTRACTING", -12, "Money or balance-sheet contraction is a liquidity headwind.")
    return driver(name, value, "pct", ["<0% = contracting", "0-5% = neutral", ">5% = expanding"], "NEUTRAL", 0, "Money growth is not strong enough to create a major liquidity impulse.")


def policy_rate_driver(name: str, series: pd.Series) -> LiquidityDriver:
    value = latest_series_value(series)
    if value is None:
        return missing_driver(name, ["Falling = supportive", "Stable = neutral", "Rising = restrictive"])
    change = series_change(series, 12)
    if change is None:
        return driver(name, value, "pct", ["Falling = supportive", "Stable = neutral", "Rising = restrictive"], "NEUTRAL", 0, "Policy-rate proxy is available, but trend history is insufficient.")
    if change < -0.05:
        return driver(name, value, "pct", ["Falling = supportive", "Stable = neutral", "Rising = restrictive"], "SUPPORTIVE", 8, "China policy-rate proxy has fallen over the past year, supporting liquidity.")
    if change > 0.05:
        return driver(name, value, "pct", ["Falling = supportive", "Stable = neutral", "Rising = restrictive"], "RESTRICTIVE", -8, "China policy-rate proxy has risen over the past year, tightening liquidity.")
    return driver(name, value, "pct", ["Falling = supportive", "Stable = neutral", "Rising = restrictive"], "NEUTRAL", 0, "China policy-rate proxy is broadly stable.")


def property_stress_driver(name: str, value: float | None) -> LiquidityDriver:
    if value is None:
        return missing_driver(name, ["<-5% = stress", "-5% to +5% = neutral", ">+5% = supportive"])
    if value < -5:
        return driver(name, value, "pct", ["<-5% = stress", "-5% to +5% = neutral", ">+5% = supportive"], "STRESS", -10, "Falling real residential property prices point to property-sector stress.")
    if value > 5:
        return driver(name, value, "pct", ["<-5% = stress", "-5% to +5% = neutral", ">+5% = supportive"], "SUPPORTIVE", 6, "Rising real residential property prices reduce property-stress pressure.")
    return driver(name, value, "pct", ["<-5% = stress", "-5% to +5% = neutral", ">+5% = supportive"], "NEUTRAL", 0, "Property-price pressure is not extreme enough to drive liquidity.")


def currency_driver(name: str, value: float | None, weak_currency_negative: bool) -> LiquidityDriver:
    if value is None:
        return missing_driver(name, ["<=-5% = supportive", "-5% to +5% = neutral", ">=+5% = pressure"])
    if abs(value) < 5:
        return driver(name, value, "pct", ["<=-5% = supportive", "-5% to +5% = neutral", ">=+5% = pressure"], "NEUTRAL", 0, "Currency pressure is contained.")
    if value >= 5:
        effect = -8 if weak_currency_negative else 4
        explanation = "Currency weakness tightens imported-inflation and capital-flow pressure." if weak_currency_negative else "EUR strength can ease imported inflation pressure."
        return driver(name, value, "pct", ["<=-5% = supportive", "-5% to +5% = neutral", ">=+5% = pressure"], "PRESSURE", effect, explanation)
    effect = 6 if weak_currency_negative else -6
    explanation = "Currency strength reduces imported-inflation pressure." if weak_currency_negative else "EUR weakness adds imported-inflation pressure."
    return driver(name, value, "pct", ["<=-5% = supportive", "-5% to +5% = neutral", ">=+5% = pressure"], "SUPPORTIVE", effect, explanation)


def latest_series_value(series: pd.Series) -> float | None:
    if series.empty:
        return None
    values = series.dropna()
    if values.empty:
        return None
    return float(values.iloc[-1])


def series_change(series: pd.Series, periods: int) -> float | None:
    values = series.dropna()
    if len(values) <= periods:
        return None
    return float(values.iloc[-1] - values.iloc[-periods - 1])


def stress_placeholder_driver(name: str) -> LiquidityDriver:
    return missing_driver(name, ["Data source not integrated = no score effect"])


def missing_driver(name: str, thresholds: list[str]) -> LiquidityDriver:
    return LiquidityDriver(
        name=name,
        value=None,
        value_kind="number",
        thresholds=thresholds,
        classification="MISSING DATA",
        score_effect=0,
        explanation=f"{name} is unavailable, so it contributes no score and lowers confidence.",
    )


def driver(
    name: str,
    value: float | None,
    value_kind: str,
    thresholds: list[str],
    classification: str,
    score_effect: float,
    explanation: str,
) -> LiquidityDriver:
    return LiquidityDriver(name, value, value_kind, thresholds, classification, score_effect, explanation)


def classify_liquidity(score: float) -> str:
    if score <= 30:
        return "Liquidity Contracting"
    if score >= 70:
        return "Liquidity Expanding"
    return "Neutral"


def coverage_confidence(available: int, expected: int, warnings: list[str]) -> int:
    coverage = available / expected if expected else 0
    confidence = 25 + (coverage * 65) - min(len(warnings) * 3, 20)
    return int(max(5, min(confidence, 90)))


def weighted_average(values: list[float], weights: list[float]) -> float:
    total_weight = sum(weights)
    if total_weight == 0:
        return 0
    return sum(value * weight for value, weight in zip(values, weights)) / total_weight


def weighted_score_contribution(economy: EconomyLiquidity) -> float:
    return economy.weight * (economy.score - 50)


def liquidity_warnings(economies: list[EconomyLiquidity], weight_total: float) -> list[str]:
    warnings = [warning for economy in economies for warning in economy.warnings]
    if abs(weight_total - 1.0) > 0.001:
        warnings.append(f"Liquidity weights sum to {weight_total:.2f}; weighted averages are normalized by total configured weight.")
    for economy in economies:
        if economy.expected_drivers and economy.available_drivers == 0:
            warnings.append(f"{economy.name} has no live liquidity drivers; its score is a neutral placeholder and confidence is low.")
        elif economy.expected_drivers and economy.available_drivers < economy.expected_drivers:
            warnings.append(
                f"{economy.name} liquidity coverage is partial ({economy.available_drivers}/{economy.expected_drivers} drivers available)."
            )
    return warnings


def top_global_drivers(economies: list[EconomyLiquidity]) -> list[str]:
    candidates: list[tuple[float, str]] = []
    for economy in economies:
        for item in economy.drivers:
            if item.classification == "MISSING DATA":
                candidates.append((economy.weight * 4, f"{economy.name} {item.name} missing; confidence reduced"))
            elif item.score_effect != 0:
                direction = "supports liquidity" if item.score_effect > 0 else "tightens liquidity"
                weighted_effect = item.score_effect * economy.weight
                candidates.append(
                    (
                        abs(weighted_effect),
                        f"{economy.name} {item.name} {direction} ({format_score(weighted_effect)} weighted points)",
                    )
                )
    candidates.sort(reverse=True, key=lambda item: item[0])
    return [text for _, text in candidates[:5]]


def save_liquidity_outputs(result: GlobalLiquidityResult) -> None:
    rows = [
        {
            "Economy": economy.code,
            "Name": economy.name,
            "Weight": economy.weight,
            "Liquidity Score": round(economy.score, 2),
            "Classification": economy.classification,
            "Confidence": economy.confidence,
            "Data Coverage": f"{economy.available_drivers}/{economy.expected_drivers}",
            "Weighted Contribution": round(weighted_score_contribution(economy), 2),
        }
        for economy in result.economies
    ]
    rows.append(
        {
            "Economy": "GLOBAL",
            "Name": "Global Liquidity",
            "Weight": 1.0,
            "Liquidity Score": round(result.score, 2),
            "Classification": result.classification,
            "Confidence": result.confidence,
            "Data Coverage": "",
            "Weighted Contribution": round(result.score - 50, 2),
        }
    )
    pd.DataFrame(rows).to_csv(result.output_dir / "global_liquidity_score.csv", index=False)
    save_breakdown_chart(result)


def save_breakdown_chart(result: GlobalLiquidityResult) -> None:
    mpl_config_dir = result.output_dir / ".mplconfig"
    mpl_config_dir.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    labels = [economy.code for economy in result.economies]
    scores = [economy.score for economy in result.economies]
    colors = ["#b23b3b" if score <= 30 else "#2f7d46" if score >= 70 else "#4f6f9f" for score in scores]
    plt.figure(figsize=(9, 4.5))
    bars = plt.bar(labels, scores, color=colors)
    plt.axhline(30, color="#888888", linewidth=0.8, linestyle="--")
    plt.axhline(70, color="#888888", linewidth=0.8, linestyle="--")
    for bar, economy in zip(bars, result.economies):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{economy.score:.0f}\n{economy.confidence}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    plt.ylim(0, 100)
    plt.ylabel("Liquidity Score")
    plt.title("Global Liquidity Breakdown (score / confidence)")
    plt.grid(True, axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(result.output_dir / "global_liquidity_breakdown.png", dpi=140)
    plt.close()


def render_liquidity_report(result: GlobalLiquidityResult, show_details: bool) -> str:
    parts = [
        "GLOBAL LIQUIDITY COMPASS",
        "=================================================",
        f"Global Liquidity Score: {result.score:.0f}/100",
        f"Classification: {result.classification}",
        f"Confidence: {result.confidence}%",
        "",
        "Country Breakdown:",
        *[
            (
                f"{economy.code}: {economy.score:.0f}/100 - {economy.classification} "
                f"({economy.confidence}% confidence, weight {economy.weight:.0%}, coverage {economy.available_drivers}/{economy.expected_drivers})"
            )
            for economy in result.economies
        ],
        "",
        "Top Drivers:",
        *[f"{index}. {driver}" for index, driver in enumerate(result.top_drivers, start=1)],
        "",
        "Coverage Caveats:",
        *coverage_caveats(result.economies),
        "",
        "Market Interpretation:",
        market_interpretation(result),
        "",
        "Saved Outputs:",
        str(result.output_dir / "global_liquidity_score.csv"),
        str(result.output_dir / "global_liquidity_breakdown.png"),
    ]

    if show_details:
        parts.extend(["", "=================================================", "DETAILS", "================================================="])
        for economy in result.economies:
            parts.extend(render_economy_details(economy))

    if result.warnings:
        parts.extend(["", "Missing-Data Warnings:", *[f"- {warning}" for warning in dedupe(result.warnings)]])
    return "\n".join(parts)


def render_economy_details(economy: EconomyLiquidity) -> list[str]:
    lines = [
        "",
        f"{economy.name} ({economy.code})",
        "-------------------------------------------------",
        f"Liquidity Score: {economy.score:.0f}/100",
        f"Classification: {economy.classification}",
        f"Confidence: {economy.confidence}%",
        f"Weight: {economy.weight:.0%}",
        f"Weighted Contribution vs Neutral: {format_score(weighted_score_contribution(economy))}",
        f"Data Coverage: {economy.available_drivers}/{economy.expected_drivers}",
        "Drivers:",
    ]
    for item in economy.drivers:
        lines.extend(
            [
                f"- {item.name}: {format_value(item.value, item.value_kind)}",
                "  Thresholds:",
                *[f"  {threshold}" for threshold in item.thresholds],
                f"  Classification: {item.classification}",
                f"  Score Effect: {format_score(item.score_effect)}",
                f"  Explanation: {item.explanation}",
            ]
        )
    return lines


def coverage_caveats(economies: list[EconomyLiquidity]) -> list[str]:
    caveats: list[str] = []
    for economy in economies:
        if economy.expected_drivers and economy.available_drivers == 0:
            caveats.append(f"- {economy.name}: placeholder only; score is neutral and confidence is low.")
        elif economy.expected_drivers and economy.available_drivers < economy.expected_drivers:
            caveats.append(f"- {economy.name}: partial coverage ({economy.available_drivers}/{economy.expected_drivers}).")
    return caveats or ["- No major coverage caveats."]


def market_interpretation(result: GlobalLiquidityResult) -> str:
    if result.classification == "Liquidity Expanding":
        return (
            "Expanding liquidity is generally supportive for equities and crypto, can ease credit conditions, "
            "and may weigh on the USD if global risk appetite improves. Gold can benefit when liquidity improves and real-rate pressure fades."
        )
    if result.classification == "Liquidity Contracting":
        return (
            "Contracting liquidity is usually a headwind for equities and crypto, can support the USD through tighter global funding conditions, "
            "and may pressure gold if real yields rise. Bonds may benefit only if contraction also raises growth risk."
        )
    return (
        "Neutral liquidity suggests markets remain sensitive to marginal shifts in central-bank guidance, yields, credit stress, and China data. "
        "Equities and crypto may need clearer liquidity expansion, while bonds, USD, and gold can move with the next inflation or growth surprise."
    )


def format_value(value: float | str | None, kind: str) -> str:
    if value is None:
        return "N/A"
    if kind == "text":
        return str(value)
    if kind == "pct":
        return f"{value:.1f}%"
    if kind == "pp":
        return f"{value:.2f} pp"
    return f"{value:.1f}"


def format_score(value: float) -> str:
    if abs(value) < 0.005:
        value = 0
    if float(value).is_integer():
        return f"{int(value):+d}"
    return f"{value:+.1f}"


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(value, maximum))


def dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            result.append(item)
            seen.add(item)
    return result
