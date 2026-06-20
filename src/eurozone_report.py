from typing import Any

from src.policy_signal import PolicySignal
from src.report import data_coverage_lines, interpretation, market_meaning, policy_decision_lines, render_block


def render_eurozone_report(signal: PolicySignal, energy: dict[str, Any], summary_only: bool = False) -> str:
    summary = render_eurozone_summary(signal, energy)
    if summary_only:
        return summary
    return "\n\n".join([summary, render_ecb_narrative_filter(signal, energy), render_eurozone_details(signal)])


def render_eurozone_summary(signal: PolicySignal, energy: dict[str, Any]) -> str:
    return "\n".join(
        [
            "EUROZONE MACRO NOISE SUMMARY",
            "=================================================",
            "Current ECB View:",
            ecb_view(signal),
            "",
            "Key Tension:",
            key_tension(signal),
            "",
            "Noise Filter:",
            noise_filter(signal),
            "",
            "Energy Context:",
            ecb_energy_context(signal, energy),
        ]
    )


def render_ecb_narrative_filter(signal: PolicySignal, energy: dict[str, Any]) -> str:
    indicators = signal.indicators
    checks = [
        ("Growth weakness", indicators.get("growth_yoy"), growth_check(signal)),
        ("Core HICP", indicators.get("core_inflation_yoy"), inflation_check(signal)),
        ("Services inflation", indicators.get("services_inflation_yoy"), "Unavailable; aggressive easing lacks services-inflation confirmation."),
        ("Wage pressure", indicators.get("wage_growth"), "Unavailable; negotiated-wage persistence remains an important evidence gap."),
        ("Bank lending", indicators.get("bank_lending_stress"), "Unavailable; no bank-lending stress is inferred from a US market proxy."),
        ("Peripheral sovereign spread", indicators.get("sovereign_spread"), sovereign_check(indicators.get("sovereign_spread"))),
        ("EUR/USD YoY", indicators.get("currency_change_yoy"), currency_check(indicators.get("currency_change_yoy"))),
        ("Oil YoY", energy.get("yoy_change"), energy_check(energy)),
    ]
    parts = [
        "ECB NARRATIVE FILTER",
        "=================================================",
        "Market Narrative Being Tested:",
        '"ECB will cut aggressively because European growth is weak."',
        "",
        "Reality Check:",
    ]
    for name, value, explanation in checks:
        parts.extend([f"- {name}: {format_optional(value)}", f"  {explanation}"])
    parts.extend(["", f"Verdict: {ecb_narrative_verdict(signal, energy)}"])
    return "\n".join(parts)


def render_eurozone_details(signal: PolicySignal) -> str:
    parts = [
        "ECB POLICY PRESSURE DETAIL",
        "=================================================",
        "Economy: Eurozone",
        "Central Bank: European Central Bank",
        "Policy Tool: ECB Policy Rates and Transmission Conditions",
        f"Current Signal: {signal.signal}",
        f"Policy Bias: {signal.policy_bias}",
        f"Confidence: {signal.confidence}%",
        "",
        "DATA COVERAGE",
        "=================================================",
        *data_coverage_lines(signal),
        "",
        "BLOCK EXPLAINABILITY",
        "=================================================",
        *render_block("ECB Inflation Persistence", signal.inflation, signed_positive=True),
        *render_block("Eurozone Labor Slack", signal.labor, signed_positive=False),
        *render_block("Eurozone Growth Weakness", signal.growth, signed_positive=False),
        *render_block("Fragmentation and Financial Stress", signal.financial, signed_positive=False),
        *render_block("EUR Inflation Pressure", signal.currency, signed_positive=True),
        "",
        "ECB POLICY DECISION LOGIC",
        "=================================================",
        *policy_decision_lines(signal),
        "",
        "ECB Interpretation:",
        ecb_reasoning(signal),
        "",
        "Market Meaning:",
        market_meaning(signal),
    ]
    if signal.warnings:
        parts.extend(["", "Data Warnings:", *[f"- {warning}" for warning in signal.warnings]])
    return "\n".join(parts)


def ecb_view(signal: PolicySignal) -> str:
    available, expected = total_coverage(signal)
    if expected == 0 or available / expected < 0.25:
        return "INSUFFICIENT DATA"
    if signal.signal == "EASING":
        return "Easing Bias"
    if signal.signal == "TIGHTENING":
        return "Tightening Bias"
    return {"Dovish": "Dovish Hold", "Hawkish": "Hawkish Hold"}.get(signal.policy_bias, "HOLD")


def total_coverage(signal: PolicySignal) -> tuple[int, int]:
    blocks = [signal.inflation, signal.labor, signal.growth, signal.financial, signal.currency]
    return sum(block.available for block in blocks), sum(block.expected for block in blocks)


def key_tension(signal: PolicySignal) -> str:
    if signal.growth.label in {"HIGH", "MODERATE"} and signal.inflation.label in {"STICKY", "HIGH"}:
        return "Growth is weak, but domestic inflation persistence is not yet soft enough to validate aggressive easing."
    if signal.growth.label in {"HIGH", "MODERATE"}:
        return "Weak growth supports easing, while missing services, wage, and credit-transmission evidence limits conviction."
    if signal.inflation.label in {"STICKY", "HIGH"}:
        return "Growth is not clearly collapsing, while persistent inflation keeps the ECB cautious."
    return "The available growth and inflation evidence does not establish a decisive ECB policy imbalance."


def noise_filter(signal: PolicySignal) -> str:
    if signal.growth.label in {"HIGH", "MODERATE"}:
        return "A 'weak Germany means aggressive ECB cuts' narrative is only partially supported unless core and services inflation, wages, bank lending, and fragmentation data confirm it."
    return "Country-level weakness does not automatically imply aggressive area-wide easing; the ECB must assess inflation persistence and monetary transmission across the currency union."


def ecb_energy_context(signal: PolicySignal, energy: dict[str, Any]) -> str:
    oil_yoy = energy.get("yoy_change")
    headline = signal.indicators.get("headline_inflation_yoy")
    core = signal.indicators.get("core_inflation_yoy")
    if oil_yoy is None:
        return "Oil data is unavailable, so headline HICP cannot be cleanly separated from energy effects."
    if oil_yoy >= 20:
        if core is not None and headline is not None and headline > core:
            return "Oil is amplifying headline HICP above core inflation. This is an energy shock, not automatic evidence of domestic services or wage persistence."
        return "Oil is adding a material headline HICP shock; core, services, and wages must confirm whether it is persistent."
    if oil_yoy <= -15:
        return "Energy disinflation is reducing headline HICP, but domestic core, services, and wage inflation still determine ECB persistence risk."
    return "Energy is not producing a major headline HICP shock, so domestic inflation persistence deserves more weight."


def ecb_narrative_verdict(signal: PolicySignal, energy: dict[str, Any]) -> str:
    indicators = signal.indicators
    values = [
        indicators.get("growth_yoy"),
        indicators.get("core_inflation_yoy"),
        indicators.get("services_inflation_yoy"),
        indicators.get("wage_growth"),
        indicators.get("bank_lending_stress"),
        indicators.get("sovereign_spread"),
        indicators.get("currency_change_yoy"),
        energy.get("yoy_change"),
    ]
    if sum(value is not None for value in values) < 3:
        return "Insufficient Data"
    support = [
        signal.growth.label in {"HIGH", "MODERATE"},
        indicators.get("core_inflation_yoy") is not None and indicators.get("core_inflation_yoy") < 2.5,
        indicators.get("sovereign_spread") is not None and indicators.get("sovereign_spread") < 2.5,
        indicators.get("currency_change_yoy") is not None and indicators.get("currency_change_yoy") > -8,
        energy.get("yoy_change") is not None and energy.get("yoy_change") < 20,
    ]
    ratio = sum(support) / len(support)
    domestic_confirmation_missing = indicators.get("services_inflation_yoy") is None or indicators.get("wage_growth") is None
    if ratio >= 0.8 and not domestic_confirmation_missing:
        return "Supported"
    if ratio >= 0.4:
        return "Partially Supported"
    return "Not Supported"


def growth_check(signal: PolicySignal) -> str:
    return "Weak activity supports an easing bias." if signal.growth.label in {"HIGH", "MODERATE"} else "Area-wide growth does not confirm aggressive easing."


def inflation_check(signal: PolicySignal) -> str:
    return "Domestic inflation persistence constrains aggressive easing." if signal.inflation.label in {"STICKY", "HIGH"} else "Core HICP does not present a strong persistence barrier."


def sovereign_check(value: float | None) -> str:
    if value is None:
        return "Unavailable; fragmentation risk cannot be assessed."
    return "Fragmentation pressure is contained." if value < 2.5 else "Wide peripheral spreads indicate material fragmentation pressure."


def currency_check(value: float | None) -> str:
    if value is None:
        return "Unavailable; the imported-inflation constraint from EUR weakness is uncertain."
    return "EUR weakness constrains aggressive easing." if value <= -8 else "EUR pressure does not create a severe imported-inflation constraint."


def energy_check(energy: dict[str, Any]) -> str:
    value = energy.get("yoy_change")
    if value is None:
        return "Unavailable; energy attribution is uncertain."
    if value >= 20:
        return "Energy is amplifying headline HICP."
    if value <= -15:
        return "Energy disinflation supports lower headline HICP."
    return "Energy is not a dominant headline shock."


def ecb_reasoning(signal: PolicySignal) -> str:
    return interpretation(signal).replace("For the Eurozone,", "For the ECB,")


def format_optional(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.2f}"
