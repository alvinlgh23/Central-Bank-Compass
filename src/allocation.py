from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import pandas as pd

from src.liquidity import (
    GlobalLiquidityResult,
    build_economy_liquidity,
    classify_liquidity,
    liquidity_warnings,
    top_global_drivers,
    weighted_average,
)


ASSET_CLASSES = ["Cash", "Government Bonds", "Equities", "Gold", "Crypto"]


ALLOCATION_RANGES: dict[str, dict[str, str]] = {
    "Liquidity Expansion": {
        "Cash": "5-15%",
        "Government Bonds": "10-25%",
        "Equities": "45-65%",
        "Gold": "5-15%",
        "Crypto": "5-15%",
    },
    "Liquidity Neutral": {
        "Cash": "15-30%",
        "Government Bonds": "20-35%",
        "Equities": "30-50%",
        "Gold": "5-15%",
        "Crypto": "0-10%",
    },
    "Liquidity Contraction": {
        "Cash": "25-45%",
        "Government Bonds": "25-45%",
        "Equities": "10-30%",
        "Gold": "5-15%",
        "Crypto": "0-5%",
    },
    "Disinflationary Expansion": {
        "Cash": "5-15%",
        "Government Bonds": "15-30%",
        "Equities": "45-65%",
        "Gold": "5-15%",
        "Crypto": "5-15%",
    },
    "Inflationary Expansion": {
        "Cash": "10-25%",
        "Government Bonds": "10-25%",
        "Equities": "35-55%",
        "Gold": "10-25%",
        "Crypto": "0-10%",
    },
    "Growth Slowdown": {
        "Cash": "20-35%",
        "Government Bonds": "25-45%",
        "Equities": "15-35%",
        "Gold": "5-15%",
        "Crypto": "0-5%",
    },
    "Stagflationary Pressure": {
        "Cash": "20-35%",
        "Government Bonds": "10-25%",
        "Equities": "15-35%",
        "Gold": "15-30%",
        "Crypto": "0-5%",
    },
    "Financial Stress Shock": {
        "Cash": "30-50%",
        "Government Bonds": "25-45%",
        "Equities": "5-25%",
        "Gold": "10-25%",
        "Crypto": "0-5%",
    },
}


@dataclass(frozen=True)
class AllocationInput:
    global_liquidity_score: float
    global_liquidity_classification: str
    liquidity_confidence: int
    us_policy_signal: str
    us_policy_bias: str
    inflation_pressure: str
    growth_weakness: str
    financial_stress: str
    missing_data_warnings: list[str]


@dataclass(frozen=True)
class AllocationResult:
    regime: str
    confidence: int
    allocation_ranges: dict[str, str]
    inputs: AllocationInput
    decision_logic: list[str]
    confidence_penalties: list[str]
    output_dir: Path


def run_allocation_framework(
    config: dict[str, Any],
    project_dir: Path,
    show_details: bool = False,
    debug_data: bool = False,
) -> str:
    output_dir = project_dir / "outputs"
    output_dir.mkdir(exist_ok=True)

    liquidity = build_global_liquidity_snapshot(config, output_dir, debug_data=debug_data)
    us = next(economy for economy in liquidity.economies if economy.code == "US")
    us_policy = extract_policy_driver(us)
    inputs = AllocationInput(
        global_liquidity_score=liquidity.score,
        global_liquidity_classification=liquidity.classification,
        liquidity_confidence=liquidity.confidence,
        us_policy_signal=us_policy[0],
        us_policy_bias=us_policy[1],
        inflation_pressure=us.inflation_label or "UNKNOWN",
        growth_weakness=us.growth_label or "UNKNOWN",
        financial_stress=us.financial_label or "UNKNOWN",
        missing_data_warnings=liquidity.warnings,
    )
    regime, decision_logic = classify_regime(inputs)
    penalties = confidence_penalties(inputs)
    confidence = calculate_confidence(inputs, penalties)
    result = AllocationResult(
        regime=regime,
        confidence=confidence,
        allocation_ranges=ALLOCATION_RANGES[regime],
        inputs=inputs,
        decision_logic=decision_logic,
        confidence_penalties=penalties,
        output_dir=output_dir,
    )
    save_allocation_outputs(result)
    return render_allocation_report(result, show_details)


def build_global_liquidity_snapshot(
    config: dict[str, Any],
    output_dir: Path,
    debug_data: bool = False,
) -> GlobalLiquidityResult:
    weights = config.get("liquidity", {}).get("weights", {})
    weight_total = sum(float(weight) for weight in weights.values())
    economies = [
        build_economy_liquidity(code, float(weight), config, debug_data=debug_data)
        for code, weight in weights.items()
    ]
    score = weighted_average([economy.score for economy in economies], [economy.weight for economy in economies])
    confidence = int(round(weighted_average([economy.confidence for economy in economies], [economy.weight for economy in economies])))
    return GlobalLiquidityResult(
        score=score,
        classification=classify_liquidity(score),
        confidence=confidence,
        economies=economies,
        top_drivers=top_global_drivers(economies),
        warnings=liquidity_warnings(economies, weight_total),
        output_dir=output_dir,
    )


def classify_regime(inputs: AllocationInput) -> tuple[str, list[str]]:
    logic = [
        "Financial Stress Shock: financial stress is HIGH.",
        "Stagflationary Pressure: inflation is HIGH and growth weakness is HIGH or MODERATE.",
        "Disinflationary Expansion: liquidity is expanding or neutral, inflation is not HIGH, growth weakness is LOW, and financial stress is LOW.",
        "Inflationary Expansion: inflation is HIGH while growth weakness and financial stress are LOW.",
        "Growth Slowdown: growth weakness is HIGH or MODERATE while financial stress is not HIGH.",
        "Liquidity Contraction: global liquidity score is <=30.",
        "Liquidity Expansion: global liquidity score is >=70.",
        "Liquidity Neutral: no stronger regime rule is triggered.",
    ]

    if inputs.financial_stress == "HIGH":
        return "Financial Stress Shock", logic
    if inputs.inflation_pressure == "HIGH" and inputs.growth_weakness in {"HIGH", "MODERATE"}:
        return "Stagflationary Pressure", logic
    if inputs.global_liquidity_score <= 30:
        return "Liquidity Contraction", logic
    if (
        inputs.global_liquidity_score >= 55
        and inputs.inflation_pressure != "HIGH"
        and inputs.growth_weakness == "LOW"
        and inputs.financial_stress == "LOW"
    ):
        return "Disinflationary Expansion", logic
    if inputs.inflation_pressure == "HIGH" and inputs.growth_weakness == "LOW" and inputs.financial_stress == "LOW":
        return "Inflationary Expansion", logic
    if inputs.growth_weakness in {"HIGH", "MODERATE"} and inputs.financial_stress != "HIGH":
        return "Growth Slowdown", logic
    if inputs.global_liquidity_score >= 70:
        return "Liquidity Expansion", logic
    return "Liquidity Neutral", logic


def calculate_confidence(inputs: AllocationInput, penalties: list[str]) -> int:
    confidence = min(inputs.liquidity_confidence, 85)
    if inputs.inflation_pressure == "UNKNOWN":
        confidence -= 12
    if inputs.growth_weakness == "UNKNOWN":
        confidence -= 12
    if inputs.financial_stress == "UNKNOWN":
        confidence -= 12
    confidence -= min(len(penalties) * 4, 20)
    return int(max(15, min(confidence, 90)))


def confidence_penalties(inputs: AllocationInput) -> list[str]:
    penalties: list[str] = []
    if inputs.liquidity_confidence < 60:
        penalties.append(f"Global liquidity confidence is {inputs.liquidity_confidence}%, so allocation regime confidence is reduced.")
    if inputs.missing_data_warnings:
        penalties.append("Some liquidity inputs are missing, especially placeholder-driven country data.")
    if inputs.inflation_pressure == "UNKNOWN":
        penalties.append("US inflation pressure is unknown.")
    if inputs.growth_weakness == "UNKNOWN":
        penalties.append("US growth weakness is unknown.")
    if inputs.financial_stress == "UNKNOWN":
        penalties.append("US financial stress is unknown.")
    return penalties


def extract_policy_driver(economy: Any) -> tuple[str, str]:
    for driver in economy.drivers:
        if "Policy Signal" in driver.name:
            text = str(driver.value or "")
            if "/" in text:
                signal, bias = text.split("/", 1)
                return signal.strip(), bias.strip()
    return economy.policy_signal or "UNKNOWN", "UNKNOWN"


def save_allocation_outputs(result: AllocationResult) -> None:
    frame = pd.DataFrame(
        [
            {
                "Regime": result.regime,
                "Asset Class": asset,
                "Educational Allocation Range": allocation_range,
                "Confidence": result.confidence,
                "Global Liquidity Score": round(result.inputs.global_liquidity_score, 2),
                "US Policy Signal": result.inputs.us_policy_signal,
                "US Policy Bias": result.inputs.us_policy_bias,
            }
            for asset, allocation_range in result.allocation_ranges.items()
        ]
    )
    frame.to_csv(result.output_dir / "allocation_framework.csv", index=False)
    save_allocation_chart(result)


def save_allocation_chart(result: AllocationResult) -> None:
    mpl_config_dir = result.output_dir / ".mplconfig"
    mpl_config_dir.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    mids = [range_midpoint(result.allocation_ranges[asset]) for asset in ASSET_CLASSES]
    lows = [range_bounds(result.allocation_ranges[asset])[0] for asset in ASSET_CLASSES]
    highs = [range_bounds(result.allocation_ranges[asset])[1] for asset in ASSET_CLASSES]
    lower_errors = [mid - low for mid, low in zip(mids, lows)]
    upper_errors = [high - mid for mid, high in zip(mids, highs)]

    plt.figure(figsize=(9, 4.8))
    plt.bar(ASSET_CLASSES, mids, color=["#6b7280", "#4f6f9f", "#2f7d46", "#c99a2e", "#7c3aed"])
    plt.errorbar(ASSET_CLASSES, mids, yerr=[lower_errors, upper_errors], fmt="none", ecolor="#222222", capsize=5)
    plt.ylim(0, 75)
    plt.ylabel("Allocation Range Midpoint (%)")
    plt.title(f"Educational Allocation Ranges: {result.regime}")
    plt.xticks(rotation=15, ha="right")
    plt.grid(True, axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(result.output_dir / "allocation_ranges.png", dpi=140)
    plt.close()


def render_allocation_report(result: AllocationResult, show_details: bool) -> str:
    inputs = result.inputs
    parts = [
        "MACRO REGIME ALLOCATION FRAMEWORK",
        "=================================================",
        f"Current Macro Regime: {result.regime}",
        f"Confidence: {result.confidence}%",
        "",
        f"Global Liquidity Score: {inputs.global_liquidity_score:.0f}/100",
        f"US Policy Signal: {inputs.us_policy_signal}",
        f"US Policy Bias: {inputs.us_policy_bias}",
        "",
        "Suggested Educational Allocation Ranges:",
        *[f"{asset}: {result.allocation_ranges[asset]}" for asset in ASSET_CLASSES],
        "",
        "Reasoning:",
        reasoning(result),
        "",
        "Risk Notes:",
        "- This is an educational macro framework, not financial advice.",
        "- Asset classes can move differently from macro expectations.",
        "- Crypto is highly volatile and should be treated as risk capital.",
        "- These are broad regime ranges, not individual-security recommendations or buy/sell signals.",
        "",
        "Saved Outputs:",
        str(result.output_dir / "allocation_framework.csv"),
        str(result.output_dir / "allocation_ranges.png"),
    ]
    if show_details:
        parts.extend(
            [
                "",
                "=================================================",
                "DETAILS",
                "=================================================",
                "Input Signals:",
                f"- Global Liquidity Classification: {inputs.global_liquidity_classification}",
                f"- Global Liquidity Confidence: {inputs.liquidity_confidence}%",
                f"- US Policy Signal: {inputs.us_policy_signal}",
                f"- US Policy Bias: {inputs.us_policy_bias}",
                f"- Inflation Pressure: {inputs.inflation_pressure}",
                f"- Growth Weakness: {inputs.growth_weakness}",
                f"- Financial Stress: {inputs.financial_stress}",
                "",
                "Thresholds:",
                "- Liquidity <=30 = Liquidity Contraction",
                "- Liquidity 30-70 = Liquidity Neutral",
                "- Liquidity >=70 = Liquidity Expansion",
                "- Financial Stress HIGH overrides into Financial Stress Shock",
                "- Inflation HIGH plus Growth Weakness HIGH/MODERATE = Stagflationary Pressure",
                "",
                "Regime Decision Logic:",
                *[f"- {line}" for line in result.decision_logic],
                "",
                "Allocation Rule Used:",
                f"- {result.regime}: " + ", ".join(f"{asset} {value}" for asset, value in result.allocation_ranges.items()),
                "",
                "Confidence Penalties:",
                *(f"- {penalty}" for penalty in result.confidence_penalties),
            ]
        )
    return "\n".join(parts)


def reasoning(result: AllocationResult) -> str:
    inputs = result.inputs
    return (
        f"The framework maps a {inputs.global_liquidity_classification.lower()} global liquidity backdrop "
        f"({inputs.global_liquidity_score:.0f}/100) with a US {inputs.us_policy_signal} signal and "
        f"{inputs.us_policy_bias.lower()} policy bias into {result.regime}. US inflation pressure is "
        f"{inputs.inflation_pressure.lower()}, growth weakness is {inputs.growth_weakness.lower()}, and "
        f"financial stress is {inputs.financial_stress.lower()}. The allocation ranges reflect broad macro "
        f"risk exposure rather than a recommendation to own specific instruments."
    )


def range_bounds(text: str) -> tuple[float, float]:
    clean = text.replace("%", "")
    left, right = clean.split("-", 1)
    return float(left), float(right)


def range_midpoint(text: str) -> float:
    low, high = range_bounds(text)
    return (low + high) / 2
