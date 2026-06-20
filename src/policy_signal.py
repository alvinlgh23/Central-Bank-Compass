from dataclasses import dataclass
from typing import Any

from src.country_profiles import EconomyProfile
from src.indicators import IndicatorSet
from src.scoring import ScoreBlock, inflation_is_accelerating, score_macro_blocks


@dataclass(frozen=True)
class MissingIndicator:
    block: str
    indicator_name: str
    expected_source: str
    reason: str
    confidence_reduced: bool


@dataclass(frozen=True)
class PolicySignal:
    profile: EconomyProfile
    signal: str
    policy_bias: str
    model_view: str
    confidence: int
    coverage_confidence_adjustment: str
    policy_score: float
    inflation: ScoreBlock
    labor: ScoreBlock
    growth: ScoreBlock
    financial: ScoreBlock
    currency: ScoreBlock
    block_scores: dict[str, float]
    decision_rules: list[str]
    indicators: IndicatorSet
    warnings: list[str]
    placeholders: list[str]
    missing_indicators: list[MissingIndicator]


def build_policy_signal(
    indicators: IndicatorSet,
    profile: EconomyProfile,
    config: dict[str, Any],
    data_warnings: list[str],
    placeholders: list[str],
) -> PolicySignal:
    blocks = score_macro_blocks(indicators, profile)
    weights = profile.scoring_weights

    block_scores = {
        "inflation": blocks["inflation"].score * weights.get("inflation", 1.0),
        "labor": -(blocks["labor"].score * weights.get("labor", 1.0)),
        "growth": -(blocks["growth"].score * weights.get("growth", 1.0)),
        "financial": -(blocks["financial"].score * weights.get("financial", 1.0)),
        "currency": blocks["currency"].score * weights.get("currency", 0.0),
    }
    policy_score = sum(block_scores.values())

    thresholds = config.get("scoring", {})
    base_signal = classify(
        policy_score,
        tightening_threshold=thresholds.get("tightening_threshold", 28),
        easing_threshold=thresholds.get("easing_threshold", -28),
    )
    signal, policy_bias, decision_rules = apply_realism_rules(base_signal, blocks, policy_score, thresholds)
    model_view = build_model_view(signal, policy_bias, profile)

    warnings = build_warnings(indicators, blocks, data_warnings, placeholders)
    missing_indicators = build_missing_indicators(profile, blocks, placeholders)
    coverage_adjustment = coverage_confidence_adjustment(blocks)
    confidence = calculate_confidence(signal, policy_score, blocks, warnings)

    return PolicySignal(
        profile=profile,
        signal=signal,
        policy_bias=policy_bias,
        model_view=model_view,
        confidence=confidence,
        coverage_confidence_adjustment=coverage_adjustment,
        policy_score=policy_score,
        inflation=blocks["inflation"],
        labor=blocks["labor"],
        growth=blocks["growth"],
        financial=blocks["financial"],
        currency=blocks["currency"],
        block_scores=block_scores,
        decision_rules=decision_rules,
        indicators=indicators,
        warnings=warnings,
        placeholders=placeholders,
        missing_indicators=missing_indicators,
    )


def classify(policy_score: float, tightening_threshold: float, easing_threshold: float) -> str:
    if policy_score > tightening_threshold:
        return "TIGHTENING"
    if policy_score <= easing_threshold:
        return "EASING"
    return "HOLD"


def apply_realism_rules(
    base_signal: str,
    blocks: dict[str, ScoreBlock],
    policy_score: float,
    thresholds: dict[str, Any],
) -> tuple[str, str, list[str]]:
    tightening_threshold = thresholds.get("tightening_threshold", 28)
    easing_threshold = thresholds.get("easing_threshold", -28)
    rules = [
        f"<= {easing_threshold:+.0f} = EASING",
        f"{easing_threshold:+.0f} to {tightening_threshold:+.0f} = HOLD",
        f"> {tightening_threshold:+.0f} = TIGHTENING CANDIDATE",
        "TIGHTENING requires high inflation, accelerating inflation, low labor weakness, low growth weakness, and low financial stress.",
    ]

    if base_signal == "EASING":
        return "EASING", "Dovish", rules
    if base_signal != "TIGHTENING":
        if policy_score > 8:
            return "HOLD", "Hawkish", rules
        if policy_score < -8:
            return "HOLD", "Dovish", rules
        return "HOLD", "Neutral", rules

    can_tighten = (
        blocks["inflation"].label == "HIGH"
        and inflation_is_accelerating(blocks["inflation"])
        and blocks["labor"].label == "LOW"
        and blocks["growth"].label == "LOW"
        and blocks["financial"].label == "LOW"
    )
    if can_tighten:
        return "TIGHTENING", "Hawkish", rules
    return "HOLD", "Hawkish", rules


def build_model_view(signal: str, policy_bias: str, profile: EconomyProfile) -> str:
    if signal == "TIGHTENING":
        return f"The model says {profile.central_bank_name} has a full tightening signal, not just a hawkish hold."
    if signal == "EASING":
        return f"The model says {profile.central_bank_name} has enough macro weakness to justify an easing stance."
    if policy_bias == "Hawkish":
        return f"The model says {profile.central_bank_name} should stay restrictive, but not necessarily tighten further."
    if policy_bias == "Dovish":
        return f"The model says {profile.central_bank_name} should lean easier, but the evidence is not strong enough for an outright easing signal."
    return f"The model says {profile.central_bank_name} has a balanced case for waiting."


def calculate_confidence(
    signal: str,
    policy_score: float,
    blocks: dict[str, ScoreBlock],
    warnings: list[str],
) -> int:
    expected = sum(block.expected for block in blocks.values())
    available = sum(block.available for block in blocks.values())
    coverage = available / expected if expected else 0

    distance = min(abs(policy_score), 45) * 0.75
    confidence = 36 + distance + (coverage * 22)

    inflation = blocks["inflation"].label
    weakness_labels = {blocks["labor"].label, blocks["growth"].label, blocks["financial"].label}
    if inflation == "HIGH" and weakness_labels.intersection({"MODERATE", "HIGH"}):
        confidence -= 12
    if signal == "HOLD":
        confidence -= 8
    if warnings:
        confidence -= min(len(warnings) * 3, 18)

    return int(max(25, min(confidence, 85)))


def coverage_confidence_adjustment(blocks: dict[str, ScoreBlock]) -> str:
    expected = sum(block.expected for block in blocks.values())
    available = sum(block.available for block in blocks.values())
    coverage = available / expected if expected else 0
    if coverage > 0.90:
        return "No penalty: data coverage is above 90%."
    if coverage >= 0.75:
        return "Small penalty: data coverage is between 75% and 90%."
    if coverage >= 0.50:
        return "Moderate penalty: data coverage is between 50% and 75%."
    return "Large penalty: data coverage is below 50%."


def build_warnings(
    indicators: IndicatorSet,
    blocks: dict[str, ScoreBlock],
    data_warnings: list[str],
    placeholders: list[str],
) -> list[str]:
    warnings = list(data_warnings)
    if indicators.latest_observation is None:
        warnings.append("No live macro observations were loaded.")
    for name, block in blocks.items():
        if block.available < block.expected:
            warnings.append(f"{name.title()} block has partial data coverage ({block.available}/{block.expected}).")
    for placeholder in placeholders:
        warnings.append(f"TODO data source not integrated: {placeholder}.")
    return warnings


def build_missing_indicators(
    profile: EconomyProfile,
    blocks: dict[str, ScoreBlock],
    placeholders: list[str],
) -> list[MissingIndicator]:
    missing: list[MissingIndicator] = []
    for block_name, block in blocks.items():
        for item in block.evidence:
            if item.value is None:
                missing.append(
                    MissingIndicator(
                        block=block_name,
                        indicator_name=item.name,
                        expected_source=expected_source_for_indicator(profile.code, item.name),
                        reason="Unavailable from configured source or source integration is TODO.",
                        confidence_reduced=True,
                    )
                )
    existing = {item.indicator_name.lower() for item in missing}
    for placeholder in placeholders:
        if placeholder.lower() not in existing:
            missing.append(
                MissingIndicator(
                    block=block_for_placeholder(placeholder),
                    indicator_name=placeholder,
                    expected_source=expected_source_for_placeholder(profile.code, placeholder),
                    reason="Placeholder-ready data source is not fully automated.",
                    confidence_reduced=True,
                )
            )
    return missing


def expected_source_for_indicator(economy_code: str, indicator_name: str) -> str:
    name = indicator_name.lower()
    if "core pce" in name or "headline inflation" in name or "claims" in name or "unemployment" in name or "gdp" in name:
        return "FRED / OECD fallback where configured"
    if "hicp" in name:
        return "FRED / ECB"
    if "cpi ex fresh food" in name:
        return "FRED / BOJ"
    if "mas core" in name:
        return "MAS / SingStat"
    if "pmi" in name or "external demand" in name:
        return "FRED / OECD / SingStat"
    if "vix" in name or "credit spread" in name or "financial stress" in name:
        return "FRED / central bank or market-data module"
    if "yield" in name:
        return "FRED / BOJ / market-data module"
    if "usd" in name or "eur" in name or "dollar" in name or "currency" in name:
        return "FRED / FX module"
    if economy_code == "SG":
        return "MAS / SingStat / FX module"
    return "Configured data source"


def expected_source_for_placeholder(economy_code: str, placeholder: str) -> str:
    text = placeholder.lower()
    if economy_code == "SG" or "sgd" in text or "mas" in text:
        return "MAS / SingStat"
    if economy_code == "EZ" or "hicp" in text or "sovereign" in text:
        return "ECB / Eurostat"
    if economy_code == "JP" or "wage" in text:
        return "BOJ / Japan official statistics"
    if "china" in text or "credit impulse" in text or "rrr" in text:
        return "PBC / NBS"
    return "Configured placeholder source"


def block_for_placeholder(placeholder: str) -> str:
    text = placeholder.lower()
    if "inflation" in text or "hicp" in text:
        return "inflation"
    if "wage" in text or "unemployment" in text:
        return "labor"
    if "external" in text or "demand" in text:
        return "growth"
    if "spread" in text or "stress" in text:
        return "financial"
    if "sgd" in text or "neer" in text or "cny" in text:
        return "currency"
    return "data"
