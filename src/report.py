from src.policy_signal import PolicySignal
from src.scoring import IndicatorEvidence, ScoreBlock


def render_report(signal: PolicySignal) -> str:
    profile = signal.profile
    parts = [
        "CENTRAL BANK COMPASS",
        "=================================================",
        f"Economy: {profile.economy_name}",
        f"Central Bank: {profile.central_bank_name}",
        f"Policy Tool: {profile.policy_tool}",
        "",
        f"Current Signal: {signal.signal}",
        f"Policy Bias: {signal.policy_bias}",
        f"Confidence: {signal.confidence}%",
        "",
        "Model View:",
        signal.model_view,
        "",
        "=================================================",
        "DATA COVERAGE",
        "=================================================",
        *data_coverage_lines(signal),
        "",
        "=================================================",
        "BLOCK EXPLAINABILITY",
        "=================================================",
        *render_block("Inflation Pressure", signal.inflation, signed_positive=True),
        *render_block("Labor Weakness", signal.labor, signed_positive=False),
        *render_block("Growth Weakness", signal.growth, signed_positive=False),
        *render_block("Financial Stress", signal.financial, signed_positive=False),
        *render_block("Currency Pressure", signal.currency, signed_positive=True),
        "",
        "=================================================",
        "POLICY DECISION LOGIC",
        "=================================================",
        *policy_decision_lines(signal),
        "",
        "Reason:",
        interpretation(signal),
        "",
        "Market Meaning:",
        market_meaning(signal),
    ]

    if signal.warnings:
        parts.extend(["", "Data Warnings:", *[f"- {warning}" for warning in signal.warnings]])

    return "\n".join(parts)


def render_block(title: str, block: ScoreBlock, signed_positive: bool) -> list[str]:
    lines = [
        "",
        "-------------------------------------------------",
        f"{title}: {block.label}",
        "Reason:",
    ]
    for item in block.evidence:
        lines.extend(render_evidence(item))

    display_score = block.score if signed_positive else -block.score
    lines.extend(
        [
            "",
            f"{short_block_name(title)} Block Score: {format_score(display_score)}",
            f"Block Classification: {block.label}",
            "Policy Meaning:",
            block.policy_meaning,
        ]
    )
    return lines


def data_coverage_lines(signal: PolicySignal) -> list[str]:
    blocks = {
        "Inflation": signal.inflation,
        "Labor": signal.labor,
        "Growth": signal.growth,
        "Financial": signal.financial,
        "Currency": signal.currency,
    }
    total_available = sum(block.available for block in blocks.values())
    total_expected = sum(block.expected for block in blocks.values())
    lines = [f"{name}: {block.available}/{block.expected}" for name, block in blocks.items()]
    overall_pct = (total_available / total_expected * 100) if total_expected else 0
    lines.append(f"Overall: {total_available}/{total_expected} ({overall_pct:.0f}%)")
    lines.append(f"Coverage Confidence Adjustment: {signal.coverage_confidence_adjustment}")

    if signal.missing_indicators:
        lines.extend(["", "Missing Indicators:"])
        for item in signal.missing_indicators:
            lines.extend(
                [
                    f"- {item.indicator_name}",
                    f"  Block: {item.block.title()}",
                    f"  Expected Source: {item.expected_source}",
                    f"  Status: {item.reason}",
                    f"  Confidence Reduced: {'Yes' if item.confidence_reduced else 'No'}",
                ]
            )
    return lines


def render_evidence(item: IndicatorEvidence) -> list[str]:
    return [
        f"- {item.name}: {format_value(item.value, item.value_kind)}",
        "  Thresholds:",
        *[f"  {threshold}" for threshold in item.thresholds],
        f"  Classification: {item.classification}",
        f"  Score Contribution: {format_score(item.score_contribution)}",
        f"  Explanation: {item.explanation}",
    ]


def policy_decision_lines(signal: PolicySignal) -> list[str]:
    block_scores = signal.block_scores
    lines = [
        f"Inflation Score: {format_score(block_scores['inflation'])}",
        f"Labor Weakness Score: {format_score(block_scores['labor'])}",
        f"Growth Weakness Score: {format_score(block_scores['growth'])}",
        f"Financial Stress Score: {format_score(block_scores['financial'])}",
        f"Currency Pressure Score: {format_score(block_scores['currency'])}",
        "",
        f"Net Policy Score: {format_score(signal.policy_score)}",
        "",
        "Decision Rules:",
        *[f"- {rule}" for rule in signal.decision_rules],
    ]
    if signal.signal == "HOLD" and signal.policy_bias == "Hawkish" and signal.policy_score > 30:
        lines.extend(
            [
                "",
                "Realism Rule Applied:",
                "- Net score is hawkish, but an outright tightening signal requires high and re-accelerating inflation with low labor, growth, and financial-stress weakness.",
            ]
        )
    return lines


def interpretation(signal: PolicySignal) -> str:
    profile = signal.profile
    policy_language = profile.interpretation_rules[signal.signal]
    inflation = signal.inflation.label.lower()
    labor = signal.labor.label.lower()
    growth = signal.growth.label.lower()
    stress = signal.financial.label.lower()
    currency = signal.currency.label.lower()
    inflation_trend = inflation_trend_classification(signal)

    if signal.signal == "TIGHTENING":
        reason = (
            f"Inflation is {inflation}, inflation momentum is {inflation_trend.lower()}, and the labor, growth, "
            f"and financial-stress blocks are calm enough for a tightening candidate. Currency pressure is {currency}."
        )
    elif signal.signal == "EASING":
        reason = (
            f"Labor weakness is {labor}, growth weakness is {growth}, and financial stress is {stress}. Those easing "
            f"forces outweigh inflation pressure, which is {inflation}, and currency pressure, which is {currency}."
        )
    elif signal.policy_bias == "Hawkish":
        reason = (
            f"Inflation remains too high or policy-sensitive pressure is positive, but the model does not have enough "
            f"evidence of re-accelerating inflation plus calm macro conditions to justify outright tightening. Inflation "
            f"momentum is {inflation_trend.lower()}."
        )
    elif signal.policy_bias == "Dovish":
        reason = (
            f"Weakness is building, but the net policy score is not low enough for a full easing signal. Labor weakness is "
            f"{labor}, growth weakness is {growth}, and financial stress is {stress}."
        )
    else:
        reason = (
            f"The tightening and easing forces are balanced. Inflation pressure is {inflation}, labor weakness is {labor}, "
            f"growth weakness is {growth}, financial stress is {stress}, and currency pressure is {currency}."
        )

    return f"{policy_language} {reason}"


def market_meaning(signal: PolicySignal) -> str:
    if signal.signal == "TIGHTENING":
        return (
            "TIGHTENING usually pressures rate-sensitive equities and long-duration assets, can push bond yields higher, "
            "and may support the local currency. Gold and crypto often face headwinds if real-rate and liquidity expectations tighten."
        )
    if signal.signal == "EASING":
        return (
            "EASING can support bonds as yields fall and may help equities if growth risk is manageable. The local currency may soften, "
            "gold can benefit from lower real-rate expectations, and crypto may respond positively to easier liquidity unless recession stress dominates."
        )
    if signal.policy_bias == "Hawkish":
        return (
            "A hawkish HOLD can keep pressure on rate-sensitive equities while supporting front-end yields and the local currency. "
            "Gold and crypto may struggle if markets expect restrictive policy to last."
        )
    if signal.policy_bias == "Dovish":
        return (
            "A dovish HOLD can help bonds and long-duration assets, but markets may wait for stronger easing evidence before fully pricing cuts or easier policy."
        )
    return (
        "A neutral HOLD keeps markets data-dependent. Equities, bonds, currency, gold, and crypto are likely to react most to the next inflation, labor, growth, or policy communication shock."
    )


def inflation_trend_classification(signal: PolicySignal) -> str:
    for item in signal.inflation.evidence:
        if item.name.startswith("Inflation Trend"):
            return item.classification
    return "UNKNOWN"


def short_block_name(title: str) -> str:
    return title.replace(" Pressure", "").replace(" Weakness", "").replace(" Stress", "")


def format_value(value: float | None, kind: str) -> str:
    if value is None:
        return "N/A"
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
