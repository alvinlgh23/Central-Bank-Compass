from dataclasses import dataclass

from src.country_profiles import EconomyProfile
from src.indicators import IndicatorSet


@dataclass(frozen=True)
class IndicatorEvidence:
    name: str
    value: float | None
    value_kind: str
    thresholds: list[str]
    classification: str
    score_contribution: float
    explanation: str


@dataclass(frozen=True)
class ScoreBlock:
    label: str
    score: float
    available: int
    expected: int
    notes: list[str]
    evidence: list[IndicatorEvidence]
    policy_meaning: str


def score_macro_blocks(indicators: IndicatorSet, profile: EconomyProfile) -> dict[str, ScoreBlock]:
    return {
        "inflation": score_inflation(indicators, profile),
        "labor": score_labor_weakness(indicators),
        "growth": score_growth_weakness(indicators),
        "financial": score_financial_stress(indicators),
        "currency": score_currency_pressure(indicators, profile),
    }


def score_inflation(indicators: IndicatorSet, profile: EconomyProfile) -> ScoreBlock:
    evidence: list[IndicatorEvidence] = []
    expected = 3

    primary_name = primary_inflation_name(profile)
    primary = first_available(indicators, ["core_inflation_yoy", "headline_inflation_yoy"])
    primary_score, primary_class = classify_inflation(primary, profile)
    evidence.append(
        IndicatorEvidence(
            name=primary_name,
            value=primary,
            value_kind="pct",
            thresholds=inflation_thresholds(profile),
            classification=primary_class,
            score_contribution=primary_score,
            explanation=inflation_explanation(primary_class, profile),
        )
    )

    headline = indicators.get("headline_inflation_yoy")
    headline_score, headline_class = classify_headline_inflation(headline, profile)
    evidence.append(
        IndicatorEvidence(
            name="Headline Inflation YoY",
            value=headline,
            value_kind="pct",
            thresholds=headline_thresholds(profile),
            classification=headline_class,
            score_contribution=headline_score,
            explanation=headline_inflation_explanation(headline_class),
        )
    )

    trend = first_available(indicators, ["core_inflation_trend", "headline_inflation_trend"])
    trend_score, trend_class = classify_inflation_trend(trend)
    evidence.append(
        IndicatorEvidence(
            name="Inflation Trend, 3M Change in YoY Rate",
            value=trend,
            value_kind="pp",
            thresholds=["<=-0.20 pp = COOLING", "-0.20 to +0.20 pp = STABLE", ">=+0.20 pp = ACCELERATING"],
            classification=trend_class,
            score_contribution=trend_score,
            explanation=inflation_trend_explanation(trend_class),
        )
    )

    available = count_available(evidence)
    score = sum(item.score_contribution for item in evidence)
    label = label_from_pressure(max(score, 0))
    return ScoreBlock(label, max(score, 0), available, expected, unavailable_notes(evidence), evidence, inflation_policy_meaning(label))


def score_labor_weakness(indicators: IndicatorSet) -> ScoreBlock:
    evidence: list[IndicatorEvidence] = []
    expected = 2

    gap = indicators.get("unemployment_gap")
    gap_score, gap_class = classify_unemployment_gap(gap)
    evidence.append(
        IndicatorEvidence(
            name="Unemployment Gap from 12M Low",
            value=gap,
            value_kind="pp",
            thresholds=["<0.3 pp = LOW", "0.3-0.5 pp = MODERATE", ">=0.5 pp = HIGH"],
            classification=gap_class,
            score_contribution=gap_score,
            explanation=labor_gap_explanation(gap_class),
        )
    )

    claims = indicators.get("claims_yoy")
    payrolls_3m = indicators.get("payrolls_3m_avg_change")
    payrolls_12m = indicators.get("payrolls_12m_avg_change")
    if claims is not None:
        claims_score, claims_class = classify_claims(claims)
        evidence.append(
            IndicatorEvidence(
                name="Initial Claims YoY",
                value=claims,
                value_kind="pct",
                thresholds=["<=0% = LOW", "0-15% = MODERATE", ">15% = HIGH"],
                classification=claims_class,
                score_contribution=claims_score,
                explanation=claims_explanation(claims_class),
            )
        )
    else:
        payroll_score, payroll_class = classify_payroll_momentum(payrolls_3m, payrolls_12m)
        evidence.append(
            IndicatorEvidence(
                name="Payroll Momentum, 3M Avg vs 12M Avg",
                value=payroll_difference(payrolls_3m, payrolls_12m),
                value_kind="number",
                thresholds=[">=0 = LOW", "-50k to 0 = MODERATE", "<-50k = HIGH"],
                classification=payroll_class,
                score_contribution=payroll_score,
                explanation=payroll_explanation(payroll_class),
            )
        )

    available = count_available(evidence)
    score = sum(item.score_contribution for item in evidence)
    label = label_from_weakness(score)
    return ScoreBlock(label, score, available, expected, unavailable_notes(evidence), evidence, weakness_policy_meaning("Labor", label))


def score_growth_weakness(indicators: IndicatorSet) -> ScoreBlock:
    evidence: list[IndicatorEvidence] = []
    expected = 2

    growth = indicators.get("growth_yoy")
    growth_score, growth_class = classify_growth(growth)
    evidence.append(
        IndicatorEvidence(
            name="GDP Growth YoY",
            value=growth,
            value_kind="pct",
            thresholds=["<1.0% = HIGH weakness", "1.0-2.0% = MODERATE weakness", ">2.0% = LOW weakness"],
            classification=growth_class,
            score_contribution=growth_score,
            explanation=growth_explanation(growth_class),
        )
    )

    pmi = indicators.get("pmi")
    is_external_growth_rate = indicators.get("external_demand_is_growth_rate") == 1.0
    pmi_score, pmi_class = classify_external_growth(pmi) if is_external_growth_rate else classify_pmi(pmi)
    pmi_thresholds = (
        ["<0.0% = HIGH weakness", "0.0-2.0% = MODERATE weakness", ">2.0% = LOW weakness"]
        if is_external_growth_rate
        else ["<48 = HIGH weakness", "48-50 = MODERATE weakness", ">=50 = LOW weakness"]
    )
    pmi_value_kind = "pct" if is_external_growth_rate else "number"
    pmi_explanation_text = (
        external_growth_explanation(pmi_class) if is_external_growth_rate else pmi_explanation(pmi_class)
    )
    evidence.append(
        IndicatorEvidence(
            name="PMI / External Demand Proxy",
            value=pmi,
            value_kind=pmi_value_kind,
            thresholds=pmi_thresholds,
            classification=pmi_class,
            score_contribution=pmi_score,
            explanation=pmi_explanation_text,
        )
    )

    available = count_available(evidence)
    score = sum(item.score_contribution for item in evidence)
    label = label_from_weakness(score)
    return ScoreBlock(label, score, available, expected, unavailable_notes(evidence), evidence, weakness_policy_meaning("Growth", label))


def score_financial_stress(indicators: IndicatorSet) -> ScoreBlock:
    evidence: list[IndicatorEvidence] = []
    expected = 2

    vix = indicators.get("vix")
    stress = indicators.get("financial_stress")
    stress_value = vix if vix is not None else stress
    stress_name = "VIX" if vix is not None else "Financial Stress Proxy"
    stress_score, stress_class = classify_financial_stress(stress_value)
    evidence.append(
        IndicatorEvidence(
            name=stress_name,
            value=stress_value,
            value_kind="number",
            thresholds=["<18 = LOW", "18-25 = MODERATE", ">25 = HIGH"],
            classification=stress_class,
            score_contribution=stress_score,
            explanation=financial_stress_explanation(stress_class),
        )
    )

    spread_change = indicators.get("credit_spread_change")
    if spread_change is not None:
        spread_score, spread_class = classify_spread_change(spread_change)
        evidence.append(
            IndicatorEvidence(
                name="Credit Spread Change, 6M",
                value=spread_change,
                value_kind="pp",
                thresholds=["<=0.0 pp = LOW", "0.0-0.25 pp = MODERATE", ">0.25 pp = HIGH"],
                classification=spread_class,
                score_contribution=spread_score,
                explanation=spread_explanation(spread_class),
            )
        )
    else:
        yield_value = indicators.get("ten_year_yield")
        yield_score, yield_class = classify_long_yield(yield_value)
        evidence.append(
            IndicatorEvidence(
                name="Long-Term Yield Stress Proxy",
                value=yield_value,
                value_kind="pct",
                thresholds=["Available = LOW stress proxy", "Unavailable = UNKNOWN"],
                classification=yield_class,
                score_contribution=yield_score,
                explanation=yield_explanation(yield_class),
            )
        )

    available = count_available(evidence)
    score = sum(item.score_contribution for item in evidence)
    label = label_from_weakness(score)
    return ScoreBlock(label, score, available, expected, unavailable_notes(evidence), evidence, weakness_policy_meaning("Financial stress", label))


def score_currency_pressure(indicators: IndicatorSet, profile: EconomyProfile) -> ScoreBlock:
    change = indicators.get("currency_change_yoy")
    score, classification = classify_currency(change, profile)
    evidence = [
        IndicatorEvidence(
            name=currency_name(profile),
            value=change,
            value_kind="pct",
            thresholds=currency_thresholds(profile),
            classification=classification,
            score_contribution=score,
            explanation=currency_explanation(classification, profile),
        )
    ]
    label = "LOW" if score < 0 else label_from_pressure(score)
    return ScoreBlock(label, score, count_available(evidence), 1, unavailable_notes(evidence), evidence, currency_policy_meaning(label, profile))


def classify_inflation(value: float | None, profile: EconomyProfile) -> tuple[float, str]:
    if value is None:
        return 0, "UNKNOWN"
    if profile.code == "JP":
        if value >= 2.5:
            return 40, "HIGH"
        if value >= 1.8:
            return 25, "MODERATE"
        return 8, "LOW"
    if value > 3.0:
        return 40, "HIGH"
    if value >= 2.3:
        return 25, "MODERATE"
    return 8, "LOW"


def classify_headline_inflation(value: float | None, profile: EconomyProfile) -> tuple[float, str]:
    if value is None:
        return 0, "UNKNOWN"
    if profile.code == "JP":
        if value >= 2.5:
            return 20, "HIGH"
        if value >= 1.8:
            return 12, "MODERATE"
        return 3, "LOW"
    if value > 3.5:
        return 25, "HIGH"
    if value >= 2.5:
        return 14, "MODERATE"
    return 3, "LOW"


def classify_inflation_trend(value: float | None) -> tuple[float, str]:
    if value is None:
        return 0, "UNKNOWN"
    if value >= 0.2:
        return 10, "ACCELERATING"
    if value <= -0.2:
        return -8, "COOLING"
    return 0, "STABLE"


def classify_unemployment_gap(value: float | None) -> tuple[float, str]:
    if value is None:
        return 0, "UNKNOWN"
    if value >= 0.5:
        return 34, "HIGH"
    if value >= 0.3:
        return 20, "MODERATE"
    return 5, "LOW"


def classify_claims(value: float | None) -> tuple[float, str]:
    if value is None:
        return 0, "UNKNOWN"
    if value > 15:
        return 14, "HIGH"
    if value > 0:
        return 8, "MODERATE"
    return 0, "LOW"


def classify_payroll_momentum(recent: float | None, trend: float | None) -> tuple[float, str]:
    difference = payroll_difference(recent, trend)
    if difference is None:
        return 0, "UNKNOWN"
    if difference < -50:
        return 14, "HIGH"
    if difference < 0:
        return 8, "MODERATE"
    return 0, "LOW"


def classify_growth(value: float | None) -> tuple[float, str]:
    if value is None:
        return 0, "UNKNOWN"
    if value < 1.0:
        return 34, "HIGH"
    if value <= 2.0:
        return 20, "MODERATE"
    return 5, "LOW"


def classify_pmi(value: float | None) -> tuple[float, str]:
    if value is None:
        return 0, "UNKNOWN"
    if value < 48:
        return 15, "HIGH"
    if value < 50:
        return 8, "MODERATE"
    return 0, "LOW"


def classify_external_growth(value: float | None) -> tuple[float, str]:
    if value is None:
        return 0, "UNKNOWN"
    if value < 0:
        return 15, "HIGH"
    if value <= 2:
        return 8, "MODERATE"
    return 0, "LOW"


def classify_financial_stress(value: float | None) -> tuple[float, str]:
    if value is None:
        return 0, "UNKNOWN"
    if value > 25:
        return 25, "HIGH"
    if value >= 18:
        return 12, "MODERATE"
    return 5, "LOW"


def classify_spread_change(value: float | None) -> tuple[float, str]:
    if value is None:
        return 0, "UNKNOWN"
    if value > 0.25:
        return 10, "HIGH"
    if value > 0:
        return 5, "MODERATE"
    return 0, "LOW"


def classify_long_yield(value: float | None) -> tuple[float, str]:
    if value is None:
        return 0, "UNKNOWN"
    return 0, "LOW"


def classify_currency(value: float | None, profile: EconomyProfile) -> tuple[float, str]:
    if value is None:
        return 0, "UNKNOWN"
    if profile.code in {"SG", "JP"}:
        if value >= 8:
            return 24, "HIGH"
        if value >= 3:
            return 12, "MODERATE"
        if value <= -5:
            return -8, "LOW"
        return 3, "LOW"
    if profile.code == "EZ":
        if value <= -8:
            return 16, "HIGH"
        if value >= 8:
            return -6, "LOW"
        return 2, "LOW"
    if profile.code == "US":
        if value >= 8:
            return 8, "MODERATE"
        if value <= -8:
            return -4, "LOW"
        return 0, "LOW"
    return 0, "LOW"


def inflation_is_accelerating(block: ScoreBlock) -> bool:
    return any(item.name.startswith("Inflation Trend") and item.classification == "ACCELERATING" for item in block.evidence)


def primary_inflation_name(profile: EconomyProfile) -> str:
    if profile.code == "US":
        return "Core PCE YoY"
    if profile.code == "EZ":
        return "HICP Inflation YoY"
    if profile.code == "JP":
        return "CPI ex Fresh Food YoY"
    if profile.code == "SG":
        return "MAS Core / Primary Inflation YoY"
    return "Primary Inflation YoY"


def inflation_thresholds(profile: EconomyProfile) -> list[str]:
    if profile.code == "JP":
        return ["<1.8% = LOW", "1.8-2.5% = MODERATE", ">=2.5% = HIGH"]
    return ["<2.3% = LOW", "2.3-3.0% = MODERATE", ">3.0% = HIGH"]


def headline_thresholds(profile: EconomyProfile) -> list[str]:
    if profile.code == "JP":
        return ["<1.8% = LOW", "1.8-2.5% = MODERATE", ">=2.5% = HIGH"]
    return ["<2.5% = LOW", "2.5-3.5% = MODERATE", ">3.5% = HIGH"]


def currency_thresholds(profile: EconomyProfile) -> list[str]:
    if profile.code in {"SG", "JP"}:
        return ["<=-5% = LOW/import disinflation", "3-8% = MODERATE weakness", ">=8% = HIGH weakness"]
    if profile.code == "EZ":
        return ["EUR +8% or more = LOW", "EUR within +/-8% = LOW/MODERATE", "EUR -8% or more = HIGH imported inflation pressure"]
    if profile.code == "US":
        return ["Broad USD <=-8% = LOW", "Broad USD within +/-8% = LOW", "Broad USD >=+8% = MODERATE tightening pressure"]
    return ["Currency channel currently has zero direct score weight for this profile."]


def currency_name(profile: EconomyProfile) -> str:
    if profile.code == "SG":
        return "USD/SGD YoY Change"
    if profile.code == "JP":
        return "USD/JPY YoY Change"
    if profile.code == "EZ":
        return "EUR/USD YoY Change"
    if profile.code == "US":
        return "Broad Dollar Index YoY Change"
    return "Currency Pressure Proxy"


def first_available(indicators: IndicatorSet, names: list[str]) -> float | None:
    for name in names:
        value = indicators.get(name)
        if value is not None:
            return value
    return None


def count_available(evidence: list[IndicatorEvidence]) -> int:
    return sum(1 for item in evidence if item.value is not None)


def unavailable_notes(evidence: list[IndicatorEvidence]) -> list[str]:
    return [f"{item.name} is unavailable." for item in evidence if item.value is None]


def payroll_difference(recent: float | None, trend: float | None) -> float | None:
    if recent is None or trend is None:
        return None
    return recent - trend


def label_from_pressure(score: float) -> str:
    if score >= 35:
        return "HIGH"
    if score >= 18:
        return "MODERATE"
    return "LOW"


def label_from_weakness(score: float) -> str:
    if score >= 30:
        return "HIGH"
    if score >= 18:
        return "MODERATE"
    return "LOW"


def inflation_explanation(classification: str, profile: EconomyProfile) -> str:
    if classification == "HIGH":
        return f"{primary_inflation_name(profile)} is above the high threshold, so it adds tightening pressure."
    if classification == "MODERATE":
        return f"{primary_inflation_name(profile)} is above target-consistent levels, so it keeps some hawkish pressure in the model."
    if classification == "LOW":
        return f"{primary_inflation_name(profile)} is below the moderate threshold, so it adds little tightening pressure."
    return "The primary inflation series is missing, so it contributes no score and lowers confidence."


def headline_inflation_explanation(classification: str) -> str:
    if classification == "HIGH":
        return "Headline inflation is above the high threshold, reinforcing tightening pressure."
    if classification == "MODERATE":
        return "Headline inflation is moderately elevated, adding some tightening pressure."
    if classification == "LOW":
        return "Headline inflation is below the moderate threshold, limiting broad price-pressure concerns."
    return "Headline inflation is missing, so it contributes no score and lowers confidence."


def inflation_trend_explanation(classification: str) -> str:
    if classification == "ACCELERATING":
        return "The year-over-year inflation rate has risen over the last three months, supporting a tightening candidate only if other blocks are calm."
    if classification == "COOLING":
        return "The year-over-year inflation rate has fallen over the last three months, reducing the case for further tightening."
    if classification == "STABLE":
        return "The year-over-year inflation rate is broadly stable, which argues for restrictive patience rather than an automatic hike."
    return "Inflation momentum is missing, so the model cannot confirm re-acceleration."


def labor_gap_explanation(classification: str) -> str:
    if classification == "HIGH":
        return "Unemployment has risen materially from its recent low, adding easing pressure."
    if classification == "MODERATE":
        return "Unemployment has risen enough to show some labor-market softening."
    if classification == "LOW":
        return "Unemployment is close to its recent low, so labor weakness adds little easing pressure."
    return "Unemployment trend data is missing, so this labor indicator contributes no score."


def claims_explanation(classification: str) -> str:
    if classification == "HIGH":
        return "Claims are sharply higher than a year ago, reinforcing labor weakness."
    if classification == "MODERATE":
        return "Claims are higher than a year ago, adding some labor weakness."
    if classification == "LOW":
        return "Claims are not higher than a year ago, so they do not add easing pressure."
    return "Claims data is missing, so payroll momentum is used when available."


def payroll_explanation(classification: str) -> str:
    if classification == "HIGH":
        return "Recent payroll momentum is much weaker than the 12-month trend, adding easing pressure."
    if classification == "MODERATE":
        return "Recent payroll momentum is softer than the 12-month trend."
    if classification == "LOW":
        return "Recent payroll momentum is not weaker than trend, limiting labor weakness."
    return "Payroll momentum data is missing, so this indicator contributes no score."


def growth_explanation(classification: str) -> str:
    if classification == "HIGH":
        return "Growth is below the weak-growth threshold, adding clear easing pressure."
    if classification == "MODERATE":
        return "Growth is modest, adding some easing pressure."
    if classification == "LOW":
        return "Growth is above the weak-growth threshold, so it adds little easing pressure."
    return "GDP growth data is missing, so this indicator contributes no score."


def pmi_explanation(classification: str) -> str:
    if classification == "HIGH":
        return "The PMI or demand proxy is contractionary, reinforcing growth weakness."
    if classification == "MODERATE":
        return "The PMI or demand proxy is slightly below neutral."
    if classification == "LOW":
        return "The PMI or demand proxy is neutral or expansionary."
    return "PMI or external demand data is missing, so this indicator contributes no score."


def external_growth_explanation(classification: str) -> str:
    if classification == "HIGH":
        return "External demand is contracting, reinforcing growth weakness."
    if classification == "MODERATE":
        return "External demand growth is positive but soft."
    if classification == "LOW":
        return "External demand growth is positive enough to limit growth-weakness pressure."
    return "External demand data is missing, so this indicator contributes no score."


def financial_stress_explanation(classification: str) -> str:
    if classification == "HIGH":
        return "Market stress is above the high threshold, adding easing pressure."
    if classification == "MODERATE":
        return "Market stress is elevated enough to add some easing pressure."
    if classification == "LOW":
        return "Market stress is below the moderate threshold, so it adds little easing pressure."
    return "Financial stress data is missing, so this indicator contributes no score."


def spread_explanation(classification: str) -> str:
    if classification == "HIGH":
        return "Credit spreads have widened meaningfully, reinforcing financial stress."
    if classification == "MODERATE":
        return "Credit spreads have widened slightly."
    if classification == "LOW":
        return "Credit spreads have not widened, limiting financial-stress pressure."
    return "Credit spread data is missing, so this indicator contributes no score."


def yield_explanation(classification: str) -> str:
    if classification == "LOW":
        return "A long-term yield is available, but without a stress threshold it does not add easing pressure."
    return "Long-term yield data is missing, so this proxy contributes no score."


def currency_explanation(classification: str, profile: EconomyProfile) -> str:
    if classification == "HIGH":
        return f"{currency_name(profile)} is beyond the high-pressure threshold, adding imported-inflation or tightening pressure."
    if classification == "MODERATE":
        return f"{currency_name(profile)} shows moderate currency pressure."
    if classification == "LOW":
        return f"{currency_name(profile)} is not adding meaningful tightening pressure."
    return "Currency data is missing, so this channel contributes no score and lowers confidence."


def inflation_policy_meaning(label: str) -> str:
    if label == "HIGH":
        return "Inflation data supports a hawkish stance."
    if label == "MODERATE":
        return "Inflation data argues against easy policy but does not alone force tightening."
    return "Inflation data does not create a strong tightening case."


def weakness_policy_meaning(name: str, label: str) -> str:
    if label == "HIGH":
        return f"{name} data supports easier policy."
    if label == "MODERATE":
        return f"{name} data adds some easing pressure."
    return f"{name} data does not create a strong easing case."


def currency_policy_meaning(label: str, profile: EconomyProfile) -> str:
    if profile.code == "SG" and label in {"HIGH", "MODERATE"}:
        return "Currency pressure supports a more hawkish SGD NEER stance."
    if label == "HIGH":
        return "Currency pressure supports a hawkish stance through imported-inflation risk."
    if label == "MODERATE":
        return "Currency pressure adds some tightening pressure through imported-inflation risk."
    return "Currency pressure does not create a strong tightening case."
