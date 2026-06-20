from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.data_sources import fetch_economy_data
from src.data_sources import commodities, pbc
from src.indicators import build_indicators, latest, yoy


@dataclass(frozen=True)
class EvidenceSpec:
    name: str
    key: str
    context: str
    supportive: str
    adverse: str


@dataclass(frozen=True)
class NarrativeDefinition:
    code: str
    economy: str
    central_bank: str
    policy_channel: str
    narratives: list[str]
    reality_check: str
    evidence: list[EvidenceSpec]
    validate: list[str]
    invalidate: list[str]
    implications: dict[str, str]


@dataclass(frozen=True)
class NarrativeResult:
    definition: NarrativeDefinition
    narrative: str
    verdict: str
    confidence: int
    values: dict[str, float | None]
    interpretations: dict[str, str]
    data_gaps: list[str]


SUPPORTED_CODES = ["EZ", "SG", "CN", "KR", "UK", "AU", "CA", "CH"]


def run_narrative_stress_test(
    code: str,
    config: dict[str, Any],
    project_dir: Path,
    market_view: str | None = None,
    debug_data: bool = False,
) -> str:
    codes = SUPPORTED_CODES if code.lower() == "all" else [code.upper()]
    invalid = [item for item in codes if item not in DEFINITIONS]
    if invalid:
        raise ValueError(f"Unsupported narrative economy: {', '.join(invalid)}")
    results = [build_narrative_result(item, config, market_view, debug_data) for item in codes]
    save_results(results, project_dir / "outputs")
    return "\n\n\n".join(render_result(result) for result in results)


def build_narrative_result(
    code: str,
    config: dict[str, Any],
    market_view: str | None,
    debug_data: bool,
) -> NarrativeResult:
    definition = DEFINITIONS[code]
    narrative = closest_narrative(market_view, definition.narratives)
    values = collect_values(code, config, debug_data)
    interpretations = {
        spec.key: interpret_evidence(code, spec.key, values.get(spec.key), narrative)
        for spec in definition.evidence
    }
    available = sum(1 for spec in definition.evidence if values.get(spec.key) is not None)
    expected = len(definition.evidence)
    gaps = [spec.name for spec in definition.evidence if values.get(spec.key) is None]
    verdict = narrative_verdict(code, narrative, values, available, expected)
    confidence = narrative_confidence(available, expected, interpretations, verdict)
    return NarrativeResult(definition, narrative, verdict, confidence, values, interpretations, gaps)


def collect_values(code: str, config: dict[str, Any], debug_data: bool) -> dict[str, float | None]:
    if code in {"EZ", "SG"}:
        data, _ = fetch_economy_data(config, code, debug_data=debug_data)
        indicators = build_indicators(code, data)
        values = dict(indicators.values)
        if code == "EZ":
            energy = commodities.get_energy_shock_indicators(config)
            values.update(
                {
                    "headline_inflation": indicators.get("headline_inflation_yoy"),
                    "core_inflation": indicators.get("core_inflation_yoy"),
                    "services_inflation": None,
                    "wage_pressure": None,
                    "growth": indicators.get("growth_yoy"),
                    "pmi": indicators.get("pmi"),
                    "peripheral_spread": indicators.get("sovereign_spread"),
                    "currency": indicators.get("currency_change_yoy"),
                    "oil_yoy": energy.get("yoy_change"),
                }
            )
        else:
            values.update(
                {
                    "core_inflation": indicators.get("core_inflation_yoy"),
                    "headline_inflation": indicators.get("headline_inflation_yoy"),
                    "import_inflation": indicators.get("import_inflation_pressure"),
                    "external_demand": indicators.get("pmi"),
                    "growth": indicators.get("growth_yoy"),
                    "unemployment": indicators.get("unemployment_rate"),
                    "currency": indicators.get("usd_sgd_change_yoy"),
                    "sgd_neer": indicators.get("sgd_neer_shadow_proxy"),
                }
            )
        return values
    if code == "CN":
        return {
            "m2_growth": yoy(pbc.fetch_m2_growth(config), 12),
            "credit_impulse": latest(pbc.fetch_credit_impulse_proxy(config)),
            "policy_rate": latest(pbc.fetch_lpr(config)),
            "rrr": latest(pbc.fetch_rrr(config)),
            "property_stress": yoy(pbc.fetch_property_stress_proxy(config), 4),
            "cny_pressure": yoy(pbc.fetch_cny_pressure(config), 12),
            "pmi": None,
            "industrial_production": None,
            "retail_sales": None,
            "exports": None,
        }
    return {spec.key: None for spec in DEFINITIONS[code].evidence}


def narrative_verdict(
    code: str,
    narrative: str,
    values: dict[str, float | None],
    available: int,
    expected: int,
) -> str:
    coverage = available / expected if expected else 0
    if available < 2 or coverage < 0.25:
        return "Insufficient Data"
    checks = narrative_checks(code, narrative, values)
    usable = [check for check in checks if check is not None]
    if len(usable) < 2:
        return "Insufficient Data"
    support = sum(1 for check in usable if check) / len(usable)
    if code == "EZ" and ("cut" in narrative.lower() or "dovish" in narrative.lower() or "weak" in narrative.lower()):
        critical_confirmation = [values.get("core_inflation"), values.get("services_inflation"), values.get("wage_pressure")]
        if any(value is None for value in critical_confirmation) or (values.get("core_inflation") or 0) >= 2.5:
            return "Partially Supported" if support >= 0.4 else "Not Supported"
    if support >= 0.75 and coverage >= 0.5:
        return "Supported"
    if support >= 0.4:
        return "Partially Supported"
    return "Not Supported"


def narrative_checks(code: str, narrative: str, values: dict[str, float | None]) -> list[bool | None]:
    text = narrative.lower()
    if code == "EZ":
        easing = "cut" in text or "dovish" in text or "weak" in text
        if easing:
            return [below(values, "growth", 1), below(values, "core_inflation", 2.5), below(values, "pmi", 100), below(values, "peripheral_spread", 2), below(values, "oil_yoy", 20)]
        return [above(values, "core_inflation", 2.5), above(values, "pmi", 99), below(values, "peripheral_spread", 2), above(values, "oil_yoy", 20)]
    if code == "SG":
        return [below(values, "headline_inflation", 2.5), below(values, "import_inflation", 2), below(values, "growth", 2), below(values, "external_demand", 2)]
    if code == "CN":
        return [above(values, "m2_growth", 5), above(values, "property_stress", -5), below(values, "cny_pressure", 5), value_exists(values, "credit_impulse")]
    return []


def interpret_evidence(code: str, key: str, value: float | None, narrative: str) -> str:
    if value is None:
        return "Missing; this weakens confidence and cannot confirm the narrative."
    if code == "EZ":
        if key == "growth":
            return "Weak Eurozone growth supports a dovish narrative." if value < 1 else "Growth is not weak enough to independently validate aggressive easing."
        if key == "core_inflation":
            return "Core inflation remains sticky, constraining aggressive easing." if value >= 2.5 else "Cooling core inflation supports easier policy pressure."
        if key == "peripheral_spread":
            return "Peripheral stress is contained." if value < 2 else "Wider peripheral spreads raise fragmentation risk."
        if key == "oil_yoy":
            return "An energy shock can lift headline HICP without proving domestic services/wage persistence." if value >= 20 else "Energy is not adding a major inflation shock, so weak growth and core inflation deserve more weight."
    if code == "SG":
        if key in {"core_inflation", "headline_inflation", "import_inflation"}:
            return "Inflation pressure remains relevant for the SGD NEER stance." if value >= 2.5 else "Cooling inflation supports a more dovish MAS narrative."
        if key in {"growth", "external_demand"}:
            return "Weak activity supports MAS easing pressure." if value < 2 else "Growth or external demand does not confirm forced easing."
    if code == "CN":
        if key == "m2_growth":
            return "Liquidity growth is supportive, but transmission still needs confirmation." if value > 5 else "Money growth does not indicate a strong liquidity impulse."
        if key == "property_stress":
            return "Property stress remains a drag on recovery." if value < -5 else "Property conditions appear closer to stabilization."
        if key == "cny_pressure":
            return "CNY pressure is manageable." if value < 5 else "CNY weakness constrains aggressive easing."
    return "Available data provides context but is not sufficient alone to validate the narrative."


def closest_narrative(market_view: str | None, narratives: list[str]) -> str:
    if not market_view:
        return narratives[0]
    tokens = set(market_view.lower().replace("/", " ").split())
    return max(narratives, key=lambda item: len(tokens.intersection(item.lower().replace("/", " ").split())))


def render_result(result: NarrativeResult) -> str:
    definition = result.definition
    parts = [
        "COUNTRY NARRATIVE STRESS TEST",
        "=================================================",
        f"Economy: {definition.economy}",
        f"Central Bank: {definition.central_bank}",
        f"Main Policy Channel: {definition.policy_channel}",
        "",
        "Market Narrative Being Tested:",
        result.narrative,
        "",
        "Central Bank Reality Check:",
        definition.reality_check,
        "",
        f"Narrative Verdict: {result.verdict}",
        f"Confidence: {result.confidence}%",
        "",
        "Key Evidence:",
    ]
    for spec in definition.evidence:
        value = result.values.get(spec.key)
        parts.extend(
            [
                f"- Indicator: {spec.name}",
                f"  Current: {format_value(value)}",
                f"  Threshold / Context: {spec.context}",
                f"  Interpretation: {result.interpretations[spec.key]}",
            ]
        )
    parts.extend(["", "What Would Validate This Narrative:", *[f"- {item}" for item in definition.validate]])
    parts.extend(["", "What Would Invalidate This Narrative:", *[f"- {item}" for item in definition.invalidate]])
    parts.extend(["", "Market Implication:"])
    parts.extend(f"- {asset}: {text}" for asset, text in definition.implications.items())
    parts.extend(["", "Data Gaps:"])
    parts.extend([f"- {gap}" for gap in result.data_gaps] or ["- None in the configured evidence set."])
    parts.extend(["", "This is a narrative stress test, not a trading signal or a prediction that the central bank will act."])
    return "\n".join(parts)


def save_results(results: list[NarrativeResult], output_dir: Path) -> None:
    output_dir.mkdir(exist_ok=True)
    path = output_dir / "narrative_stress_tests.csv"
    new_rows = pd.DataFrame(
        {
            "Economy": [result.definition.code for result in results],
            "Narrative": [result.narrative for result in results],
            "Verdict": [result.verdict for result in results],
            "Confidence": [result.confidence for result in results],
            "Available Indicators": [
                sum(result.values.get(spec.key) is not None for spec in result.definition.evidence)
                for result in results
            ],
            "Expected Indicators": [len(result.definition.evidence) for result in results],
            "Data Gaps": ["; ".join(result.data_gaps) for result in results],
        }
    )
    if path.exists():
        existing = pd.read_csv(path)
        if "Economy" not in existing.columns:
            existing = pd.DataFrame(columns=new_rows.columns)
        else:
            existing = existing[existing["Economy"].isin(SUPPORTED_CODES)]
        existing = existing[~existing["Economy"].isin(new_rows["Economy"])]
        new_rows = pd.concat([existing, new_rows], ignore_index=True)
    new_rows.to_csv(path, index=False)


def narrative_confidence(available: int, expected: int, interpretations: dict[str, str], verdict: str) -> int:
    coverage = available / expected if expected else 0
    confidence = 20 + coverage * 60
    if verdict == "Insufficient Data":
        confidence = min(confidence, 35)
    return int(max(15, min(confidence, 80)))


def below(values: dict[str, float | None], key: str, threshold: float) -> bool | None:
    value = values.get(key)
    return None if value is None else value < threshold


def above(values: dict[str, float | None], key: str, threshold: float) -> bool | None:
    value = values.get(key)
    return None if value is None else value > threshold


def value_exists(values: dict[str, float | None], key: str) -> bool | None:
    return None if values.get(key) is None else True


def format_value(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.2f}"


def evidence(name: str, key: str, context: str) -> EvidenceSpec:
    return EvidenceSpec(name, key, context, "Supports narrative", "Challenges narrative")


def definition(
    code: str,
    economy: str,
    central_bank: str,
    channel: str,
    narratives: list[str],
    reality: str,
    evidence_specs: list[tuple[str, str, str]],
    validate: list[str],
    invalidate: list[str],
    implications: dict[str, str],
) -> NarrativeDefinition:
    return NarrativeDefinition(code, economy, central_bank, channel, narratives, reality, [evidence(*item) for item in evidence_specs], validate, invalidate, implications)


DEFINITIONS = {
    "EZ": definition("EZ", "Eurozone", "European Central Bank", "Policy rates, bank lending, and fragmentation control", ["ECB will cut aggressively because European growth is weak.", "ECB cannot cut because services inflation and wages are sticky.", "Germany weakness means Eurozone policy must turn dovish.", "Peripheral spreads will constrain ECB tightening."], "ECB must separate domestic core/services and wage persistence from energy-driven HICP volatility while monitoring weak growth, bank lending, peripheral spreads, and EUR pressure.", [("Core HICP", "core_inflation", "Below roughly 2.5% supports easing"), ("Services Inflation", "services_inflation", "Cooling services inflation is required for aggressive easing"), ("Wage / Negotiated Wage Pressure", "wage_pressure", "Cooling wages are required for aggressive easing"), ("GDP Growth", "growth", "Below 1% indicates weak growth"), ("PMI / Confidence Proxy", "pmi", "Below neutral supports slowdown narrative"), ("Peripheral Sovereign Spread", "peripheral_spread", "Wider spreads raise fragmentation risk"), ("EUR/USD Pressure", "currency", "Sharp EUR weakness can constrain easing"), ("Oil YoY Change", "oil_yoy", ">=20% is an energy shock; <=-15% is energy disinflation")], ["Growth and PMI remain weak.", "Core/services inflation and wage pressure cool.", "Peripheral spreads remain stable.", "Energy prices do not create a renewed headline shock."], ["Core/services inflation remains sticky.", "Wage pressure persists.", "Financial stress or EUR weakness rises.", "A headline HICP rise is confirmed by domestic inflation breadth rather than oil alone."], {"Local Equities": "Easing can support duration-sensitive sectors, but weak earnings can offset it.", "Currency": "A more dovish ECB can pressure EUR unless global risk improves.", "Bonds": "Cooling inflation and weak growth generally support sovereign bonds.", "Global Risk Sentiment": "Fragmentation stress would be negative for European and global risk appetite."}),
    "SG": definition("SG", "Singapore", "Monetary Authority of Singapore", "SGD NEER policy band", ["Fed easing means MAS will ease too.", "SGD strength means MAS is already too tight.", "Singapore growth weakness will force MAS dovish.", "Imported inflation means MAS must stay tight."], "MAS targets the SGD NEER path, so imported inflation, external demand, growth, and the exchange-rate stance matter more than Fed policy alone.", [("MAS Core Inflation", "core_inflation", "Cooling core inflation supports easing"), ("Headline CPI", "headline_inflation", "Below roughly 2.5% reduces inflation pressure"), ("Import Inflation", "import_inflation", "Cooling import prices support easing"), ("External Demand", "external_demand", "Weak external demand supports easing"), ("GDP Growth", "growth", "Sub-2% growth adds easing pressure"), ("Unemployment", "unemployment", "Rising unemployment supports easing"), ("USD/SGD", "currency", "SGD weakness can constrain easing"), ("SGD NEER Shadow Proxy", "sgd_neer", "Restrictive SGD conditions can support easing")], ["Core and import inflation cool.", "External demand and GDP weaken.", "SGD NEER conditions are already restrictive."], ["Imported inflation remains firm.", "External demand improves.", "SGD weakness raises inflation risk."], {"Local Equities": "A dovish MAS stance can help domestic cyclicals if growth is not collapsing.", "Currency": "MAS easing would reduce SGD appreciation pressure.", "Bonds": "Lower inflation pressure can support Singapore government bonds and rates.", "Regional Risk Sentiment": "SGD policy is a useful regional inflation and FX signal.", "Singapore REITs": "Easier financial conditions can help rate-sensitive property assets conceptually."}),
    "CN": definition("CN", "China", "People's Bank of China", "Liquidity transmission, credit policy, FX stability, and targeted support", ["PBOC easing means China growth recovery is back.", "LPR cuts mean a major stimulus cycle has started.", "Property weakness means Beijing must launch bazooka stimulus.", "CNY weakness prevents meaningful easing."], "China recovery depends on credit transmission, property stabilization, real activity, and manageable CNY pressure, not policy-rate cuts alone.", [("M2 Growth", "m2_growth", "Above 5% is a positive liquidity signal"), ("Credit Impulse", "credit_impulse", "A positive turn is needed for stronger recovery confirmation"), ("Policy Rate / LPR Proxy", "policy_rate", "Cuts indicate easing intent, not transmission"), ("RRR", "rrr", "Cuts can release bank liquidity"), ("Property Stress", "property_stress", "Stabilization above severe contraction supports recovery"), ("CNY Pressure", "cny_pressure", "Large depreciation pressure constrains easing"), ("PMI", "pmi", "Improvement above neutral confirms activity"), ("Industrial Production", "industrial_production", "Acceleration confirms real-economy transmission"), ("Retail Sales", "retail_sales", "Consumer improvement broadens recovery"), ("Exports", "exports", "External demand can support the cycle")], ["Credit impulse turns positive.", "Property stress stabilizes.", "PMI, production, and retail activity improve.", "CNY pressure remains manageable."], ["Liquidity fails to reach private demand.", "Property stress worsens.", "CNY pressure intensifies.", "Real activity remains weak."], {"Local Equities": "Recovery requires earnings and credit transmission, not just policy announcements.", "Currency": "CNY weakness can constrain the scale of easing.", "Bonds": "Persistent weak demand can support government bonds despite easing measures.", "Commodities": "A real recovery would need construction and industrial confirmation.", "Global Risk Sentiment": "China transmission matters for Asia, Korea, Australia, and global cyclicals."}),
}


def generic_definition(code: str, economy: str, bank: str, channel: str, narratives: list[str], reality: str, indicators: list[str], implications: dict[str, str]) -> NarrativeDefinition:
    specs = [(name, name.lower().replace(" / ", "_").replace(" ", "_"), "Country-specific source integration required") for name in indicators]
    return definition(code, economy, bank, channel, narratives, reality, specs, ["Required domestic indicators confirm the narrative consistently."], ["Key inflation, labor, growth, or currency indicators contradict the narrative."], implications)


DEFINITIONS.update(
    {
        "KR": generic_definition("KR", "South Korea", "Bank of Korea", "Policy rate, household debt, and KRW stability", ["Strong semiconductor exports mean Korea is fine.", "Fed cuts mean BOK will cut too.", "KRW weakness prevents BOK easing.", "Household debt keeps BOK cautious."], "Korea combines semiconductor/export sensitivity with household debt and FX constraints.", ["Core CPI", "Headline CPI", "Exports", "Semiconductor Exports", "China Demand Proxy", "KRW/USD", "Household Debt", "Unemployment", "GDP Growth"], {"Local Equities": "Export strength helps KOSPI, but domestic debt and FX risks still matter.", "Currency": "KRW pressure can constrain easing.", "Bonds": "Cooling inflation and domestic weakness would support bonds.", "Global Risk Sentiment": "Korea is a useful Asia technology-cycle signal."}),
        "UK": generic_definition("UK", "United Kingdom", "Bank of England", "Bank Rate, services inflation, wages, and mortgage transmission", ["Headline CPI is falling, so BoE can cut quickly.", "Services inflation means BoE must stay hawkish.", "Wage growth keeps UK inflation sticky.", "Mortgage stress will force easing."], "BoE policy is unusually sensitive to services inflation, wages, labor slack, and mortgage resets.", ["Headline CPI", "Core CPI", "Services CPI", "Wage Growth", "Unemployment", "Vacancies", "GDP Growth", "Retail Sales", "Mortgage Stress", "GBP/USD", "Gilt Yields"], {"Local Equities": "FTSE effects differ between domestic demand and global exporters.", "Currency": "Sticky services inflation can support GBP.", "Bonds": "Cooling wages and services inflation would support gilts.", "Global Risk Sentiment": "UK mortgage stress can signal developed-market consumer pressure."}),
        "AU": generic_definition("AU", "Australia", "Reserve Bank of Australia", "Cash rate, housing transmission, AUD, and China/commodity exposure", ["China stimulus means Australia will benefit.", "Housing and rents keep RBA hawkish.", "CPI cooling means RBA can cut.", "AUD weakness imports inflation."], "Australia combines services/rent inflation, mortgages, labor resilience, AUD pressure, and China/commodity demand.", ["CPI", "Trimmed Mean CPI", "Services Inflation", "Wage Growth", "Unemployment", "Housing Inflation", "Mortgage Stress", "AUD/USD", "China Demand", "Commodity Proxy"], {"Local Equities": "ASX sensitivity spans banks, housing, and commodities.", "Currency": "AUD needs China and commodity confirmation.", "Bonds": "Broad inflation cooling and labor softness support bonds.", "Global Risk Sentiment": "Australia transmits China and commodity-cycle signals."}),
        "CA": generic_definition("CA", "Canada", "Bank of Canada", "Overnight rate, housing/mortgages, CAD, and US spillovers", ["Canada cannot cut before the Fed.", "Mortgage renewal shock forces BoC dovish.", "Oil price strength supports CAD and limits easing.", "Housing stress will dominate policy."], "Canada is highly exposed to household debt, mortgage resets, housing, oil, CAD, and US spillovers.", ["Core CPI", "Headline CPI", "Unemployment", "GDP Growth", "Housing Prices", "Housing Sales", "Mortgage Stress", "Oil Price", "CAD/USD", "Fed-BoC Differential"], {"Local Equities": "TSX mixes financial, housing, and energy sensitivity.", "Currency": "Oil and rate differentials influence CAD.", "Bonds": "Domestic stress can support Canadian bonds even if Fed policy differs.", "Global Risk Sentiment": "Canada is a useful housing and household-leverage signal."}),
        "CH": generic_definition("CH", "Switzerland", "Swiss National Bank", "Policy rate and CHF management", ["SNB will ease because inflation is low.", "CHF strength allows SNB to cut.", "Safe-haven flows complicate policy.", "Europe stress drives CHF and SNB reaction."], "Switzerland is a low-inflation, FX-sensitive economy affected by safe-haven flows and European stress.", ["CPI", "Core Inflation", "CHF Strength", "EUR/CHF", "GDP Growth", "Unemployment", "Europe Stress", "Global Risk Stress"], {"Local Equities": "Swiss exporters are sensitive to CHF strength.", "Currency": "Safe-haven flows can override domestic policy signals.", "Bonds": "Low inflation supports bonds, subject to global rate spillovers.", "Global Risk Sentiment": "CHF strength is often a global stress signal."}),
    }
)
