from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EconomyProfile:
    code: str
    economy_name: str
    central_bank_name: str
    policy_tool: str
    inflation_indicators: list[str]
    labor_indicators: list[str]
    growth_indicators: list[str]
    financial_indicators: list[str]
    currency_indicators: list[str]
    scoring_weights: dict[str, float]
    interpretation_rules: dict[str, str]


PROFILES: dict[str, EconomyProfile] = {
    "US": EconomyProfile(
        code="US",
        economy_name="United States",
        central_bank_name="Federal Reserve",
        policy_tool="Federal Funds Rate",
        inflation_indicators=["Core PCE", "CPI", "Core CPI"],
        labor_indicators=["Unemployment rate", "Payrolls", "Initial claims"],
        growth_indicators=["Real GDP"],
        financial_indicators=["Treasury yields", "VIX", "Credit spreads"],
        currency_indicators=["US dollar conditions"],
        scoring_weights={"inflation": 1.0, "labor": 1.0, "growth": 0.9, "financial": 0.8, "currency": 0.4},
        interpretation_rules={
            "EASING": "For the United States, EASING means the macro data support a rate cut bias.",
            "HOLD": "For the United States, HOLD means the macro data support no change in the federal funds rate.",
            "TIGHTENING": "For the United States, TIGHTENING means the macro data support a rate hike bias.",
        },
    ),
    "SG": EconomyProfile(
        code="SG",
        economy_name="Singapore",
        central_bank_name="Monetary Authority of Singapore",
        policy_tool="Singapore Dollar Nominal Effective Exchange Rate Policy Band",
        inflation_indicators=["MAS core inflation", "CPI", "Import inflation pressure"],
        labor_indicators=["Unemployment rate"],
        growth_indicators=["GDP growth", "External demand"],
        financial_indicators=["Domestic financial conditions"],
        currency_indicators=["SGD NEER proxy", "USD/SGD"],
        scoring_weights={"inflation": 1.0, "labor": 0.7, "growth": 1.0, "financial": 0.5, "currency": 0.9},
        interpretation_rules={
            "EASING": "For Singapore, EASING means reduce slope, re-center lower, or widen the SGD NEER band dovishly.",
            "HOLD": "For Singapore, HOLD means maintain the SGD NEER policy band.",
            "TIGHTENING": "For Singapore, TIGHTENING means increase slope, re-center higher, or keep a hawkish SGD appreciation bias.",
        },
    ),
    "EZ": EconomyProfile(
        code="EZ",
        economy_name="Eurozone",
        central_bank_name="European Central Bank",
        policy_tool="ECB Policy Rates",
        inflation_indicators=["HICP inflation", "Core HICP"],
        labor_indicators=["Unemployment rate"],
        growth_indicators=["GDP growth", "PMI"],
        financial_indicators=["Sovereign spreads", "Financial stress"],
        currency_indicators=["EUR strength or weakness"],
        scoring_weights={"inflation": 1.0, "labor": 0.8, "growth": 1.0, "financial": 0.8, "currency": 0.5},
        interpretation_rules={
            "EASING": "For the Eurozone, EASING means the macro data support an ECB rate cut bias.",
            "HOLD": "For the Eurozone, HOLD means the macro data support no change in ECB policy rates.",
            "TIGHTENING": "For the Eurozone, TIGHTENING means the macro data support an ECB rate hike bias.",
        },
    ),
    "JP": EconomyProfile(
        code="JP",
        economy_name="Japan",
        central_bank_name="Bank of Japan",
        policy_tool="Policy Rate and Yield Curve / Normalization Framework",
        inflation_indicators=["CPI ex fresh food", "Inflation expectations"],
        labor_indicators=["Unemployment rate", "Wage growth"],
        growth_indicators=["GDP growth"],
        financial_indicators=["10Y JGB yield"],
        currency_indicators=["JPY weakness"],
        scoring_weights={"inflation": 1.0, "labor": 0.6, "growth": 0.8, "financial": 0.6, "currency": 1.0},
        interpretation_rules={
            "EASING": "For Japan, EASING means a rate cut or more dovish policy bias.",
            "HOLD": "For Japan, HOLD means no change in the policy stance.",
            "TIGHTENING": "For Japan, TIGHTENING means rate hike pressure or policy normalization pressure.",
        },
    ),
}


def get_profile(code: str) -> EconomyProfile:
    normalized = code.upper()
    if normalized not in PROFILES:
        valid = ", ".join(sorted(PROFILES))
        raise ValueError(f"Unknown economy '{code}'. Choose one of: {valid}.")
    return PROFILES[normalized]


def profile_config(config: dict[str, Any], code: str) -> dict[str, Any]:
    return config.get("economies", {}).get(code.upper(), {})
