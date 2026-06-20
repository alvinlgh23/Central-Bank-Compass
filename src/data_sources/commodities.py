from typing import Any

import pandas as pd

from src.data_sources import fred


SERIES = {
    "wti": "DCOILWTICO",
    "brent": "DCOILBRENTEU",
    "gasoline": "GASREGW",
    "energy_cpi": "CPIENGSL",
}


def get_oil_price_data(config: dict[str, Any]) -> dict[str, pd.Series]:
    return {name: fred.fetch_series(series_id, config) for name, series_id in SERIES.items()}


def get_energy_shock_indicators(config: dict[str, Any]) -> dict[str, Any]:
    data = get_oil_price_data(config)
    primary_name = "WTI" if not data["wti"].empty else "Brent"
    primary = data["wti"] if not data["wti"].empty else data["brent"]
    latest_value = latest(primary)
    yoy_change = date_change(primary, 12)
    three_month_change = date_change(primary, 3)
    six_month_change = date_change(primary, 6)
    return {
        "primary_name": primary_name,
        "latest": latest_value,
        "yoy_change": yoy_change,
        "three_month_change": three_month_change,
        "six_month_change": six_month_change,
        "classification": classify_energy(yoy_change),
        "short_term_classification": classify_short_term(three_month_change),
        "gasoline_latest": latest(data["gasoline"]),
        "energy_cpi_yoy": date_change(data["energy_cpi"], 12),
        "missing": primary.empty,
    }


def render_energy_shock_monitor(config: dict[str, Any]) -> str:
    energy = get_energy_shock_indicators(config)
    if energy["missing"]:
        return "\n".join(
            [
                "ENERGY SHOCK MONITOR",
                "=================================================",
                "WTI / Brent Oil: N/A",
                "Warning: Stable FRED commodity data was unavailable. Energy-shock confidence is reduced.",
            ]
        )
    return "\n".join(
        [
            "ENERGY SHOCK MONITOR",
            "=================================================",
            f"{energy['primary_name']} Oil:",
            f"Latest: {format_price(energy['latest'])}",
            f"YoY Change: {format_pct(energy['yoy_change'])}",
            f"3M Change: {format_pct(energy['three_month_change'])}",
            f"6M Change: {format_pct(energy['six_month_change'])}",
            f"Classification: {energy['classification']}",
            f"Short-Term Classification: {energy['short_term_classification']}",
            "",
            "Inflation Meaning:",
            inflation_meaning(energy),
            "",
            "Market Meaning:",
            "Energy shocks can raise headline inflation, reduce real household income, and increase stagflation risk. They should not be treated as proof of broad core inflation.",
        ]
    )


def render_energy_check(config: dict[str, Any]) -> str:
    energy = get_energy_shock_indicators(config)
    if energy["missing"]:
        return "Energy Shock Check:\nCommodity data unavailable; no energy attribution is applied."
    return "\n".join(
        [
            "Energy Shock Check:",
            f"{energy['primary_name']} YoY: {format_pct(energy['yoy_change'])}",
            f"3M Change: {format_pct(energy['three_month_change'])}",
            f"Classification: {energy['classification']}",
            f"Short-Term: {energy['short_term_classification']}",
        ]
    )


def us_inflation_energy_interpretation(config: dict[str, Any], core_inflation: float | None, headline_inflation: float | None) -> str:
    energy = get_energy_shock_indicators(config)
    if energy["missing"]:
        return "Inflation Interpretation:\nEnergy data is unavailable, so headline/core inflation attribution remains uncertain."
    shock = energy["classification"] == "ENERGY_SHOCK" or energy["short_term_classification"] == "SHORT_TERM_ENERGY_SHOCK"
    if not shock:
        return "Inflation Interpretation:\nOil is not currently signaling a major energy shock; core inflation remains the main policy-pressure anchor."
    if core_inflation is not None and core_inflation >= 2.5:
        return "Inflation Interpretation:\nBroad inflation pressure plus energy shock. Headline inflation may be amplified by energy prices, while elevated core inflation keeps underlying hawkish pressure intact."
    if headline_inflation is not None and headline_inflation >= 2.5:
        return "Inflation Interpretation:\nHeadline inflation is partly energy-driven. Core inflation does not confirm equally broad pressure, so the headline move receives a less hawkish interpretation."
    return "Inflation Interpretation:\nEnergy prices are rising sharply, but broad underlying inflation confirmation is limited."


def allocation_energy_context(config: dict[str, Any]) -> str:
    energy = get_energy_shock_indicators(config)
    if energy["missing"]:
        return "Energy Shock Context: Unavailable; commodity data did not load."
    if energy["classification"] == "ENERGY_SHOCK" or energy["short_term_classification"] == "SHORT_TERM_ENERGY_SHOCK":
        return "Energy Shock Context: Active. Stagflationary pressure risk is higher because energy can lift headline inflation while reducing real income. This is context, not an individual-security recommendation."
    if energy["classification"] == "ENERGY_DISINFLATION":
        return "Energy Shock Context: Energy disinflation is reducing headline inflation pressure, though core inflation still determines underlying persistence."
    return "Energy Shock Context: Neutral. Oil is not currently a dominant macro-regime driver."


def energy_context_summary(config: dict[str, Any]) -> str:
    energy = get_energy_shock_indicators(config)
    if energy["missing"]:
        return "Commodity data is unavailable, so the summary cannot attribute headline inflation to energy."
    if energy["classification"] == "ENERGY_SHOCK" or energy["short_term_classification"] == "SHORT_TERM_ENERGY_SHOCK":
        return f"{energy['primary_name']} is signaling an energy shock. Headline inflation may be amplified, but core inflation still determines underlying persistence."
    if energy["classification"] == "ENERGY_DISINFLATION":
        return f"{energy['primary_name']} is providing energy disinflation, reducing headline pressure without automatically resolving core inflation."
    return f"{energy['primary_name']} is not signaling a major energy shock, so core inflation remains the main policy-pressure anchor."


def classify_energy(yoy_change: float | None) -> str:
    if yoy_change is None:
        return "UNKNOWN"
    if yoy_change <= -15:
        return "ENERGY_DISINFLATION"
    if yoy_change >= 20:
        return "ENERGY_SHOCK"
    return "NEUTRAL"


def classify_short_term(change: float | None) -> str:
    if change is None:
        return "UNKNOWN"
    if change >= 15:
        return "SHORT_TERM_ENERGY_SHOCK"
    if change <= -15:
        return "SHORT_TERM_ENERGY_RELIEF"
    return "NEUTRAL"


def inflation_meaning(energy: dict[str, Any]) -> str:
    if energy["classification"] == "ENERGY_SHOCK":
        return "Oil is adding meaningful headline-inflation pressure. Core and services inflation must still confirm whether the shock is broad or persistent."
    if energy["classification"] == "ENERGY_DISINFLATION":
        return "Falling oil prices are reducing headline inflation pressure and supporting real income."
    return "Oil is not currently producing a large year-over-year headline inflation impulse."


def date_change(series: pd.Series, months: int) -> float | None:
    values = clean(series)
    if values.empty:
        return None
    current_date = values.index[-1]
    prior = values.loc[values.index <= current_date - pd.DateOffset(months=months)]
    if prior.empty or prior.iloc[-1] == 0:
        return None
    return float((values.iloc[-1] / prior.iloc[-1] - 1) * 100)


def latest(series: pd.Series) -> float | None:
    values = clean(series)
    return None if values.empty else float(values.iloc[-1])


def clean(series: pd.Series) -> pd.Series:
    if series is None or series.empty:
        return pd.Series(dtype="float64")
    return series.astype(float).dropna().sort_index()


def format_pct(value: float | None) -> str:
    return "N/A" if value is None else f"{value:+.1f}%"


def format_price(value: float | None) -> str:
    return "N/A" if value is None else f"${value:.2f}"
