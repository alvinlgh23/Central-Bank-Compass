import os
from typing import Any

import pandas as pd

from src.data_sources import boj, ecb, fred, fx, mas, oecd, pbc, singstat


PLACEHOLDER_FETCHERS = {
    "ecb": {
        "core_hicp": ecb.fetch_core_hicp,
        "sovereign_spreads": ecb.fetch_sovereign_spread_proxy,
    },
    "boj": {
        "wage_growth": boj.fetch_wage_growth,
        "inflation_expectations": boj.fetch_inflation_expectations,
    },
    "mas": {
        "mas_core_inflation": mas.fetch_core_inflation,
        "sgd_neer": mas.fetch_sgd_neer_position,
        "sgd_neer_shadow_proxy": mas.fetch_sgd_neer_shadow_proxy,
        "import_inflation_pressure": mas.fetch_import_inflation_proxy,
    },
    "singstat": {
        "external_demand": singstat.fetch_external_demand_proxy,
        "singapore_cpi": singstat.fetch_cpi,
        "singapore_gdp": singstat.fetch_gdp_growth,
        "singapore_unemployment": singstat.fetch_unemployment,
    },
    "pbc": {
        "m2_growth": pbc.fetch_m2_growth,
        "credit_impulse": pbc.fetch_credit_impulse_proxy,
        "lpr": pbc.fetch_lpr,
        "rrr": pbc.fetch_rrr,
        "cny_pressure": pbc.fetch_cny_pressure,
        "property_stress": pbc.fetch_property_stress_proxy,
    },
    "oecd": {
        "gdp": oecd.fetch_gdp,
        "cpi": oecd.fetch_cpi,
        "unemployment": oecd.fetch_unemployment,
        "industrial_production": oecd.fetch_industrial_production,
        "business_confidence": oecd.fetch_business_confidence,
    },
}


def fetch_economy_data(
    config: dict[str, Any],
    economy_code: str,
    debug_data: bool = False,
) -> tuple[dict[str, pd.Series], list[str]]:
    economy_config = config.get("economies", {}).get(economy_code, {})
    indicator_map = economy_config.get("indicator_map", economy_config.get("fred_series", {}))
    warnings: list[str] = []
    data: dict[str, pd.Series] = {}

    if not indicator_map:
        warnings.append(f"No data series are configured for {economy_code}.")
        return data, warnings

    if uses_fred(indicator_map) and not os.getenv("FRED_API_KEY"):
        warnings.append("No FRED data was loaded. Set FRED_API_KEY in a .env file to enable live FRED data.")

    for indicator_name, spec in indicator_map.items():
        series, warning = fetch_indicator_series(indicator_name, spec, config, economy_code)
        if warning and should_show_fetch_warning(warning, debug_data):
            warnings.append(warning)
        data[indicator_name] = series

    return data, warnings


def fetch_indicator_series(
    indicator_name: str,
    spec: Any,
    config: dict[str, Any],
    economy_code: str,
) -> tuple[pd.Series, str | None]:
    if isinstance(spec, str):
        series = fred.fetch_series(spec, config)
        warning = fred_warning(indicator_name, spec, series)
        return series, warning

    source = spec.get("source", "fred")
    series_id = spec.get("series_id")
    status = spec.get("status", "active")
    expected_source = spec.get("expected_source", source.upper())

    if source == "fred":
        if not series_id:
            return empty(indicator_name), unavailable_warning(indicator_name, expected_source, status)
        series = fred.fetch_series(series_id, config)
        return series, fred_warning(indicator_name, series_id, series)

    if source == "fx":
        series = fx.fetch_fx_series(series_id or indicator_name, config)
        if series.empty:
            return series, unavailable_warning(indicator_name, "FRED / FX module", status)
        fred_series_id = fx.fred_series_id(series_id or indicator_name)
        return series, fred.cache_warning(fred_series_id) if fred_series_id else None

    if source in PLACEHOLDER_FETCHERS:
        series = fetch_placeholder_source(source, series_id or indicator_name, economy_code, config)
        if series.empty or status == "TODO":
            return series, unavailable_warning(indicator_name, expected_source, status)
        return series, None

    return empty(indicator_name), unavailable_warning(indicator_name, expected_source, status)


def fetch_placeholder_source(source: str, key: str, economy_code: str, config: dict[str, Any]) -> pd.Series:
    fetchers = PLACEHOLDER_FETCHERS[source]
    fetcher = fetchers.get(key)
    if fetcher is None:
        return empty(key)
    if source == "oecd":
        return fetcher(economy_code)
    return fetcher(config)


def configured_placeholders(config: dict[str, Any], economy_code: str) -> list[str]:
    economy_config = config.get("economies", {}).get(economy_code, {})
    placeholders = list(economy_config.get("placeholders", []))
    for indicator_name, spec in economy_config.get("indicator_map", {}).items():
        if isinstance(spec, dict) and spec.get("status") == "TODO":
            label = spec.get("label", indicator_name)
            if label not in placeholders:
                placeholders.append(label)
    return placeholders


def expected_indicator_metadata(config: dict[str, Any], economy_code: str) -> dict[str, dict[str, str]]:
    economy_config = config.get("economies", {}).get(economy_code, {})
    metadata: dict[str, dict[str, str]] = {}
    for indicator_name, spec in economy_config.get("indicator_map", economy_config.get("fred_series", {})).items():
        if isinstance(spec, str):
            metadata[indicator_name] = {
                "label": indicator_name,
                "expected_source": "FRED",
                "status": "active",
            }
        else:
            metadata[indicator_name] = {
                "label": spec.get("label", indicator_name),
                "expected_source": spec.get("expected_source", spec.get("source", "unknown").upper()),
                "status": spec.get("status", "active"),
            }
    return metadata


def fetch_fred_series(series_id: str, api_key: str, config: dict[str, Any]) -> pd.Series:
    return fred.fetch_series_with_key(series_id, api_key, config)


def uses_fred(indicator_map: dict[str, Any]) -> bool:
    for spec in indicator_map.values():
        if isinstance(spec, str) or spec.get("source") in {"fred", "fx"}:
            return True
    return False


def fred_warning(indicator_name: str, series_id: str, series: pd.Series) -> str | None:
    if series.empty:
        return f"FRED series {series_id} for {indicator_name} is unavailable or returned no data."
    return fred.cache_warning(series_id)


def unavailable_warning(indicator_name: str, expected_source: str, status: str) -> str:
    return f"Indicator unavailable: {indicator_name}. Expected source: {expected_source}. Status: {status}."


def should_show_fetch_warning(warning: str, debug_data: bool) -> bool:
    if debug_data:
        return True
    return warning.startswith("No ") or warning.startswith("No data series") or warning.startswith("Live FRED fetch failed") or warning.startswith("Offline mode")


def empty(name: str) -> pd.Series:
    return pd.Series(dtype="float64", name=name)
