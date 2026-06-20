from typing import Any

import pandas as pd

from src.data_sources import fred


FRED_FX_SERIES = {
    "broad_dollar": "DTWEXBGS",
    "eur_usd": "DEXUSEU",
    "usd_jpy": "DEXJPUS",
    "usd_sgd": "DEXSIUS",
    "usd_cny": "CCUSSP02CNM650N",
}


def classify_yoy_change(value: float | None) -> str:
    if value is None:
        return "MISSING DATA"
    if value >= 5:
        return "APPRECIATING / PRESSURE"
    if value <= -5:
        return "DEPRECIATING / SUPPORTIVE"
    return "NEUTRAL"


def current_value(series: pd.Series) -> float | None:
    values = series.dropna()
    if values.empty:
        return None
    return float(values.iloc[-1])


def yoy_change(series: pd.Series, periods: int = 260) -> float | None:
    values = series.dropna()
    if len(values) <= periods:
        return None
    prior = values.iloc[-periods - 1]
    if prior == 0:
        return None
    return float(((values.iloc[-1] / prior) - 1) * 100)


def describe_currency_indicator(indicator_key: str, config: dict[str, Any]) -> dict[str, float | str | None]:
    series = fetch_fx_series(indicator_key, config)
    change = yoy_change(series)
    return {
        "indicator": indicator_key,
        "current_value": current_value(series),
        "yoy_change": change,
        "classification": classify_yoy_change(change),
    }


def get_inflation_data(config: dict[str, Any]) -> dict[str, pd.Series]:
    return {}


def get_growth_data(config: dict[str, Any]) -> dict[str, pd.Series]:
    return {}


def get_labor_data(config: dict[str, Any]) -> dict[str, pd.Series]:
    return {}


def get_financial_data(config: dict[str, Any]) -> dict[str, pd.Series]:
    return {}


def get_currency_data(config: dict[str, Any]) -> dict[str, pd.Series]:
    return {key: fetch_fx_series(key, config) for key in FRED_FX_SERIES}


def fetch_fx_series(indicator_key: str, config: dict[str, Any]) -> pd.Series:
    series_id = FRED_FX_SERIES.get(indicator_key)
    if not series_id:
        return placeholder_series(indicator_key)
    return fred.fetch_series(series_id, config)


def fred_series_id(indicator_key: str) -> str | None:
    return FRED_FX_SERIES.get(indicator_key)


def placeholder_series(indicator_key: str) -> pd.Series:
    # TODO: Add a stable institutional FX source if FRED coverage is not enough.
    return pd.Series(dtype="float64", name=indicator_key)
