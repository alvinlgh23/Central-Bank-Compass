from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class IndicatorSet:
    values: dict[str, float | None] = field(default_factory=dict)
    latest_observation: pd.Timestamp | None = None

    def get(self, name: str) -> float | None:
        return self.values.get(name)


def build_indicators(economy_code: str, data: dict[str, pd.Series]) -> IndicatorSet:
    latest_dates = [series.index.max() for series in data.values() if not series.empty]
    latest_observation = max(latest_dates) if latest_dates else None

    builders = {
        "US": build_us_indicators,
        "SG": build_sg_indicators,
        "EZ": build_ez_indicators,
        "JP": build_jp_indicators,
    }
    values = builders[economy_code](data)
    return IndicatorSet(values=values, latest_observation=latest_observation)


def build_us_indicators(data: dict[str, pd.Series]) -> dict[str, float | None]:
    ten_year = latest(data.get("ten_year_yield"))
    two_year = latest(data.get("two_year_yield"))
    broad_dollar_change = yoy(data.get("broad_dollar"), 260)
    return {
        "core_inflation_yoy": yoy(data.get("core_pce"), 12),
        "core_inflation_trend": yoy_change(data.get("core_pce"), 12, 3),
        "headline_inflation_yoy": yoy(data.get("cpi"), 12),
        "headline_inflation_trend": yoy_change(data.get("cpi"), 12, 3),
        "secondary_inflation_yoy": yoy(data.get("core_cpi"), 12),
        "unemployment_rate": latest(data.get("unemployment")),
        "unemployment_gap": unemployment_gap_from_12m_low(data.get("unemployment")),
        "payrolls_3m_avg_change": average_change(data.get("payrolls"), 3),
        "payrolls_12m_avg_change": average_change(data.get("payrolls"), 12),
        "claims_yoy": yoy(data.get("initial_claims"), 52),
        "growth_yoy": yoy(data.get("real_gdp"), 4),
        "ten_year_yield": ten_year,
        "two_year_yield": two_year,
        "yield_curve": subtract(ten_year, two_year),
        "credit_spread": latest(data.get("credit_spread")),
        "credit_spread_change": point_change(data.get("credit_spread"), 126),
        "vix": latest(data.get("vix")),
        "currency_change_yoy": broad_dollar_change,
    }


def build_sg_indicators(data: dict[str, pd.Series]) -> dict[str, float | None]:
    usd_sgd_change = yoy(data.get("usd_sgd"), 260)
    import_inflation = yoy(data.get("import_inflation_pressure"), 12)
    external_demand = latest(data.get("external_demand"))
    return {
        "core_inflation_yoy": latest_or_yoy(data.get("mas_core_inflation"), 12),
        "core_inflation_trend": None,
        "headline_inflation_yoy": latest(data.get("cpi")),
        "headline_inflation_trend": point_change(data.get("cpi"), 1),
        "secondary_inflation_yoy": import_inflation,
        "unemployment_rate": latest(data.get("unemployment")),
        "unemployment_gap": unemployment_gap_from_12m_low(data.get("unemployment")),
        "claims_yoy": None,
        "growth_yoy": latest(data.get("real_gdp")),
        "pmi": external_demand,
        "external_demand_is_growth_rate": 1.0,
        "financial_stress": None,
        "currency_change_yoy": usd_sgd_change,
        "usd_sgd_change_yoy": usd_sgd_change,
        "import_inflation_pressure": import_inflation,
        "sgd_neer_shadow_proxy": yoy(data.get("sgd_neer_shadow_proxy"), 260),
    }


def build_ez_indicators(data: dict[str, pd.Series]) -> dict[str, float | None]:
    core_hicp = data.get("core_hicp")
    headline_hicp = data.get("hicp")
    sovereign_spread = latest(data.get("sovereign_spreads"))
    return {
        "core_inflation_yoy": yoy(core_hicp, 12),
        "core_inflation_trend": yoy_change(core_hicp, 12, 3),
        "headline_inflation_yoy": yoy(headline_hicp, 12),
        "headline_inflation_trend": yoy_change(headline_hicp, 12, 3),
        "secondary_inflation_yoy": yoy(headline_hicp, 12),
        "unemployment_rate": latest(data.get("unemployment")),
        "unemployment_gap": unemployment_gap_from_12m_low(data.get("unemployment")),
        "claims_yoy": None,
        "growth_yoy": yoy(data.get("real_gdp"), 4),
        "pmi": latest(data.get("pmi")),
        "financial_stress": sovereign_spread,
        "sovereign_spread": sovereign_spread,
        "currency_change_yoy": yoy(data.get("eur_usd"), 260),
    }


def build_jp_indicators(data: dict[str, pd.Series]) -> dict[str, float | None]:
    # TODO: Add wage growth and inflation expectations when reliable source mappings are configured.
    usd_jpy_change = yoy(data.get("usd_jpy"), 260)
    return {
        "core_inflation_yoy": yoy(data.get("cpi_ex_fresh_food"), 12),
        "core_inflation_trend": yoy_change(data.get("cpi_ex_fresh_food"), 12, 3),
        "headline_inflation_yoy": yoy(data.get("cpi_ex_fresh_food"), 12),
        "headline_inflation_trend": yoy_change(data.get("cpi_ex_fresh_food"), 12, 3),
        "secondary_inflation_yoy": None,
        "unemployment_rate": latest(data.get("unemployment")),
        "unemployment_gap": unemployment_gap_from_12m_low(data.get("unemployment")),
        "claims_yoy": None,
        "growth_yoy": yoy(data.get("real_gdp"), 4),
        "ten_year_yield": latest(data.get("ten_year_yield")),
        "financial_stress": None,
        "currency_change_yoy": usd_jpy_change,
        "usd_jpy_change_yoy": usd_jpy_change,
        "wage_growth": None,
    }


def clean(series: pd.Series | None) -> pd.Series:
    if series is None or series.empty:
        return pd.Series(dtype="float64")
    return series.astype(float).replace([np.inf, -np.inf], np.nan).dropna()


def latest(series: pd.Series | None) -> float | None:
    values = clean(series)
    if values.empty:
        return None
    return float(values.iloc[-1])


def latest_or_yoy(series: pd.Series | None, periods: int) -> float | None:
    value = yoy(series, periods)
    if value is not None:
        return value
    return latest(series)


def yoy(series: pd.Series | None, periods: int) -> float | None:
    values = clean(series)
    if len(values) <= periods:
        return None
    current = values.iloc[-1]
    prior = values.iloc[-periods - 1]
    if prior == 0 or pd.isna(prior):
        return None
    return float(((current / prior) - 1) * 100)


def yoy_change(series: pd.Series | None, yoy_periods: int, change_periods: int) -> float | None:
    values = clean(series)
    if len(values) <= yoy_periods + change_periods:
        return None
    current_prior = values.iloc[-yoy_periods - 1]
    previous_current = values.iloc[-change_periods - 1]
    previous_prior = values.iloc[-yoy_periods - change_periods - 1]
    if current_prior == 0 or previous_prior == 0 or pd.isna(current_prior) or pd.isna(previous_prior):
        return None
    current_yoy = ((values.iloc[-1] / current_prior) - 1) * 100
    previous_yoy = ((previous_current / previous_prior) - 1) * 100
    return float(current_yoy - previous_yoy)


def point_change(series: pd.Series | None, periods: int) -> float | None:
    values = clean(series)
    if len(values) <= periods:
        return None
    return float(values.iloc[-1] - values.iloc[-periods - 1])


def average_change(series: pd.Series | None, periods: int) -> float | None:
    values = clean(series)
    if len(values) <= periods:
        return None
    changes = values.diff().dropna()
    if len(changes) < periods:
        return None
    return float(changes.tail(periods).mean())


def unemployment_gap_from_12m_low(series: pd.Series | None) -> float | None:
    values = clean(series)
    if values.empty:
        return None
    window = values.tail(12)
    if window.empty:
        return None
    return float(values.iloc[-1] - window.min())


def subtract(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return float(left - right)
