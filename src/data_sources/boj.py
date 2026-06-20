import pandas as pd

from src.data_sources import fred


def get_inflation_data(config: dict) -> dict[str, pd.Series]:
    return {
        "cpi_ex_fresh_food": fetch_cpi_ex_fresh_food(config),
        "inflation_expectations": fetch_inflation_expectations(config),
    }


def get_growth_data(config: dict) -> dict[str, pd.Series]:
    return {"gdp": fetch_gdp_growth(config)}


def get_labor_data(config: dict) -> dict[str, pd.Series]:
    return {
        "unemployment": fetch_unemployment(config),
        "wage_growth": fetch_wage_growth(config),
    }


def get_financial_data(config: dict) -> dict[str, pd.Series]:
    return {
        "jgb_10y": fetch_jgb_10y_yield(config),
        "policy_rate": fetch_policy_rate(config),
    }


def get_currency_data(config: dict) -> dict[str, pd.Series]:
    return {"usd_jpy": fred.fetch_series("DEXJPUS", config)}


def fetch_cpi_ex_fresh_food(config: dict) -> pd.Series:
    return fred.fetch_series("JPNCPIALLMINMEI", config)


def fetch_wage_growth(config: dict) -> pd.Series:
    # TODO: Integrate Japan wage growth from MHLW or BOJ when a stable API is configured.
    return placeholder_series("japan_wage_growth")


def fetch_unemployment(config: dict) -> pd.Series:
    return fred.fetch_series("LRHUTTTTJPM156S", config)


def fetch_gdp_growth(config: dict) -> pd.Series:
    return fred.fetch_series("JPNRGDPEXP", config)


def fetch_jgb_10y_yield(config: dict) -> pd.Series:
    return fred.fetch_series("IRLTLT01JPM156N", config)


def fetch_policy_rate(config: dict) -> pd.Series:
    return fred.fetch_series("IRSTCB01JPM156N", config)


def fetch_inflation_expectations(config: dict) -> pd.Series:
    # TODO: Integrate BOJ inflation expectations when a stable official series is configured.
    return placeholder_series("japan_inflation_expectations")


def placeholder_series(name: str) -> pd.Series:
    return pd.Series(dtype="float64", name=name)
