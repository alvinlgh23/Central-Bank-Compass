import pandas as pd

from src.data_sources import fred


def get_inflation_data(config: dict) -> dict[str, pd.Series]:
    return {"cpi": fetch_cpi(config)}


def get_growth_data(config: dict) -> dict[str, pd.Series]:
    return {
        "gdp": fetch_gdp_growth(config),
        "external_demand": fetch_external_demand_proxy(config),
    }


def get_labor_data(config: dict) -> dict[str, pd.Series]:
    return {"unemployment": fetch_unemployment(config)}


def get_financial_data(config: dict) -> dict[str, pd.Series]:
    return {}


def get_currency_data(config: dict) -> dict[str, pd.Series]:
    return {}


def fetch_cpi(config: dict) -> pd.Series:
    return fred.fetch_series("FPCPITOTLZGSGP", config)


def fetch_gdp_growth(config: dict) -> pd.Series:
    return fred.fetch_series("SGPNGDPRPCPPPT", config)


def fetch_unemployment(config: dict) -> pd.Series:
    return fred.fetch_series("SGPURAMS", config)


def fetch_external_demand_proxy(config: dict) -> pd.Series:
    return fred.fetch_series("SGPTXRPCPPPT", config)


def placeholder_series(name: str) -> pd.Series:
    return pd.Series(dtype="float64", name=name)
