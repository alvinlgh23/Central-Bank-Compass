import pandas as pd

from src.data_sources import fred


def get_inflation_data(config: dict) -> dict[str, pd.Series]:
    return {
        "mas_core_inflation": fetch_core_inflation(config),
        "import_inflation_proxy": fetch_import_inflation_proxy(config),
    }


def get_growth_data(config: dict) -> dict[str, pd.Series]:
    return {}


def get_labor_data(config: dict) -> dict[str, pd.Series]:
    return {}


def get_financial_data(config: dict) -> dict[str, pd.Series]:
    return {"sgd_neer_shadow_proxy": fetch_sgd_neer_shadow_proxy(config)}


def get_currency_data(config: dict) -> dict[str, pd.Series]:
    return {"usd_sgd": fred.fetch_series("DEXSIUS", config)}


def fetch_core_inflation(config: dict) -> pd.Series:
    # TODO: Integrate MAS core inflation from MAS or SingStat.
    return placeholder_series("mas_core_inflation")


def fetch_sgd_neer_position(config: dict) -> pd.Series:
    # TODO: Add official SGD NEER band position hook if MAS publishes a stable machine-readable series.
    return placeholder_series("sgd_neer_position")


def fetch_sgd_neer_shadow_proxy(config: dict) -> pd.Series:
    # This is only a shadow FX proxy, not the official MAS SGD NEER band.
    return fred.fetch_series("DEXSIUS", config)


def fetch_import_inflation_proxy(config: dict) -> pd.Series:
    return fred.fetch_series("PLMCPPSGA670NRUG", config)


def placeholder_series(name: str) -> pd.Series:
    return pd.Series(dtype="float64", name=name)
