import pandas as pd

from src.data_sources import fred


def get_inflation_data(config: dict) -> dict[str, pd.Series]:
    return {
        "headline_hicp": fetch_hicp_inflation(config),
        "core_hicp": fetch_core_hicp(config),
    }


def get_growth_data(config: dict) -> dict[str, pd.Series]:
    return {"gdp": fetch_gdp_growth(config)}


def get_labor_data(config: dict) -> dict[str, pd.Series]:
    return {"unemployment": fetch_unemployment(config)}


def get_financial_data(config: dict) -> dict[str, pd.Series]:
    return {
        "deposit_facility_rate": fetch_deposit_facility_rate(config),
        "sovereign_spread_proxy": fetch_sovereign_spread_proxy(config),
    }


def get_currency_data(config: dict) -> dict[str, pd.Series]:
    return {"eur_usd": fred.fetch_series("DEXUSEU", config)}


def fetch_hicp_inflation(config: dict) -> pd.Series:
    return fred.fetch_series("CP0000EZ19M086NEST", config)


def fetch_core_hicp(config: dict) -> pd.Series:
    return fred.fetch_series("TOTNRGFOODEA20MI15XM", config)


def fetch_unemployment(config: dict) -> pd.Series:
    return fred.fetch_series("LRHUTTTTEZM156S", config)


def fetch_gdp_growth(config: dict) -> pd.Series:
    return fred.fetch_series("CLVMNACSCAB1GQEA19", config)


def fetch_deposit_facility_rate(config: dict) -> pd.Series:
    return fred.fetch_series("ECBDFR", config)


def fetch_sovereign_spread_proxy(config: dict) -> pd.Series:
    italy = fred.fetch_series("IRLTLT01ITM156N", config)
    germany = fred.fetch_series("IRLTLT01DEM156N", config)
    if italy.empty or germany.empty:
        return placeholder_series("eurozone_sovereign_spread_proxy")
    frame = pd.concat([italy, germany], axis=1, join="inner").dropna()
    if frame.empty:
        return placeholder_series("eurozone_sovereign_spread_proxy")
    spread = frame.iloc[:, 0] - frame.iloc[:, 1]
    return spread.rename("eurozone_sovereign_spread_proxy")


def placeholder_series(name: str) -> pd.Series:
    return pd.Series(dtype="float64", name=name)
