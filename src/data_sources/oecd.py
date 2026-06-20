import pandas as pd


def get_inflation_data(config: dict, economy_code: str) -> dict[str, pd.Series]:
    return {"cpi": fetch_cpi(economy_code)}


def get_growth_data(config: dict, economy_code: str) -> dict[str, pd.Series]:
    return {
        "gdp": fetch_gdp(economy_code),
        "industrial_production": fetch_industrial_production(economy_code),
        "business_confidence": fetch_business_confidence(economy_code),
    }


def get_labor_data(config: dict, economy_code: str) -> dict[str, pd.Series]:
    return {"unemployment": fetch_unemployment(economy_code)}


def get_financial_data(config: dict, economy_code: str) -> dict[str, pd.Series]:
    return {}


def get_currency_data(config: dict, economy_code: str) -> dict[str, pd.Series]:
    return {}


def fetch_gdp(economy_code: str) -> pd.Series:
    # TODO: Add OECD SDMX fallback for GDP.
    return placeholder_series(f"{economy_code.lower()}_oecd_gdp")


def fetch_cpi(economy_code: str) -> pd.Series:
    # TODO: Add OECD SDMX fallback for CPI.
    return placeholder_series(f"{economy_code.lower()}_oecd_cpi")


def fetch_unemployment(economy_code: str) -> pd.Series:
    # TODO: Add OECD SDMX fallback for unemployment.
    return placeholder_series(f"{economy_code.lower()}_oecd_unemployment")


def fetch_industrial_production(economy_code: str) -> pd.Series:
    # TODO: Add OECD SDMX fallback for industrial production.
    return placeholder_series(f"{economy_code.lower()}_oecd_industrial_production")


def fetch_business_confidence(economy_code: str) -> pd.Series:
    # TODO: Add OECD SDMX fallback for business confidence.
    return placeholder_series(f"{economy_code.lower()}_oecd_business_confidence")


def placeholder_series(name: str) -> pd.Series:
    return pd.Series(dtype="float64", name=name)
