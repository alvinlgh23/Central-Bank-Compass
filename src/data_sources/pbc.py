import pandas as pd

from src.data_sources import fred


def get_inflation_data(config: dict) -> dict[str, pd.Series]:
    return {}


def get_growth_data(config: dict) -> dict[str, pd.Series]:
    return {"property_stress": fetch_property_stress_proxy(config)}


def get_labor_data(config: dict) -> dict[str, pd.Series]:
    return {}


def get_financial_data(config: dict) -> dict[str, pd.Series]:
    return {
        "m2_growth": fetch_m2_growth(config),
        "credit_impulse": fetch_credit_impulse_proxy(config),
        "lpr": fetch_lpr(config),
        "rrr": fetch_rrr(config),
    }


def get_currency_data(config: dict) -> dict[str, pd.Series]:
    return {"cny_pressure": fetch_cny_pressure(config)}


def fetch_m2_growth(config: dict) -> pd.Series:
    return fred.fetch_series("MYAGM2CNM189N", config)


def fetch_credit_impulse_proxy(config: dict) -> pd.Series:
    # TODO: Build credit impulse proxy from total social financing or credit aggregates.
    return placeholder_series("china_credit_impulse")


def fetch_lpr(config: dict) -> pd.Series:
    # FRED central bank rate is used as a public policy-rate proxy, not exact LPR.
    return fred.fetch_series("IRSTCB01CNM156N", config)


def fetch_rrr(config: dict) -> pd.Series:
    # TODO: Integrate required reserve ratio from PBC.
    return placeholder_series("china_rrr")


def fetch_cny_pressure(config: dict) -> pd.Series:
    return fred.fetch_series("CCUSSP02CNM650N", config)


def fetch_property_stress_proxy(config: dict) -> pd.Series:
    return fred.fetch_series("QCNR628BIS", config)


def placeholder_series(name: str) -> pd.Series:
    return pd.Series(dtype="float64", name=name)
