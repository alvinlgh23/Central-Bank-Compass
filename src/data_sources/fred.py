import os
from pathlib import Path
from typing import Any

import pandas as pd
import requests


_FETCH_DIAGNOSTICS: dict[str, dict[str, Any]] = {}
CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache" / "fred"


def get_inflation_data(config: dict[str, Any]) -> dict[str, pd.Series]:
    return {
        "core_pce": fetch_series("PCEPILFE", config),
        "cpi": fetch_series("CPIAUCSL", config),
    }


def get_growth_data(config: dict[str, Any]) -> dict[str, pd.Series]:
    return {"real_gdp": fetch_series("GDPC1", config)}


def get_labor_data(config: dict[str, Any]) -> dict[str, pd.Series]:
    return {
        "unemployment": fetch_series("UNRATE", config),
        "payrolls": fetch_series("PAYEMS", config),
        "initial_claims": fetch_series("ICSA", config),
    }


def get_financial_data(config: dict[str, Any]) -> dict[str, pd.Series]:
    return {
        "ten_year_yield": fetch_series("DGS10", config),
        "two_year_yield": fetch_series("DGS2", config),
        "credit_spread": fetch_series("BAA10Y", config),
        "vix": fetch_series("VIXCLS", config),
    }


def get_currency_data(config: dict[str, Any]) -> dict[str, pd.Series]:
    return {"broad_dollar": fetch_series("DTWEXBGS", config)}


def fetch_series(series_id: str, config: dict[str, Any]) -> pd.Series:
    api_key = os.getenv("FRED_API_KEY")
    if offline_mode():
        return load_cached_series(series_id, request_status="OFFLINE")
    if not api_key:
        return load_cached_series(series_id, request_status="API key unavailable")
    return fetch_series_with_key(series_id, api_key, config)


def fetch_series_with_key(
    series_id: str,
    api_key: str,
    config: dict[str, Any],
    allow_cache: bool = True,
) -> pd.Series:
    if offline_mode() and allow_cache:
        return load_cached_series(series_id, request_status="OFFLINE")
    fred_config = config.get("fred", {})
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": fred_config.get("observation_start", "2000-01-01"),
        "sort_order": "asc",
    }

    endpoint = fred_config.get("base_url", "https://api.stlouisfed.org/fred/series/observations")
    diagnostic: dict[str, Any] = {
        "source": "FRED",
        "series_id": series_id,
        "endpoint": endpoint,
        "request_status": "Not attempted",
        "exception_type": None,
        "raw_observations": 0,
        "parsed_row_count": 0,
        "latest_observation_available": False,
        "latest_non_null_date": None,
        "api_key_in_params": "api_key" in params and bool(params.get("api_key")),
        "api_key_length": len(api_key),
        "api_key_prefix": api_key[:4],
        "api_key_suffix": api_key[-4:],
        "response_error_body": None,
        "data_source": "FRED_LIVE",
        "cache_used": False,
        "cache_latest_date": None,
        "cache_warning": None,
    }

    try:
        response = requests.get(
            endpoint,
            params=params,
            timeout=10,
        )
        diagnostic["request_status"] = f"HTTP {response.status_code}"
        if response.status_code != 200:
            diagnostic["response_error_body"] = response.text
        response.raise_for_status()
        payload = response.json()
    except (ValueError, requests.RequestException) as exc:
        diagnostic["exception_type"] = type(exc).__name__
        _FETCH_DIAGNOSTICS[series_id] = diagnostic
        if allow_cache and cache_eligible_failure(diagnostic, exc):
            return load_cached_series(series_id, diagnostic=diagnostic)
        return pd.Series(dtype="float64", name=series_id)

    observations = payload.get("observations", [])
    diagnostic["raw_observations"] = len(observations)
    if not observations:
        _FETCH_DIAGNOSTICS[series_id] = diagnostic
        return pd.Series(dtype="float64", name=series_id)

    frame = pd.DataFrame(observations)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["value"] = pd.to_numeric(frame["value"].replace(".", pd.NA), errors="coerce")
    frame = frame.dropna(subset=["date"]).set_index("date").sort_index()
    series = frame["value"].dropna().rename(series_id)
    diagnostic["parsed_row_count"] = len(series)
    diagnostic["latest_observation_available"] = not series.empty
    diagnostic["latest_non_null_date"] = series.index[-1].date().isoformat() if not series.empty else None
    _FETCH_DIAGNOSTICS[series_id] = diagnostic
    if not series.empty:
        try:
            save_cached_series(series_id, series)
        except OSError as exc:
            diagnostic["cache_write_error"] = f"{type(exc).__name__}: {exc}"
            _FETCH_DIAGNOSTICS[series_id] = diagnostic
    return series


def get_fetch_diagnostic(series_id: str) -> dict[str, Any]:
    return dict(_FETCH_DIAGNOSTICS.get(series_id, {}))


def reset_fetch_diagnostics() -> None:
    _FETCH_DIAGNOSTICS.clear()


def offline_mode() -> bool:
    return os.getenv("CBC_FRED_OFFLINE", "").lower() in {"1", "true", "yes"}


def cache_path(series_id: str, cache_dir: Path | None = None) -> Path:
    safe_id = "".join(character for character in series_id if character.isalnum() or character in {"-", "_"})
    return (cache_dir or CACHE_DIR) / f"{safe_id}.csv"


def save_cached_series(series_id: str, series: pd.Series, cache_dir: Path | None = None) -> None:
    destination_dir = cache_dir or CACHE_DIR
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = cache_path(series_id, destination_dir)
    temporary = destination.with_suffix(".csv.tmp")
    frame = series.rename("value").rename_axis("date").reset_index()
    frame.to_csv(temporary, index=False, date_format="%Y-%m-%d")
    temporary.replace(destination)


def load_cached_series(
    series_id: str,
    diagnostic: dict[str, Any] | None = None,
    request_status: str | None = None,
) -> pd.Series:
    path = cache_path(series_id)
    current = dict(diagnostic or {})
    current.setdefault("source", "FRED")
    current.setdefault("series_id", series_id)
    current.setdefault("endpoint", "cache-only")
    current.setdefault("request_status", request_status or "Live fetch failed")
    current.setdefault("exception_type", None)
    current.setdefault("raw_observations", 0)
    current.setdefault("api_key_in_params", False)
    current.setdefault("api_key_length", len(os.getenv("FRED_API_KEY", "")))
    current.setdefault("api_key_prefix", os.getenv("FRED_API_KEY", "")[:4])
    current.setdefault("api_key_suffix", os.getenv("FRED_API_KEY", "")[-4:])
    current["data_source"] = "CACHE_FALLBACK"
    current["cache_used"] = False
    if not path.is_file():
        current.update({"parsed_row_count": 0, "latest_observation_available": False, "latest_non_null_date": None})
        _FETCH_DIAGNOSTICS[series_id] = current
        return pd.Series(dtype="float64", name=series_id)
    try:
        frame = pd.read_csv(path)
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
        series = frame.dropna(subset=["date", "value"]).set_index("date")["value"].sort_index().rename(series_id)
    except (OSError, ValueError, KeyError, pd.errors.ParserError) as exc:
        current["exception_type"] = f"Cache{type(exc).__name__}"
        current.update({"parsed_row_count": 0, "latest_observation_available": False, "latest_non_null_date": None})
        _FETCH_DIAGNOSTICS[series_id] = current
        return pd.Series(dtype="float64", name=series_id)
    latest_date = series.index[-1].date().isoformat() if not series.empty else None
    current.update(
        {
            "cache_used": not series.empty,
            "cache_latest_date": latest_date,
            "parsed_row_count": len(series),
            "latest_observation_available": not series.empty,
            "latest_non_null_date": latest_date,
            "cache_warning": cache_use_warning(current.get("request_status"), latest_date),
        }
    )
    _FETCH_DIAGNOSTICS[series_id] = current
    return series


def cache_use_warning(request_status: str | None, latest_date: str | None) -> str | None:
    if not latest_date:
        return None
    if request_status == "OFFLINE":
        return f"Offline mode; using cached FRED data from {latest_date}."
    return f"Live FRED fetch failed; using cached data from {latest_date}."


def cache_eligible_failure(diagnostic: dict[str, Any], exc: Exception) -> bool:
    status_text = str(diagnostic.get("request_status", ""))
    status = int(status_text.removeprefix("HTTP ")) if status_text.startswith("HTTP ") else None
    return status == 403 or (status is not None and status >= 500) or isinstance(exc, (requests.Timeout, requests.ConnectionError))


def cache_warning(series_id: str) -> str | None:
    return get_fetch_diagnostic(series_id).get("cache_warning")


def configured_series_ids(config: dict[str, Any]) -> list[str]:
    from src.data_sources.fx import FRED_FX_SERIES
    from src.data_sources.commodities import SERIES as COMMODITY_SERIES

    series_ids: set[str] = set(COMMODITY_SERIES.values())
    for economy in config.get("economies", {}).values():
        for spec in economy.get("indicator_map", {}).values():
            if isinstance(spec, str):
                series_ids.add(spec)
            elif spec.get("source") == "fred" and spec.get("series_id"):
                series_ids.add(spec["series_id"])
            elif spec.get("source") == "fx" and spec.get("series_id") in FRED_FX_SERIES:
                series_ids.add(FRED_FX_SERIES[spec["series_id"]])
        series_ids.update(economy.get("actual_policy_series", {}).values())
    for economy_series in config.get("liquidity", {}).get("fred_series", {}).values():
        series_ids.update(economy_series.values())
    return sorted(series_id for series_id in series_ids if series_id)


def cache_entry(series_id: str) -> tuple[bool, str | None, int]:
    path = cache_path(series_id)
    if not path.is_file():
        return False, None, 0
    try:
        frame = pd.read_csv(path, usecols=["date", "value"])
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
        frame = frame.dropna(subset=["date", "value"])
    except (OSError, ValueError, pd.errors.ParserError):
        return False, None, 0
    latest = frame["date"].max().date().isoformat() if not frame.empty else None
    return not frame.empty, latest, len(frame)


def render_cache_status(config: dict[str, Any]) -> str:
    lines = [
        "FRED CACHE STATUS",
        "=================================================",
        "Series ID | Cached? | Latest Date | Row Count",
    ]
    for series_id in configured_series_ids(config):
        cached, latest, rows = cache_entry(series_id)
        lines.append(f"{series_id} | {'yes' if cached else 'no'} | {latest or 'N/A'} | {rows}")
    return "\n".join(lines)


def refresh_cache(config: dict[str, Any]) -> str:
    api_key = os.getenv("FRED_API_KEY")
    series_ids = configured_series_ids(config)
    lines = ["FRED CACHE REFRESH", "================================================="]
    if not api_key:
        return "\n".join([*lines, "FRED_API_KEY is unavailable; no live refresh was attempted."])
    succeeded: list[str] = []
    failed: list[str] = []
    for series_id in series_ids:
        series = fetch_series_with_key(series_id, api_key, config, allow_cache=False)
        if series.empty:
            failed.append(series_id)
        else:
            succeeded.append(series_id)
    lines.extend(
        [
            f"Successful updates: {len(succeeded)}",
            f"Failed updates: {len(failed)}",
            f"Updated: {', '.join(succeeded) if succeeded else 'none'}",
            f"Failed (existing cache retained): {', '.join(failed) if failed else 'none'}",
        ]
    )
    return "\n".join(lines)


def render_data_source_status() -> str:
    diagnostics = list(_FETCH_DIAGNOSTICS.values())
    statuses = [str(item.get("request_status", "")) for item in diagnostics]
    if offline_mode():
        live_status = "offline (not attempted)"
    elif any(status == "HTTP 200" for status in statuses):
        live_status = "ok"
    elif any(status == "HTTP 403" for status in statuses):
        live_status = "blocked"
    else:
        live_status = "unavailable"
    cache_used = any(item.get("cache_used") for item in diagnostics)
    cached = []
    for path in sorted(CACHE_DIR.glob("*.csv")) if CACHE_DIR.exists() else []:
        exists, latest, rows = cache_entry(path.stem)
        if exists and latest:
            cached.append((path.stem, latest, rows))
    oldest = min(cached, key=lambda item: item[1]) if cached else None
    latest = max(cached, key=lambda item: item[1]) if cached else None
    lines = [
        "DATA SOURCE STATUS",
        "=================================================",
        f"FRED Live: {live_status}",
        f"Cache Fallback: {'used' if cache_used else 'not used'}",
        f"Oldest Cached Series: {oldest[0] + ' (' + oldest[1] + ')' if oldest else 'N/A'}",
        f"Latest Cached Series: {latest[0] + ' (' + latest[1] + ')' if latest else 'N/A'}",
    ]
    used_cache = [item for item in diagnostics if item.get("cache_used") and item.get("cache_latest_date")]
    for item in used_cache:
        age = (pd.Timestamp.today().normalize() - pd.Timestamp(item["cache_latest_date"])).days
        if age > 120:
            lines.append(f"SEVERE STALE WARNING: {item['series_id']} cache is {age} days old (>120 days).")
        elif age > 45:
            lines.append(f"Stale warning: {item['series_id']} cache is {age} days old (46-120 days).")
    return "\n".join(lines)


def test_fred_connection(config: dict[str, Any], env_path: Path) -> str:
    api_key = os.getenv("FRED_API_KEY")
    fred_config = config.get("fred", {})
    endpoint = fred_config.get("base_url", "https://api.stlouisfed.org/fred/series/observations")
    params = {
        "series_id": "UNRATE",
        "api_key": api_key,
        "file_type": "json",
        "observation_start": fred_config.get("observation_start", "2000-01-01"),
        "sort_order": "asc",
    }
    sanitized_params = {key: ("<redacted>" if key == "api_key" else value) for key, value in params.items()}
    lines = [
        "FRED CONNECTION TEST",
        "=================================================",
        f".env found: {'yes' if env_path.is_file() else 'no'} ({env_path})",
        f"FRED_API_KEY exists: {'yes' if bool(api_key) else 'no'}",
        f"api_key_in_params: {'yes' if bool(params.get('api_key')) else 'no'}",
        f"api_key_length: {len(api_key or '')}",
        f"api_key_prefix: {api_key[:4] if api_key else 'N/A'}",
        f"api_key_suffix: {api_key[-4:] if api_key else 'N/A'}",
        f"Endpoint: {endpoint}",
        f"Sanitized URL params: {sanitized_params}",
    ]
    if not api_key:
        return "\n".join([*lines, "HTTP status: not requested", "Response error: FRED_API_KEY is missing."])
    try:
        response = requests.get(endpoint, params=params, timeout=10)
        lines.append(f"HTTP status: {response.status_code}")
        if response.status_code != 200:
            lines.extend(["Response error body:", response.text])
            return "\n".join(lines)
        payload = response.json()
        observations = payload.get("observations", [])
        frame = pd.DataFrame(observations)
        if frame.empty:
            lines.extend(["Rows returned: 0", "Latest observation: unavailable"])
            return "\n".join(lines)
        frame["value"] = pd.to_numeric(frame["value"].replace(".", pd.NA), errors="coerce")
        frame = frame.dropna(subset=["value"])
        latest = frame.iloc[-1] if not frame.empty else None
        lines.extend(
            [
                f"Rows returned: {len(frame)}",
                f"Latest observation: {latest['date']} = {latest['value']}" if latest is not None else "Latest observation: unavailable",
            ]
        )
    except (ValueError, requests.RequestException) as exc:
        lines.extend([f"HTTP status: unavailable", f"Response exception: {type(exc).__name__}: {exc}"])
    return "\n".join(lines)
