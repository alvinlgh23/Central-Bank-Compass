from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import pandas as pd

from src.data_sources.fred import CACHE_DIR, configured_series_ids, save_cached_series


SERIES_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
DATE_COLUMNS = {"date", "observation_date"}
VALUE_COLUMNS = {"value"}


@dataclass(frozen=True)
class SeedItem:
    filename: str
    series_id: str | None
    status: str
    reason: str
    rows: int = 0
    invalid_rows: int = 0
    start_date: str | None = None
    end_date: str | None = None


@dataclass(frozen=True)
class SeedResult:
    items: list[SeedItem]

    @property
    def imported(self) -> list[SeedItem]:
        return [item for item in self.items if item.status == "IMPORTED"]

    @property
    def skipped(self) -> list[SeedItem]:
        return [item for item in self.items if item.status == "SKIPPED"]


def seed_fred_cache(
    import_dir: Path,
    config: dict[str, Any],
    force: bool = False,
    cache_dir: Path = CACHE_DIR,
) -> SeedResult:
    if not import_dir.is_dir():
        return SeedResult([SeedItem(str(import_dir), None, "SKIPPED", "Import directory does not exist.")])
    mappings, mapping_errors = load_mapping(import_dir / "mapping.csv")
    items = list(mapping_errors)
    allowed_ids = set(configured_series_ids(config))
    csv_files = sorted(path for path in import_dir.glob("*.csv") if path.name.lower() != "mapping.csv")
    if not csv_files:
        items.append(SeedItem(str(import_dir), None, "SKIPPED", "No CSV files found."))
        return SeedResult(items)
    for path in csv_files:
        series_id = mappings.get(path.name, path.stem)
        validation_error = validate_series_id(series_id, allowed_ids)
        if validation_error:
            items.append(SeedItem(path.name, series_id, "SKIPPED", validation_error))
            continue
        items.append(import_series_file(path, series_id, force, cache_dir))
    return SeedResult(items)


def load_mapping(path: Path) -> tuple[dict[str, str], list[SeedItem]]:
    if not path.is_file():
        return {}, []
    try:
        frame = pd.read_csv(path, dtype=str)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        return {}, [SeedItem(path.name, None, "SKIPPED", f"Invalid mapping file: {type(exc).__name__}.")]
    columns = {str(column).strip().lower(): column for column in frame.columns}
    if "filename" not in columns or "series_id" not in columns:
        return {}, [SeedItem(path.name, None, "SKIPPED", "Mapping file requires filename and series_id columns.")]
    mappings: dict[str, str] = {}
    errors: list[SeedItem] = []
    for _, row in frame.iterrows():
        filename = str(row[columns["filename"]]).strip()
        series_id = str(row[columns["series_id"]]).strip()
        if not filename or not series_id or filename.lower() == "nan" or series_id.lower() == "nan":
            errors.append(SeedItem(path.name, None, "SKIPPED", "Mapping row has an empty filename or series_id."))
            continue
        mappings[filename] = series_id
    return mappings, errors


def validate_series_id(series_id: str, allowed_ids: set[str]) -> str | None:
    if not SERIES_ID_PATTERN.fullmatch(series_id):
        return "Invalid series ID format."
    if series_id not in allowed_ids:
        return "Series ID is not configured by Central Bank Compass."
    return None


def import_series_file(path: Path, series_id: str, force: bool, cache_dir: Path) -> SeedItem:
    try:
        frame = pd.read_csv(path)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        return SeedItem(path.name, series_id, "SKIPPED", f"CSV read failed: {type(exc).__name__}.")
    columns = {str(column).strip().lower(): column for column in frame.columns}
    date_column = next((columns[name] for name in DATE_COLUMNS if name in columns), None)
    value_column = next((columns[name] for name in VALUE_COLUMNS if name in columns), None)
    if value_column is None:
        value_column = next((column for column in frame.columns if str(column).strip().upper() == series_id.upper()), None)
    if date_column is None or value_column is None:
        return SeedItem(path.name, series_id, "SKIPPED", "CSV requires DATE/VALUE or observation_date/value columns.")
    parsed = pd.DataFrame(
        {
            "date": pd.to_datetime(frame[date_column], errors="coerce"),
            "value": pd.to_numeric(frame[value_column].replace(".", pd.NA), errors="coerce"),
        }
    )
    invalid_rows = int(parsed[["date", "value"]].isna().any(axis=1).sum())
    parsed = parsed.dropna(subset=["date", "value"]).drop_duplicates(subset=["date"], keep="last").sort_values("date")
    if parsed.empty:
        return SeedItem(path.name, series_id, "SKIPPED", "CSV contains no valid date/value rows.", invalid_rows=invalid_rows)
    imported_start = parsed["date"].iloc[0].date().isoformat()
    imported_end = parsed["date"].iloc[-1].date().isoformat()
    existing_latest = existing_cache_latest(cache_dir / f"{series_id}.csv")
    if existing_latest is not None and existing_latest >= parsed["date"].iloc[-1] and not force:
        return SeedItem(
            path.name,
            series_id,
            "SKIPPED",
            f"Existing valid cache is equally recent or newer ({existing_latest.date().isoformat()}); use --force to overwrite.",
            rows=len(parsed),
            invalid_rows=invalid_rows,
            start_date=imported_start,
            end_date=imported_end,
        )
    series = parsed.set_index("date")["value"].rename(series_id)
    try:
        save_cached_series(series_id, series, cache_dir=cache_dir)
    except OSError as exc:
        return SeedItem(path.name, series_id, "SKIPPED", f"Atomic cache write failed: {type(exc).__name__}.")
    return SeedItem(
        path.name,
        series_id,
        "IMPORTED",
        "Cache created or updated.",
        rows=len(series),
        invalid_rows=invalid_rows,
        start_date=imported_start,
        end_date=imported_end,
    )


def existing_cache_latest(path: Path) -> pd.Timestamp | None:
    if not path.is_file():
        return None
    try:
        frame = pd.read_csv(path, usecols=["date", "value"])
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
        frame = frame.dropna(subset=["date", "value"])
    except (OSError, ValueError, pd.errors.ParserError):
        return None
    return None if frame.empty else pd.Timestamp(frame["date"].max())


def render_seed_result(result: SeedResult) -> str:
    lines = ["FRED CACHE SEED", "================================================="]
    for item in result.items:
        identity = f"{item.filename} -> {item.series_id}" if item.series_id else item.filename
        range_text = f" | Range: {item.start_date} to {item.end_date}" if item.start_date and item.end_date else ""
        lines.append(
            f"{item.status}: {identity} | Rows: {item.rows} | Invalid Rows: {item.invalid_rows}{range_text} | {item.reason}"
        )
    lines.extend(
        [
            "",
            f"Imported Series: {len(result.imported)}",
            f"Skipped Series: {len(result.skipped)}",
            f"Invalid Rows: {sum(item.invalid_rows for item in result.items)}",
        ]
    )
    return "\n".join(lines)
