from pathlib import Path
import tempfile
import unittest

import pandas as pd

from src.data_sources.fred_seed import seed_fred_cache


CONFIG = {
    "economies": {
        "US": {
            "indicator_map": {
                "unemployment": {"source": "fred", "series_id": "UNRATE"},
            }
        }
    }
}


class FredSeedTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.import_dir = root / "imports"
        self.cache_dir = root / "cache"
        self.import_dir.mkdir()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_seeds_valid_fred_csv(self) -> None:
        (self.import_dir / "UNRATE.csv").write_text("DATE,VALUE\n2024-01-01,3.7\n2024-02-01,3.8\n", encoding="utf-8")

        result = seed_fred_cache(self.import_dir, CONFIG, cache_dir=self.cache_dir)

        self.assertEqual(len(result.imported), 1)
        cached = pd.read_csv(self.cache_dir / "UNRATE.csv")
        self.assertEqual(list(cached.columns), ["date", "value"])
        self.assertEqual(len(cached), 2)

    def test_skips_bad_csv_columns(self) -> None:
        (self.import_dir / "UNRATE.csv").write_text("month,reading\n2024-01,3.7\n", encoding="utf-8")

        result = seed_fred_cache(self.import_dir, CONFIG, cache_dir=self.cache_dir)

        self.assertEqual(len(result.imported), 0)
        self.assertIn("requires DATE/VALUE", result.skipped[0].reason)
        self.assertFalse((self.cache_dir / "UNRATE.csv").exists())

    def test_reports_and_removes_empty_values(self) -> None:
        (self.import_dir / "UNRATE.csv").write_text(
            "observation_date,value\n2024-01-01,3.7\n2024-02-01,\n2024-03-01,.\n",
            encoding="utf-8",
        )

        result = seed_fred_cache(self.import_dir, CONFIG, cache_dir=self.cache_dir)

        item = result.imported[0]
        self.assertEqual(item.rows, 1)
        self.assertEqual(item.invalid_rows, 2)

    def test_force_overwrites_newer_cache(self) -> None:
        self.cache_dir.mkdir()
        (self.cache_dir / "UNRATE.csv").write_text("date,value\n2025-01-01,4.1\n", encoding="utf-8")
        (self.import_dir / "UNRATE.csv").write_text("DATE,VALUE\n2024-01-01,3.7\n", encoding="utf-8")

        skipped = seed_fred_cache(self.import_dir, CONFIG, cache_dir=self.cache_dir)
        forced = seed_fred_cache(self.import_dir, CONFIG, force=True, cache_dir=self.cache_dir)

        self.assertEqual(len(skipped.imported), 0)
        self.assertEqual(len(forced.imported), 1)
        cached = pd.read_csv(self.cache_dir / "UNRATE.csv")
        self.assertEqual(cached.loc[0, "date"], "2024-01-01")


if __name__ == "__main__":
    unittest.main()
