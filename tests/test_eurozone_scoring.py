import unittest

import pandas as pd

from src.country_profiles import get_profile
from src.indicators import IndicatorSet, build_ez_indicators
from src.scoring import score_macro_blocks


class EurozoneScoringTests(unittest.TestCase):
    def test_moderate_headline_without_core_is_not_high(self) -> None:
        indicators = IndicatorSet(
            values={
                "core_inflation_yoy": None,
                "headline_inflation_yoy": 2.5,
                "headline_inflation_trend": 0.0,
            }
        )

        inflation = score_macro_blocks(indicators, get_profile("EZ"))["inflation"]

        self.assertEqual(inflation.label, "MODERATE")

    def test_sticky_core_without_acceleration_is_not_high(self) -> None:
        indicators = IndicatorSet(
            values={
                "core_inflation_yoy": 2.7,
                "headline_inflation_yoy": 2.6,
                "core_inflation_trend": 0.0,
            }
        )

        inflation = score_macro_blocks(indicators, get_profile("EZ"))["inflation"]

        self.assertEqual(inflation.label, "STICKY")

    def test_confidence_index_is_transformed_to_yoy_change(self) -> None:
        dates = pd.date_range("2024-01-01", periods=13, freq="MS")
        confidence = pd.Series([100.0] * 12 + [99.0], index=dates)

        values = build_ez_indicators({"pmi": confidence})

        self.assertIsNone(values["pmi"])
        self.assertAlmostEqual(values["business_confidence_yoy"], -1.0)

    def test_financial_block_does_not_use_vix_thresholds(self) -> None:
        indicators = IndicatorSet(values={"sovereign_spread": 1.0, "vix": 40.0})

        financial = score_macro_blocks(indicators, get_profile("EZ"))["financial"]

        self.assertEqual(financial.label, "LOW")
        self.assertNotIn("VIX", [item.name for item in financial.evidence])


if __name__ == "__main__":
    unittest.main()
