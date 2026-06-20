import argparse
from pathlib import Path

import yaml
from dotenv import load_dotenv

from src.allocation import run_allocation_framework
from src.data_sources.commodities import allocation_energy_context, energy_context_summary, get_energy_shock_indicators, render_energy_check, render_energy_shock_monitor, us_inflation_energy_interpretation
from src.country_profiles import get_profile
from src.backtest import run_backtest
from src.data_sources import configured_placeholders, fetch_economy_data
from src.data_sources import fred
from src.indicators import build_indicators
from src.eurozone_report import render_eurozone_report
from src.japan_pressure import run_japan_policy_pressure_report
from src.liquidity import run_liquidity_compass
from src.narrative_filter import run_narrative_stress_test
from src.policy_signal import build_policy_signal
from src.probability_model import build_us_probability_signal, render_policy_constraint_check, render_probability_details, render_probability_summary, run_probability_backtest
from src.report import render_report


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Central Bank Compass CLI macro policy signal model.")
    parser.add_argument(
        "--economy",
        choices=["US", "SG", "EZ", "JP"],
        help="Economy profile to evaluate: US, SG, EZ, or JP.",
    )
    parser.add_argument(
        "--liquidity",
        action="store_true",
        help="Run the Global Liquidity Compass instead of an economy policy report.",
    )
    parser.add_argument(
        "--allocation",
        action="store_true",
        help="Run the Macro Regime Allocation Framework.",
    )
    parser.add_argument(
        "--details",
        action="store_true",
        help="Show detailed drivers. Used with --liquidity or --allocation.",
    )
    parser.add_argument("--summary", action="store_true", help="Show the concise US Macro Noise Summary only.")
    parser.add_argument(
        "--debug-data",
        action="store_true",
        help="Show all attempted data-fetch failures, including unused configured series.",
    )
    parser.add_argument(
        "--legacy-rules",
        action="store_true",
        help="Use the original rule-based policy signal model.",
    )
    parser.add_argument(
        "--market-view",
        help="Manual market narrative for a policy or country narrative stress test.",
    )
    parser.add_argument("--narrative", help="Run a country narrative stress test: EZ, SG, CN, KR, UK, AU, CA, CH, or all.")
    parser.add_argument("--energy-shock", action="store_true", help="Run the oil and energy shock monitor.")
    parser.add_argument(
        "--backtest",
        action="store_true",
        help="Run a historical monthly backtest instead of the current signal report.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=2010,
        help="Backtest start year. Used only with --backtest.",
    )
    return parser.parse_args()


def project_context() -> tuple[Path, dict]:
    project_dir = Path(__file__).resolve().parent
    load_dotenv(project_dir / ".env")
    return project_dir, load_config(project_dir / "config.yaml")


def run_policy_report(
    economy_code: str,
    config: dict,
    debug_data: bool = False,
    legacy_rules: bool = False,
    market_view: str | None = None,
    summary_only: bool = False,
    details_only: bool = False,
) -> str:
    fred.reset_fetch_diagnostics()
    if economy_code == "US" and not legacy_rules:
        signal = build_us_probability_signal(config, debug_data=debug_data, market_view=market_view)
        summary = render_probability_summary(signal, energy_context_summary(config))
        summary += "\n\n" + fred.render_data_source_status()
        if summary_only:
            return summary
        core = next((feature.value for feature in signal.features if feature.key == "core_pce_yoy"), None)
        headline = next((feature.value for feature in signal.features if feature.key == "cpi_yoy"), None)
        details = "\n\n".join(
            [
                render_policy_constraint_check(signal),
                render_probability_details(signal),
                render_energy_check(config) + "\n" + us_inflation_energy_interpretation(config, core, headline),
            ]
        )
        details += "\n\n" + fred.render_data_source_status()
        return details if details_only else summary + "\n\n" + details
    if economy_code == "JP" and not legacy_rules:
        report = run_japan_policy_pressure_report(config, debug_data=debug_data, market_view=market_view)
        return report + "\n\n" + fred.render_data_source_status()
    profile = get_profile(economy_code)
    data, data_warnings = fetch_economy_data(config, economy_code, debug_data=debug_data)
    placeholders = configured_placeholders(config, economy_code)
    indicators = build_indicators(economy_code, data)
    signal = build_policy_signal(indicators, profile, config, data_warnings, placeholders)
    if economy_code == "EZ":
        report = render_eurozone_report(signal, get_energy_shock_indicators(config), summary_only=summary_only)
        return report + "\n\n" + fred.render_data_source_status()
    report = render_report(signal) + "\n\n" + fred.render_data_source_status()
    if economy_code == "US" and legacy_rules:
        return "Model Type: Rule-Based Threshold Model\nLegacy Mode: --legacy-rules\n\n" + report
    return report


def run_backtest_report(
    economy_code: str,
    config: dict,
    project_dir: Path,
    start_year: int,
    legacy_rules: bool = False,
) -> str:
    if economy_code == "US" and not legacy_rules:
        return run_probability_backtest(config, project_dir, start_year)
    profile = get_profile(economy_code)
    return run_backtest(economy_code, profile, config, project_dir, start_year)


def run_liquidity_report(
    config: dict,
    project_dir: Path,
    show_details: bool = False,
    debug_data: bool = False,
) -> str:
    fred.reset_fetch_diagnostics()
    report = run_liquidity_compass(config, project_dir, show_details=show_details, debug_data=debug_data)
    return report + "\n\n" + fred.render_data_source_status()


def run_allocation_report(
    config: dict,
    project_dir: Path,
    show_details: bool = False,
    debug_data: bool = False,
) -> str:
    fred.reset_fetch_diagnostics()
    report = run_allocation_framework(config, project_dir, show_details=show_details, debug_data=debug_data)
    if show_details:
        report += "\n\n" + allocation_energy_context(config)
    return report + "\n\n" + fred.render_data_source_status()


def run_narrative_report(
    code: str,
    config: dict,
    project_dir: Path,
    market_view: str | None = None,
    debug_data: bool = False,
) -> str:
    return run_narrative_stress_test(code, config, project_dir, market_view=market_view, debug_data=debug_data)


def run_energy_shock_report(config: dict) -> str:
    return render_energy_shock_monitor(config)


def run_full_macro_dashboard(config: dict, project_dir: Path, debug_data: bool = False) -> str:
    print("Running US Policy...", flush=True)
    policy_report = run_policy_report("US", config, debug_data=debug_data, summary_only=True)
    policy_report += "\n\nRun python model.py --economy US --details for full evidence."

    print("Running Liquidity...", flush=True)
    liquidity_report = run_liquidity_report(config, project_dir, show_details=False, debug_data=debug_data)

    print("Running Eurozone Policy...", flush=True)
    eurozone_report = run_policy_report("EZ", config, debug_data=debug_data, summary_only=True)

    print("Running Allocation...", flush=True)
    allocation_report = run_allocation_report(config, project_dir, show_details=False, debug_data=debug_data)

    sections = [
        policy_report,
        eurozone_report,
        liquidity_report,
        allocation_report,
    ]
    return "\n\n\n".join(sections)


def main() -> None:
    project_dir, config = project_context()
    args = parse_args()

    if args.allocation:
        print(run_allocation_report(config, project_dir, show_details=args.details, debug_data=args.debug_data))
        return

    if args.energy_shock:
        print(run_energy_shock_report(config))
        return

    if args.narrative:
        print(run_narrative_report(args.narrative, config, project_dir, market_view=args.market_view, debug_data=args.debug_data))
        return

    if args.liquidity:
        print(run_liquidity_report(config, project_dir, show_details=args.details, debug_data=args.debug_data))
        return

    if not args.economy:
        raise SystemExit("Error: --economy is required unless --liquidity or --allocation is used.")

    economy_code = args.economy.upper()

    if args.backtest:
        print(run_backtest_report(economy_code, config, project_dir, args.start, legacy_rules=args.legacy_rules))
        return

    print(
        run_policy_report(
            economy_code,
            config,
            debug_data=args.debug_data,
            legacy_rules=args.legacy_rules,
            market_view=args.market_view,
            summary_only=args.summary,
        )
    )


if __name__ == "__main__":
    main()
