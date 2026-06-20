import argparse
import os
from pathlib import Path

from src.data_sources.fred import refresh_cache, render_cache_status, test_fred_connection
from src.data_sources.fred_seed import render_seed_result, seed_fred_cache
from app import (
    project_context,
    run_allocation_report,
    run_backtest_report,
    run_energy_shock_report,
    run_full_macro_dashboard,
    run_liquidity_report,
    run_narrative_report,
    run_policy_report,
)


ECONOMY_CHOICES = {
    "1": ("US", "United States / Federal Reserve"),
    "2": ("SG", "Singapore / MAS"),
    "3": ("EZ", "Eurozone / ECB"),
    "4": ("JP", "Japan / BOJ"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Central Bank Compass interactive and direct CLI.")
    parser.add_argument("--economy", choices=["US", "SG", "EZ", "JP"], help="Run a policy report for an economy.")
    parser.add_argument("--backtest", action="store_true", help="Run a policy-signal backtest.")
    parser.add_argument("--start", type=int, default=2010, help="Backtest start year.")
    parser.add_argument("--liquidity", action="store_true", help="Run the Global Liquidity Compass.")
    parser.add_argument("--allocation", action="store_true", help="Run the Macro Regime Allocation Framework.")
    parser.add_argument("--details", action="store_true", help="Show details for liquidity or allocation.")
    parser.add_argument("--summary", action="store_true", help="Show the concise US Macro Noise Summary only.")
    parser.add_argument(
        "--debug-data",
        action="store_true",
        help="Show all attempted data-fetch failures, including unused configured series.",
    )
    parser.add_argument("--legacy-rules", action="store_true", help="Use the original rule-based policy model.")
    parser.add_argument(
        "--market-view",
        help="Manual market narrative for policy or country narrative stress tests.",
    )
    parser.add_argument("--narrative", help="Run country narrative stress test: EZ, SG, CN, KR, UK, AU, CA, CH, or all.")
    parser.add_argument("--energy-shock", action="store_true", help="Run the oil and energy shock monitor.")
    parser.add_argument("--test-fred", action="store_true", help="Test the sanitized FRED request path with UNRATE.")
    parser.add_argument("--cache-status", action="store_true", help="Show local FRED cache coverage.")
    parser.add_argument("--refresh-cache", action="store_true", help="Refresh all configured FRED cache files.")
    parser.add_argument("--offline", action="store_true", help="Use FRED cache only; do not make live FRED requests.")
    parser.add_argument("--seed-fred-cache", type=str, metavar="PATH", help="Seed the FRED cache from downloaded CSV files.")
    parser.add_argument("--force", action="store_true", help="Allow cache seeding to overwrite an equally recent or newer valid cache.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.offline:
        os.environ["CBC_FRED_OFFLINE"] = "1"
    project_dir, config = project_context()

    try:
        if direct_command_requested(args):
            output = run_direct_command(args, config, project_dir)
            if output:
                print(output)
            return
        interactive_menu(config, project_dir)
    except KeyboardInterrupt:
        print("\nExiting Central Bank Compass.")


def direct_command_requested(args: argparse.Namespace) -> bool:
    return bool(args.economy or args.backtest or args.liquidity or args.allocation or args.narrative or args.energy_shock or args.test_fred or args.cache_status or args.refresh_cache or args.seed_fred_cache)


def run_direct_command(args: argparse.Namespace, config: dict, project_dir) -> str:
    if args.seed_fred_cache:
        result = seed_fred_cache(Path(args.seed_fred_cache).expanduser().resolve(), config, force=args.force)
        return render_seed_result(result)
    if args.cache_status:
        return render_cache_status(config)
    if args.refresh_cache:
        return refresh_cache(config)
    if args.test_fred:
        return test_fred_connection(config, project_dir / ".env")
    if args.energy_shock:
        return run_energy_shock_report(config)
    if args.narrative:
        return run_narrative_report(args.narrative, config, project_dir, market_view=args.market_view, debug_data=args.debug_data)
    if args.allocation:
        return run_allocation_report(config, project_dir, show_details=args.details, debug_data=args.debug_data)
    if args.liquidity:
        return run_liquidity_report(config, project_dir, show_details=args.details, debug_data=args.debug_data)
    if args.backtest:
        if not args.economy:
            raise SystemExit("Error: --economy is required with --backtest.")
        return run_backtest_report(args.economy.upper(), config, project_dir, args.start, legacy_rules=args.legacy_rules)
    if args.economy:
        if args.economy.upper() == "US" and not args.legacy_rules and not args.summary and not args.details and not args.debug_data:
            print(
                run_policy_report(
                    "US",
                    config,
                    debug_data=args.debug_data,
                    market_view=args.market_view,
                    summary_only=True,
                )
            )
            if ask_show_details():
                return run_policy_report(
                    "US",
                    config,
                    debug_data=args.debug_data,
                    market_view=args.market_view,
                    details_only=True,
                )
            return ""
        return run_policy_report(
            args.economy.upper(),
            config,
            debug_data=args.debug_data,
            legacy_rules=args.legacy_rules,
            market_view=args.market_view,
            summary_only=args.summary,
            details_only=args.debug_data,
        )
    raise SystemExit("Error: choose a report mode.")


def interactive_menu(config: dict, project_dir) -> None:
    while True:
        print_menu()
        choice = input("Enter choice: ").strip()
        if choice == "0":
            print("Exiting Central Bank Compass.")
            return
        if choice == "1":
            print(run_policy_report("US", config, summary_only=True))
            if ask_show_details():
                print(run_policy_report("US", config, details_only=True))
            continue
        if choice == "2":
            economy = choose_economy(verbose=True, include_us=False)
            if economy:
                print(run_policy_report(economy, config))
            continue
        if choice == "3":
            economy = choose_economy(verbose=False)
            if economy:
                start_year = choose_start_year()
                print(run_backtest_report(economy, config, project_dir, start_year))
            continue
        if choice == "4":
            print(run_liquidity_report(config, project_dir, show_details=choose_details()))
            continue
        if choice == "5":
            print(run_allocation_report(config, project_dir, show_details=choose_details()))
            continue
        if choice == "6":
            print(run_full_macro_dashboard(config, project_dir))
            continue
        if choice == "7":
            narrative_code = choose_narrative_country()
            if narrative_code:
                print(run_narrative_report(narrative_code, config, project_dir))
            continue
        print("Invalid choice. Please choose a number from 0 to 7.")


def print_menu() -> None:
    print(
        "\n".join(
            [
                "CENTRAL BANK COMPASS",
                "=================================================",
                "Choose a report:",
                "",
                "1. United States Policy Pressure & Narrative Filter",
                "2. Other Central Bank Policy Signals",
                "3. Backtest Policy Pressure Model",
                "4. Global Liquidity Compass",
                "5. Macro Regime Allocation Framework",
                "6. Run Full Macro Dashboard",
                "7. Country-Specific Narrative Stress Test",
                "0. Exit",
                "",
            ]
        )
    )


def choose_economy(verbose: bool, include_us: bool = True) -> str | None:
    print("Choose economy:")
    if include_us:
        choices = ECONOMY_CHOICES
    else:
        non_us = [value for value in ECONOMY_CHOICES.values() if value[0] != "US"]
        choices = {str(index + 1): value for index, value in enumerate(non_us)}
    if verbose:
        for key, (_, label) in choices.items():
            print(f"{key}. {label}")
    else:
        for key, (code, _) in choices.items():
            print(f"{key}. {code}")
    choice = input("Enter choice: ").strip()
    if choice not in choices:
        print("Invalid economy choice. Returning to main menu.")
        return None
    return choices[choice][0]


def choose_start_year() -> int:
    raw = input("Start year? Press Enter for default 2010: ").strip()
    if not raw:
        return 2010
    try:
        return int(raw)
    except ValueError:
        print("Invalid year. Using default 2010.")
        return 2010


def choose_details() -> bool:
    print("Show details?")
    print("1. Summary")
    print("2. Details")
    choice = input("Enter choice: ").strip()
    if choice == "2":
        return True
    if choice != "1":
        print("Invalid choice. Showing summary.")
    return False


def choose_narrative_country() -> str | None:
    choices = {
        "1": ("EZ", "Eurozone / ECB"),
        "2": ("SG", "Singapore / MAS"),
        "3": ("CN", "China / PBOC"),
        "4": ("KR", "South Korea / BOK"),
        "5": ("UK", "United Kingdom / BoE"),
        "6": ("AU", "Australia / RBA"),
        "7": ("CA", "Canada / BoC"),
        "8": ("CH", "Switzerland / SNB"),
        "9": ("all", "All supported countries"),
    }
    print("Choose narrative stress test:")
    for key, (_, label) in choices.items():
        print(f"{key}. {label}")
    choice = input("Enter choice: ").strip()
    if choice not in choices:
        print("Invalid narrative choice. Returning to main menu.")
        return None
    return choices[choice][0]


def ask_show_details() -> bool:
    while True:
        answer = input("Show detailed evidence? (Y/N): ").strip().lower()
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no", ""}:
            return False
        print("Please enter Y or N.")


if __name__ == "__main__":
    main()
