# Central Bank Compass

Central Bank Compass is a standalone CLI research project for comparing macro policy pressure across central banks. It answers a practical question:

> Does current macro data support policy easing, holding, or tightening?

The project also includes a historical backtesting layer, a global liquidity layer, and an educational macro allocation framework. It is intentionally CLI-only: no Streamlit, no web app, and no integration into a larger market-intelligence system.

## What It Does

- Generates a US Historical Policy Pressure & Narrative Filter, a BOJ Normalization Pressure model, and rule-based signals for other supported central banks.
- Explains every signal with indicator values, thresholds, classifications, score contributions, and policy meaning.
- Backtests monthly historical policy signals against actual policy-rate moves where data is available.
- Scores global liquidity across the US, China, Eurozone, Japan, and Singapore.
- Maps macro regimes into broad educational allocation ranges for cash, government bonds, equities, gold, and crypto.

This is a research and portfolio project, not financial advice.

## Why Central Banks Differ

Central banks respond to inflation, labor markets, growth, financial conditions, and currency pressure, but their policy tools differ.

- United States / Federal Reserve: Federal Funds Rate
- Eurozone / European Central Bank: ECB policy rates
- Japan / Bank of Japan: policy rate and policy-normalization framework
- Singapore / Monetary Authority of Singapore: SGD nominal effective exchange rate policy band

For that reason, the model uses generic labels:

- `EASING`
- `HOLD`
- `TIGHTENING`

Singapore is treated separately because MAS does not mainly use a domestic policy rate. For Singapore, `EASING` means a lower/flatter or more dovish SGD NEER band stance, while `TIGHTENING` means a stronger/higher or more hawkish SGD appreciation bias.

## Supported Economies

Policy signal model:

- `US`: United States / Federal Reserve
- `SG`: Singapore / MAS
- `EZ`: Eurozone / ECB
- `JP`: Japan / BOJ

Global liquidity layer:

- United States: 40%
- China: 25%
- Eurozone: 15%
- Japan: 15%
- Singapore: 5%

## Project Architecture

```text
central-bank-compass/
├── app.py
├── model.py
├── config.yaml
├── requirements.txt
├── README.md
├── .env.example
├── .gitignore
└── src/
    ├── __init__.py
    ├── allocation.py
    ├── backtest.py
    ├── country_profiles.py
    ├── data_sources/
    │   ├── __init__.py
    │   ├── boj.py
    │   ├── commodities.py
    │   ├── ecb.py
    │   ├── fred.py
    │   ├── fx.py
    │   ├── mas.py
    │   ├── oecd.py
    │   ├── pbc.py
    │   └── singstat.py
    ├── indicators.py
    ├── liquidity.py
    ├── narrative_filter.py
    ├── japan_pressure.py
    ├── policy_signal.py
    ├── probability_model.py
    ├── report.py
    └── scoring.py
```

Module responsibilities:

- `model.py`: primary interactive CLI entry point.
- `app.py`: legacy/advanced direct CLI routing.
- `country_profiles.py`: economy metadata, policy tools, scoring weights, and interpretation rules.
- `data_sources/`: multi-source data layer with FRED, FX, central-bank, national-statistics, PBC, and OECD modules.
- `narrative_filter.py`: country-specific market-narrative stress tests for non-US, non-Japan economies.
- `indicators.py`: indicator transformations such as YoY changes, gaps, spreads, and trends.
- `scoring.py`: explainable block and indicator-level scoring.
- `probability_model.py`: US historical policy-pressure model, market narrative filter, and probability-style backtest.
- `japan_pressure.py`: BOJ-specific easing/hold/normalization pressure model.
- `policy_signal.py`: legacy rule-based policy score aggregation, confidence, bias, and decision rules.
- `report.py`: current-signal CLI report formatting.
- `backtest.py`: monthly historical replay, policy-action comparison, metrics, CSVs, and charts.
- `liquidity.py`: global liquidity scoring, confidence, details, CSV, and chart.
- `allocation.py`: educational macro-regime allocation ranges, CSV, and chart.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Then edit `.env`:

```bash
FRED_API_KEY=your_fred_api_key_here
```

FRED API keys are available from the Federal Reserve Bank of St. Louis:

https://fred.stlouisfed.org/docs/api/api_key.html

The CLI still runs without a valid key, but live data will be unavailable and confidence will be reduced.

### Manual FRED Cache Seeding

If FRED blocks the current network path, manually downloaded FRED CSV files can initialize the local cache:

1. Download each required series as CSV from FRED.
2. Place the files in `data/imports/fred/`.
3. Name each file with its configured series ID, for example `UNRATE.csv`, `PCEPILFE.csv`, or `DGS10.csv`.
4. Seed and inspect the cache:

```bash
python model.py --seed-fred-cache data/imports/fred/
python model.py --cache-status
python model.py --offline --economy US --summary
```

Accepted column layouts include:

```text
DATE,VALUE
observation_date,value
observation_date,UNRATE
```

For downloaded files whose names are not the series ID, add `data/imports/fred/mapping.csv`:

```csv
filename,series_id
unemployment_download.csv,UNRATE
core_pce_download.csv,PCEPILFE
```

Series IDs are validated against the project's configured FRED series. Invalid dates and empty or nonnumeric observations are removed and reported. An import does not replace an equally recent or newer valid cache unless explicitly forced:

```bash
python model.py --seed-fred-cache data/imports/fred/ --force
```

Cache files are normalized to `data/cache/fred/<SERIES_ID>.csv` with `date,value` columns and written atomically. Original import files remain unchanged.

## CLI Usage

Primary interactive menu:

```bash
python model.py
```

The menu lets you choose:

```text
1. United States Policy Pressure & Narrative Filter
2. Other Central Bank Policy Signals
3. Backtest Policy Pressure Model
4. Global Liquidity Compass
5. Macro Regime Allocation Framework
6. Run Full Macro Dashboard
0. Exit
```

Direct command alternatives:

Current policy signal:

```bash
python model.py --economy US
python model.py --economy US --summary
python model.py --economy US --details
python model.py --economy US --market-view dovish
python model.py --economy US --market-view aggressive_easing
python model.py --economy SG
python model.py --economy EZ
python model.py --economy JP
```

The plain US command displays a concise Macro Noise Summary and asks whether to show detailed evidence. `--summary` skips the prompt and stops after the summary. `--details` skips the prompt and prints the full coverage, narrative, scenario, explainability, and energy sections.

Country-specific narrative stress tests:

```bash
python model.py --narrative EZ
python model.py --narrative SG
python model.py --narrative CN
python model.py --narrative KR
python model.py --narrative UK
python model.py --narrative AU
python model.py --narrative CA
python model.py --narrative CH
python model.py --narrative all
python model.py --narrative CN --market-view "PBOC easing means China recovery is back"
```

Active narrative countries are Eurozone, Singapore, China, South Korea, United Kingdom, Australia, Canada, and Switzerland. India is intentionally outside the current implementation scope.

Energy shock monitor:

```bash
python model.py --energy-shock
```

Legacy US rule-based mode:

```bash
python model.py --economy US --legacy-rules
```

Backtesting:

```bash
python model.py --backtest --economy US
python model.py --backtest --economy US --start 2015
```

Global liquidity:

```bash
python model.py --liquidity
python model.py --liquidity --details
```

Macro regime allocation framework:

```bash
python model.py --allocation
python model.py --allocation --details
```

Full macro dashboard:

```bash
python model.py
# choose option 5
```

## Legacy / Advanced CLI Usage

`app.py` remains available for backward compatibility:

```bash
python app.py --economy US
python app.py --economy SG
python app.py --economy EZ
python app.py --economy JP
python app.py --economy US --backtest --start 2015
python app.py --liquidity
python app.py --liquidity --details
python app.py --allocation
python app.py --allocation --details
```

## Policy Pressure Model

Central Bank Compass does not predict the next Federal Reserve meeting. It evaluates whether current macroeconomic data supports, contradicts, or leaves uncertain the policy narrative currently influencing markets.

The default US model is a Historical Policy Pressure Model. It estimates:

- Easing Pressure
- Hold Pressure
- Tightening Pressure

These pressure estimates sum to 100%, but they are not exact FOMC odds. They are macro-policy pressure estimates based on current data, historical feature distributions, and forward 3M/6M policy-direction labels.

US features include inflation, labor, growth, financial conditions, and currency pressure. Each feature is shown with current value, historical percentile, z-score, trend, policy effect, and explanation.

### Market Narrative Filter

Live Fed futures are not scraped. Instead, a manual market-view input can compare a market narrative against macro pressure:

```bash
python model.py --economy US --market-view dovish
python model.py --economy US --market-view neutral
python model.py --economy US --market-view hawkish
```

Allowed values:

- `aggressive_easing`
- `dovish`
- `neutral`
- `hawkish`
- `aggressive_tightening`

The narrative filter reports whether the manual market view appears more dovish, aligned, or more hawkish than macro conditions.

### BOJ Normalization Pressure

Japan uses a central-bank-specific BOJ Normalization Pressure model, not a Fed-style rate-cut/rate-hike template. It estimates:

- Easing Pressure
- Hold Pressure
- Normalization / Hike Pressure

Japan inputs include CPI ex fresh food, wage/services/inflation-expectation hooks where available, unemployment, GDP growth, USD/JPY pressure, 10Y JGB yield, and BOJ balance-sheet/JGB-purchase hooks. The report explains whether inflation looks more demand/wage-driven or import/currency-driven.

## Country-Specific Narrative Stress Tests

Narrative stress tests ask whether current macro conditions support, partially support, contradict, or cannot yet evaluate a common market narrative. They are not central-bank meeting predictions and do not produce trading recommendations.

Each country definition reflects its actual policy constraints: ECB fragmentation and services inflation, MAS SGD NEER policy, PBOC credit transmission, BOK household debt and KRW sensitivity, BoE wages/services inflation, RBA housing and China exposure, BoC mortgages and oil, and SNB safe-haven FX flows.

Where source integrations are incomplete, the report returns `Insufficient Data`, lists every missing indicator, and lowers confidence.

## Energy Shock Monitor

The energy monitor uses stable FRED series for WTI, Brent where available, gasoline, and energy CPI. It classifies year-over-year and short-term oil moves as energy disinflation, neutral, or an energy shock.

Oil matters because it can move headline inflation, real household income, central-bank reaction functions, and stagflation risk. An oil shock is not treated as equivalent to broad core or services inflation. US and ECB interpretation layers distinguish energy-driven headline pressure from persistent underlying inflation.

### Legacy Rule-Based Model

The policy model groups indicators into five blocks:

- Inflation pressure
- Labor weakness
- Growth weakness
- Financial stress
- Currency pressure

The model shows each indicator's current value, thresholds, classification, score contribution, and explanation. The net policy score is:

```text
inflation_score
- labor_weakness_score
- growth_weakness_score
- financial_stress_score
+ currency_pressure_score
```

Decision rules:

- `<= -30`: `EASING`
- `-30 to +30`: `HOLD`
- `> +30`: `TIGHTENING CANDIDATE`

Additional realism rule:

`TIGHTENING` requires high inflation, accelerating inflation, low labor weakness, low growth weakness, and low financial stress. If the score is hawkish but those conditions are not met, the model can return `HOLD` with a `Hawkish` policy bias.

Every policy report also includes a `DATA COVERAGE` section showing block-level coverage and each missing indicator's expected source, status, and confidence impact.

## Data Sources

Central Bank Compass uses a modular data layer under `src/data_sources/`.

Current modules:

- `fred.py`: primary live data source for US indicators and several international public series, with local CSV cache and offline fallback support.
- `fred_seed.py`: validates manually downloaded FRED exports and converts them into the normalized cache format.
- `fx.py`: FX and broad-dollar proxies, using FRED first where possible.
- `ecb.py`: Eurozone adapter with FRED-backed ECB/HICP and sovereign-spread proxies plus placeholder hooks.
- `boj.py`: Japan adapter with FRED-backed CPI, labor, GDP, JGB, policy-rate, and FX mappings plus placeholder hooks.
- `mas.py`: MAS adapter with SGD NEER hooks, a USD/SGD shadow proxy, import-inflation proxy, and MAS-core placeholder.
- `singstat.py`: Singapore statistics adapter with FRED-backed CPI, GDP, and external-demand proxy mappings plus an unemployment hook.
- `pbc.py`: China/PBC/NBS adapter with FRED-backed M2, policy-rate proxy, CNY pressure, property-stress proxy, and missing credit/RRR hooks.
- `oecd.py`: placeholder-ready OECD fallback module for GDP, CPI, unemployment, industrial production, and business confidence.

The project does not use unstable scraping or random free APIs. If a stable source is not integrated, the indicator remains missing and is shown as `TODO`.

### Coverage Table

| Economy | Current Coverage | Main Live Sources | Remaining Gaps / Placeholder Hooks |
| --- | --- | --- | --- |
| United States | Strong | FRED, FRED FX proxies | OECD fallback hooks |
| Eurozone | Moderate | FRED, FRED FX proxies, ECB/FRED HICP and spread proxies | Official ECB SDMX integration |
| Japan | Moderate | FRED, FRED FX proxies, JGB and policy-rate mappings | Wage growth, inflation expectations |
| Singapore | Partial to moderate | FRED, FRED FX proxy, SingStat/FRED macro proxies, import-price proxy | MAS core inflation, unemployment, financial-stress proxy, official SGD NEER band position |
| China | Partial liquidity coverage only | FRED-backed M2, policy-rate proxy, CNY pressure, property-stress proxy | Credit impulse, RRR, official PBC/NBS automation |

### Current Status

Latest validation snapshot:

| Economy / Layer | Coverage |
| --- | --- |
| US policy signal | 9/10, 90% |
| Eurozone policy signal | 8/10, 80% |
| Japan policy signal | 7/10, 70% |
| Singapore policy signal | 6/10, 60% |
| China liquidity layer | 4/6 drivers, 67% coverage with 62% confidence |

These figures can change if FRED availability, the API key, or upstream series coverage changes. Missing indicators remain visible in the CLI and reduce confidence.

### Data Source Roadmap

- Add official ECB SDMX integration for HICP, deposit facility rate, and stress/spread proxies.
- Add BOJ/Japan official-source integration for wage growth, CPI ex fresh food, JGB yields, and policy-rate data.
- Add MAS/SingStat integration for MAS core inflation, CPI, GDP, unemployment, imports, exports, and SGD NEER hooks.
- Add fuller PBC/NBS integration for China credit impulse, LPR, RRR, M2, CNY pressure, and property stress.
- Add OECD fallback queries for GDP, CPI, unemployment, industrial production, and business confidence.

## Short Example Outputs

Policy signal:

```text
CENTRAL BANK COMPASS
=================================================
Economy: United States
Central Bank: Federal Reserve
Policy Tool: Federal Funds Rate

Current Signal: TIGHTENING
Policy Bias: Hawkish
Confidence: 81%
```

Backtest:

```text
BACKTEST RESULTS
=================================================
Economy: US
Period: 2015-present

Signal Counts:
EASING: 16
HOLD: 108
TIGHTENING: 13
```

Global liquidity:

```text
GLOBAL LIQUIDITY COMPASS
=================================================
Global Liquidity Score: 48/100
Classification: Neutral
Confidence: 59%
```

Allocation framework:

```text
MACRO REGIME ALLOCATION FRAMEWORK
=================================================
Current Macro Regime: Inflationary Expansion
Confidence: 51%

Suggested Educational Allocation Ranges:
Cash: 10-25%
Government Bonds: 10-25%
Equities: 35-55%
Gold: 10-25%
Crypto: 0-10%
```

Example values depend on the latest data available from FRED and may differ when you run the project.

## Backtesting Methodology

The backtest replays the signal engine month by month from a configurable start year. For each month, it uses only observations dated on or before that month-end.

It produces:

- Signal history
- Policy score history
- Data coverage and missing-data notes
- Same-month policy move comparison
- Forward 3M, 6M, and 12M hit rates
- Regime comparison
- Lead/lag analysis

Important: this is not a true real-time vintage-data backtest. It filters by observation date using currently available revised series, but it does not reconstruct historical release calendars, publication lags, or data revisions.

## Global Liquidity Layer

The liquidity layer scores global liquidity from 0 to 100:

- `0-30`: Liquidity Contracting
- `30-70`: Neutral
- `70-100`: Liquidity Expanding

Inputs include policy signals, yields, volatility, credit stress, money supply or balance-sheet trends when available, and currency pressure. China now has several public proxy series, but credit impulse and RRR remain explicit placeholders. Singapore has public macro and FX proxies, while official SGD NEER band data remains a placeholder hook.

Missing data never becomes fake data. The model keeps neutral placeholders visible and lowers confidence.

## Macro Allocation Framework

The allocation framework translates macro regimes into educational allocation ranges across:

- Cash
- Government Bonds
- Equities
- Gold
- Crypto

Possible regimes:

- Liquidity Expansion
- Liquidity Neutral
- Liquidity Contraction
- Disinflationary Expansion
- Inflationary Expansion
- Growth Slowdown
- Stagflationary Pressure
- Financial Stress Shock

The output is broad macro scenario guidance only. It does not recommend individual securities and does not generate buy/sell signals.

## Outputs

Generated files are written to `outputs/`, which is ignored by Git.

Policy signal and backtest outputs:

- `{economy}_signal_history.csv`: monthly signal, score, confidence, data coverage, and missing-data fields.
- `{economy}_policy_comparison.csv`: model signal versus policy-rate action comparison.
- `{economy}_policy_score.png`: policy score through time.
- `{economy}_actual_policy_rate.png`: actual policy-rate chart when available.
- `{economy}_signal_changes.png`: signal changes through time.
- `{economy}_signal_vs_policy_regime.png`: signal versus actual policy regime where available.
- `{economy}_forward_hit_results.csv`: forward-window evaluation.
- `{economy}_regime_comparison.csv`: regime comparison table.
- `{economy}_lead_lag_cycles.csv`: lead/lag cycle analysis.

Global liquidity outputs:

- `global_liquidity_score.csv`
- `global_liquidity_breakdown.png`

Allocation outputs:

- `allocation_framework.csv`
- `allocation_ranges.png`

## Data Limitations

- The backtest uses revised data, not true real-time vintage data.
- Some economies have partial data coverage.
- Singapore MAS policy actions are not fully automated.
- China liquidity coverage is partial: M2, policy-rate proxy, CNY pressure, and property-stress proxy are available, while credit impulse and RRR remain missing.
- Several data-source modules are placeholder-ready but not fully automated.
- Some non-US indicators depend on imperfect public FRED mappings.
- Policy decisions are institutional and judgment-based, not purely mechanical.
- Market reactions can differ from macro expectations for positioning, valuation, geopolitical, and liquidity-friction reasons.

## Missing-Data Transparency

Central Bank Compass is designed to show missing data explicitly:

- Missing indicators contribute no fake score.
- Confidence is reduced when data coverage is weak.
- Placeholder drivers are labeled as placeholders.
- Data warnings are printed in CLI reports.

This is especially important for China credit/RRR inputs, Singapore MAS policy actions and SGD NEER band data, Japan wage growth, and inflation expectations.

## Not Financial Advice

This project is for educational, analytical, and portfolio demonstration purposes only. It is not financial advice, investment advice, trading advice, or a forecast of a central bank's next decision. The allocation framework provides broad educational ranges, not recommendations to buy, sell, or hold any asset.
