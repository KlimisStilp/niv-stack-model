# niv-stack-model

Data acquisition (and, later, modelling) for a **stacked probabilistic forecast of GB Net Imbalance
Volume (NIV), 2–40 h ahead**, with calibrated, regime-adaptive uncertainty.
Theory and methodology: [`docs/NIV_stack_model.tex`](docs/NIV_stack_model.tex).

## What it downloads (default: last 365 complete days)

| Source | Content | Key needed |
|---|---|---|
| Elexon Insights (BMRS) | NIV and system prices (target), indicated imbalance, wind/demand/margin forecasts (**every publication kept**), market index, NETBSAD/DISBSAD, fuel outturn (incl. interconnector flows as `INT*` fuel types) | no |
| Open-Meteo | archived NWP wind/solar/temperature at 13 GB points, several models | no (free tier) |
| ENTSO-E | neighbouring-zone day-ahead prices and load/wind/solar forecasts | `ENTSOE_API_KEY` |
| NESO portal | resource discovery + generic CKAN downloader | no |

Opt-in heavy Elexon datasets (`remit`, `acceptances`, `bid_offer`) via `--include-heavy`.
See `niv list` and [`docs/REFERENCES.md`](docs/REFERENCES.md).

## Quick start (VS Code)

```bash
git clone <your-repo-url> && cd niv-stack-model
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"            # add ,entsoe for ENTSO-E
cp .env.example .env               # only needed for ENTSO-E / Open-Meteo key
bash scripts/setup_editor_config.sh   # installs .vscode/ settings and the GitHub CI workflow
niv verify                         # checks every Elexon endpoint/parameter against the live OpenAPI spec
niv fetch --source elexon --days 365 --dry-run
niv fetch --source elexon --days 365
niv fetch --source weather --weather-mode previous_runs
```

Open the folder in VS Code and accept the recommended extensions; use the *Run and Debug* entries
and *Tasks* (`test`, `lint`, `verify endpoints`, `fetch 1y core data`).

Output: raw chunks in `data/raw/<source>/<dataset>/`, consolidated tables in
`data/processed/<source>/<dataset>.parquet` (git-ignored). Runs are **resumable** — rerun after a failure.

## Where the commands live

* `pyproject.toml` -> `[project.scripts] niv = "niv.cli:main"` creates the `niv` command on `pip install -e .`
* `src/niv/cli.py` -> implementation of `niv list | verify | fetch | summary | neso-search`
* `scripts/fetch_year.sh` -> one-year Elexon download with `caffeinate` and a log file in `logs/`
* `editor-config/vscode/tasks.json` -> the same commands as VS Code tasks (after `scripts/setup_editor_config.sh`)

## Modelling pipeline

```bash
niv build-dataset        # leak-free (target period, lead) table -> data/features/niv_dataset.parquet
niv backtest --quick     # ~3 min smoke run of the whole pipeline
niv backtest --test-start 2026-01   # full backtest (test period Jan 2026 -> end of data) -> data/backtest/{predictions.parquet,*.csv,report.md,figures/*.pdf}
niv figures              # regenerate the publication PDFs from stored forecasts (no refit)
```

| Module | Role |
|---|---|
| `niv.dataset` | point-in-time features at 12 lead times (2-40 h); as-of join on `publishTime <= issue` |
| `niv.models` | base learners: climatology, NESO-calibrated, LightGBM quantiles, **GAMLSS (Johnson SU) via [`ondil`](https://github.com/simon-hirsch/ondil)** |
| `niv.stack` | softmax-gated quantile averaging + regime-dependent scale head (pinball loss, analytic gradient) |
| `niv.conformal` | delayed-feedback adaptive conformal recalibration (CQR + ACI), per lead time |
| `niv.backtest`, `niv.reports` | monthly expanding folds, 48 h embargo, sequential meta-training, tables and figures |
| `niv.metrics` | pinball, CRPS proxy, coverage, Winkler, Diebold-Mariano (Newey-West) |

GAMLSS exists natively only in R; here it is provided by `ondil` (Hirsch, Berrisch & Ziel,
arXiv:2407.08750 - please verify the reference). **Licence note:** `ondil` is GPL-3.0; check that
this is compatible with how you intend to distribute/use this repository.

## Important caveats

* **Run `niv verify` first.** Endpoint *paths* were checked against Elexon's client inventory, but
  dataset parameter names are discovered from the live spec, and the `processType` value for the
  wind/solar day-ahead endpoint (`"Day Ahead"`) is unverified. This package was developed in a
  sandbox without access to these APIs, so **no live download has been run yet**; the logic is
  covered by offline unit tests only.
* **Vintages matter.** `publish`-windowed Elexon datasets keep each forecast revision. Open-Meteo's
  `historical_forecast` mode is *not* lead-specific; prefer `previous_runs` for training features.
  Start archiving live forecasts now — history for many feeds cannot be recovered later.
* Settlement dates are GB-local; use `startTime` (UTC) for joins.

## Development

```bash
pytest          # offline tests (fake HTTP session)
ruff check src tests && ruff format src tests
```

## Push to GitHub

```bash
git init -b main && git add -A && git commit -m "Initial NIV data pipeline"
gh repo create niv-stack-model --private --source=. --push      # or: git remote add origin <url> && git push -u origin main
```
