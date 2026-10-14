r"""Command-line interface: ``niv <command>``.

Commands
--------
``niv list``
    Show the Elexon catalogue (dataset keys, paths, whether in the default pull).
``niv summary``
    Show row counts and time coverage of every consolidated dataset.
``niv verify``
    Download Elexon's OpenAPI spec and report which time parameters each endpoint uses.
``niv fetch --source {elexon,weather,entsoe} [--days 365] [--end YYYY-MM-DD] ...``
    Download data for a window (default: the 365 complete days before today) and
    consolidate it into ``data/processed``.
``niv neso-search QUERY`` / ``niv fetch --source neso --resource ID --name NAME``
    Discover and download NESO Data Portal resources.

Examples
--------
.. code-block:: console

    niv verify
    niv fetch --source elexon --days 365 --dry-run
    niv fetch --source elexon --datasets system_prices,indicated_imbalance
    niv fetch --source weather --weather-mode previous_runs

The first full year can take a while (per-day endpoints need ~365 requests each); runs are
resumable because every window is cached.

"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, timedelta

from niv.config import DEFAULT_DAYS, data_dir, default_window, load_dotenv
from niv.sources import elexon, entsoe, neso, weather
from niv.storage import Store

log = logging.getLogger("niv")


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser."""
    p = argparse.ArgumentParser(prog="niv", description=__doc__.split("\n")[0])
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="list the Elexon dataset catalogue")
    sub.add_parser("summary", help="rows and time coverage of downloaded datasets")
    sub.add_parser("verify", help="check catalogue against Elexon's OpenAPI spec (needs network)")

    s = sub.add_parser("neso-search", help="search the NESO Data Portal")
    s.add_argument("query")

    g = sub.add_parser("figures", help="regenerate publication PDFs from stored forecasts")
    g.add_argument("--dir", help="backtest directory (default <data>/backtest)")

    sub.add_parser("build-dataset", help="build the leak-free feature/target table")

    b = sub.add_parser("backtest", help="expanding-window backtest of base learners and the stack")
    b.add_argument("--quick", action="store_true", help="smaller/faster base learners")
    b.add_argument("--test-start", default="2026-01", help="first test month YYYY-MM (2026-01)")
    b.add_argument("--oof-months", type=int, default=6, help="out-of-fold months before test")
    b.add_argument("--exclude-prefix", action="append", default=[], help="drop features by prefix")
    b.add_argument("--max-months", type=int, help="limit the number of folds (debug)")
    b.add_argument("--out", help="output directory (default <data>/backtest)")

    f = sub.add_parser("fetch", help="download data")
    f.add_argument("--source", choices=["elexon", "weather", "entsoe", "neso"], required=True)
    f.add_argument("--days", type=int, default=DEFAULT_DAYS, help="look-back in days (365)")
    f.add_argument("--end", type=date.fromisoformat, help="exclusive end date (default today)")
    f.add_argument("--datasets", help="comma-separated Elexon keys (default: all default ones)")
    f.add_argument("--include-heavy", action="store_true", help="also fetch opt-in big datasets")
    f.add_argument("--force", action="store_true", help="re-download cached chunks")
    f.add_argument("--dry-run", action="store_true", help="print the plan only")
    f.add_argument(
        "--weather-mode",
        choices=["historical_forecast", "previous_runs"],
        default="historical_forecast",
    )
    f.add_argument("--weather-models", help="comma-separated Open-Meteo models")
    f.add_argument("--weather-vars", help="comma-separated hourly variables")
    f.add_argument("--weather-days", default="1,2", help="previous-day offsets (default 1,2)")
    f.add_argument("--zones", help="ENTSO-E zones, comma-separated")
    f.add_argument("--resource", help="NESO CKAN resource id")
    f.add_argument("--name", help="NESO dataset name for storage")
    return p


def _selected_elexon(args: argparse.Namespace) -> list[elexon.Endpoint]:
    """Resolve ``--datasets`` / ``--include-heavy`` to catalogue entries."""
    if args.datasets:
        keys = [k.strip() for k in args.datasets.split(",") if k.strip()]
        unknown = [k for k in keys if k not in elexon.CATALOGUE]
        if unknown:
            raise SystemExit(f"Unknown dataset(s): {unknown}. See `niv list`.")
        return [elexon.CATALOGUE[k] for k in keys]
    return [e for e in elexon.CATALOGUE.values() if e.default or args.include_heavy]


def cmd_list() -> None:
    """Print the catalogue."""
    for e in elexon.CATALOGUE.values():
        flag = "default" if e.default else "opt-in "
        print(f"{e.key:30s} {flag}  {e.prefer:10s} {e.path}\n{'':40s}{e.description}")


def cmd_verify() -> int:
    """Print the verification report; non-zero exit if anything is wrong."""
    client = elexon.ElexonClient()
    cache = data_dir() / "cache" / "elexon_openapi.json"
    if client.load_spec(cache) is None:
        print("OpenAPI spec unavailable (no network and no cache).")
        return 2
    bad = 0
    for row in elexon.verify_catalogue(client):
        ok = row["path_found"] and (row["time_params"] is not None or "{" in row["path"])
        ok = ok and row["extra_params_present"]
        bad += not ok
        print(
            f"{'OK ' if ok else 'BAD'} {row['key']:30s} time={row['time_params']} "
            f"extra_ok={row['extra_params_present']}"
        )
        if not ok:
            print(f"    path={row['path']} found={row['path_found']} params={row['params']}")
    return 1 if bad else 0


def cmd_fetch(args: argparse.Namespace) -> int:
    """Run a download for the requested source."""
    start, end = default_window(args.days, args.end)
    store = Store(data_dir())
    log.info("Window %s -> %s (UTC, end exclusive)", start.date(), end.date())

    if args.source == "elexon":
        client = elexon.ElexonClient()
        if not args.dry_run:
            client.load_spec(data_dir() / "cache" / "elexon_openapi.json")
        for ep in _selected_elexon(args):
            log.info("== %s", ep.key)
            try:
                n = client.fetch(ep, start, end, store, force=args.force, dry_run=args.dry_run)
            except Exception as exc:  # noqa: BLE001 - keep going with other datasets
                log.error("%s failed: %s", ep.key, exc)
                continue
            if not args.dry_run:
                store.consolidate(elexon.SOURCE, ep.key)
            log.info("%s: %d chunks", ep.key, n)
    elif args.source == "weather":
        weather.fetch_weather(
            store,
            start.date(),
            end.date() - timedelta(days=1),
            mode=args.weather_mode,
            models=args.weather_models.split(",") if args.weather_models else None,
            hourly=args.weather_vars.split(",") if args.weather_vars else None,
            previous_days=tuple(int(k) for k in args.weather_days.split(",")),
            force=args.force,
            dry_run=args.dry_run,
        )
        if not args.dry_run:
            store.consolidate(weather.SOURCE, f"weather_{args.weather_mode}")
    elif args.source == "entsoe":
        zones = args.zones.split(",") if args.zones else None
        entsoe.fetch_entsoe(
            store, start.date(), end.date(), zones=zones, force=args.force, dry_run=args.dry_run
        )
        if not args.dry_run:
            for ds in entsoe.DATASETS:
                store.consolidate(entsoe.SOURCE, ds)
    elif args.source == "neso":
        if not (args.resource and args.name):
            raise SystemExit("--resource and --name are required for --source neso")
        n = neso.fetch_resource(args.resource, args.name, store)
        store.consolidate(neso.SOURCE, args.name)
        log.info("neso %s: %d rows", args.name, n)
    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    """Run the backtest, print the headline tables and write CSV/markdown/figures."""
    from pathlib import Path

    from niv.backtest import evaluate, run_backtest, save
    from niv.reports import make_figures

    out = Path(args.out) if args.out else data_dir() / "backtest"
    res = run_backtest(
        data_dir(),
        quick=args.quick,
        test_start=args.test_start,
        oof_months=args.oof_months,
        max_months=args.max_months,
        exclude_prefixes=tuple(args.exclude_prefix),
    )
    tables = evaluate(res)
    save(res, tables, out)
    make_figures(out / "predictions.parquet", out / "weights.csv", out / "figures")
    print(tables["scores"].round(3).to_string())
    print(tables["by_lead"].round(2).to_string())
    print(tables["dm"].round(4).to_string(index=False))
    log.info("results written to %s", out)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``niv`` console script."""
    load_dotenv()
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    if args.cmd == "list":
        cmd_list()
        return 0
    if args.cmd == "summary":
        summ = Store(data_dir()).summary()
        print(summ.to_string(index=False) if len(summ) else "No processed data yet.")
        return 0
    if args.cmd == "verify":
        return cmd_verify()
    if args.cmd == "build-dataset":
        from niv.dataset import build_dataset, save_dataset

        df = build_dataset(data_dir())
        log.info("dataset %s saved to %s", df.shape, save_dataset(df, data_dir()))
        return 0
    if args.cmd == "backtest":
        return cmd_backtest(args)
    if args.cmd == "figures":
        from pathlib import Path

        from niv.reports import make_figures

        d = Path(args.dir) if args.dir else data_dir() / "backtest"
        for p in make_figures(d / "predictions.parquet", d / "weights.csv", d / "figures"):
            print(p)
        return 0
    if args.cmd == "neso-search":
        for pkg in neso.search_packages(args.query):
            print(pkg["package"], "-", pkg["title"])
            for r in pkg["resources"]:
                print("    ", r["id"], r["format"], r["name"])
        return 0
    return cmd_fetch(args)


if __name__ == "__main__":
    sys.exit(main())
