#!/usr/bin/env bash
# Download one year of Elexon data, keep the Mac awake, and log to logs/.
# Resumable: re-run the same command after any interruption.
# Usage (from the repo root, with .venv active):
#   bash scripts/fetch_year.sh                 # default datasets, 365 days
#   bash scripts/fetch_year.sh --include-heavy # also REMIT/BOALF/BOD
set -euo pipefail
mkdir -p logs
LOG="logs/fetch_elexon_$(date +%Y%m%d_%H%M%S).log"
CMD=(niv fetch --source elexon --days 365 "$@")
if command -v caffeinate >/dev/null 2>&1; then CMD=(caffeinate -i "${CMD[@]}"); fi
"${CMD[@]}" 2>&1 | tee "$LOG"
niv summary | tee -a "$LOG"
