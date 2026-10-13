#!/usr/bin/env bash
# Copy VS Code settings and the GitHub Actions workflow from editor-config/ into the
# dot-folders that VS Code and GitHub read. Run once from the repository root:
#   bash scripts/setup_editor_config.sh
set -euo pipefail
mkdir -p .vscode .github/workflows
cp editor-config/vscode/*.json .vscode/
cp editor-config/github-workflows/ci.yml .github/workflows/ci.yml
echo "Installed .vscode/ and .github/workflows/ci.yml"
