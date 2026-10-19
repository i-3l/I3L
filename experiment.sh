#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if [[ $# -eq 0 ]]; then
  exec python scripts/experiment_wizard.py
fi

case "${1}" in
  wizard|--wizard|-w)
    shift
    exec python scripts/experiment_wizard.py "$@"
    ;;
  cards|list-cards|--cards)
    shift
    exec python scripts/experiment_wizard.py --list-cards "$@"
    ;;
  *)
    exec python scripts/experiment_pipeline.py "$@"
    ;;
esac
