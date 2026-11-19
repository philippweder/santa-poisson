#!/usr/bin/env bash
# Run the smoke scripts of the conda environment <env>; the first failure stops the run.
#
# Usage, from the repository root: bash environments/smoke.sh <env>
# This is the one list of smoke scripts per environment.
set -euo pipefail

usage="usage: bash environments/smoke.sh <advent|advent-cut|advent-iga>"

if [ "$#" -ne 1 ]; then
  echo "$usage" >&2
  exit 2
fi
env_name="$1"

case "$env_name" in
  advent)
    scripts="poisson_exact.py"
    ;;
  advent-cut)
    scripts="poisson_exact.py cut_order.py"
    ;;
  advent-iga)
    scripts="iga_exact.py"
    ;;
  *)
    echo "smoke.sh: unknown environment '$env_name'" >&2
    echo "$usage" >&2
    exit 2
    ;;
esac

for script in $scripts; do
  echo "smoke.sh: $env_name: $script"
  conda run -n "$env_name" --no-capture-output python "environments/smoke/$script" \
    --env-file "environments/$env_name.yml"
done
