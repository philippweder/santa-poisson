#!/usr/bin/env bash
# Create the conda environment <env> from its lock file, then run its smoke check.
#
# Usage, from the repository root: bash environments/create.sh <env>
# <env> is one of: advent, advent-cut, advent-iga. Needs `conda` and `mamba` on PATH
# (miniforge ships both).
set -euo pipefail

usage="usage: bash environments/create.sh <advent|advent-cut|advent-iga>"

if [ "$#" -ne 1 ]; then
  echo "$usage" >&2
  exit 2
fi
env_name="$1"

case "$env_name" in
  advent | advent-cut | advent-iga) ;;
  *)
    echo "create.sh: unknown environment '$env_name'" >&2
    echo "$usage" >&2
    exit 2
    ;;
esac

if [ ! -f "environments/locks/$env_name.conda-lock.yml" ]; then
  echo "create.sh: no environments/locks/$env_name.conda-lock.yml; run from the repository root" >&2
  exit 2
fi

if ! command -v mamba > /dev/null; then
  echo "create.sh: mamba is not on PATH (miniforge ships it)" >&2
  exit 2
fi

# 1. The tool environment that holds conda-lock.
environments_listed="$(conda env list)"
if ! echo "$environments_listed" | awk '$1 == "advent-lock" { found = 1 } END { exit !found }'; then
  mamba create -y -n advent-lock -c conda-forge python=3.12 conda-lock=4.1.0
fi

# 2. The environment itself, from the lock. mamba does the install.
conda run -n advent-lock --no-capture-output conda-lock install \
  --conda "$(command -v mamba)" -n "$env_name" "environments/locks/$env_name.conda-lock.yml"

# 3. The library, from this checkout.
conda run -n "$env_name" --no-capture-output pip install --no-deps -e .

# 4. The tooling, only in the development repository: the public repository has no tools/.
if [ "$env_name" = "advent" ] && [ -d tools ]; then
  conda run -n "$env_name" --no-capture-output pip install --no-deps -e tools
fi

# 5. CutFEMx, which no package provides.
if [ "$env_name" = "advent-cut" ]; then
  conda run -n advent-cut --no-capture-output bash environments/cutfemx/build.sh
fi

# 6. The smoke check.
bash environments/smoke.sh "$env_name"
