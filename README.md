# santa-poisson

The scripts behind the [Poisson advent calendar](https://www.philippweder.com/advent/): from
December 1 to 24, 2026, one window a day, each showing a numerical method for the Poisson problem
with a picture, a short text and the script that makes the picture. This repository holds the
scripts of the days published so far; a new day appears here when its window opens.

## Layout

- `advent/` — the small shared library: plotting style, export, Christmas shapes.
- `environments/` — the conda environments the days run in.
- `days/NN/` — the script of day NN (`solve.py`) and a short README.

## Running a day

`environments/create.sh` needs `conda` with `mamba` on `PATH` (miniforge ships both). From the
repository root:

```bash
environments/create.sh <env>
conda run -n <env> python days/NN/solve.py
```

`<env>` is the environment named in `days/NN/README.md` (`advent`, `advent-cut` or `advent-iga`).

## Licence

The code is licensed under the MIT licence (see `LICENSE`). The figures and texts of the calendar
are licensed under CC BY 4.0, attributed to Philipp Weder and each day's authors.
