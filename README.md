# Kakeya geometry sets attainable limits on ballistic transport

Code and numerical data for the manuscript by Kangqiao Liu.

## Requirements and usage

Python 3.9 or later. Figure generation requires LaTeX with `latex`, `dvipng`, Computer Modern, `amsmath`, and `bm`.

Install the dependencies and run the calculations and figure generation:

```sh
python3 -m pip install -r requirements.txt
python3 reproduce.py
```

Generate figures from the supplied loss-geometry data:

```sh
python3 reproduce.py --figures-only
```

PDF and PNG files are written to `fig1/` and `loss_geometry/figures/`.

## Contents

| Path | Contents |
|---|---|
| `fig1/` | Figure 1 scripts and source data |
| `loss_geometry/geometry.py` | Tube scanning, overlap integrals, and profile optimization |
| `loss_geometry/run_*.py` | Numerical calculations |
| `loss_geometry/build_summary.py` | Collection of numerical results |
| `loss_geometry/make_figures.py` | Figures 2–4 and the peak-rate plot |
| `loss_geometry/results/` | Field arrays, responses, parameters, and calculation results |

## Data conventions

Loss-geometry calculations use `L = v_F = 1`. The `.npz` field files contain `field`, pixel-center coordinates `x`, half-circle angles `theta` in radians, `response`, and maximizing `centers`. Optimized fields also contain `history`. Figure 1 data use the length unit `ellstar`.

The full-circle angular norm is `sqrt(2*pi*mean(response**2))`. The spatial norm is `h*sqrt(sum(field**2))`, where `h` is the pixel width. CSV files contain column names; JSON files contain parameters and numerical summaries.
