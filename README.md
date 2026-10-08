# Kakeya geometry constrains directionally optimized ballistic transport

Numerical code, source data, and figure-generation scripts for the work by Kangqiao Liu.

A resolved logarithmic transmission measurement gives the integral of a nonnegative local extinction rate over a finite-width ballistic tube. Translating the tube at each direction gives a directional Kakeya maximal response. This repository evaluates its exact finite-width limit, constructs attaining and finitely layered landscapes, and compares spatial arrangements at fixed rate statistics.

## Reproduction

The calculations use Python 3.9 or later. The recorded environment uses Python 3.9.6 and the package versions in `requirements.txt`. Install these packages into a Python environment:

```sh
python3 -m pip install -r requirements.txt
```

Figure generation also requires a working LaTeX installation, including `latex`, `dvipng`, Computer Modern, `amsmath`, `bm`, and Matplotlib's TeX support packages. A full TeX Live or MacTeX distribution supplies them. Numerical calculations run without LaTeX.

From the repository directory, run the complete workflow:

```sh
python3 reproduce.py
```

To redraw the four manuscript figures and the additional peak-rate plot using the supplied loss-geometry data:

```sh
python3 reproduce.py --figures-only
```

The complete workflow includes numerical optimization, independent spatial and angular scans, and figure export. It can use several gigabytes of RAM during the largest scans. Scripts run sequentially and print the calculation currently being performed. All output paths are local to this repository.

## Figure and data map

| Figure | Source | Principal data | Output |
|---|---|---|---|
| 1: transport correspondence | `fig1/kakeya_fig1.py`; `fig1/generate_source_data_fig1.py` | `fig1/source_data_fig1.csv`, `fig1/source_data_fig1_landscape.npz` | `fig1/kakeya_fig1.pdf` and `.png` |
| 2: exact limit and annular realization | `loss_geometry/make_figures.py` | `annular_realization.csv`, `annular_profiles_w005.json`; exact formula in `geometry.py` | `loss_geometry/figures/01_exact_limit_and_design.pdf` and `.png` |
| 3: fixed-statistics comparison | same | `radial_optimum.npz`, `fixed_support_radial_reverse.npz`, `fixed_support_shuffle.npz` | `loss_geometry/figures/02_fixed_statistics_response.pdf` and `.png` |
| 4: binary geometry and coverage | same | `binary_disk.npz`, `binary_deltoid.npz`, `binary_deltoid_optimized.npz` | `loss_geometry/figures/03_binary_geometry_and_coverage.pdf` and `.png` |
| Additional plot: bounded peak rate | same | `rate_cap_radial_family.csv` | `loss_geometry/figures/04_peak_rate_constraint.pdf` and `.png` |

Data filenames in the final four rows are relative to `loss_geometry/results/`. Figure descriptions are in `loss_geometry/FIGURE_CAPTIONS.md`.

## Model and normalization

`loss_geometry/geometry.py` uses length `L = 1` and Fermi speed `v_F = 1` by default. The tube width is `w`; directions span a half circle because reversing a rectangular tube leaves its support unchanged. The angular norm uses ordinary arc-length measure on the **full** circle:

```text
D(theta) = sup_c integral_{c + T_theta} gamma(x) dx / (v_F w)
B^2 = integral gamma(x)^2 dx
Q = v_F ||D||_{L2(S1)} / B
```

For a half-circle grid of `N` directions, `||D||^2` is approximated by `2*pi*mean(D**2)`. For square pixels of side `h`, `B^2 = h**2 * sum(field**2)`.

The exact coefficient is computed independently by radial quadrature, polygon-overlap quadrature, and a closed expression. At `w/L = 0.05` it is `4.24053007817`. The attaining rate is proportional to the angular occupancy of centered rectangles. An eight-level concentric approximation retains `0.992498887` of this response at the same spatial norm. The capped design is optimized within radial nonincreasing profiles at fixed spatial norm and local peak rate.

The two-dimensional scanner samples rectangular kernel area with subpixels, normalizes each kernel, and performs zero-padded linear convolution. Its maximum covers all grid-spaced centers whose tubes intersect the supplied field. Histogram-preserving optimization assigns the same pixel values according to a response-gradient score. Independent readout holds the resulting piecewise-constant field fixed while refining spatial sampling and shifting the angular mesh.

## Stored results

- Each landscape `.npz` contains `field`, pixel-center coordinates `x`, half-circle angles `theta` in radians, translation-maximized `response`, and maximizing `centers`. Optimized landscapes also contain an objective `history`.
- `exploration.csv`, `exploration_config.json`, and `optimization_traces.json` record the calculated patterns, discretization, random seed, and optimization trajectories.
- `three_way_exact_check.csv`, `overlap_formula_check.csv`, and `independent_checks.json` compare the analytical coefficient with independent integrations and the weighted-angle construction.
- `radial_grid_convergence.csv`, `independent_mesh_responses.csv`, `fixed_support_independent_readout.csv`, and `annular_independent_readout.csv` contain refined spatial and angular readouts.
- `fixed_support_shuffle_ensemble.csv` records twelve assignments with the same support and rate histogram.
- `summary.json` collects quantitative results and the numerical package versions.

The plotted curves and refined values have their sampling specified separately in the figure captions and data files. Random generators use explicit seeds. Run times in `exploration_config.json` depend on the machine.

## Citation

Kangqiao Liu, *Kakeya geometry constrains directionally optimized ballistic transport*, data and code (2026). Repository: https://github.com/KangqiaoLiu/kakeya_paper. Machine-readable metadata are provided in `CITATION.cff`.
