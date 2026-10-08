# Figure descriptions

The descriptions below use the manuscript notation. LaTeX expressions are retained for mathematical symbols.

## Figure 2

A graded loss landscape attains the collective limit and admits an accurate finite-level realization. (a) Exact coefficient $C_\star(w/L)$ and responses of eight-level annular profiles at the same spatial loss norm. (b) Radial occupancy and its eight-level approximation at $w/L=0.05$, each normalized to its own peak. (c) Fractional response shortfall for two to thirty-two concentric levels, including the central disk. Radial boundaries outside the core are geometrically spaced. Eight levels retain 99.25\% of the optimum.

## Figure 3

Spatial correlations control collective attenuation at fixed local-rate statistics. The same nonzero pixel values occupy the same circular support in (a) radial occupancy order, (b) reversed radial order, and (c) a random permutation. All three fields have the same peak, integrated rate, area, and spatial $L^2$ norm, with $w/L=0.05$. (d) Translation-maximized attenuation around the half circle, normalized by the common spatial norm. The plotted scans use 60 directions for (a) and 90 shifted directions for (b,c). Independent scans at finer spatial resolution and 180 shifted directions are recorded in `results/fixed_support_independent_readout.csv`.

## Figure 4

Spatial organization changes attenuation and angular coverage at fixed binary loss area. (a) Disk, (b) deltoid, and (c) a pattern obtained by rearranging the deltoid's active pixels. Each field has rate $\gamma_0$ on area $0.1L^2$, with $w/L=0.05$. (d) Translation-maximized responses for 60 half-circle directions. (e) Fraction of these directions reaching $\Dop\ge\beta$. Dashed lines mark $\beta v_F/(\gamma_0L)=0.4$. The deltoid and rearranged pattern reach this target at every sampled direction. The refined response values are in `results/independent_mesh_responses.csv`.

## Additional plot: bounded peak rate

An available peak rate controls the response of radial designs. (a) Exact unrestricted coefficient and optimal response within the capped radial nonincreasing family, at $B/v_F=0.08$ and $\gamma_{\max}L/v_F=1$. The profile is redesigned at each width. (b) Peak rate required by the unrestricted optimum and the available rate. Their crossing at $w/L=0.04696$ marks the onset of a saturated core. These curves describe the rigid-ray model; the physical width range also depends on diffraction.

