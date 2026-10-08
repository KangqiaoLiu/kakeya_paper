"""Collect reported quantities from saved calculations."""
from pathlib import Path
import json
import platform

import numpy as np
import pandas as pd
import scipy
import matplotlib
from scipy.optimize import brentq

from geometry import exact_q_closed

OUT = Path(__file__).parent/'results'


def main():
    exact = pd.read_csv(OUT/'three_way_exact_check.csv')
    mesh = pd.read_csv(OUT/'radial_grid_convergence.csv')
    controls = pd.read_csv(OUT/'fixed_support_independent_readout.csv').set_index('name')
    ensemble = pd.read_csv(OUT/'fixed_support_shuffle_ensemble.csv')
    binary = pd.read_csv(OUT/'independent_mesh_responses.csv').query('refinement == 2').set_index('name')
    rings = pd.read_csv(OUT/'annular_realization.csv').query('w == 0.05').set_index('rings')
    annular_2d = pd.read_csv(OUT/'annular_independent_readout.csv').set_index('rings')
    B, cap, R = .08, 1., .5
    rc = brentq(lambda r: np.pi*cap**2*r**2*(1+2*np.log(R/r))-B**2,1e-10,R)
    t = np.log(R/rc)
    cap_limit = 2*np.sqrt(2)*(1+t)/np.sqrt(1+2*t)
    activation = brentq(lambda w:w*exact_q_closed(w)/np.sqrt(2*np.pi)-B/cap,.001,.99)
    summary = dict(date='2026-10-08', L=1.0, v_F=1.0, w=0.05,
        exact_q=exact_q_closed(.05),
        omnidirectional_min_budget_squared_coefficient=2*np.pi/exact_q_closed(.05)**2,
        exact_formula_max_relative_spread=float(exact.relative_spread.max()),
        finest_2d_relative_error=float(mesh.iloc[-1].relative_error),
        fixed_support_refined_q=controls.q.to_dict(),
        fixed_support_radial_over_shuffle=float(controls.loc['radial_optimum','q']/controls.loc['fixed_support_shuffle','q']),
        fixed_support_shuffle_mean_q=float(ensemble.q.mean()),
        fixed_support_shuffle_sample_std_q=float(ensemble.q.std(ddof=1)),
        fixed_support_shuffle_count=len(ensemble),
        eight_annuli_exact_retention=float(rings.loc[8,'retention']),
        eight_annuli_2d_retention=float(annular_2d.loc[8,'retention_2d']),
        binary_refined_q=binary.loc[['binary_disk','binary_deltoid','binary_deltoid_optimized','binary_random_optimized'],'q'].to_dict(),
        binary_optimized_over_disk=float(binary.loc['binary_deltoid_optimized','q']/binary.loc['binary_disk','q']),
        capped_radial_design=dict(budget=B, cap=cap, activation_width=activation,
            small_width_core_radius=rc, small_width_q_limit=cap_limit),
        versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
            pandas=pd.__version__,matplotlib=matplotlib.__version__))
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
