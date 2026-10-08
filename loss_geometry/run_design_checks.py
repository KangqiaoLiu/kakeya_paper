"""Independent 2D readout of annular designs and fixed-footprint controls."""
from pathlib import Path
import gc
import json

import numpy as np
import pandas as pd

from geometry import TubeScanner, angular_occupancy, exact_q

OUT = Path(__file__).parent / 'results'


def main():
    rows = []
    old = np.load(OUT / 'radial_optimum.npz')
    h = old['x'][1] - old['x'][0]
    n = len(old['x'])
    factor = 2
    radius = (n*h - h/factor)/2
    scanner = TubeScanner(n=n*factor, radius=radius, w=0.05,
                          nangles=180, oversample=6, angle_offset=0.5)
    names = ['radial_optimum', 'fixed_support_shuffle',
             'fixed_support_radial_reverse']
    values = np.sort(old['field'], axis=None)
    for name in names:
        data = np.load(OUT / (name + '.npz'))
        field = np.repeat(np.repeat(data['field'], factor, axis=0), factor, axis=1)
        result = scanner.scan(field)
        row = dict(name=name, q=result.q, n=scanner.n, angles=180,
                   min_response=float(result.response.min()),
                   max_response=float(result.response.max()),
                   norm2=float(np.sum(field**2)*scanner.h**2),
                   mass=float(np.sum(field)*scanner.h**2),
                   support_area=float(np.count_nonzero(field)*scanner.h**2),
                   identical_histogram=bool(np.array_equal(
                       np.sort(data['field'], axis=None), values)))
        rows.append(row)
        print('FIXED_SUPPORT_REFINED', row, flush=True)
    pd.DataFrame(rows).to_csv(OUT/'fixed_support_independent_readout.csv', index=False)
    del scanner
    gc.collect()

    profiles = json.loads((OUT/'annular_profiles_w005.json').read_text())
    rows = []
    scanner = TubeScanner(n=513, w=0.05, nangles=90, oversample=6, angle_offset=0.5)
    radius = np.hypot(scanner.X, scanner.Y)
    for count in [4, 8, 16]:
        profile = profiles[str(count)]
        edges, levels = np.asarray(profile['edges']), np.asarray(profile['levels'])
        indices = np.searchsorted(edges[1:], radius, side='right')
        field = np.r_[levels, 0.0][indices]
        result = scanner.scan(field)
        row = dict(rings=count, q_2d=result.q, q_radial=profile['q'],
                   relative_error=result.q/profile['q']-1,
                   retention_2d=result.q/exact_q(0.05),
                   angular_relative_spread=float(np.ptp(result.response)/np.mean(result.response)),
                   max_center_displacement=float(np.linalg.norm(result.centers, axis=1).max()))
        rows.append(row)
        print('ANNULAR_2D', row, flush=True)
        if count == 8:
            np.savez_compressed(OUT/'eight_annuli_2d.npz', field=field, x=scanner.x,
                theta=scanner.theta, response=result.response, centers=result.centers)
    pd.DataFrame(rows).to_csv(OUT/'annular_independent_readout.csv', index=False)


if __name__ == '__main__':
    main()
