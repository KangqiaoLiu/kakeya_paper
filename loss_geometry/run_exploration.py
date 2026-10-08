"""Reproduce loss-geometry responses and retain every optimization trace."""
from pathlib import Path
import json
import time

import numpy as np
import pandas as pd

from geometry import (TubeScanner, angular_occupancy, exact_q, exact_q_overlap,
                      equal_area_shape, optimize, rearrange)


OUT = Path(__file__).parent / 'results'
OUT.mkdir(exist_ok=True)


def main():
    start = time.time()
    exact_rows = []
    for w in [1.0, 0.5, 0.2, 0.1, 0.05, 0.025, 0.01, 0.001, 0.0001]:
        q = exact_q(w)
        overlap = exact_q_overlap(w)
        exact_rows.append(dict(w=w, q=q, q_overlap=overlap,
                               overlap_relative_difference=abs(q-overlap)/q,
                               remainder=q*q-4*np.log(1/w)))
    pd.DataFrame(exact_rows).to_csv(OUT/'exact_envelope.csv', index=False)
    print('EXACT', exact_rows, flush=True)

    scan = TubeScanner(n=193, w=0.05, nangles=60, oversample=4)
    radial = angular_occupancy(np.hypot(scan.X, scan.Y), scan.w)
    rng = np.random.default_rng(20261008)
    fields = {'radial_optimum':radial,
              'same_histogram_ellipse':rearrange(radial, -(scan.X**2/4+4*scan.Y**2)),
              'same_histogram_random':rng.permutation(radial.ravel()).reshape(radial.shape)}
    for shape in ['disk', 'stripe', 'deltoid']:
        fields['binary_'+shape] = equal_area_shape(scan, 0.1, shape)
    fields['binary_random'] = rng.permutation(fields['binary_disk'].ravel()).reshape(radial.shape)
    rows = []
    histories = {}
    for name, field in list(fields.items()):
        result = scan.scan(field)
        rows.append(dict(name=name, stage='initial', q=result.q,
                         min_response=result.response.min(), max_response=result.response.max(),
                         mass=np.sum(field)*scan.h**2,
                         norm2=np.sum(field**2)*scan.h**2))
        print('INITIAL',rows[-1],flush=True)
        np.savez_compressed(OUT/(name+'.npz'), field=field, x=scan.x,
                            theta=scan.theta, response=result.response, centers=result.centers)
        if name in ['same_histogram_random', 'same_histogram_ellipse',
                    'binary_disk', 'binary_deltoid', 'binary_random']:
            evolved, ev, hist = optimize(scan, field, steps=25, constraint='histogram')
            histories[name] = hist.tolist()
            fields[name+'_optimized'] = evolved
            rows.append(dict(name=name, stage='optimized', q=ev.q,
                min_response=ev.response.min(), max_response=ev.response.max(),
                mass=np.sum(evolved)*scan.h**2,
                norm2=np.sum(evolved**2)*scan.h**2))
            np.savez_compressed(OUT/(name+'_optimized.npz'), field=evolved, x=scan.x,
                theta=scan.theta, response=ev.response, centers=ev.centers, history=hist)
            print('OPTIMIZED',rows[-1], 'monotone', bool(np.all(np.diff(hist)>=-1e-10)),flush=True)
    initial = fields['same_histogram_random']
    evolved, ev, hist = optimize(scan, initial, steps=30, constraint='l2')
    histories['free_l2_random'] = hist.tolist()
    rows.append(dict(name='free_l2_random',stage='optimized',q=ev.q,
        min_response=ev.response.min(),max_response=ev.response.max(),
        mass=np.sum(evolved)*scan.h**2,norm2=np.sum(evolved**2)*scan.h**2))
    np.savez_compressed(OUT/'free_l2_random_optimized.npz', field=evolved,x=scan.x,
        theta=scan.theta,response=ev.response,centers=ev.centers,history=hist)
    print('FREE_L2',rows[-1],flush=True)
    pd.DataFrame(rows).to_csv(OUT/'exploration.csv',index=False)
    (OUT/'optimization_traces.json').write_text(json.dumps(histories,indent=2)+'\n')
    (OUT/'exploration_config.json').write_text(json.dumps(dict(
        grid_n=193, domain_radius=0.8, w=0.05, L=1.0, angles=60,
        kernel_subpixels=4, seed=20261008, binary_area_target=0.1,
        elapsed_seconds=time.time()-start),indent=2)+'\n')


if __name__ == '__main__':
    main()
