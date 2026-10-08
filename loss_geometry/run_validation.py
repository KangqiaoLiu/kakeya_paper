"""Independent polygon, spectral, grid and direct-integration checks."""
from pathlib import Path
import json
import gc

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.special import roots_legendre

from geometry import (angular_occupancy, exact_q, exact_q_closed,
                      exact_q_overlap, intersection_area, TubeScanner)

OUT = Path(__file__).parent/'results'


def main():
    exact = []
    for w in [1.0, 0.5, 0.2, 0.1, 0.05, 0.025, 0.01, 0.001]:
        vals = [exact_q(w), exact_q_overlap(w), exact_q_closed(w)]
        exact.append(dict(w=w, radial=vals[0], polygon=vals[1], closed=vals[2],
            relative_spread=(max(vals)-min(vals))/vals[0]))
    pd.DataFrame(exact).to_csv(OUT/'three_way_exact_check.csv', index=False)

    overlap_rows=[]
    for delta in [0.01, 0.05, 0.1, 0.5, 0.99, 1.0]:
        for th in np.linspace(1e-6, np.pi/2-1e-6, 71):
            area=((delta-(1+delta**2)*np.tan(th/2)/2)/np.cos(th)
                  if th < 2*np.arctan(delta) else delta**2/np.sin(th))
            polygon=intersection_area(th, delta)
            overlap_rows.append(dict(w=delta, theta=th, closed_area=area,
                polygon_area=polygon, absolute_error=abs(area-polygon)))
    pd.DataFrame(overlap_rows).to_csv(OUT/'overlap_formula_check.csv', index=False)

    rng = np.random.default_rng(9427)
    differences=[]
    for w in [0.025, 0.05, 0.1, 0.3, 1.0]:
        for _ in range(200):
            th = rng.uniform(0, np.pi)
            shift = rng.uniform(-0.7, 0.7, size=2)
            centered = intersection_area(th,w)
            shifted = intersection_area(th,w,shift=shift)
            differences.append(shifted-centered)

    # An irregular, unequally weighted angular detector also has a centered
    # Gram-matrix optimum. Numerical eigenvectors are checked independently.
    n=17
    theta=np.sort(rng.uniform(0,np.pi,n))
    weights=rng.uniform(0.2,1.2,n)
    weights*=2*np.pi/weights.sum()
    w=0.05
    centers=rng.uniform(-0.2,0.2,(n,2))
    base=np.empty((n,n)); moved=np.empty((n,n))
    for i in range(n):
        for j in range(n):
            scale=np.sqrt(weights[i]*weights[j])/w**2
            base[i,j]=scale*intersection_area(theta[j]-theta[i],w)
            c,s=np.cos(theta[i]),np.sin(theta[i])
            shift=(centers[j]-centers[i]) @ np.array([[c,-s],[s,c]])
            moved[i,j]=scale*intersection_area(theta[j]-theta[i],w,shift=shift)
    eig,vec=np.linalg.eigh(base)
    v=vec[:,-1]
    spectral=dict(centered_lambda=float(eig[-1]),shifted_lambda=float(np.linalg.eigvalsh(moved)[-1]),
        eigen_residual=float(np.linalg.norm(base@v-eig[-1]*v)),
        max_shifted_overlap_excess=float(max(differences)),
        shift_trials=len(differences),max_entrywise_gram_excess=float(np.max(moved-base)))
    print('SPECTRAL',spectral,flush=True)

    # Direct Cartesian quadrature on one tube; split at the analytic radial
    # breakpoints to resolve the circular core and outer corner layer.
    def line_integral(y):
        breaks=[0.0,0.5]
        for radius in [w/2,0.5]:
            if radius > abs(y):
                x=np.sqrt(radius**2-y**2)
                if 0 < x < 0.5:
                    breaks.append(x)
        breaks=sorted(breaks)
        return 2*sum(quad(lambda x: float(angular_occupancy(np.hypot(x,y),w)),a,b,
            epsabs=1e-10,epsrel=1e-9)[0] for a,b in zip(breaks[:-1],breaks[1:]))
    tube_integral=2*quad(line_integral,0,w/2,epsabs=1e-10,epsrel=1e-9)[0]
    expected=(w*exact_q(w))**2/(2*np.pi)
    spectral.update(direct_tube_integral=tube_integral,
        radial_occupancy_norm2=expected,direct_relative_error=abs(tube_integral-expected)/expected)
    (OUT/'independent_checks.json').write_text(json.dumps(spectral,indent=2)+'\n')

    rows=[]
    for n,angles,oversample in [(129,60,4),(193,60,4),(257,90,4),(385,90,6),(513,90,6)]:
        scan=TubeScanner(n=n,w=0.05,nangles=angles,oversample=oversample)
        field=angular_occupancy(np.hypot(scan.X,scan.Y),scan.w)
        result=scan.scan(field)
        row=dict(n=n,nangles=angles,subpixels=oversample,q=result.q,
            exact_q=exact_q(scan.w),relative_error=(result.q/exact_q(scan.w)-1),
            angular_relative_spread=np.ptp(result.response)/np.mean(result.response),
            max_center_displacement=float(np.linalg.norm(result.centers,axis=1).max()))
        rows.append(row);print('GRID',row,flush=True)
        del scan
        gc.collect()
    pd.DataFrame(rows).to_csv(OUT/'radial_grid_convergence.csv',index=False)

    # Hold the original piecewise-constant pixels fixed, refine each into
    # identical subpixels, and use an angular mesh independent of training.
    rows=[]
    names=['binary_disk','binary_deltoid','binary_deltoid_optimized',
           'binary_random_optimized','same_histogram_ellipse','radial_optimum',
           'same_histogram_random_optimized']
    for factor,angles in [(1,120),(2,180)]:
        old=np.load(OUT/'radial_optimum.npz')
        n=len(old['x']);h=old['x'][1]-old['x'][0]
        radius=(n*h-h/factor)/2
        scan=TubeScanner(n=n*factor,radius=radius,w=0.05,nangles=angles,
                          oversample=4,angle_offset=0.5)
        for name in names:
            d=np.load(OUT/(name+'.npz'))
            field=np.repeat(np.repeat(d['field'],factor,axis=0),factor,axis=1)
            res=scan.scan(field)
            row=dict(name=name,refinement=factor,angles=angles,q=res.q,
                min_response=res.response.min(),max_response=res.response.max(),
                coverage_040=float(np.mean(res.response>=0.4)),
                norm2=float(np.sum(field**2)*scan.h**2))
            rows.append(row);print('INDEPENDENT_MESH',row,flush=True)
        del scan
        gc.collect()
    pd.DataFrame(rows).to_csv(OUT/'independent_mesh_responses.csv',index=False)


if __name__=='__main__':
    main()
