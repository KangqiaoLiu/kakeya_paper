"""Finite annular fabrication levels, rate cap, and fixed-footprint controls."""
from pathlib import Path
import gc
import json

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import brentq

from geometry import angular_occupancy, exact_q, radial_integral, TubeScanner

OUT=Path(__file__).parent/'results'


def annular_projection(w, count):
    edges=np.r_[0.0,np.geomspace(w/2,np.hypot(1,w)/2,count)]
    levels=[]
    for a,b in zip(edges[:-1],edges[1:]):
        value=quad(lambda r: 2*np.pi*r*float(angular_occupancy(r,w)),a,b,
            epsabs=1e-12,epsrel=1e-10,points=[0.5] if a<0.5<b else None)[0]
        levels.append(value/(np.pi*(b*b-a*a)))
    areas=np.pi*np.diff(edges**2)
    norm2=np.sum(areas*np.array(levels)**2)
    q=np.sqrt(2*np.pi*norm2)/w
    return edges,np.array(levels),q


def capped_profile(w, budget=0.08, cap=1.0):
    """Optimal capped profile within the radial nonincreasing design family."""
    def integrals(alpha):
        edges=[0.0,w/2,0.5,np.hypot(1,w)/2]
        if alpha>cap:
            rc=brentq(lambda r: float(angular_occupancy(r,w))-cap/alpha,
                       w/2,np.hypot(1,w)/2,xtol=1e-14)
            edges=sorted(edges+[rc])
        def value(power, product=False):
            return sum(quad(lambda r:2*np.pi*r*min(cap,alpha*float(angular_occupancy(r,w)))**power
                *(float(angular_occupancy(r,w)) if product else 1.0),a,b,
                epsabs=1e-12,epsrel=1e-9)[0] for a,b in zip(edges[:-1],edges[1:]))
        return value(2),value(1,True)
    alpha=brentq(lambda a:integrals(a)[0]-budget**2,0.0,1e4,xtol=1e-11)
    norm2,response_integral=integrals(alpha)
    q=np.sqrt(2*np.pi)*response_integral/(w*np.sqrt(norm2))
    return alpha,q,norm2


def main():
    rows=[];profiles={}
    for w in [0.2,0.1,0.05,0.025,0.01,0.005]:
        for count in [2,4,8,16,32]:
            edges,levels,q=annular_projection(w,count)
            rows.append(dict(w=w,rings=count,q=q,exact_q=exact_q(w),retention=q/exact_q(w)))
            if w==0.05:
                profiles[str(count)]=dict(edges=edges.tolist(),levels=levels.tolist(),q=q)
    pd.DataFrame(rows).to_csv(OUT/'annular_realization.csv',index=False)
    (OUT/'annular_profiles_w005.json').write_text(json.dumps(profiles,indent=2)+'\n')
    rows=[]
    for w in np.geomspace(0.003,0.3,30):
        alpha,q,norm2=capped_profile(w)
        rows.append(dict(w=w,budget=0.08,cap=1.0,alpha=alpha,q=q,
                         exact_q=exact_q(w),norm2=norm2))
    pd.DataFrame(rows).to_csv(OUT/'rate_cap_radial_family.csv',index=False)
    print('RINGS',profiles['8'],flush=True)
    print('RATE_CAP',rows[0],rows[-1],flush=True)

    scanner=TubeScanner(n=193,w=0.05,nangles=90,oversample=4,angle_offset=0.5)
    field=angular_occupancy(np.hypot(scanner.X,scanner.Y),0.05)
    mask=field>0
    support_values=field[mask]
    rows=[]
    for seed in range(12):
        rng=np.random.default_rng(61008+seed)
        scrambled=np.zeros_like(field)
        scrambled[mask]=rng.permutation(support_values)
        res=scanner.scan(scrambled)
        rows.append(dict(seed=seed,q=res.q,mass=np.sum(scrambled)*scanner.h**2,
            norm2=np.sum(scrambled**2)*scanner.h**2,
            support_area=np.count_nonzero(scrambled)*scanner.h**2))
        if seed==0:
            np.savez_compressed(OUT/'fixed_support_shuffle.npz',field=scrambled,x=scanner.x,
                theta=scanner.theta,response=res.response,centers=res.centers)
    pd.DataFrame(rows).to_csv(OUT/'fixed_support_shuffle_ensemble.csv',index=False)
    print('FIXED_SUPPORT',pd.DataFrame(rows).q.describe().to_dict(),flush=True)
    reversed_field=np.zeros_like(field)
    reversed_field[mask]=np.sort(support_values)[np.argsort(np.argsort(-field[mask]))]
    res=scanner.scan(reversed_field)
    np.savez_compressed(OUT/'fixed_support_radial_reverse.npz',field=reversed_field,x=scanner.x,
        theta=scanner.theta,response=res.response,centers=res.centers)
    print('RADIAL_REVERSE',res.q,flush=True)


if __name__=='__main__':
    main()
