#!/usr/bin/env python3
import numpy as np
import pandas as pd
from scipy.signal import fftconvolve

N = 340
xmin, xmax = -1.42, 1.42
x = np.linspace(xmin, xmax, N)
y = np.linspace(xmin, xmax, N)
d = (xmax - xmin) / (N - 1)
X, Y = np.meshgrid(x, y, indexing="xy")

# cx, cy, ell, b, amplitude, phi(deg)
ridges = [
    ( 0.00,  0.00, 0.62, 0.085, 1.00,  22),
    (-0.30,  0.34, 0.50, 0.075, 0.86, 100),
    ( 0.34, -0.30, 0.46, 0.070, 0.80, 158),
    ( 0.46,  0.44, 0.34, 0.065, 0.62,  70),
    (-0.52, -0.34, 0.36, 0.065, 0.58,  10),
    (-0.10, -0.62, 0.30, 0.060, 0.45, 130),
]

g0 = np.zeros_like(X)
for cx, cy, ell, b, amp, phi_deg in ridges:
    phi = np.deg2rad(phi_deg)
    u = (X-cx)*np.cos(phi) + (Y-cy)*np.sin(phi)
    v = -(X-cx)*np.sin(phi) + (Y-cy)*np.cos(phi)
    g0 += amp*np.exp(-0.5*((u/ell)**4 + (v/b)**2))

g = g0 / g0.max()

L = 1.24
w = 0.124
records = []

for j in range(60):
    theta = j*np.pi/60.0

    # Only shifts capable of lying inside the L x w rectangle are needed.
    R = int(np.ceil((L/2 + w/2)/d)) + 2
    ind = np.arange(-R, R+1)
    DX, DY = np.meshgrid(ind*d, ind*d, indexing="xy")
    para = DX*np.cos(theta) + DY*np.sin(theta)
    perp = -DX*np.sin(theta) + DY*np.cos(theta)
    mask = (np.abs(para) <= L/2) & (np.abs(perp) <= w/2)

    tube_mean = fftconvolve(g, mask[::-1, ::-1].astype(float), mode="same") / mask.sum()

    # Translation domain in Methods: Model landscape and scan in Figure 1.
    sub = tube_mean[76:264, 76:264]
    iy, ix = np.unravel_index(np.argmax(sub), sub.shape)
    iy += 76
    ix += 76

    records.append({
        "direction_index": j,
        "theta_deg": 180.0*theta/np.pi,
        "theta_rad": theta,
        "max_tube_mean": float(tube_mean[iy, ix]),
        "normalized_response": 0.0,
        "center_x_over_ellstar": float(x[ix]),
        "center_y_over_ellstar": float(y[iy]),
        "tube_mask_points": int(mask.sum()),
    })

mx = max(r["max_tube_mean"] for r in records)
for r in records:
    r["normalized_response"] = r["max_tube_mean"] / mx

df = pd.DataFrame(records)
df.to_csv("source_data_fig1.csv", index=False)

np.savez_compressed(
    "source_data_fig1_landscape.npz",
    x=x, y=y, normalized_extinction=g,
    L_over_ellstar=L, w_over_ellstar=w
)

print(f"min={df.max_tube_mean.min():.12f}")
print(f"max={df.max_tube_mean.max():.12f}")
print(f"ratio={df.max_tube_mean.max()/df.max_tube_mean.min():.12f}")
print(f"max_direction_deg={df.loc[df.max_tube_mean.idxmax(), 'theta_deg']:.1f}")
