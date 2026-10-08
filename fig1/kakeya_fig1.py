#!/usr/bin/env python3
"""Fig. 1 for the Kakeya PRL -- all geometry computed, not hand-placed."""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.transforms import Affine2D
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from scipy.signal import fftconvolve

mpl.rcParams.update({
    "text.usetex": True, "font.family": "serif",
    "text.latex.preamble": r"\usepackage{amsmath}\usepackage{bm}",
    "font.size": 7.6, "axes.linewidth": 0.5, "lines.linewidth": 0.7,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 1.8, "ytick.major.size": 1.8,
    "xtick.labelsize": 7.0, "ytick.labelsize": 7.0,
})

PEACH_E = "#b8641f"; STEEL = "#1f4e79"; STEEL_F = "#bcd6ec"
CRIMSON = "#b4232a"; INK = "#1b1b1b"; GREY = "#8d8d8d"
GAMMAP = LinearSegmentedColormap.from_list(
    "gam", ["#ffffff", "#fdeada", "#fac89b", "#ee9d55"])
CMAP = plt.get_cmap("twilight")

# ---------------------------------------------------------- the landscape
N, R = 340, 1.42
xs = np.linspace(-R, R, N); ys = np.linspace(-R, R, N)
X, Y = np.meshgrid(xs, ys); h = xs[1] - xs[0]

def ridge(cx, cy, ln, wd, amp, rot):
    c, s = np.cos(rot), np.sin(rot)
    u = (X - cx) * c + (Y - cy) * s
    v = -(X - cx) * s + (Y - cy) * c
    return amp * np.exp(-0.5 * ((u / ln) ** 4 + (v / wd) ** 2))

# an anisotropic patterned sink: a few oriented ridges, so the angular
# response has real structure rather than being nearly isotropic
GAM = (ridge( 0.00, 0.00, 0.62, 0.085, 1.00, np.deg2rad(  22))
     + ridge(-0.30, 0.34, 0.50, 0.075, 0.86, np.deg2rad( 100))
     + ridge( 0.34,-0.30, 0.46, 0.070, 0.80, np.deg2rad( 158))
     + ridge( 0.46, 0.44, 0.34, 0.065, 0.62, np.deg2rad(  70))
     + ridge(-0.52,-0.34, 0.36, 0.065, 0.58, np.deg2rad(  10))
     + ridge(-0.10,-0.62, 0.30, 0.060, 0.45, np.deg2rad( 130)))
GAM /= GAM.max()

L, W = 1.24, 0.124
NT = 60
def kern(th):
    half = int(np.ceil(L / (2 * h))) + 3
    a = np.arange(-half, half + 1) * h
    KX, KY = np.meshgrid(a, a)
    c, s = np.cos(th), np.sin(th)
    k = ((np.abs(KX * c + KY * s) <= L / 2) &
         (np.abs(-KX * s + KY * c) <= W / 2)).astype(float)
    return k / k.sum()

thetas = np.linspace(0.0, np.pi, NT, endpoint=False)
D = np.empty(NT); CX = np.empty(NT); CY = np.empty(NT)
pad = int(L / (2 * h)) + 2
for i, th in enumerate(thetas):
    avg = fftconvolve(GAM, kern(th), mode="same")
    m = np.full_like(avg, -np.inf); m[pad:-pad, pad:-pad] = avg[pad:-pad, pad:-pad]
    j = np.unravel_index(np.argmax(m), m.shape)
    D[i] = m[j]; CY[i] = ys[j[0]]; CX[i] = xs[j[1]]
print(f"delta = {W/L:.3f}   D in [{D.min():.3f}, {D.max():.3f}]  "
      f"anisotropy = {D.max()/D.min():.2f}x")

# =============================================================== drawing
import matplotlib.patheffects as pe
HALO = [pe.withStroke(linewidth=1.6, foreground="white")]

def tube(ax, cx, cy, th, fc, ec, lw=0.6, ls="-", z=3, alpha=1.0):
    r = Rectangle((-L/2, -W/2), L, W, facecolor=fc, edgecolor=ec, linewidth=lw,
                  linestyle=ls, zorder=z, alpha=alpha, joinstyle="miter")
    r.set_transform(Affine2D().rotate(th).translate(cx, cy) + ax.transData)
    ax.add_patch(r); return r

def land(ax, a=1.0):
    ax.imshow(GAM, extent=[-R, R, -R, R], origin="lower", cmap=GAMMAP,
              vmin=0, vmax=1.02, alpha=a, zorder=0, interpolation="bilinear")
    ax.contour(X, Y, GAM, levels=[0.25, 0.55, 0.85], colors=PEACH_E,
               linewidths=0.30, alpha=0.70*a, zorder=1)

fig = plt.figure(figsize=(7.05, 2.08))
axA = fig.add_axes([0.002, 0.012, 0.328, 0.960])
axB = fig.add_axes([0.335, 0.012, 0.328, 0.960])
cax = fig.add_axes([0.4153, 0.055, 0.138, 0.030])  # bar+label centred under (b)
axC = fig.add_axes([0.712, 0.075, 0.258, 0.820], projection="polar")

# ------------------------------------------------------------------- (a)
k  = int(np.argmax(D)); th0 = thetas[k]; cx0, cy0 = CX[k], CY[k]
e  = np.array([np.cos(th0), np.sin(th0)])
ep = np.array([-np.sin(th0), np.cos(th0)])
C0 = np.array([cx0, cy0])
land(axA)
Z = 0.86                                        # zoom half-width
axA.set_xlim(cx0 - Z, cx0 + Z); axA.set_ylim(cy0 - Z, cy0 + Z)
axA.set_aspect("equal"); axA.axis("off")

GHOST = ((0.31, 0.02), (0.05, 0.34))             # transverse and longitudinal
for dt, dl in GHOST:
    tube(axA, *(C0 + dt*ep + dl*e), th0, "none", GREY, 0.45, (0, (2.4, 1.9)), 2)
tube(axA, cx0, cy0, th0, STEEL_F, STEEL, 0.85, "-", 4)
for t in (-0.30, 0.30):
    p0 = C0 + t*W*ep - (L/2)*e
    axA.plot(*np.c_[p0, p0 + L*e], color=STEEL, lw=0.35, alpha=0.8, zorder=5)

pa = C0 - (L/2 - 0.03)*e
axA.add_patch(FancyArrowPatch(pa, pa + 0.30*e, arrowstyle="-|>",
              mutation_scale=6.5, lw=0.95, color=STEEL, zorder=6))
axA.text(*(pa + 0.17*e - 0.105*ep), r"$\hat{\bm e}$", color=STEEL,
         fontsize=8.0, ha="center", va="center", zorder=7, path_effects=HALO)

for t, sgn in ((-0.55, +1), (0.05, +1), (0.62, +1), (-0.32, -1), (0.62, -1)):
    ep_ = sgn*ep                                 # escape leaves the channel
    b = C0 + t*(L/2)*e + sgn*(W/2)*ep
    u = np.linspace(0, 1, 30)
    wg = b[:, None] + u*0.17*ep_[:, None] + 0.015*np.sin(u*3.2*np.pi)*e[:, None]
    axA.plot(wg[0], wg[1], color=CRIMSON, lw=0.6, zorder=6)
    axA.add_patch(FancyArrowPatch(wg[:, -2], wg[:, -1], arrowstyle="-|>",
                  mutation_scale=4.6, lw=0.55, color=CRIMSON, zorder=6))
axA.text(*(C0 - 0.78*e - 0.20*ep), r"$\gamma(\bm r)$", color=CRIMSON,
         fontsize=8.2, ha="center", va="center", zorder=7, path_effects=HALO)

q0 = C0 - (L/2)*e - 0.265*ep                     # L
axA.annotate("", q0 + L*e, q0, arrowprops=dict(arrowstyle="<|-|>", lw=0.55,
             color=INK, mutation_scale=4.6, shrinkA=0, shrinkB=0))
axA.text(*(q0 + 0.5*L*e - 0.055*ep), r"$L$", fontsize=7.6, ha="center",
         va="center", color=INK, rotation=np.rad2deg(th0), zorder=7,
         path_effects=HALO)
m0 = C0 + (L/2 + 0.075)*e - (W/2)*ep             # w
axA.annotate("", m0 + W*ep, m0, arrowprops=dict(arrowstyle="<|-|>", lw=0.55,
             color=INK, mutation_scale=3.8, shrinkA=0, shrinkB=0))
axA.text(*(m0 + 0.5*W*ep + 0.065*e), r"$w$", fontsize=7.6, ha="left",
         va="center", color=INK, zorder=7, path_effects=HALO)

for dt, dl in GHOST:                             # centres of the translates
    c = C0 + dt*ep + dl*e
    axA.plot(*c, "o", ms=2.1, mfc="white", mec=GREY, mew=0.6, zorder=6)
    axA.annotate("", c, C0, arrowprops=dict(arrowstyle="-|>", lw=0.6, color=GREY,
                 mutation_scale=4.6, shrinkA=2.2, shrinkB=2.2), zorder=6)
axA.plot(*C0, "o", ms=2.6, mfc=STEEL, mec="white", mew=0.6, zorder=8)
axA.text(*(C0 - 0.155*ep + 0.30*e), r"$\bm r_0$", color=STEEL,
         fontsize=8.2, ha="center", va="center", zorder=9, path_effects=HALO)
axA.text(0.0, 1.0, r"(a)", transform=axA.transAxes, ha="left", va="top",
         fontsize=8.4, color=INK)

# ------------------------------------------------------------------- (b)
land(axB, 0.85)
_hx = 0.5*L*np.abs(np.cos(thetas)) + 0.5*W*np.abs(np.sin(thetas))
_hy = 0.5*L*np.abs(np.sin(thetas)) + 0.5*W*np.abs(np.cos(thetas))
_bx = max(np.abs(CX + _hx).max(), np.abs(CX - _hx).max())
_by = max(np.abs(CY + _hy).max(), np.abs(CY - _hy).max())
_b  = min(R, max(_bx, _by) + 0.06)
axB.set_xlim(-_b, _b); axB.set_ylim(-_b, _b)
axB.set_aspect("equal"); axB.axis("off")
for i in np.argsort(D):
    c = CMAP(thetas[i]/np.pi)
    tube(axB, CX[i], CY[i], thetas[i], c, "none", 0, "-", 3, 0.12)
for i in range(0, NT, 3):
    c = CMAP(thetas[i]/np.pi)
    tube(axB, CX[i], CY[i], thetas[i], "none", c, 0.45, "-", 4, 0.95)
axB.text(0.0, 1.0, r"(b)", transform=axB.transAxes, ha="left", va="top",
         fontsize=8.4, color=INK)

sm = ScalarMappable(cmap=CMAP, norm=mpl.colors.Normalize(0, np.pi))
cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
cb.outline.set_linewidth(0.4)
cb.set_ticks([0, np.pi/2, np.pi])
cb.set_ticklabels([r"$0$", r"$\pi/2$", r"$\pi$"])
cb.ax.xaxis.set_ticks_position("top")
cb.ax.xaxis.set_label_position("top")
cb.ax.tick_params(width=0.4, length=1.6, labelsize=7.0, pad=0.8)
cax.text(1.14, 0.45, r"$\hat{\bm e}$", transform=cax.transAxes,
         ha="left", va="center", fontsize=8.2, color=INK)

# ------------------------------------------------------------------- (c)
Dn = D / D.max()                                 # model gamma: normalise
tt = np.concatenate([thetas, thetas + np.pi, [thetas[0]]])
dd = np.concatenate([Dn, Dn, [Dn[0]]])
axC.fill(tt, dd, color=STEEL_F, alpha=0.6, zorder=2)
axC.plot(tt, dd, color=INK, lw=0.7, zorder=4)
for i in range(0, NT, 2):
    for o in (0.0, np.pi):
        axC.plot([thetas[i] + o], [Dn[i]], "o", ms=1.8,
                 color=CMAP(thetas[i]/np.pi), zorder=5)
axC.set_ylim(0, 1.14); axC.set_yticks([])
axC.grid(alpha=0.20, lw=0.3)
axC.set_xticks(np.deg2rad([0, 90, 180, 270]))
axC.set_xticklabels([r"$0$", "", r"$\pi$", ""], fontsize=7.0)
axC.tick_params(pad=-1.5)
axC.spines["polar"].set_linewidth(0.45)
fig.text(0.686, 0.012 + 0.960, r"(c)", ha="left", va="top",
         fontsize=8.4, color=INK)          # same baseline as (a) and (b)
axC.text(0.5, 1.05, r"$\mathcal{D}_{L,w}(\hat{\bm e})/\mathcal{D}^{\max}$",
         transform=axC.transAxes, ha="center", va="bottom", fontsize=7.8, color=INK)

import os
OUT = os.path.dirname(os.path.abspath(__file__)) or "."
fig.savefig(os.path.join(OUT, "kakeya_fig1.pdf"))
fig.savefig(os.path.join(OUT, "kakeya_fig1.png"), dpi=430)
print("wrote", os.path.join(OUT, "kakeya_fig1.pdf"))
