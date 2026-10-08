"""Finite-width attenuation, with L = v_F = 1 unless specified.

The spatial field is zero outside the supplied grid. Full linear convolution
scans every center whose rectangle intersects that grid. Angles cover [0, pi);
the reported L2 norm uses arc-length measure on the full circle.
"""
from dataclasses import dataclass

import numpy as np
from scipy.fft import rfft2, irfft2, next_fast_len
from scipy.integrate import quad


def angular_occupancy(r, w, L=1.0):
    """Fraction of centered, rotated L x w rectangles containing radius r."""
    r = np.asarray(r, dtype=float)
    safe_r = np.maximum(r, np.finfo(float).tiny)
    upper = np.arcsin(np.minimum(1.0, w / (2.0 * safe_r)))
    lower = np.arccos(np.minimum(1.0, L / (2.0 * safe_r)))
    return (2.0 / np.pi) * np.maximum(0.0, upper - lower)


def radial_integral(fn, w, L=1.0):
    edges = [0.0, w/2.0, L/2.0, np.hypot(L, w)/2.0]
    return sum(quad(lambda r: 2*np.pi*r*fn(r), a, b,
                    epsabs=1e-12, epsrel=1e-10)[0]
               for a, b in zip(edges[:-1], edges[1:]))


def exact_q(w, L=1.0):
    return np.sqrt(2*np.pi*radial_integral(
        lambda r: angular_occupancy(r, w, L)**2, w, L))/w


def exact_q_closed(w, L=1.0):
    """Exact rectangular coefficient for 0 < w/L <= 1."""
    delta = w/L
    if delta == 1.0:
        return np.sqrt(8*np.log(2))
    if delta <= 0.01:
        # The even series evaluates the closed expression to double precision.
        d2 = delta*delta
        regular = 6-d2*(1/3+d2*(1/15+d2*(1/42+d2/90)))
        return np.sqrt(regular-4*np.log(delta))
    value = 2*((1+delta)**2*np.log1p(delta)
               +(1-delta)**2*np.log1p(-delta))/delta**2 - 4*np.log(delta)
    return np.sqrt(value)


def rectangle(theta, w, L=1.0, center=(0.0, 0.0)):
    c, s = np.cos(theta), np.sin(theta)
    vertices = np.array([[-L/2, -w/2], [L/2, -w/2],
                         [L/2, w/2], [-L/2, w/2]])
    return vertices @ np.array([[c, s], [-s, c]]) + center


def polygon_intersection(subject, clip):
    """Sutherland-Hodgman clipping for counterclockwise convex polygons."""
    out = [p.copy() for p in subject]
    cross = lambda a, b: a[0]*b[1] - a[1]*b[0]
    for a, b in zip(clip, np.roll(clip, -1, axis=0)):
        inp, out = out, []
        if len(inp) == 0:
            break
        edge = b-a
        prev = inp[-1]
        prev_dist = cross(edge, prev-a)
        for cur in inp:
            cur_dist = cross(edge, cur-a)
            if (cur_dist >= 0) != (prev_dist >= 0):
                out.append(prev+(cur-prev)*prev_dist/(prev_dist-cur_dist))
            if cur_dist >= 0:
                out.append(cur)
            prev, prev_dist = cur, cur_dist
    return np.asarray(out)


def intersection_area(theta, w, L=1.0, shift=(0.0, 0.0)):
    p = polygon_intersection(rectangle(0.0, w, L),
                             rectangle(theta, w, L, shift))
    if len(p) < 3:
        return 0.0
    return abs(np.sum(p[:, 0]*np.roll(p[:, 1], -1)
                      - p[:, 1]*np.roll(p[:, 0], -1)))/2


def exact_q_overlap(w, L=1.0):
    angular_integral = 4*quad(lambda th: intersection_area(th, w, L),
        0, np.pi/2, epsabs=1e-11, epsrel=1e-10,
        points=[2*np.arctan(w/L)], limit=300)[0]
    return np.sqrt(angular_integral)/w


@dataclass
class Scan:
    response: np.ndarray
    centers: np.ndarray
    gradient: np.ndarray
    q: float


class TubeScanner:
    def __init__(self, n=193, radius=0.8, w=0.05, nangles=60,
                 oversample=4, angle_offset=0.0, L=1.0):
        self.n, self.w, self.L = n, w, L
        self.x = np.linspace(-radius, radius, n)
        self.h = self.x[1]-self.x[0]
        self.theta = (np.arange(nangles)+angle_offset)*np.pi/nangles
        self.X, self.Y = np.meshgrid(self.x, self.x)
        half = int(np.ceil(np.hypot(L, w)/(2*self.h)))+1
        self.half = half
        offsets = np.arange(-half, half+1)*self.h
        kx, ky = np.meshgrid(offsets, offsets)
        subs = ((np.arange(oversample)+0.5)/oversample-0.5)*self.h
        kernels = []
        for th in self.theta:
            c, s = np.cos(th), np.sin(th)
            k = np.zeros_like(kx)
            for ox in subs:
                for oy in subs:
                    u = (kx+ox)*c+(ky+oy)*s
                    v = -(kx+ox)*s+(ky+oy)*c
                    k += (np.abs(u) <= L/2) & (np.abs(v) <= w/2)
            kernels.append(k/k.sum())
        self.kernels = np.asarray(kernels)
        self.full_n = n + 2*half
        self.fft_n = next_fast_len(self.full_n)
        self.kernel_fft = rfft2(self.kernels, s=(self.fft_n, self.fft_n))

    def scan(self, field, gradient=False):
        ft = rfft2(field, s=(self.fft_n, self.fft_n))
        values, centers = [], []
        grad = np.zeros_like(field)
        for k, fk in zip(self.kernels, self.kernel_fft):
            conv = irfft2(ft*fk, s=(self.fft_n, self.fft_n))[:self.full_n, :self.full_n]
            ij = np.unravel_index(np.argmax(conv), conv.shape)
            value = conv[ij]
            values.append(value*self.L)
            center_index = np.array(ij)-self.half
            centers.append(self.x[0]+self.h*center_index[::-1])
            if gradient:
                start = center_index-self.half
                stop = start+len(k)
                lo, hi = np.maximum(start, 0), np.minimum(stop, self.n)
                klo, khi = lo-start, hi-start
                grad[lo[0]:hi[0], lo[1]:hi[1]] += value*k[klo[0]:khi[0], klo[1]:khi[1]]
        values = np.asarray(values)
        norm = np.linalg.norm(field)*self.h
        q = np.sqrt(2*np.pi*np.mean(values**2))/norm
        return Scan(values, np.asarray(centers), grad, float(q))


def rearrange(values, score):
    """Maximize a linear functional over permutations of fixed pixel values."""
    out = np.empty(values.size)
    out[np.argsort(score, axis=None, kind='stable')] = np.sort(values, axis=None)
    return out.reshape(values.shape)


def optimize(scanner, field, steps=30, constraint='histogram'):
    history = []
    norm = np.linalg.norm(field)
    values = field.copy()
    for iteration in range(steps+1):
        result = scanner.scan(field, gradient=True)
        history.append(result.q)
        if iteration == steps:
            break
        if constraint == 'histogram':
            field = rearrange(values, result.gradient)
        elif constraint == 'l2':
            field = result.gradient*(norm/np.linalg.norm(result.gradient))
    return field, result, np.asarray(history)


def equal_area_shape(scanner, area, shape):
    X, Y = scanner.X, scanner.Y
    if shape == 'disk':
        score = -(X**2+Y**2)
    elif shape == 'stripe':
        score = -np.maximum(np.abs(X)/0.5, np.abs(Y)/(area/2))
    elif shape == 'deltoid':
        # Implicit 3-cusped hypocycloid with boundary
        # x+iy = r*(2*exp(it)+exp(-2it)), area 2*pi*r^2.
        r = np.sqrt(area/(2*np.pi))
        q = (X**2+Y**2)**2 + 18*r*r*(X**2+Y**2) - 8*r*(X**3-3*X*Y**2)-27*r**4
        score = -q
    count = round(area/scanner.h**2)
    values = np.zeros_like(X)
    values.flat[:count] = 1.0
    return rearrange(values, score)
