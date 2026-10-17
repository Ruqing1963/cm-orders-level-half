"""
Linear-sieve functions F (upper) and f (lower) of Jurkat-Richert / Iwaniec, floating point, for exploration.

    F(s) = 2 e^gamma / s                     for 0 < s <= 3,
    f(s) = 2 e^gamma log(s - 1) / s          for 2 <= s <= 4,
    (s F(s))' = f(s - 1)  for s > 3,         (s f(s))' = F(s - 1)  for s > 4,
    F(s), f(s) -> 1 as s -> infinity.

Implemented on a grid of step h by integrating the delay-differential equations (trapezoid rule).  Accuracy ~1e-6
for h = 1e-4; this module is for exploration only.  Certified values (interval arithmetic) are produced in
certify_G.py, where the parameters are chosen so that only the closed forms above are needed.
"""
import numpy as np

EG = float(np.exp(np.euler_gamma))
H = 1e-4
S_MAX = 40.0

_s = np.arange(0.0, S_MAX + H / 2, H)
_sF = np.empty_like(_s)         # s F(s)
_sf = np.zeros_like(_s)         # s f(s)
_sF[0] = 2 * EG
_i3, _i2, _i4 = int(round(3 / H)), int(round(2 / H)), int(round(4 / H))
_sF[: _i3 + 1] = 2 * EG
_sf[_i2: _i4 + 1] = 2 * EG * np.log(np.maximum(_s[_i2: _i4 + 1] - 1, 1e-300))
_sf[_i2] = 0.0
_di = int(round(1 / H))
for i in range(_i3, len(_s) - 1):
    # (sF)' = f(s-1) = sf(s-1)/(s-1)
    g0 = _sf[i - _di] / _s[i - _di]
    g1 = _sf[i + 1 - _di] / _s[i + 1 - _di]
    _sF[i + 1] = _sF[i] + H * (g0 + g1) / 2
    if i >= _i4:
        F0 = _sF[i - _di] / _s[i - _di]
        F1 = _sF[i + 1 - _di] / _s[i + 1 - _di]
        _sf[i + 1] = _sf[i] + H * (F0 + F1) / 2


def F(s):
    s = np.asarray(s, dtype=float)
    out = np.where(s <= 3, 2 * EG / np.maximum(s, 1e-12), 1.0)
    big = s > 3
    if np.any(big):
        sb = np.minimum(s[big], S_MAX)
        out = out.copy()
        out[big] = np.interp(sb, _s, _sF) / sb
    return out if out.ndim else float(out)


def f(s):
    s = np.asarray(s, dtype=float)
    out = np.zeros_like(s)
    mid = (s >= 2) & (s <= 4)
    out[mid] = 2 * EG * np.log(s[mid] - 1) / s[mid]
    big = s > 4
    if np.any(big):
        sb = np.minimum(s[big], S_MAX)
        out[big] = np.interp(sb, _s, _sf) / sb
    return out if out.ndim else float(out)


if __name__ == "__main__":
    print("F(3) =", F(3.0), " (2e^gamma/3 =", 2 * EG / 3, ")")
    print("f(4) =", f(4.0), " (2e^gamma log3/4 =", 2 * EG * np.log(3) / 4, ")")
    print("F(4) =", F(4.0), " f(5) =", f(5.0), " F(10) =", F(10.0), " f(10) =", f(10.0))
