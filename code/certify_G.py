"""
Certified evaluation of the weighted-sieve functional of Xie (arXiv:2504.18732, (4.9)) at level alpha,

    G(theta, delta, lambda) = f(alpha/theta) - lambda * int_theta^delta F((alpha - t)/theta) (1 - t/delta) dt/t,

in the parameter range  alpha/4 <= theta < delta < alpha  (so 3 <= alpha/theta <= 4 and 0 < (alpha-t)/theta <= 3 on
the whole range of integration), where the linear-sieve functions are given by their closed forms
    F(s) = 2 e^gamma / s            (0 < s <= 3),
    f(s) = 2 e^gamma log(s-1) / s   (2 <= s <= 4).
Partial fractions  (1 - t/delta) / (t (alpha - t)) = (1/alpha) [ 1/t + (1 - alpha/delta) / (alpha - t) ]  give

    G = (2 e^gamma theta / alpha) * [ log(alpha/theta - 1) - lambda * ( log(delta/theta) - (alpha/delta - 1) log((alpha-theta)/(alpha-delta)) ) ].

All parameters are rationals; the bracket is evaluated with mpmath interval arithmetic (iv), so the printed lower
bound G_lo is rigorous.  The sieve conclusion (Xie (4.8)) is  Omega(a) < 1/lambda + (l-1)/delta  for the elements
a ~ x^{l-1} of the sifted sequence, provided G > 0.

Usage:  python certify_G.py            (reproduces the table in the paper)
        python certify_G.py alpha theta delta inv_lambda   (rationals as strings, e.g. 0.499 1/8 9/20 17/10)
"""
import sys
from fractions import Fraction
from mpmath import iv, mp

iv.prec = 100
mp.prec = 100


def R(fr):
    """exact interval for a Fraction."""
    return iv.mpf(fr.numerator) / iv.mpf(fr.denominator)


def G_interval(alpha, theta, delta, inv_lambda):
    alpha, theta, delta, inv_lambda = (Fraction(v) for v in (alpha, theta, delta, inv_lambda))
    assert alpha / 4 <= theta < delta < alpha, "need alpha/4 <= theta < delta < alpha for the closed form"
    assert theta <= alpha / 2
    a, th, de, il = R(alpha), R(theta), R(delta), R(inv_lambda)
    lam = 1 / il
    pref = 2 * iv.exp(iv.euler) * th / a
    bracket = iv.log(a / th - 1) - lam * (iv.log(de / th) - (a / de - 1) * iv.log((a - th) / (a - de)))
    return pref * bracket


def report(alpha, theta, delta, inv_lambda, ells=(2, 3, 5, 7, 11, 13)):
    Gi = G_interval(alpha, theta, delta, inv_lambda)
    al, th, de, il = (Fraction(v) for v in (alpha, theta, delta, inv_lambda))
    print(f"alpha = {alpha}, theta = {theta} (alpha/theta = {float(al / th):.4f}), delta = {delta}, "
          f"1/lambda = {inv_lambda}")
    print(f"   G in [{float(Gi.a):+.6f}, {float(Gi.b):+.6f}]  ->  {'POSITIVE' if Gi.a > 0 else 'NOT certified'}")
    for ell in ells:
        b = il + (ell - 1) / de
        print(f"   l = {ell:2d}: 1/lambda + (l-1)/delta = {b} = {float(b):.4f}  ->  Omega <= {int(b) if b.denominator > 1 else int(b) - 1}"
              f"{'  (integer value: increase lambda slightly)' if b.denominator == 1 else ''}")
    return Gi


if __name__ == "__main__":
    if len(sys.argv) == 5:
        report(*sys.argv[1:5])
    else:
        print("Parameters used in the paper (level alpha = 0.499):")
        report("0.499", "1/8", "9/20", "17/10", ells=(2,))
        report("0.499", "1/8", "20/41", "6/5")
        print("\nAlternative at alpha = 0.49 with alpha/theta = 4 exactly:")
        report("0.49", "49/400", "9/20", "17/10", ells=(2,))
        report("0.49", "49/400", "20/41", "6/5", ells=(3,))
