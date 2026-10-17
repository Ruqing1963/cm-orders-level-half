"""
First-principles numerical check of the bookkeeping in prop51_rederivation.md, Sec. 7 (and of IJU Sec. 3/5):

  * build the switched set  Omega(x) = {(1+i)^3 eps pi1 pi2 pi3 : z <= N(pi3) < y <= N(pi2) < N(pi1), N <= x}
    and the primary primes pi with N(pi) <= x  (sequence A);
  * check the inclusion-exclusion identity (7.1) for split p:  #{p | N(1+w)} = #{pi|1+w} + #{conj pi|1+w} - #{p|1+w};
  * compare B_d = #{w : d | N(1+w)} with |Omega| g(d),  and A_d = #{pi : d | N(pi-1)/8} with Pi(x) g(d),
    g(q) = 1/(q^2-1) (inert), g(p) = 2/(p-1) - 1/(p-1)^2 (split);
  * look at the discrepancy  E(x; alpha, a) = #{w = alpha mod a} - #{(w,a)=1}/Phi(a)  for a few ideals a.
Everything is plain counting; no sieve theory is assumed.
"""
import sys
import numpy as np

x = int(float(sys.argv[1])) if len(sys.argv) > 1 else 10 ** 8
z = x ** (1 / 8)
y = x ** (1 / 3)
Lx = np.log(x)


def primes_upto(n):
    s = np.ones(n + 1, dtype=bool)
    s[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]:
            s[i * i::i] = False
    return np.nonzero(s)[0]


def primary_primes(bound):
    """all primary Gaussian primes a+bi (b even, a+b = 1 mod 4) with a^2+b^2 <= bound, as (a, b, p) sorted by p."""
    isp = np.zeros(bound + 1, dtype=bool)
    isp[primes_upto(bound)] = True
    out = []
    B = int(bound ** 0.5) + 2
    B += B % 2                                              # b runs over EVEN integers
    for b in range(-B, B + 1, 2):
        amax = int((bound - b * b) ** 0.5) if b * b <= bound else -1
        if amax < 0:
            continue
        a = np.arange(-amax, amax + 1)
        a = a[(a % 2 != 0) & ((a + b) % 4 == 1)]
        p = a * a + b * b
        ok = isp[p]
        out.append(np.stack([a[ok], np.full(ok.sum(), b), p[ok]], axis=1))
    P = np.concatenate(out)
    return P[np.argsort(P[:, 2], kind="stable")]


# ---------- sequence A: primary primes with norm <= x ----------
PA = primary_primes(x)
Pi_x = len(PA)
print(f"x = {x:.3e}: Pi(x) = #primary primes = {Pi_x}  (x/log x = {x / Lx:.0f})")
assert Pi_x % 2 == 0 and np.all(PA[0::2, 2] == PA[1::2, 2]), "each split p must have exactly two primary primes"
a_A = ((PA[:, 0] - 1) ** 2 + PA[:, 1] ** 2) // 8          # N(pi-1)/8
assert np.all(((PA[:, 0] - 1) ** 2 + PA[:, 1] ** 2) % 8 == 0)

# ---------- switched set Omega(x) ----------
bound1 = int(x / (8 * z * y)) + 1
PS = primary_primes(bound1)
norms = PS[:, 2]
p3 = PS[(norms >= z) & (norms < y)]
om = []
for a3, b3, n3 in p3:
    n2max = (x / (8 * n3)) ** 0.5
    sel2 = PS[(norms >= y) & (norms < n2max)]
    for a2, b2, n2 in sel2:
        n1max = x / (8 * n3 * n2)
        sel1 = PS[(norms > n2) & (norms <= n1max)]
        if len(sel1) == 0:
            continue
        # beta = pi1 * pi2 * pi3 (vectorised over pi1)
        c23r, c23i = a2 * a3 - b2 * b3, a2 * b3 + b2 * a3
        br = sel1[:, 0] * c23r - sel1[:, 1] * c23i
        bi = sel1[:, 0] * c23i + sel1[:, 1] * c23r
        # (1+i)^3 = -2 + 2i ;  times units eps in {1, i, -1, -i}
        wr, wi = -2 * br - 2 * bi, 2 * br - 2 * bi
        for _ in range(4):
            om.append(np.stack([wr, wi], axis=1))
            wr, wi = -wi, wr                                 # multiply by i
W = np.concatenate(om).astype(np.int64)
assert np.all(W[:, 0] ** 2 + W[:, 1] ** 2 <= x)
nW = len(W)
uniq = len(np.unique(W[:, 0] * (4 * int(x ** 0.5) + 10) + W[:, 1]))
print(f"|Omega(x)| = {nW}, distinct = {uniq} (injectivity (0.1): {'OK' if uniq == nW else 'FAIL'})")
c_chen = 0.36308372924830460041
print(f"|Omega|/Pi(x) = {nW / Pi_x:.4f}   (asymptotic upper bound c/2 = {c_chen / 2:.4f}; slow convergence expected)")
normB = (W[:, 0] + 1) ** 2 + W[:, 1] ** 2                   # N(1+w)
assert np.all(normB % 2 == 1), "N(1+w) must be odd"


def g_of(d, fac):
    v = 1.0
    for p in fac:
        v *= 1 / (p * p - 1) if p % 4 == 3 else 2 / (p - 1) - 1 / (p - 1) ** 2
    return v


def divisible_by_ideal(wr, wi, gr, gi):
    """w = 0 mod (gr + i gi):  w * conj(gamma) = 0 mod N(gamma) in both components."""
    n = gr * gr + gi * gi
    return ((wr * gr + wi * gi) % n == 0) & ((wi * gr - wr * gi) % n == 0)


# ---------- identity (7.1) for split p, both sequences ----------
print("\nIdentity (7.1), split p: #{p | N(1+w)} vs #{pi|1+w} + #{pibar|1+w} - #{p|1+w}   [and same for pi-1 in A]")
small = primary_primes(400)
for p in [5, 13, 17, 29, 37, 41]:
    a, b = small[small[:, 2] == p][0][:2]
    lhsB = int(np.sum(normB % p == 0))
    t1 = int(divisible_by_ideal(W[:, 0] + 1, W[:, 1], a, b).sum())
    t2 = int(divisible_by_ideal(W[:, 0] + 1, W[:, 1], a, -b).sum())
    t3 = int((((W[:, 0] + 1) % p == 0) & (W[:, 1] % p == 0)).sum())
    lhsA = int(np.sum((8 * a_A) % p == 0))
    s1 = int(divisible_by_ideal(PA[:, 0] - 1, PA[:, 1], a, b).sum())
    s2 = int(divisible_by_ideal(PA[:, 0] - 1, PA[:, 1], a, -b).sum())
    s3 = int((((PA[:, 0] - 1) % p == 0) & (PA[:, 1] % p == 0)).sum())
    print(f"  p={p:3d}: B: {lhsB} = {t1} + {t2} - {t3} -> {t1 + t2 - t3}  {'OK' if lhsB == t1 + t2 - t3 else 'FAIL'}"
          f"   | A: {lhsA} = {s1} + {s2} - {s3} -> {s1 + s2 - s3}  {'OK' if lhsA == s1 + s2 - s3 else 'FAIL'}")

# ---------- densities g(d) ----------
print("\nDensities: B_d/|Omega| and A_d/Pi(x) against g(d)   (d odd square-free)")
print(f"{'d':>6} {'factors':>12} {'g(d)':>10} {'B_d/|Om|':>10} {'ratio':>7} {'A_d/Pi':>10} {'ratio':>7}")
tests = [(3, [3]), (5, [5]), (7, [7]), (11, [11]), (13, [13]), (15, [3, 5]), (21, [3, 7]), (35, [5, 7]),
         (39, [3, 13]), (65, [5, 13]), (105, [3, 5, 7]), (195, [3, 5, 13]), (221, [13, 17]), (1105, [5, 13, 17])]
for d, fac in tests:
    g = g_of(d, fac)
    Bd = np.sum(normB % d == 0) / nW
    Ad = np.sum(a_A % d == 0) / Pi_x
    print(f"{d:>6} {str(fac):>12} {g:10.6f} {Bd:10.6f} {Bd / g:7.4f} {Ad:10.6f} {Ad / g:7.4f}")

# ---------- the coprimality term of Sec. 7 (iii): for split p in the pi3-range, many w are divisible by pi_p ----------
print("\nSplit p: B_p against the prediction WITH the coprimality correction,"
      "  2 #{(w,pi)=1}/(p-1) - #{(w,p)=1}/(p-1)^2   [z <= p < y means p can be N(pi3)]")
print(f"  z = {z:.1f}, y = {y:.1f}")
for p in [5, 13, 17, 29, 37, 41, 53]:
    a, b = small[small[:, 2] == p][0][:2]
    Bp = int(np.sum(normB % p == 0))
    cop_pi = (~divisible_by_ideal(W[:, 0], W[:, 1], a, b)).sum()
    cop_pibar = (~divisible_by_ideal(W[:, 0], W[:, 1], a, -b)).sum()
    cop_p = (~(divisible_by_ideal(W[:, 0], W[:, 1], a, b) | divisible_by_ideal(W[:, 0], W[:, 1], a, -b))).sum()
    pred = (cop_pi + cop_pibar) / (p - 1) - cop_p / (p - 1) ** 2
    naive = nW * g_of(p, [p])
    print(f"  p={p:3d}: B_p = {Bp:6d}   corrected prediction = {pred:9.1f} (ratio {Bp / pred:.4f})"
          f"   naive |Om| g(p) = {naive:9.1f} (ratio {Bp / naive:.4f})   #{{pi|w}} = {nW - cop_pi}")

# ---------- discrepancies E(x; alpha, a) ----------
print("\nE(x; alpha, a) for Omega(x): max over classes alpha coprime to a, versus main term #{(w,a)=1}/Phi(a)")
# (gamma, label, prime ideal factors of gamma)
ideals = [((3, 0), "(3) inert", [(3, 0)]), ((7, 0), "(7) inert", [(7, 0)]),
          ((3, 2), "pi_13", [(3, 2)]), ((3, -2), "pibar_13", [(3, -2)]), ((5, 2), "pi_29", [(5, 2)]),
          ((13, 0), "(13) = pi pibar", [(3, 2), (3, -2)]), ((21, 0), "(21)", [(3, 0), (7, 0)]),
          ((9, 2), "9+2i = (2+i)(4-i), N=85", [(2, 1), (4, -1)])]
for (gr, gi), label, fi in ideals:
    n = gr * gr + gi * gi
    # class label of w mod (gamma): the pair (w * conj(gamma)) mod n is faithful on Z[i]/(gamma)
    ur = (W[:, 0] * gr + W[:, 1] * gi) % n
    ui = (W[:, 1] * gr - W[:, 0] * gi) % n
    # (w, gamma) = 1 iff no prime ideal factor of gamma divides w (gcd(N(w), n) = 1 would be wrong for split p)
    cop = np.ones(nW, dtype=bool)
    for (fr, fi_) in fi:
        cop &= ~divisible_by_ideal(W[:, 0], W[:, 1], fr, fi_)
    Phi = n
    for (fr, fi_) in fi:
        Phi = Phi * (1 - 1 / (fr * fr + fi_ * fi_))
    Phi = int(round(Phi))
    main = cop.sum() / Phi
    key = ur * n + ui
    keys, counts = np.unique(key[cop], return_counts=True)
    assert len(keys) == Phi, (label, len(keys), Phi)
    dev = counts - main
    print(f"  a = {label:>28}: N={n:4d} Phi={Phi:4d}  main/class = {main:10.1f}  max|E| = {np.abs(dev).max():8.1f}"
          f"  rms = {np.sqrt((dev ** 2).mean()):8.1f}  sqrt(main) = {main ** 0.5:7.1f}")
