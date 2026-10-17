"""
Exploration (floating point): how the almost-prime bound of Xie's weighted linear sieve depends on the level of
distribution alpha.  For the sequence of elements a ~ x^{l-1} (l = 1 for |E(F_p)|/8, l = 2 for N(pi+1)/4,
l >= 3 for N(Phi_l(pi)) = |E(F_{p^l})|/|E(F_p)|) the sieve gives

    Omega(a) < 1/lambda + (l-1)/delta       whenever  G(theta, delta, lambda) > 0,
    G = f(alpha/theta) - lambda * int_theta^delta F((alpha-t)/theta) (1 - t/delta) dt/t,   0 < theta < delta < alpha,

(for l = 1, 2 the weight on 1/delta is 1).  For each alpha we minimise B(alpha) = 1/lambda + w/delta over theta,
delta with lambda chosen so that G = 0+, for w = 1 (|E(F_p)|/8 and N(pi+1)/4) and w = 2, 4 (l = 3, 5).

Outputs: ../data/level_vs_bound.csv and ../figures/level_vs_bound.pdf.
Uses linear_sieve.py (F, f extended by the delay-differential equations, so that all theta are allowed).
"""
import csv
import os
import numpy as np
from scipy.integrate import quad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from linear_sieve import F, f

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def best(alpha, w, n=70):
    out = (np.inf, None)
    for theta in np.linspace(0.02, alpha / 2, n):
        f0 = float(f(alpha / theta))
        if f0 <= 0:
            continue
        for delta in np.linspace(theta + 1e-3, alpha - 1e-4, n):
            I = quad(lambda t: float(F((alpha - t) / theta)) * (1 - t / delta) / t, theta, delta, limit=400)[0]
            lam = f0 / I
            val = 1 / lam + w / delta
            if val < out[0]:
                out = (val, (theta, delta, lam))
    return out


WEIGHTS = {1: r"$w=1$:  $|E(\mathbb{F}_p)|/8$,  $|E(\mathbb{F}_{p^2})|/(4|E(\mathbb{F}_p)|)$",
           2: r"$w=2$:  $\ell=3$", 4: r"$w=4$:  $\ell=5$"}
CSV = os.path.join(ROOT, "data", "level_vs_bound.csv")


def compute():
    alphas = np.round(np.arange(0.25, 0.4999, 0.0125), 4).tolist() + [0.499]
    table = {w: [] for w in WEIGHTS}
    for a in alphas:
        for w in WEIGHTS:
            v, p = best(a, w)
            table[w].append((a, v, *p))
            print(f"alpha = {a:.4f}  w = {w}:  min 1/lambda + w/delta = {v:7.3f}  "
                  f"(theta, delta, 1/lambda) = ({p[0]:.4f}, {p[1]:.4f}, {1 / p[2]:.4f})", flush=True)
    with open(CSV, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["alpha", "w", "min_bound", "theta", "delta", "inv_lambda", "Omega_le"])
        for w in WEIGHTS:
            for a, v, th, de, lam in table[w]:
                wr.writerow([a, w, f"{v:.4f}", f"{th:.4f}", f"{de:.4f}", f"{1 / lam:.4f}", int(np.floor(v - 1e-9))])


def plot():
    """one axis, three series in the fixed categorical order of the validated palette (slots 1-3), direct labels."""
    table = {w: [] for w in WEIGHTS}
    with open(CSV, newline="") as fh:
        for row in csv.DictReader(fh):
            table[int(row["w"])].append((float(row["alpha"]), float(row["min_bound"])))
    colors = {1: "#2a78d6", 2: "#eb6834", 4: "#1baf7a"}
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    for w in WEIGHTS:
        xs = [r[0] for r in table[w]]
        ys = [r[1] for r in table[w]]
        ax.plot(xs, ys, color=colors[w], lw=2, label=WEIGHTS[w])
        ax.annotate(f"w = {w}", xy=(xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=9, color="#52514e")
    for lvl, txt in [(0.25, "level 1/4 (Xie 2025)"), (0.5, "level 1/2 (this paper)")]:
        ax.axvline(lvl, color="#c3c2b7", lw=1, ls=":")
        ax.text(lvl, 2.3, txt, ha="center", va="bottom", fontsize=8, color="#898781")
    ax.set_xlabel("level of distribution  α", color="#52514e")
    ax.set_ylabel("min  1/λ + w/δ    (Ω ≤ ⌊·⌋)", color="#52514e")
    ax.set_xlim(0.22, 0.55)
    ax.set_ylim(2, 19)
    ax.grid(axis="y", color="#e1e0d9", lw=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#c3c2b7")
    ax.tick_params(colors="#898781", labelsize=8)
    ax.legend(frameon=False, fontsize=8, loc="upper right", labelcolor="#52514e")
    ax.set_title("Weighted linear sieve: almost-prime bound against the level", fontsize=10, color="#0b0b0b", loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "figures", "level_vs_bound.pdf"))
    fig.savefig(os.path.join(ROOT, "figures", "level_vs_bound.png"))
    print("written figures/level_vs_bound.pdf")


if __name__ == "__main__":
    import sys
    if "--plot-only" not in sys.argv:
        compute()
    plot()
