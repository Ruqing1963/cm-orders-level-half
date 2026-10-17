# Recovering the level of distribution x^{1/2} for the orders of y² = x³ − x modulo p

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23428952.svg)](https://doi.org/10.5281/zenodo.23428952)

Code, data and manuscript for the note

> R. Chen, *Recovering the level of distribution x^{1/2} for the orders of y² = x³ − x modulo p*, preprint, 2026.

## What the paper does

Iwaniec and Jiménez Urroz (Ann. Sc. Norm. Super. Pisa 9 (2010) 815–832) proved that for ≫ x/(log x)² primes
p ≡ 1 (mod 4) the order of the reduction of y² = x³ − x modulo p is eight times a P₂. Xie (arXiv:2504.18732,
Remark 4.1) observed that their level-of-distribution estimate (3.9) is not justified: the inclusion–exclusion that
converts d | N(π − 1) into congruences in Z[i] produces ideals of norm up to d², so the Bombieri–Vinogradov theorem
for Q(i) only gives level x^{1/4}.

The note shows that (3.9), and its analogue (5.9) for the switched sequence, are true at level x^{1/2}(log x)^{−B}.
A square-free ideal 𝔞 factors as (c)·𝔤 with c the largest rational integer dividing it; then N(𝔞) = c·d(𝔞), where
d(𝔞) is the least positive integer in 𝔞. Ideals with c ≤ (log x)^{B₀} are covered by the Bombieri–Vinogradov
theorem; those with larger c are bounded trivially by counting lattice points in a disc. Consequently

* Theorem 1.2 of Iwaniec–Jiménez Urroz (|E(F_p)|/8 = P₂ for ≫ x/(log x)² primes) holds as stated;
* Lemma 3.3 of Xie holds with level x^{1/2−ε}, which improves her bounds for orders over prime power fields:
  |E(F_{p²})|/(4|E(F_p)|) = P₃ (previously P₅), and Ω(|E(F_{p^ℓ})|/|E(F_p)|) ≤ ⌊2.05(ℓ−1) + 1.2⌋ for odd primes ℓ
  (previously ⌊4.1(ℓ−1) + 1.2⌋).

## Repository layout

```
paper/      level_half_cm.tex, level_half_cm.pdf (manuscript), arxiv_level_half_cm.zip (arXiv source package)
code/       certify_G.py              rigorous (interval arithmetic) evaluation of the weighted-sieve functional G
            linear_sieve.py           linear-sieve functions F, f (floating point, delay-differential equations)
            scan_levels.py            almost-prime bound as a function of the level alpha -> data/, figures/
            check_omega_distribution.py   first-principles numerical check of the bookkeeping (x = 1e8)
data/       level_vs_bound.csv        min 1/lambda + w/delta against alpha (w = 1, 2, 4)
            prime_power_bounds.csv    bounds of Xie (level 1/4) versus this paper (level 1/2) for each l
figures/    level_vs_bound.pdf/.png   the same, as a figure
results/    certified_G_values.txt    output of certify_G.py (the numbers quoted in Section 3.5 of the paper)
            scan_levels_output.txt    output of scan_levels.py
            omega_distribution_x1e8.txt   output of check_omega_distribution.py
```

## Reproducing the numbers in the paper

Requirements: Python ≥ 3.10 with `numpy`, `scipy`, `mpmath`, `matplotlib`. No other dependencies.

```
cd code
python certify_G.py                      # Section 3.5: G(1/8, 9/20, 10/17) = 0.4217..., G(1/8, 20/41, 5/6) = 0.0253...
python certify_G.py 0.499 1/8 20/41 6/5  # any parameters with alpha/4 <= theta < delta < alpha
python check_omega_distribution.py 1e8   # Section 3.3 bookkeeping: identity (3), densities g(d), discrepancies (~3 min)
python scan_levels.py                    # exploration: bound against level, writes data/ and figures/ (~10 min)
python scan_levels.py --plot-only        # redraw figures/level_vs_bound.pdf from data/level_vs_bound.csv
```

`certify_G.py` evaluates the closed form

    G = (2e^γ θ/α) [ log(α/θ − 1) − λ ( log(δ/θ) − (α/δ − 1) log((α−θ)/(α−δ)) ) ],

valid for α/4 ≤ θ < δ < α (then 3 ≤ α/θ ≤ 4 and 0 < (α − t)/θ ≤ 3, so only the closed forms
F(s) = 2e^γ/s and f(s) = 2e^γ log(s−1)/s are needed), with exact rational inputs and `mpmath` interval
arithmetic, so the printed lower bounds are rigorous.

`check_omega_distribution.py` builds the switched set Ω(x) of the Chen-type argument and the primary primes of
norm ≤ x, verifies the inclusion–exclusion identity (3) exactly for p = 5, 13, 17, 29, 37, 41, compares the divisor
densities with g(d), and measures the discrepancy E(x; α, 𝔞) in residue classes. It uses only counting; no sieve
theory is assumed.

## The constants of Iwaniec–Jiménez Urroz

The three constants of their Chen-type weight were recomputed (Section 3.4 of the paper):
½e^γ log 3 (lower bound sieve), ½e^γ log 6 (the W₂ integral equals log 6 exactly), and
c = ∫_{1/8}^{1/3} ∫_{1/3}^{(1−u)/2} du dv / (uv(1−u−v)) = 0.36308372…, giving the margin
log 3 − ½ log 6 − c/2 = 0.02119… > 0. A one-line `mpmath` check:

```python
from mpmath import mp, quad, log
mp.dps = 20
c = quad(lambda u: quad(lambda v: 1/((1-u-v)*u*v), [mp.mpf(1)/3, (1-u)/2]), [mp.mpf(1)/8, mp.mpf(1)/3])
print(c, log(3) - log(6)/2 - c/2)
```

## Citation

See `CITATION.cff`. The manuscript is archived at Zenodo (preprint, version v1, published 17 October 2026):
DOI [10.5281/zenodo.23428952](https://doi.org/10.5281/zenodo.23428952), record page
<https://zenodo.org/records/23428952>. The code and data of this repository correspond to tag `v1.0`.

## License

Code: MIT (see `LICENSE`). Manuscript (`paper/`): CC BY 4.0.
