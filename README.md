# Implementaion for paper **Tight Privacy Audit in One run** (to appear in 2027 IEEE Symposium on Security and Privacy).
---

Given an audit that plants `n` canaries, releases its `r` most confident guesses
and gets `u` of them wrong, this library answers:

> what is the largest `eps` we can reject at 95 % confidence?

The core of our method is the order statistics modeling when `r < n`.

## Install

Requires `numpy` and `scipy` only. Could also use the tensor counterpart in `torch` to speed up in GPU.

```bash
pip install numpy scipy
```

## Quick start

```python
import onerun as oa

n, r, errors = 10_000, 2_500, 900

eps, mu = oa.epsilon_lower_bound(
    lambda t: oa.gaussian(t),          # null family: mu-GDP
    errors=errors, r=r, n=n, delta=1e-5, confidence=0.95)

print(f"eps >= {eps:.4f} at delta=1e-5   (fitted mu = {mu:.4f})")
```

## Layout

```
onerun/
  profile.py      score profiles: gamma, gammabar, G, in quantile coordinates
  mechanisms.py   Gaussian, Laplace, sub-sampled Gaussian, (eps,delta)-DP
  tail.py         the tail bound (exact)
  audit.py        p-values and the epsilon lower-bound search
  validate.py     brute-force Monte Carlo, for checking the closed forms
examples/         examples of usage
tests/            test the correctness of the script
compare/          compare method with previous work
docs/
  USAGE.md          how to audit each mechanism, and how to add your own
  VALIDATION.md     checking the tail against brute-force simulation
  IMPLEMENTATION.md every key numerical operation is explained
```
