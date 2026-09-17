# Validating the tail bound

`onerun/validate.py` simulates the audit directly, so that the closed form can be
checked. This document says
what the check is, how to run it, and
what the current run shows, including where the simulation has limitations.

---

## 1. Why simulation is the right check here

`exact_tail` computes

```
P[T <= u] = E_W[ BinCDF(u; r, gammabar(W)) ],   W ~ Beta(n-r, r+1)
```

which rests on the cut decomposition: conditioned on the highest discarded score,
the `r` released guesses are i.i.d. That identity is the whole content of the
theorem.

`simulate` is as a crosscheck. It draws `n` i.i.d. score quantiles, sorts them,
keeps the top `r`, and flips one independent coin per released guess with that
guess's own error probability (which is computed based on the scores):

```python
w   = rng.random((m, n))                       # n i.i.d. quantiles
top = np.partition(w, n - r, axis=1)[:, n - r:]   # the released r, by rank
gam = profile.gamma[searchsorted(profile.w_hi, top)]
T   = (rng.random(top.shape) < gam).sum(axis=1)
```

One is by analysis and one is by pure simulation. If
the two agree across mechanisms and thresholds, the decomposition is doing real
work correctly.

---

## 2. The API

```python
import onerun as oa

prof = oa.gaussian(1.0).profile()

# one threshold
p_hat, se = oa.monte_carlo(prof, u=90, r=500, n=2000, reps=200_000, seed=0)

# a whole tail from one simulation -- much cheaper
samples = oa.simulate(prof, r=500, n=2000, reps=200_000, seed=0)
for u in (60, 75, 90, 105):
    p_hat, se = oa.tail_from_samples(samples, u)
```

`simulate` returns the raw released-error counts, so checking `k` thresholds
costs one simulation rather than `k`. `monte_carlo` is a thin wrapper for the
single-threshold case.

---

## 3. Running the sweep

```bash
python examples/validate_all.py                 # ~3 min
python examples/validate_all.py --reps 500000   # tighter, slower
```

Eight channels (two parameter settings for each of the four families) times three
`(n, r)` settings — including `r = n`, where the cut degenerates — times four
thresholds placed at 0.80, 0.90, 1.00 and 1.10 of the expected error count. The
last recorded run is in [`../examples/validate_all_output.txt`](../examples/validate_all_output.txt).

