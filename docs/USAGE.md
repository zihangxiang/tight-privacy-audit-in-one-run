# Computing privacy lower bounds

How to audit each supported mechanism, how to add a new one, and how to build an
audit the bound actually applies to.

---

## 1. The audit, in three quantities

An audit plants `n` canaries, each included independently with probability 1/2,
runs the mechanism once, guesses every membership bit, and **releases** only the
`r` guesses with the highest confidence scores. It observes `u` mistakes among
those `r`.

| symbol | meaning |
|---|---|
| `n` | number of canaries |
| `r` | number of released guesses (`r <= n`; `r = n` means no filtering) |
| `u` | mistakes among the released guesses |

The library turns `(n, r, u)` into a lower bound on `eps` by asking: *what is the
most private null hypothesis that would still produce this few mistakes only
rarely?*

---

## 2. p-values

```python
import onerun as oa

# Is less than or equal to 900 mistakes in 2500 released guesses surprising for a 1-GDP mechanism?
p = oa.p_value(oa.gaussian(1.0), u=900, r=2500, n=10_000)
```

Reject the null when `p <= 1 - confidence`.

---

## 3. Lower bounds, mechanism by mechanism

`epsilon_lower_bound` bisects over a **family** of nulls. You supply a function
mapping one scalar parameter to a channel, with larger = less private.

### Gaussian mechanism (`mu`-GDP)

`P = N(0,1)`, `Q = N(mu,1)`; sensitivity 1 with noise `sigma` gives `mu = 1/sigma`.

```python
eps, mu_hat = oa.epsilon_lower_bound(
    lambda t: oa.gaussian(t),
    errors=364, r=2500, n=10_000, delta=1e-5, confidence=0.95)
# eps >= 4.1371   (true eps of 1-GDP at delta=1e-5 is 4.3772)
```

### Laplace mechanism

`P = Laplace(0,b)`, `Q = Laplace(mu,b)`; pure DP with `eps = mu/b` at `delta = 0`.

```python
eps, _ = oa.epsilon_lower_bound(
    lambda t: oa.laplace(t, b=1.0),
    errors=672, r=2500, n=10_000, delta=1e-5)
```

Report it at `delta = 0` if you want the pure-DP statement:
`oa.laplace(2.0).eps_from_delta(0.0)` returns `2.0`.

### Sub-sampled Gaussian (DP-SGD's per-step mechanism)

`P = N(0,1)`, `Q = (1-q) N(0,1) + q N(mu,1)`. The sampling rate `q` is held
fixed; the search is over `mu`.

```python
eps, _ = oa.epsilon_lower_bound(
    lambda t: oa.subsampled_gaussian(t, q=0.5),
    errors=776, r=2500, n=10_000, delta=1e-5)
```

`q = 1` reproduces the plain Gaussian exactly.

### (eps, delta)-DP channel

The tight four-atom pair. The `delta` of the *null family* is fixed; the search
is over `eps`.

```python
eps, _ = oa.epsilon_lower_bound(
    lambda t: oa.eps_delta(t, delta=1e-6),
    errors=672, r=2500, n=10_000, delta=1e-5)
```

This channel is entirely atomic — the score takes one finite value — which is
why it is the sharpest test of the implementation's handling of atoms.

---

## 4. Adding your own mechanism

A mechanism is two arrays on a common index set:

```python
from onerun.mechanisms import Channel

log_lr = ...   # log dQ/dP at each outcome  (+inf allowed: a delta-atom)
mass   = ...   # each outcome's probability under the mixture M = (P+Q)/2
ch = Channel(log_lr, mass, name="my mechanism", param_range=(0.01, 32.0))
```

In other words, we discretize it and we operate in the discret distribution (approximation). More accurate result is possible with the more points we have. Everything else, including the score profile, `delta(eps)`, `eps_from_delta` and the tail 
is derived generically. 

Verify a new mechanism against the simulator before trusting it:

```python
prof = ch.profile()
print(oa.exact_tail(prof, u=90, r=500, n=2000))
print(oa.monte_carlo(prof, u=90, r=500, n=2000, reps=200_000))
```

---

## 5. Reading the result

The reported `eps` is a **lower bound** for the targeted DP algorithm in the form of `(eps,delta)-DP`: the audit approaches the mechanism's
true `eps` from below, so a larger number is a tighter audit, and a number above
the true `eps` means something is wrong: the targeted algorithm may be not 
private as claimed.

---

## 6. How tight is it?
Please refer to [`/compare/README.md`](../compare/README.md)
