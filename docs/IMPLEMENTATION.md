# Implementation notes

Every numerical operation behind the tail bound. Please refer to `onerun/profile.py`, `onerun/tail.py`,
`onerun/mechanisms.py`, `onerun/audit.py`.

---

## 1. What is computed

For an audit that plants `n` canaries, releases the `r` most confident guesses
and makes `T` mistakes among them,

```
P[T <= u]  =  E_W[ BinCDF(u; r, gammabar(W)) ],      W ~ Beta(n-r, r+1)     (1)
```

`W` is the score quantile of the **cut**: the highest-scoring guess that is
*not* released. `gammabar(w)` is the mean error rate above quantile `w`.

Everything relies on three primitives:

| primitive | where |
|---|---|
| `gammabar(w)`, from the mechanism | `profile.py` |
| the Beta density of the cut | `tail.py::_cut_grid_beta` |
| `BinCDF(u; r, p)` | `tail.py::log_binom_cdf` |

---

## 2. Representing a mechanism

A channel is two arrays on a common index set:

```
log_lr[j]  =  log dQ/dP at outcome j        (+-inf permitted)
mass[j]    =  P_M[outcome j],  M = (P+Q)/2  (sums to 1)
```
In other words, we discretize it and we operate in the discret distribution (approximation). More accurate result is possible with the more points we have. 

The mixture `M` is the right base distribution because the audit draws the secret bit
uniformly: an outcome is seen with probability `(p+q)/2`. When the secret bit is not 
uniformly drawn, e.g., `1/3` and `2/3`, the mixture could be `(p+2q)/3`, but this case
is never tested.

A continuous channel is discretised on a **midpoint grid**: cell centres
`x_j = lo + (j+1/2) dx`, mass `density(x_j) * dx`. Midpoints rather than
endpoints avoid double-counting the boundary cells, and the masses then sum to 1.

An **atom** is a single entry. The `delta`-atom of `(eps,delta)`-DP is
`log_lr = +inf`, `mass = delta/2`. We have `/2` here because the mixture halves it. 

---

## 3. Building the score profile

`ScoreProfile.__init__` does five things.

**3.1 Error probability from the likelihood ratio.** For the optimal (MAP)
decoder the conditional error at an outcome is

```
gamma = 1 / (1 + exp|log_lr|)
```

This is computed numerical-stably.

**3.2 Sorting.** Entries are sorted by **decreasing `gamma`**, which is
increasing score, because `gamma` is a decreasing function of the score. The top
of the score range, which is what the filtering keeps, is the tail of the array.


**3.3 Quantile bands.** Entry `j` occupies `[w_lo[j], w_hi[j])` with

```
w_lo = concatenate([[0], cumsum(mass)[:-1]])
w_hi = w_lo + mass
```

so a continuous channel gets a band per grid cell and an atom gets one wide band.
The rest of the library never needs to know which it is dealing with.

**3.4 Error mass above each entry.**

```
err      = gamma * mass
G_above  = err[::-1].cumsum()[::-1] - err        # strictly above entry j
```

The reverse cumulative sum is `O(len)` and done once; every later query is a
lookup.

**3.5 The optimal single-bit error.** `e_opt = sum(err) = G(0)`. This is the
value the audit compares against, and it is worth checking against a closed form
whenever one exists.

---

## 4. Evaluating `G(w)` and `gammabar(w)`

```
G(w) = gamma[j] * (w_hi[j] - w) + G_above[j],   j = searchsorted(w_hi, w, 'right')
```

The first term is the part of the straddled entry lying above `w`; the second is
everything above it. This is **exact inside an atom** — if `w` falls partway
through a wide band, the linear term interpolates the band correctly — which
matters because for the Laplace mechanism one atom carries 68 % of the mass.

```
gammabar(w) = G(w) / max(1 - w, tiny)
```

> **Trap.** Do not precompute and store `gammabar` per entry. The final entry has
> `w_hi = 1`, so a stored value is `0/0` there; whatever you clamp it to
> (typically `0.5`) then leaks into every interpolation that touches the last
> node, and the audit saturates at large privacy parameters. Store `G`, which is
> well behaved everywhere, and divide by the **querying** `w`.

---

## 5. The outer integral

`_cut_grid_beta(n, r)` builds nodes for `W ~ Beta(n-r, r+1)`:

```
a, b   = n - r, r + 1
mean   = a / (a + b)
sd     = sqrt(a*b / ((a+b)^2 (a+b+1)))
w      = linspace(mean - 18 sd, mean + 18 sd)  clipped to (0,1),  n_points
log f  = (a-1) log w + (b-1) log1p(-w) + gammaln(a+b) - gammaln(a) - gammaln(b)
```

**Log-sum-exp shift.** The integrand is assembled in logs and exponentiated once:

```
M = max(log_integrand)
I = exp(M) * trapezoid(exp(log_integrand - M), w)
```

---

## 6. The inner binomial CDF

```
BinCDF(u; r, p) = I_{1-p}(r-u, u+1)
```

the regularised incomplete beta. We call `scipy.stats.binom.logcdf`, which uses
that identity internally, rather than summing `u+1` pmf terms.

---

## 7. Degenerate case `r = n`

Nothing is discarded, there is no cut.

```
P[T <= u] = BinCDF(u; n, e_opt)
```


---

## 8. Compute delta and eps

For a pair `(P,Q)` with mixture `M` and ratio `L = dQ/dP`,

```
delta(eps) = int (q - e^eps p)_+
           = E_M[ 2 (L - e^eps)_+ / (1 + L) ]                             
```

*Derivation.* With `m = (p+q)/2` we have `q = 2mL/(1+L)` and `p = 2m/(1+L)`, so
`q - e^eps p = 2m (L - e^eps)/(1+L)`.

The above Equation is evaluated as `2 (1 - e^{eps - logL}) / (1 + e^{-logL})`, which is
finite for every `logL` and tends to `2` as `logL -> +inf`. 

---

## 9. The epsilon lower-bound search

`epsilon_lower_bound` bisects the null's privacy parameter `theta` for the
largest null still rejected, then converts that null to `(eps, delta)`. The
p-value is increasing in `theta`, so the bracket moves up when
`p(theta) < 1 - confidence`.

There is an implicit assumption that the targeted DP algorithm should be
within the family `theta` represents. For example, for Gaussian mechanism, we have `theta=\mu`;
for subsampled Gaussian mechanism, we have `theta=\mu, q`


---

## 10. Possible Pitfalls

| symptom | cause |
|---|---|
| `gammabar(w)` flat near `e_opt` for all `w` | sorted on `log(1-gamma)`, which underflowed |
| audit saturates at large privacy parameters | stored `gammabar` per node; `0/0` at `w = 1` |
| p-value collapses to `0` at large `n` | no log-sum-exp shift, or profile nodes reused for the Beta |
| tail integral is `0` and very slow | reused profile nodes for the outer quadrature |
| `e_opt` slightly too large, audit slightly conservative | dropped point mass of `Q` (deficit `delta/2`, grid-independent) |
| large spurious difference between two methods | mismatched bisection ranges |
| `NaN` or `0` bound for sub-sampled Gaussian | `log(q_density) - log(p_density)` underflowed; use `logaddexp` |
