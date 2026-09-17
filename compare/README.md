# Comparison with Steinke et al.

How much of a mechanism's true `eps` each method recovers from the same audit
outcome, on the Gaussian and sub-sampled Gaussian mechanisms.

```bash
python compare/compare_methods.py             # 60 setups
python compare/compare_methods.py --quick     # 20 setups
python compare/compare_methods.py --markdown  # also emit the tables below
```

Recorded run: [`RESULTS.txt`](RESULTS.txt).

## Contents

| file | what it is |
|---|---|
| `compare_methods.py` | the driver: runs both methods on identical outcomes |
| `steinke.py` | port of Steinke et al.'s reference implementation |
| `RESULTS.txt` | recorded output of the full 60-setup run |

## Protocol

Both methods receive **the same** `n`, the same `r`, and the same number of
mistakes, and are held to **the same significance level**, 0.05. They differ only
in how that outcome becomes a bound.

The observed outcome is not chosen adversarially or luckily: it is
`u = round(E[T])`, the number of mistakes the mechanism actually produces, when the auditor is doing the best. An audit approaches the true `eps` from below, so a
**larger** reported value is a tighter audit, and a value above the true `eps`
would indicate an invalid bound.

Sweep: 5 mechanisms x 3 canary counts (`1e3`, `1e4`, `1e5`) x 4 release
fractions (`r/n` = 0.1, 0.25, 0.5, 1.0) = 60 setups, `delta = 1e-5`, 95 %
confidence.

`steinke.py` is a faithful port of `p_value_DP_audit` / `get_eps_audit` from the
reference implementation of Steinke, Nasr and Jagielski, *Privacy Auditing with
One (1) Training Run*, NeurIPS 2023. 

## Advantages

**Releasing more guesses.** Steinke et al. degrade as `r` grows; this library
does not.

| `r/n` | Steinke et al. | ours | ratio |
|---|---|---|---|
| 0.10 | 0.341 | 0.825 | 2.90x |
| 0.25 | 0.276 | 0.847 | 4.38x |
| 0.50 | 0.214 | 0.855 | 5.94x |
| 1.00 | 0.126 | 0.850 | 9.50x |

Their bound pays for the dependence introduced by filtering, so it prefers to
release few guesses and discards evidence to do so. Our order-statistics
treatment is a more faithful modeling for the filtering.

**More canaries.** The library gives tighter results when `n` gets larger. The
baseline largely cannot.

| `n` | Steinke et al. | ours | ratio |
|---|---|---|---|
| 1,000 | 0.201 | 0.674 | 5.03x |
| 10,000 | 0.251 | 0.893 | 5.87x |
| 100,000 | 0.266 | 0.965 | 6.15x |

At `n = 1e5` the audit recovers **96.5 %** of the true `eps` on average, against
26.6 %.

**Mechanisms with a mixture component.** The sub-sampled Gaussian is where the
baseline loses the most.

| family | true `eps` | Steinke et al. | ours | ratio |
|---|---|---|---|---|
| Gaussian `mu=0.5` | 1.993 | 0.315 | 0.853 | 3.05x |
| Gaussian `mu=1` | 4.377 | 0.319 | 0.912 | 3.26x |
| Gaussian `mu=2` | 9.997 | 0.312 | 0.904 | 3.37x |
| SGM `mu=1 q=0.5` | 3.534 | 0.178 | 0.820 | 5.29x |
| SGM `mu=2 q=0.2` | 7.620 | 0.071 | 0.732 | 13.43x |

## Full results

`eps` lower bound at `delta = 1e-5`, 95 % confidence. `u` is the number of
mistakes the mechanism actually makes; both methods see it.

### Gaussian `mu = 0.5`

True `eps` = 1.993 at `delta = 1e-5`.

| n | r | mistakes u | Steinke | ours | gain |
|---|---|---|---|---|---|
| 1000 | 100 | 26 | 0.6519 | **1.1926** | +83% |
| 1000 | 250 | 75 | 0.6124 | **1.4203** | +132% |
| 1000 | 500 | 172 | 0.4866 | **1.4689** | +202% |
| 1000 | 1000 | 401 | 0.2931 | **1.4140** | +382% |
| 10000 | 1000 | 259 | 0.9277 | **1.7437** | +88% |
| 10000 | 2500 | 755 | 0.7643 | **1.8052** | +136% |
| 10000 | 5000 | 1721 | 0.5948 | **1.8279** | +207% |
| 10000 | 10000 | 4013 | 0.3661 | **1.8074** | +394% |
| 100000 | 10000 | 2585 | 1.0128 | **1.9166** | +89% |
| 100000 | 25000 | 7545 | 0.8149 | **1.9348** | +137% |
| 100000 | 50000 | 17213 | 0.6283 | **1.9406** | +209% |
| 100000 | 100000 | 40129 | 0.3892 | **1.9345** | +397% |

### Gaussian `mu = 1`

True `eps` = 4.377 at `delta = 1e-5`.

| n | r | mistakes u | Steinke | ours | gain |
|---|---|---|---|---|---|
| 1000 | 100 | 10 | 1.6265 | **3.1645** | +95% |
| 1000 | 250 | 36 | 1.4767 | **3.6327** | +146% |
| 1000 | 500 | 103 | 1.1630 | **3.7557** | +223% |
| 1000 | 1000 | 309 | 0.6902 | **3.6700** | +432% |
| 10000 | 1000 | 97 | 2.0466 | **4.0141** | +96% |
| 10000 | 2500 | 364 | 1.6736 | **4.1371** | +147% |
| 10000 | 5000 | 1034 | 1.2858 | **4.1789** | +225% |
| 10000 | 10000 | 3085 | 0.7711 | **4.1592** | +439% |
| 100000 | 10000 | 967 | 2.1713 | **4.2662** | +96% |
| 100000 | 25000 | 3641 | 1.7377 | **4.3013** | +148% |
| 100000 | 50000 | 10343 | 1.3250 | **4.3144** | +226% |
| 100000 | 100000 | 30854 | 0.7954 | **4.3079** | +442% |

### Gaussian `mu = 2`

True `eps` = 9.997 at `delta = 1e-5`.

| n | r | mistakes u | Steinke | ours | gain |
|---|---|---|---|---|---|
| 1000 | 100 | 1 | 3.0052 | **5.9018** | +96% |
| 1000 | 250 | 4 | 3.2737 | **7.9541** | +143% |
| 1000 | 500 | 20 | 2.7928 | **8.9216** | +219% |
| 1000 | 1000 | 159 | 1.5210 | **9.0038** | +492% |
| 10000 | 1000 | 5 | 4.4836 | **8.7173** | +94% |
| 10000 | 2500 | 36 | 3.9322 | **9.4097** | +139% |
| 10000 | 5000 | 202 | 3.0457 | **9.6526** | +217% |
| 10000 | 10000 | 1587 | 1.6223 | **9.6881** | +497% |
| 100000 | 10000 | 51 | 4.8674 | **9.5899** | +97% |
| 100000 | 25000 | 362 | 4.1144 | **9.8094** | +138% |
| 100000 | 50000 | 2016 | 3.1291 | **9.8905** | +216% |
| 100000 | 100000 | 15866 | 1.6535 | **9.9002** | +499% |

### Sub-sampled Gaussian `mu = 1, q = 0.5`

True `eps` = 3.534 at `delta = 1e-5`.

| n | r | mistakes u | Steinke | ours | gain |
|---|---|---|---|---|---|
| 1000 | 100 | 24 | 0.7484 | **2.1584** | +188% |
| 1000 | 250 | 78 | 0.5583 | **2.1201** | +280% |
| 1000 | 500 | 176 | 0.4522 | **2.2774** | +404% |
| 1000 | 1000 | 404 | 0.2808 | **2.2223** | +692% |
| 10000 | 1000 | 242 | 1.0154 | **3.1102** | +206% |
| 10000 | 2500 | 776 | 0.7254 | **3.0878** | +326% |
| 10000 | 5000 | 1761 | 0.5598 | **3.1149** | +456% |
| 10000 | 10000 | 4043 | 0.3536 | **3.0963** | +776% |
| 100000 | 10000 | 2423 | 1.0982 | **3.3995** | +210% |
| 100000 | 25000 | 7763 | 0.7740 | **3.3907** | +338% |
| 100000 | 50000 | 17606 | 0.5938 | **3.4007** | +473% |
| 100000 | 100000 | 40427 | 0.3768 | **3.3944** | +801% |

### Sub-sampled Gaussian `mu = 2, q = 0.2`

True `eps` = 7.620 at `delta = 1e-5`.

| n | r | mistakes u | Steinke | ours | gain |
|---|---|---|---|---|---|
| 1000 | 100 | 22 | 0.8493 | **4.6207** | +444% |
| 1000 | 250 | 87 | 0.4015 | **3.5077** | +774% |
| 1000 | 500 | 199 | 0.2594 | **2.8572** | +1001% |
| 1000 | 1000 | 432 | 0.1666 | **2.8743** | +1625% |
| 10000 | 1000 | 216 | 1.1576 | **6.7008** | +479% |
| 10000 | 2500 | 870 | 0.5570 | **6.1668** | +1007% |
| 10000 | 5000 | 1987 | 0.3679 | **5.8117** | +1480% |
| 10000 | 10000 | 4317 | 0.2413 | **5.8610** | +2329% |
| 100000 | 10000 | 2156 | 1.2476 | **7.3328** | +488% |
| 100000 | 25000 | 8697 | 0.6054 | **7.1450** | +1080% |
| 100000 | 50000 | 19868 | 0.4009 | **7.0063** | +1648% |
| 100000 | 100000 | 43173 | 0.2640 | **7.0250** | +2561% |
## Scope

The comparison covers the Gaussian and sub-sampled Gaussian families at five
parameter settings, with `delta = 1e-5` and 95 % confidence throughout. It is a
comparison of bounds on identical audit outcomes, not of end-to-end audits of a
training pipeline.
