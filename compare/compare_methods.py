"""Compare this library's lower bound with Steinke et al.'s, on the same audits.

Both methods are given exactly the same observation -- same canary count n, same
number of released guesses r, same number of mistakes u -- and asked the same
question: what is the largest eps rejectable at 95% confidence, reported at
delta = 1e-5?

The observation is not invented.  For each mechanism we compute the number of
mistakes it actually produces, E[T], under the exact law, and feed
u = round(E[T]) to both methods.  This is the audit a typical run would see, and
because an audit approaches the true eps from below, a larger reported eps is a
strictly better audit -- provided it stays below the true eps, which is checked
in the summary.

    python compare/compare_methods.py            # 60 setups, ~5 min
    python compare/compare_methods.py --quick    # 20 setups
    python compare/compare_methods.py --markdown # also emit the per-mechanism tables
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import numpy as np
from scipy.stats import beta as Beta

import onerun as oa
import steinke

DELTA, CONF, NGRID = 1e-5, 0.95, 100_000
QUICK = "--quick" in sys.argv
MARKDOWN = "--markdown" in sys.argv

# (label, factory over the searched parameter, the true mechanism)
FAMILIES = [
    ("Gaussian mu=0.5",  lambda t: oa.gaussian(t, n_grid=NGRID),                 oa.gaussian(0.5, n_grid=NGRID)),
    ("Gaussian mu=1",    lambda t: oa.gaussian(t, n_grid=NGRID),                 oa.gaussian(1.0, n_grid=NGRID)),
    ("Gaussian mu=2",    lambda t: oa.gaussian(t, n_grid=NGRID),                 oa.gaussian(2.0, n_grid=NGRID)),
    ("SGM mu=1 q=0.5",   lambda t: oa.subsampled_gaussian(t, 0.5, n_grid=NGRID), oa.subsampled_gaussian(1.0, 0.5, n_grid=NGRID)),
    ("SGM mu=2 q=0.2",   lambda t: oa.subsampled_gaussian(t, 0.2, n_grid=NGRID), oa.subsampled_gaussian(2.0, 0.2, n_grid=NGRID)),
]
N_LIST = [1_000, 10_000] if QUICK else [1_000, 10_000, 100_000]
R_RATIOS = [0.25, 0.5] if QUICK else [0.10, 0.25, 0.50, 1.00]


def expected_errors(prof, n, r):
    """E[T] under the exact law: the mistakes this mechanism actually makes."""
    if r == n:
        return n * prof.e_opt
    w = np.linspace(1e-12, 1 - 1e-12, 100_001)
    return r * np.trapezoid(Beta.pdf(w, n - r, r + 1) * prof.gammabar(w), w)


def main():
    print(__doc__.split("\n\n")[0])
    print("\neps lower bound at delta = 1e-5, 95% confidence.")
    print("u is the number of mistakes the mechanism actually makes; both methods see it.\n")
    print(f"{'mechanism':<17} {'true eps':>8} {'n':>7} {'r':>7} {'u':>7} | "
          f"{'Steinke':>9} {'ours':>9} | {'ratio':>7}  {'gain':>8}")
    print("-" * 96)
    rows = []
    for label, make, truth in FAMILIES:
        prof = truth.profile()
        eps_true = truth.eps_from_delta(DELTA)
        for n in N_LIST:
            for ratio in R_RATIOS:
                r = max(2, int(round(n * ratio)))
                u = int(round(expected_errors(prof, n, r)))
                v = r - u                                   # correct guesses
                # steinke.epsilon_lower_bound takes a SIGNIFICANCE level,
                # not a confidence level -- 0.05, the same one we use
                e_st = steinke.epsilon_lower_bound(n, r, v, DELTA, 1 - CONF)
                e_ours, _ = oa.epsilon_lower_bound(
                    make, errors=u, r=r, n=n, delta=DELTA,
                    confidence=CONF, warn=False)
                ratio_ = e_ours / e_st if e_st > 1e-9 else float("inf")
                rows.append((label, eps_true, n, r, ratio, u, e_st, e_ours))
                gain = f"{100*(e_ours-e_st)/e_st:+.0f}%" if e_st > 1e-9 else "  --"
                rr = f"{ratio_:7.2f}" if np.isfinite(ratio_) else "    inf"
                print(f"{label:<17} {eps_true:>8.4f} {n:>7} {r:>7} {u:>7} | "
                      f"{e_st:>9.4f} {e_ours:>9.4f} | {rr}  {gain:>8}")
        print()

    # ---------------- summary ----------------
    a = np.array([(t, s, o) for _, t, _, _, _, _, s, o in rows])
    true_, st, ours = a[:, 0], a[:, 1], a[:, 2]
    N = len(rows)
    print("=" * 96)
    print(f"{N} configurations\n")
    print(f"  ours > Steinke                : {int((ours > st).sum())}/{N}")
    print(f"  both are valid (<= true eps)  : Steinke {int((st <= true_ + 1e-9).sum())}/{N}, "
          f"ours {int((ours <= true_ + 1e-9).sum())}/{N}")
    fin = st > 1e-9
    rt = ours[fin] / st[fin]
    print(f"\n  ratio ours/Steinke  : median {np.median(rt):.2f}x, "
          f"range {np.min(rt):.2f}x - {np.max(rt):.2f}x")
    print(f"  fraction of true eps: Steinke mean {np.mean(st/true_):.1%} "
          f"(median {np.median(st/true_):.1%}, best {np.max(st/true_):.1%}, "
          f"worst {np.min(st/true_):.1%})")
    print(f"                        ours    mean {np.mean(ours/true_):.1%} "
          f"(median {np.median(ours/true_):.1%}, best {np.max(ours/true_):.1%}, "
          f"worst {np.min(ours/true_):.1%})")
    print("\n  by canary count (mean fraction of the true eps recovered):")
    print(f"    {'n':>8} {'Steinke':>10} {'ours':>10} {'ratio':>8}")
    for n in N_LIST:
        m_ = np.array([row[2] == n for row in rows])
        print(f"    {n:>8} {np.mean(st[m_]/true_[m_]):>9.1%} "
              f"{np.mean(ours[m_]/true_[m_]):>9.1%} {np.mean(ours[m_]/st[m_]):>7.2f}x")
    print("\n  by release fraction r/n:")
    print(f"    {'r/n':>8} {'Steinke':>10} {'ours':>10} {'ratio':>8}")
    for ratio in R_RATIOS:
        m_ = np.array([abs(row[4] - ratio) < 1e-9 for row in rows])
        print(f"    {ratio:>8.2f} {np.mean(st[m_]/true_[m_]):>9.1%} "
              f"{np.mean(ours[m_]/true_[m_]):>9.1%} {np.mean(ours[m_]/st[m_]):>7.2f}x")
    print("\n  by mechanism family:")
    print(f"    {'family':<18} {'Steinke':>10} {'ours':>10} {'ratio':>8}")
    for label, _, _ in FAMILIES:
        m_ = np.array([row[0] == label for row in rows])
        print(f"    {label:<18} {np.mean(st[m_]/true_[m_]):>9.1%} "
              f"{np.mean(ours[m_]/true_[m_]):>9.1%} {np.mean(ours[m_]/st[m_]):>7.2f}x")

    if MARKDOWN:
        print("\n\n" + "=" * 96 + "\nMARKDOWN\n" + "=" * 96)
        cur = None
        for label, et, n, r, _, u, s, o in rows:
            if label != cur:
                cur = label
                print(f"\n**{label}** (true eps = {et:.3f} at delta = 1e-5)\n")
                print("| n | r | mistakes u | Steinke | ours | gain |")
                print("|---|---|---|---|---|---|")
            g = f"{100*(o-s)/s:+.0f}%" if s > 1e-9 else "--"
            print(f"| {n} | {r} | {u} | {s:.4f} | **{o:.4f}** | {g} |")


if __name__ == "__main__":
    main()
