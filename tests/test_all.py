"""Correctness checks.  Run:  python tests/test_all.py"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy.stats import norm, binom
import onerun as oa

FAIL = []
def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
    if not ok: FAIL.append(name)

print("1. Against closed forms")
for mu in (0.5, 1.0, 2.0):
    ch = oa.gaussian(mu)
    for eps in (0.5, 1.0, 2.0):
        want = norm.cdf(-eps/mu + mu/2) - np.exp(eps)*norm.cdf(-eps/mu - mu/2)
        check(f"delta(eps) mu={mu} eps={eps}",
              abs(ch.delta_of_eps(eps) - want) < 1e-8)
check("e_opt Gaussian mu=1 = Phi(-1/2)",
      abs(oa.gaussian(1.0).profile().e_opt - norm.cdf(-0.5)) < 1e-8)
check("e_opt (1,1e-5)-DP = (1-d)/(1+e^eps)",
      abs(oa.eps_delta(1.0,1e-5).profile().e_opt - (1-1e-5)/(1+np.exp(1.0))) < 1e-12)
check("Laplace mu=2 is 2-DP at delta=0",
      abs(oa.laplace(2.0).eps_from_delta(0.0) - 2.0) < 1e-4)

print("\n2. the tail at r = n reduces to a binomial")
for ch, name in [(oa.gaussian(1.0),"Gaussian"), (oa.eps_delta(1.0,1e-3),"(eps,delta)")]:
    p = ch.profile()
    for u in (40, 60):
        check(f"r=n {name} u={u}",
              abs(oa.exact_tail(p,u,200,200) - binom.cdf(u,200,p.e_opt)) < 1e-12)

print("\n3. monotonicity and range")
p = oa.gaussian(1.0).profile()
vals = [oa.exact_tail(p,u,250,1000) for u in range(0,120,10)]
check("tail is non-decreasing in u", all(a<=b+1e-12 for a,b in zip(vals,vals[1:])))
check("tail lies in [0,1]", all(0<=v<=1 for v in vals))

print("\n4. exact tail vs brute-force Monte Carlo")
for ch,name,n,r,u in [(oa.gaussian(0.8),"Gaussian mu=0.8",2000,500,90),
                      (oa.laplace(1.0),"Laplace mu=1",2000,400,96),
                      (oa.eps_delta(1.0,1e-3),"(1,1e-3)-DP",1000,250,60)]:
    pr=ch.profile()
    ex=oa.exact_tail(pr,u,r,n); mc,se=oa.monte_carlo(pr,u,r,n,reps=120000,seed=11)
    z=(ex-mc)/max(se,1e-12)
    check(f"MC {name}", abs(z)<4, f"exact {ex:.6f} vs MC {mc:.6f} (z={z:+.2f})")

print("\n5. epsilon lower bound never exceeds the truth at a fair observation")
for make, truth, nm in [(lambda t: oa.gaussian(t), oa.gaussian(1.0), "Gaussian mu=1"),
                        (lambda t: oa.laplace(t), oa.laplace(1.0), "Laplace mu=1")]:
    pr=truth.profile(); n,r=4000,1000
    w=np.linspace(1e-12,1-1e-12,100001)
    from scipy.stats import beta as B
    u=int(round(r*np.trapezoid(B.pdf(w,n-r,r+1)*pr.gammabar(w),w)))
    eps,_=oa.epsilon_lower_bound(make,errors=u,r=r,n=n,delta=1e-5,warn=False)
    t=truth.eps_from_delta(1e-5)
    check(f"lower bound <= true eps ({nm})", eps<=t+1e-6, f"{eps:.4f} <= {t:.4f}")

print(f"\n{'ALL CHECKS PASSED' if not FAIL else 'FAILURES: ' + ', '.join(FAIL)}")
sys.exit(1 if FAIL else 0)
