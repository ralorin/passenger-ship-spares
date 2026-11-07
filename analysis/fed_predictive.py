"""
What the differences between the centralised and the federated fits of
federated.py do to a plan.

The 100 equivalence fleets are regenerated with the same seed. Both outer
optimisers are run again; since the two objectives are the same function of
theta, each fit's plan is built from the centralised machinery at that fit's
theta (hglmm.at_theta), so any difference comes from where the optimisers
stopped. The plan is the joint 99 % vector for the four references of an
unseen ship (covariate at zero, unit exposure), and its cover is exact under
the true parameters of the fleet.

    python fed_predictive.py
"""
import json
import math
import os

import numpy as np
import pandas as pd

import hglmm
from federated import BlockFederatedFit, fleet
from glmm import Design
from plan_cover import Zone, plan_joint_draws_robust

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "data", "v2")
LEVEL = 0.99
R = 40000
TAU, OM, S_MK, EPS = 0.4, 0.4, 0.3, 0.3


def plan(f, P, rng):
    d = f.d
    L = np.linalg.cholesky(f.cov_a + 1e-12 * np.eye(f.cov_a.shape[0]))
    za = f.z[:f.na] + (L @ rng.standard_normal((f.na, R))).T
    eta = np.tile(za[:, :1], (1, P))
    for p in range(1, P):
        eta[:, p] += za[:, d.fixed.index(f"pr{p}") + 1]
    mk = d.offsets[d.random.index("maker")]
    lev = {l: j for j, l in enumerate(d.levels["maker"])}
    for p in range(P):
        eta[:, p] += za[:, mk + lev[p % 3]]
    tau, om = f.sigma[d.random.index("ship")], f.sigma[d.random.index("sp")]
    eta += rng.standard_normal((R, 1)) * tau + rng.standard_normal((R, P)) * om
    mu = np.exp(eta) * rng.gamma(f.phi, 1.0 / f.phi, (R, P))
    return plan_joint_draws_robust(rng.poisson(mu), LEVEL)


def main():
    rng = np.random.default_rng(20261102)
    rows = []
    P, S = 4, 20
    for rep in range(100):
        st = np.random.default_rng()
        st.bit_generator.state = rng.bit_generator.state
        st.normal(0, TAU, S)
        st.normal(0, OM, (S, P))
        mk = st.normal(0, S_MK, 3)
        d = fleet(S, P, 4, rng)
        fixed = ["temp"] + [f"pr{k}" for k in range(1, P)]
        random = ["maker", "ship", "sp"]
        c = hglmm.Fit(Design(d, fixed, random)).fit()
        fed = BlockFederatedFit(Design(d, fixed, random)).fit(np.r_[np.full(3, math.log(0.5)), math.log(5.0)])
        # truth for an unseen ship: maker effects known, ship, interaction and mission terms not
        lams = np.exp(mk[np.arange(P) % 3] - S_MK ** 2 / 2)
        z = Zone(lams, TAU, math.hypot(OM, EPS))
        r = dict(rep=rep, d_sigma=float(np.max(np.abs(c.sigma - np.exp(fed.theta[:3])))),
                 d_loglik=abs(c.loglik - c.laplace(fed.theta)[0]))
        for name, th in (("central", c.theta), ("federated", fed.theta)):
            g = hglmm.at_theta(c, th)
            s = plan(g, P, np.random.default_rng([rep, 7]))
            r[f"units_{name}"] = int(s.sum())
            r[f"cover_{name}"] = z.joint(s)
        rows.append(r)
        print(rep, r, flush=True)
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(OUT, "fed_predictive.csv"), index=False)
    du = T.units_federated - T.units_central
    dc = T.cover_federated - T.cover_central
    summ = dict(n=len(T), same_plan=int((du == 0).sum()), units_diff_max=int(du.abs().max()),
                units_diff_mean=float(du.abs().mean()), cover_diff_max=float(dc.abs().max()),
                cover_central_mean=float(T.cover_central.mean()),
                cover_federated_mean=float(T.cover_federated.mean()))
    with open(os.path.join(OUT, "fed_predictive.json"), "w") as f:
        json.dump(summ, f, indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
