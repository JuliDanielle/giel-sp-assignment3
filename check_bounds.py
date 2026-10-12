"""
check_bounds.py  --  ENLR 5131 Assignment 3, Part H (H4 check)
Which width limit holds the design back?  Starting from the optimized design,
relax ONE width limit at a time and let the local optimizer (SLSQP) improve the
design.  A big improvement means that limit was binding.

How to run: put this file in the same folder as chip_shear_optimizer.py and
chip_network_solver.py, then
    python check_bounds.py          (about 1-3 minutes)
"""
import numpy as np
from scipy.optimize import minimize
import chip_shear_optimizer as O

# Optimized design from the full run, width in um for each mirror-image group
DESIGN = {(1, 2, 66, 67): 1000, (4, 5, 63, 64): 1000, (6, 7, 61, 62): 960,
          (8, 11, 57, 60): 950, (9, 10, 58, 59): 200, (13, 16, 52, 55): 960,
          (14, 15, 53, 54): 200, (17, 18, 50, 51): 760, (19, 22, 46, 49): 200,
          (20, 21, 47, 48): 960, (24, 27, 41, 44): 200, (25, 26, 42, 43): 960,
          (28, 29, 39, 40): 760, (30, 33, 35, 38): 960, (31, 32, 36, 37): 200}

def design_vector():
    """Put the widths in the order the optimizer uses (O.DESIGN_GROUPS)."""
    x = []
    for g in O.DESIGN_GROUPS:
        members = tuple(sorted(int(i) + 1 for i in np.where(O.GROUP == g)[0]))
        x.append(DESIGN[members])
    return np.array(x, float)

def local_polish(x0, w_lo, w_hi):
    """Same SLSQP step as chip_shear_optimizer.optimize(), started from x0:
    minimise (b - a) with a <= ln(tau_i) <= b, keeping dP <= 500 Pa and the
    shear window 4-22 dyn/cm^2."""
    nx = len(x0)
    lt = lambda z: np.log(O.evaluate(z[:nx])["tau"][O.SEEDED])
    z0 = np.r_[x0, lt(x0).min(), lt(x0).max()]
    cons = [{"type": "ineq", "fun": lambda z: lt(z) - z[nx]},
            {"type": "ineq", "fun": lambda z: z[nx + 1] - lt(z)},
            {"type": "ineq", "fun": lambda z: O.DP_MAX - O.evaluate(z[:nx])["dP_total"]},
            {"type": "ineq", "fun": lambda z: np.array([z[nx] - np.log(O.TAU_MIN),
                                                         np.log(O.TAU_MAX) - z[nx + 1]])}]
    res = minimize(lambda z: z[nx + 1] - z[nx], z0, method="SLSQP",
                   bounds=[(w_lo, w_hi)]*nx + [(None, None)]*2, constraints=cons,
                   options={"maxiter": 200, "ftol": 1e-12, "eps": 1e-4})
    x = np.clip(res.x[:nx], w_lo, w_hi)
    r = O.evaluate(x); t = r["tau"][O.SEEDED]
    return t.max()/t.min(), r["dP_total"], t.min()*10, t.max()*10

if __name__ == "__main__":
    x0 = design_vector()
    r0 = O.evaluate(x0); t0 = r0["tau"][O.SEEDED]
    print("Starting design (10 um grid): ratio %.4f, dP %.0f Pa" % (t0.max()/t0.min(), r0["dP_total"]))
    print("\n%-26s %8s %9s %18s" % ("Width limits", "ratio", "dP [Pa]", "shear [dyn/cm2]"))
    for lo, hi, note in [(200, 1000, "control: original limits"),
                         (180, 1000, "minimum relaxed"),
                         (200, 1100, "maximum relaxed")]:
        ratio, dp, tmin, tmax = local_polish(x0, lo, hi)
        print("%4d-%-5d um %-13s %8.4f %9.0f %9.2f-%.2f" % (lo, hi, "", ratio, dp, tmin, tmax), " ", note)
