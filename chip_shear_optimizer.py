"""
chip_shear_optimizer.py  --  ENLR 5131, Assignment 3, Part H
Redesign the honeycomb chip so every seeded segment sees nearly the same floor
wall shear stress, by choosing the WIDTH of each straight segment.

Language / libraries : Python 3.9+, numpy, scipy (>= 1.9), matplotlib
                       Tested with Python 3.12.3, numpy 2.4.4, scipy 1.17.1, matplotlib 3.10.8
How to run           : put this file in the same folder as chip_network_solver.py (Part B), then
                           pip install numpy scipy matplotlib
                           python chip_shear_optimizer.py              full run, about 3-4 minutes
                           python chip_shear_optimizer.py --quick      shorter search, about 1 minute
                           python chip_shear_optimizer.py --variants   also re-optimizes with other width limits
Outputs              : printed report, partH_results.csv, partH_widths_map.png,
                       partH_shear_before_after.png
Units                : SI inside the code (m, Pa, s); widths are printed in micrometres.
"""
import sys
import csv
import numpy as np
from scipy.optimize import differential_evolution, minimize, NonlinearConstraint
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

import chip_network_solver as partB          # the team's Part B code, reused as-is

# --------------------------------------------------------------------------
# 1. DESIGN SPECIFICATION (Part H table) AND SETTINGS
# --------------------------------------------------------------------------
Q_DESIGN  = 1.0e-6/60.0       # inlet flow from Part E: 1.0 mL/min           [m^3/s]
H_FIX     = 200e-6            # channel height, fixed (single-layer)          [m]
L_FIX     = 2.0e-3            # segment length, fixed                         [m]
W_MIN, W_MAX = 200e-6, 1000e-6   # allowed segment widths                     [m]
TAU_MIN, TAU_MAX = 0.4, 2.2   # target shear window, Yourek et al. 4-22 dyn/cm^2 [Pa]
DP_MAX    = 500.0             # maximum total pressure drop A -> B            [Pa]
RE_MAX    = 2000.0            # laminar limit (conservative)                  [-]
W_PORT    = 500e-6            # outlet port width used for the exit loss      [m]
W_UNSEEDED = 500e-6           # width kept for the 7 zero-flow mid-line segments [m]
SEED      = 1                 # random seed so the search is repeatable
QUICK     = "--quick" in sys.argv
VARIANTS  = "--variants" in sys.argv


# --------------------------------------------------------------------------
# 2. CORRECTED PHYSICS (valid for any h <= w, i.e. the whole Part H range)
# --------------------------------------------------------------------------
_K = np.arange(1, 200, 2)[:, None]            # odd k = 1, 3, ..., 199 (column vector)
# Tail of sum_{odd k > 199} 1/k^3 (where tanh = 1), approx 1/(4*200^2); added so the
# truncation error is ~1e-9 instead of ~1e-5.
_TAIL3 = 1.0/(4*200.0**2)

def flow_factor_exact(w, h):
    """Exact factor f in Q = f * w h^3 G / (12 mu) for a rectangular duct (exact for any
    aspect ratio; converges fastest when h <= w).
    f = 1 - (192/pi^5)(h/w) * sum_{k odd} tanh(k pi w / 2h) / k^5   (White; Bruus)
    (Bruus's approximation is f ~ 1 - 0.630 h/w.)"""
    w = np.atleast_1d(w).astype(float); h = np.atleast_1d(h).astype(float)*np.ones_like(w)
    a = _K*np.pi*w[None, :]/(2*h[None, :])
    return 1.0 - (192/np.pi**5)*(h/w)*np.sum(np.tanh(a)/_K**5, axis=0)

def resistance_exact(mu, w, h, L):
    """Hydraulic resistance R = 12 mu L / (w h^3 f), exact series f.  [Pa s/m^3]"""
    return 12*mu*L/(w*h**3*flow_factor_exact(w, h))

def floor_shear_exact(dP_per_L, w, h, nterms=None):
    """Floor wall shear averaged across the width [Pa], same formula as Part B,
       tau = (4 h G/pi^2) * sum_{k odd} (1/k^2)(1 - tanh(a)/a),  a = k pi w/(2h),
    but with sum(1/k^2) = pi^2/8 done exactly, so only the fast 1/k^3 part is
    summed numerically.  (nterms is accepted so the Part B call still works.)"""
    w = np.atleast_1d(w).astype(float); h = np.atleast_1d(h).astype(float)*np.ones_like(w)
    G = np.abs(np.atleast_1d(dP_per_L).astype(float))
    a = _K*np.pi*w[None, :]/(2*h[None, :])
    S = np.pi**2/8 - np.sum(np.tanh(a)/(a*_K**2), axis=0) - (2*h/(np.pi*w))*_TAIL3
    return 4*h*G/np.pi**2*S

def centreline_shear(dP_per_L, w, h):
    """Floor shear at the middle of the floor (the highest value any cell sees) [Pa]."""
    w = np.atleast_1d(w).astype(float); h = np.atleast_1d(h).astype(float)*np.ones_like(w)
    G = np.abs(np.atleast_1d(dP_per_L).astype(float))
    a = _K*np.pi*w[None, :]/(2*h[None, :])
    sech = 2*np.exp(-a)/(1 + np.exp(-2*a))       # 1/cosh(a), written so it cannot overflow
    S = np.pi**2/8 - np.sum(sech/_K**2, axis=0)
    return 4*h*G/np.pi**2*S


# --------------------------------------------------------------------------
# 3. ONE CALL = ONE NETWORK SOLUTION  (wraps the Part B solver)
# --------------------------------------------------------------------------
PHYSICS = {"partB": (partB.resistance, partB.floor_avg_shear),       # original formulas
           "exact": (resistance_exact, floor_shear_exact)}           # corrected formulas

def solve(segs, n_nodes, w, h, L, q_in=Q_DESIGN, physics="exact", w_port=W_PORT,
          minor_losses=True):
    """Run the Part B solver with the chosen formulas, then add Part H outputs.
    Returns the Part B result dictionary plus:
        dP_total  total pressure drop A -> B incl. node and exit losses   [Pa]
        tau_c     centreline floor shear in every segment                 [Pa]"""
    R_fun, tau_fun = PHYSICS[physics]
    saved = partB.resistance, partB.floor_avg_shear
    partB.resistance, partB.floor_avg_shear = R_fun, tau_fun   # swap formulas in...
    try:
        res = partB.solve_network(segs, n_nodes, w, h, L, q_in=q_in, minor_losses=minor_losses)
    finally:
        partB.resistance, partB.floor_avg_shear = saved        # ...and always put them back
    if w_port is None:                       # Part B convention: port = feeding segment
        feed = [i for i, s in enumerate(segs) if n_nodes - 1 in s][0]
        w_port = w[feed]
    v_exit = q_in/(w_port*H_FIX)
    res["dP_exit"] = partB.K_EXIT*partB.RHO*v_exit**2/2
    res["dP_total"] = res["p"][0] + res["dP_exit"]             # p at A (B = 0) + exit loss
    res["tau_c"] = centreline_shear(np.abs(res["dP_f"])/L, w, h)
    return res


# --------------------------------------------------------------------------
# 4. NETWORK, SYMMETRY GROUPS, AND WHICH SEGMENTS ARE SEEDED
# --------------------------------------------------------------------------
PTS, SEGS = partB.build_honeycomb()
N_NODES, N_SEG = len(PTS), len(SEGS)
MID = np.array([(PTS[a] + PTS[b])/2 for a, b in SEGS])        # segment midpoints (drawing units)

def symmetry_groups():
    """Segments that are mirror images (top/bottom, left/right) share one width.
    Returns an array: group number of every segment."""
    xsum = PTS[:, 0].min() + PTS[:, 0].max()
    lookup = {tuple(np.round(m, 3)): i for i, m in enumerate(MID)}
    grp = -np.ones(N_SEG, int); g = 0
    for i, m in enumerate(MID):
        if grp[i] >= 0:
            continue
        for p in [m, (m[0], -m[1]), (xsum - m[0], m[1]), (xsum - m[0], -m[1])]:
            grp[lookup[tuple(np.round(p, 3))]] = g
        g += 1
    return grp

GROUP = symmetry_groups()
# The 7 vertical segments on the horizontal mid-line (3, 12, 23, 34, 45, 56, 65) carry
# zero flow in ANY top/bottom-symmetric design, so we do not seed them and do not
# optimize their width.  Every other segment (60 of them) is seeded.
MIDLINE = np.abs(MID[:, 1]) < 1e-6
SEEDED = ~MIDLINE
DESIGN_GROUPS = sorted(set(GROUP[SEEDED]))                   # 15 independent widths

def widths_from_x(x_um):
    """Design vector (15 widths in um) -> width of all 67 segments [m]."""
    w = np.full(N_SEG, W_UNSEEDED)
    for g, xv in zip(DESIGN_GROUPS, x_um):
        w[GROUP == g] = xv*1e-6
    return w

def x_from_widths(w):
    return np.array([w[GROUP == g][0]*1e6 for g in DESIGN_GROUPS])


# --------------------------------------------------------------------------
# 5. OBJECTIVE AND CONSTRAINTS
# --------------------------------------------------------------------------
_cache = {}
def evaluate(x_um, h=H_FIX, L=L_FIX, q_in=Q_DESIGN):
    """Solve the network for design x (cached, because the optimizer asks for the
    objective and the constraints of the same design separately)."""
    key = (np.asarray(x_um, float).tobytes(), h, L, q_in)
    if key not in _cache:
        if len(_cache) > 5000:
            _cache.clear()
        w = widths_from_x(x_um)
        _cache[key] = solve(SEGS, N_NODES, w, np.full(N_SEG, h), np.full(N_SEG, L), q_in=q_in)
    return _cache[key]

def shear_ratio(res, seeded=SEEDED):
    t = res["tau"][seeded]
    return t.max()/t.min()

def objective(x_um):
    return np.log(shear_ratio(evaluate(x_um)))            # log makes the search better behaved

def constraint_values(x_um):
    r = evaluate(x_um); t = r["tau"][SEEDED]
    return np.array([r["dP_total"], t.min(), t.max(), r["Re"].max()])

CON_LO = [-np.inf, TAU_MIN, -np.inf, -np.inf]
CON_HI = [DP_MAX, np.inf, TAU_MAX, RE_MAX]


def optimize(w_lo=W_MIN, w_hi=W_MAX, label="", verbose=True):
    """Global search (differential evolution) then local polish (SLSQP).
    The polish uses the 'epigraph' form: minimise (b - a) subject to
    a <= ln(tau_i) <= b for every seeded segment, which is smooth even though
    max/min are not."""
    nx = len(DESIGN_GROUPS)
    nlc = NonlinearConstraint(constraint_values, CON_LO, CON_HI)
    de = differential_evolution(objective, [(w_lo*1e6, w_hi*1e6)]*nx, constraints=(nlc,),
                                seed=SEED, maxiter=80 if QUICK else 300, popsize=15,
                                tol=1e-8, polish=False)
    lt = lambda z: np.log(evaluate(z[:nx])["tau"][SEEDED])
    z0 = np.r_[de.x, lt(de.x).min(), lt(de.x).max()]
    cons = [{"type": "ineq", "fun": lambda z: lt(z) - z[nx]},            # ln tau_i >= a
            {"type": "ineq", "fun": lambda z: z[nx + 1] - lt(z)},        # ln tau_i <= b
            {"type": "ineq", "fun": lambda z: DP_MAX - evaluate(z[:nx])["dP_total"]},
            {"type": "ineq", "fun": lambda z: np.array([z[nx] - np.log(TAU_MIN),
                                                         np.log(TAU_MAX) - z[nx + 1]])}]
    bnds = [(w_lo*1e6, w_hi*1e6)]*nx + [(None, None)]*2
    loc = minimize(lambda z: z[nx + 1] - z[nx], z0, method="SLSQP", bounds=bnds,
                   constraints=cons, options={"maxiter": 400, "ftol": 1e-12, "eps": 1e-4})
    x = np.clip(loc.x[:nx], w_lo*1e6, w_hi*1e6)
    ok = np.all(constraint_values(x) <= np.array(CON_HI) + 1e-9) and \
         np.all(constraint_values(x) >= np.array(CON_LO) - 1e-9)
    if not ok or objective(x) > de.fun:                  # keep the DE answer if polish failed
        x = de.x
    if verbose:
        print("  %-28s DE ratio %.4f -> polished %.4f  (%s)" %
              (label, np.exp(de.fun), np.exp(objective(x)), loc.message))
    return x


# --------------------------------------------------------------------------
# 6. VERIFICATION  (report section H2)
# --------------------------------------------------------------------------
def check_reproduces_partB():
    """(a) With the ORIGINAL formulas, the wrapper must give Part B's numbers exactly."""
    w = np.full(N_SEG, partB.W0); h = np.full(N_SEG, partB.H0); L = np.full(N_SEG, partB.L0)
    ref = partB.solve_network(SEGS, N_NODES, w, h, L)
    ip = partB.inlet_pressure(ref, SEGS, w, h)
    new = solve(SEGS, N_NODES, w, h, L, q_in=partB.Q_IN, physics="partB", w_port=None)
    dq = np.max(np.abs(new["q"] - ref["q"]))/partB.Q_IN
    dt = np.max(np.abs(new["tau"] - ref["tau"]))
    ddp = abs(new["dP_total"] - ip["dP_total"])
    print("  (a) original formulas: max |dQ|/Q = %.1e, max |d tau| = %.1e Pa, |d dP| = %.1e Pa"
          % (dq, dt, ddp))
    print("      Part B: tau(seg 1) = %.4f Pa, dP(A->B) = %.2f Pa" % (ref["tau"][0], ip["dP_total"]))
    return ref, ip

def check_physics_change(ref, ip):
    """(b) With the CORRECTED formulas on the original geometry: what moves, and why."""
    w = np.full(N_SEG, partB.W0); h = np.full(N_SEG, partB.H0); L = np.full(N_SEG, partB.L0)
    new = solve(SEGS, N_NODES, w, h, L, q_in=partB.Q_IN, physics="exact")
    flow = SEEDED & (np.abs(ref["q"]) > 1e-15)
    rel_tau = new["tau"][flow]/ref["tau"][flow] - 1
    f_b = 1 - 0.630*0.4; f_e = flow_factor_exact(partB.W0, partB.H0)[0]
    print("  (b) corrected formulas: tau changes by %+.2f%% to %+.2f%% (series tail),"
          % (100*rel_tau.min(), 100*rel_tau.max()))
    print("      R changes by %+.3f%% (Bruus f = %.4f vs exact f = %.5f)"
          % (100*(f_b/f_e - 1), f_b, f_e))
    print("      tau(seg 1) = %.4f Pa;  dP(A->B) = %.2f Pa (Part B %.2f);  max |dQ|/Q = %.1e"
          % (new["tau"][0], new["dP_total"], ip["dP_total"],
             np.max(np.abs(new["q"] - ref["q"]))/partB.Q_IN))
    return new

def check_square_duct():
    """(c) Independent physics check: in a square duct all four walls are alike, so the
    average shear on the floor equals the perimeter average, G*A/P = G*h/4, exactly."""
    G, h = 1.0e4, 200e-6
    exact = G*h/4
    print("  (c) square duct, floor shear / (G h/4): Part B = %.4f, corrected = %.6f"
          % (partB.floor_avg_shear(G, h, h)[0]/exact, floor_shear_exact(G, h, h)[0]/exact))

def check_two_parallel_channels():
    """(d) Known answer: two identical-section channels of length L1 and L2 in parallel
    split the flow as Q1/Q = L2/(L1 + L2) (no node losses: both enter the outlet)."""
    segs = [(0, 1), (0, 1)]
    w = np.array([500e-6, 500e-6]); h = np.full(2, 200e-6); L = np.array([2e-3, 3e-3])
    r = solve(segs, 2, w, h, L, q_in=Q_DESIGN, physics="exact", w_port=500e-6)
    q1 = Q_DESIGN*L[1]/L.sum()
    dP = float(resistance_exact(partB.MU, w[0], h[0], L[0])[0])*q1
    print("  (d) two parallel channels: Q1/Q = %.6f (expected %.6f); dP = %.4f Pa (expected %.4f)"
          % (r["q"][0]/Q_DESIGN, q1/Q_DESIGN, r["p"][0], dP))

def check_optimizer_two_channels():
    """(e) Known answer for the OPTIMIZER: in two parallel channels both see the same
    pressure drop, so equal floor shear needs c(w1)/c(w2) = L1/L2, where
    c = floor shear / (G h/2) depends only on h/w.  The optimizer should reach a
    shear ratio of 1 and satisfy that condition."""
    segs = [(0, 1), (0, 1)]; h = np.full(2, 200e-6); L = np.array([2.0e-3, 2.4e-3])
    def tau_of(xw):
        r = solve(segs, 2, np.asarray(xw)*1e-6, h, L, physics="exact", w_port=500e-6)
        return r["tau"]
    f = lambda xw: np.log(tau_of(xw).max()/tau_of(xw).min())
    r = minimize(f, [500, 500], bounds=[(200, 1000)]*2, method="L-BFGS-B")
    c = lambda wm: floor_shear_exact(1.0, wm*1e-6, 200e-6)[0]/(1.0*200e-6/2)
    print("  (e) optimizer on two channels: ratio = %.5f; c(w1)/c(w2) = %.4f (target L1/L2 = %.4f)"
          % (np.exp(r.fun), c(r.x[0])/c(r.x[1]), L[0]/L[1]))

def check_solution(res):
    """Mass conservation and symmetry for any solved design (same tests as Part B)."""
    return partB.check_mass(res, Q_DESIGN), partB.check_symmetry(res, PTS, SEGS)


# --------------------------------------------------------------------------
# 7. ROBUSTNESS  (report section H4)
# --------------------------------------------------------------------------
def fabrication_monte_carlo(x_um, sigma_um=5.0, n=2000):
    """Random width errors (normal, sigma) on EVERY segment independently, so the
    mirror symmetry is broken too.  Returns ratio percentiles and how often a
    constraint is violated."""
    rng = np.random.default_rng(SEED)
    w0 = widths_from_x(x_um); ratios, viol = [], 0
    for _ in range(n):
        w = np.clip(w0 + rng.normal(0, sigma_um*1e-6, N_SEG), 150e-6, None)
        r = solve(SEGS, N_NODES, w, np.full(N_SEG, H_FIX), np.full(N_SEG, L_FIX))
        t = r["tau"][SEEDED]; ratios.append(t.max()/t.min())
        viol += (r["dP_total"] > DP_MAX) or (t.min() < TAU_MIN) or (t.max() > TAU_MAX)
    ratios = np.array(ratios)
    return np.percentile(ratios, [50, 95]), viol/n

def junction_length_sensitivity(x_um):
    """Unmodelled geometry: where channels meet at 120 deg, the straight walls of a
    channel of width w end about w/(2*sqrt(3)) short of the node centre.  If the mask
    keeps the node positions, each straight segment shortens by about w/sqrt(3)."""
    w = widths_from_x(x_um); L = L_FIX - w/np.sqrt(3)
    r = solve(SEGS, N_NODES, w, np.full(N_SEG, H_FIX), L)
    return shear_ratio(r), r["dP_total"]


# --------------------------------------------------------------------------
# 8. OUTPUT: TABLE AND FIGURES
# --------------------------------------------------------------------------
def write_csv(base, opt, w_opt, fname="partH_results.csv"):
    hdr = ["seg", "from", "to", "seeded", "w_orig [um]", "w_opt [um]",
           "tau_orig [dyn/cm2]", "tau_opt [dyn/cm2]", "tau_opt_centre [dyn/cm2]",
           "Q_opt [m3/s]", "Re_opt [-]"]
    with open(fname, "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(hdr)
        for i, (a, b) in enumerate(SEGS):
            wr.writerow([i + 1, a + 1, b + 1, int(SEEDED[i]), 500, round(w_opt[i]*1e6),
                         round(base["tau"][i]*10, 3), round(opt["tau"][i]*10, 3),
                         round(opt["tau_c"][i]*10, 3), "%.4e" % abs(opt["q"][i]),
                         round(opt["Re"][i], 2)])

def plot_widths_map(w, res, fname="partH_widths_map.png"):
    """Labeled copy of Figure 1: line thickness ~ width, colour = shear, label = width."""
    fig, ax = plt.subplots(figsize=(14, 7.2))
    lines = [[PTS[a], PTS[b]] for a, b in SEGS]
    tau_d = res["tau"]*10
    seeded_lines = [l for l, s in zip(lines, SEEDED) if s]
    lc = LineCollection(seeded_lines, array=tau_d[SEEDED], cmap="viridis",
                        linewidths=1.5 + 9*(w[SEEDED]*1e6 - 200)/800)
    lc.set_clim(7.5, 9.5); ax.add_collection(lc)
    ax.add_collection(LineCollection([l for l, s in zip(lines, SEEDED) if not s],
                                     colors="0.6", linestyles="dashed", linewidths=1.2))
    for i in range(N_SEG):
        m = MID[i]; vertical = abs(PTS[SEGS[i][0]][0] - PTS[SEGS[i][1]][0]) < 1e-6
        dx, dy = (0.16, 0.0) if vertical else (0.0, 0.17 if m[1] >= 0 else -0.17)
        txt = "%d\n%d" % (i + 1, round(w[i]*1e6)) if SEEDED[i] else "%d\n(no flow)" % (i + 1)
        ax.text(m[0] + dx, m[1] + dy, txt, fontsize=6.3, ha="left" if vertical else "center",
                va="center", color="k" if SEEDED[i] else "0.45")
    ax.autoscale(); ax.set_aspect("equal"); ax.axis("off")
    cb = fig.colorbar(lc, ax=ax, shrink=0.6); cb.set_label("floor shear stress [dyn/cm$^2$]")
    ax.text(PTS[0][0] - 0.35, PTS[0][1], "A", fontsize=14, fontweight="bold", va="center")
    ax.text(PTS[-1][0] + 0.15, PTS[-1][1], "B", fontsize=14, fontweight="bold", va="center")
    ax.set_title("Optimized widths (label: segment number / width in um; line thickness ~ width)",
                 fontsize=10)
    plt.tight_layout(); plt.savefig(fname, dpi=200); plt.close()

def plot_before_after(base, opt, fname="partH_shear_before_after.png"):
    idx = np.where(SEEDED)[0]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.axhspan(4, 22, color="tab:green", alpha=0.12, label="target 4-22 dyn/cm$^2$ (Yourek et al.)")
    ax.semilogy(idx + 1, base["tau"][idx]*10, "o", ms=5, mfc="none", label="original (all 500 um)")
    ax.semilogy(idx + 1, opt["tau"][idx]*10, "s", ms=4, label="optimized widths")
    ax.set_xlabel("segment number (60 seeded segments)"); ax.set_ylabel("floor shear [dyn/cm$^2$]")
    ax.set_xlim(0, 68); ax.set_ylim(0.1, 40); ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower left", fontsize=8)          # lower left is empty, so no points are hidden
    plt.tight_layout(); plt.savefig(fname, dpi=200); plt.close()


# --------------------------------------------------------------------------
# 9. MAIN PROGRAM
# --------------------------------------------------------------------------
if __name__ == "__main__":
    print("H2. VERIFICATION")
    ref, ip = check_reproduces_partB()
    base = check_physics_change(ref, ip)
    check_square_duct()
    check_two_parallel_channels()
    check_optimizer_two_channels()

    print("\nH3. OPTIMIZATION  (%d design widths, %d seeded segments)" % (len(DESIGN_GROUPS), SEEDED.sum()))
    x_opt = optimize(label="widths 200-1000 um")
    x_fab = np.round(x_opt/10)*10                         # snap to a 10 um mask grid
    opt = evaluate(x_fab); w_fab = widths_from_x(x_fab)
    m, s = check_solution(opt)
    flow = SEEDED
    print("\n  %-34s %12s %12s" % ("", "original", "optimized"))
    rows = [("shear ratio max/min (seeded)", shear_ratio(base), shear_ratio(opt)),
            ("min shear [dyn/cm2]", base["tau"][flow].min()*10, opt["tau"][flow].min()*10),
            ("max shear [dyn/cm2]", base["tau"][flow].max()*10, opt["tau"][flow].max()*10),
            ("centreline-shear ratio", base["tau_c"][flow].max()/base["tau_c"][flow].min(),
             opt["tau_c"][flow].max()/opt["tau_c"][flow].min()),
            ("segments inside 4-22 dyn/cm2", np.sum((base["tau"][flow] >= 0.4) & (base["tau"][flow] <= 2.2)),
             np.sum((opt["tau"][flow] >= 0.4) & (opt["tau"][flow] <= 2.2))),
            ("total dP A->B [Pa]", base["dP_total"], opt["dP_total"]),
            ("max Reynolds number", base["Re"].max(), opt["Re"].max())]
    for name, a, b in rows:
        print("  %-34s %12.4g %12.4g" % (name, a, b))
    print("  optimizer (unrounded) ratio %.4f; rounded to 10 um ratio %.4f" %
          (np.exp(objective(x_opt)), shear_ratio(opt)))
    print("  checks on optimized design: max node imbalance/Q = %.1e, max mirror difference/Q = %.1e" % (m, s))
    print("\n  widths by symmetry group (um):")
    for g, xv in zip(DESIGN_GROUPS, x_fab):
        members = [i + 1 for i in np.where(GROUP == g)[0]]
        print("    segments %-18s %5.0f um   tau = %.2f dyn/cm2" %
              (",".join(map(str, members)), xv, opt["tau"][GROUP == g][0]*10))
    at_lo = np.sum(np.isclose(x_fab, W_MIN*1e6)); at_hi = np.sum(x_fab >= W_MAX*1e6 - 10)
    print("  groups at 200 um: %d of %d;  groups within 10 um of 1000 um: %d" % (at_lo, len(x_fab), at_hi))

    print("\nH4. ROBUSTNESS")
    for hh in (0.9*H_FIX, 1.1*H_FIX):
        r = evaluate(x_fab, h=hh)
        print("  height %.0f um: ratio %.3f, shear %.2f-%.2f dyn/cm2, dP %.0f Pa" %
              (hh*1e6, shear_ratio(r), r["tau"][SEEDED].min()*10, r["tau"][SEEDED].max()*10, r["dP_total"]))
    for qq in (0.5, 1.27):
        r = evaluate(x_fab, q_in=qq*1e-6/60)
        print("  Q = %.2f mL/min: shear %.2f-%.2f dyn/cm2, dP %.0f Pa" %
              (qq, r["tau"][SEEDED].min()*10, r["tau"][SEEDED].max()*10, r["dP_total"]))
    pct, viol = fabrication_monte_carlo(x_fab, n=500 if QUICK else 2000)
    print("  width errors sigma = 5 um: ratio median %.3f, 95th pct %.3f; constraint violated in %.1f%% of chips"
          % (pct[0], pct[1], 100*viol))
    jr, jdp = junction_length_sensitivity(x_fab)
    print("  junction shortening L_eff = L - w/sqrt(3): ratio %.3f, dP %.0f Pa (not re-optimized)" % (jr, jdp))
    print("  cells per segment at 8e3 cells/cm2: 200 um wide = %.0f, 1000 um wide = %.0f" %
          (8e3*2e-1*0.02, 8e3*2e-1*0.1))

    if VARIANTS:
        print("\n  Re-optimized with other width limits:")
        for lo, hi in [(250e-6, 950e-6), (150e-6, 1200e-6)]:
            xv = optimize(lo, hi, label="widths %.0f-%.0f um" % (lo*1e6, hi*1e6))
            rv = evaluate(xv)
            print("    ratio %.3f, dP %.0f Pa, widths %s" %
                  (shear_ratio(rv), rv["dP_total"], np.round(xv).astype(int)))

    write_csv(base, opt, w_fab)
    plot_widths_map(w_fab, opt)
    plot_before_after(base, opt)
    print("\nWrote partH_results.csv, partH_widths_map.png, partH_shear_before_after.png")