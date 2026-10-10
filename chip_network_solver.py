"""
chip_network_solver.py  --  ENLR 5131, Assignment 3, Part B
Flow through a honeycomb microfluidic network (hydraulic-electric analogy).

Language / libraries : Python 3.9+, numpy, matplotlib   (pip install numpy matplotlib)
How to run           : python chip_network_solver.py
Units                : SI everywhere (m, kg, s, Pa) unless a name says otherwise.

IDEA IN ONE LINE
    Each straight segment is a "resistor":  dP = R * Q   (like V = I * R).
    Conservation of mass at every node (flow in = flow out) gives one equation
    per node.  Solving all of them together gives every node pressure, and
    from the pressures we get every segment's flow, speed and shear stress.
"""
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")                      # draw to file, no window needed
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# --------------------------------------------------------------------------
# 1. INPUTS (given data, already in SI units)
# --------------------------------------------------------------------------
MU        = 0.72e-3      # viscosity of medium at 37 C            [Pa s]
RHO       = 1000.0       # density                                 [kg/m^3]
Q_IN      = 1.0e-6/60.0  # 1.0 mL/min -> m^3/s                     [m^3/s]
P_OUT     = 1.0e5        # pressure at outlet B, 1 bar absolute    [Pa]
W0, H0, L0 = 500e-6, 200e-6, 2.0e-3   # default width, height, length of EVERY segment [m]
K_NODE    = 0.4          # loss coefficient, flow entering a node
K_EXIT    = 1.0          # loss coefficient, exit at outlet B
PA_TO_DYN_CM2 = 10.0     # 1 Pa = 10 dyn/cm^2


# --------------------------------------------------------------------------
# 2. NETWORK DESCRIPTION  (list of segments, each joining two numbered nodes)
# --------------------------------------------------------------------------
def build_honeycomb():
    """Honeycomb of Figure 1: rows of 5, 6, 5 hexagons plus a triangle at each end.
    Returns node coordinates (drawing units only, used for plotting) and a list of
    segments (node_from, node_to) using 0-based node numbers.
    Node 0 is inlet A; the last node is outlet B."""
    s = 1.0; d = np.sqrt(3)/2*s                       # hexagon edge, half-width
    def corners(cx, cy):
        return [(cx, cy+s), (cx+d, cy+s/2), (cx+d, cy-s/2),
                (cx, cy-s), (cx-d, cy-s/2), (cx-d, cy+s/2)]
    centers  = [(2*d*k + d,  1.5*s) for k in range(5)]    # top row
    centers += [(2*d*k,      0.0)   for k in range(6)]    # middle row
    centers += [(2*d*k + d, -1.5*s) for k in range(5)]    # bottom row
    key = lambda p: (round(p[0], 3), round(p[1], 3))
    edges = set()
    for c in centers:
        v = corners(*c)
        for i in range(6):
            edges.add(tuple(sorted([key(v[i]), key(v[(i+1) % 6])])))
    xl = min(p[0] for e in edges for p in e); xr = max(p[0] for e in edges for p in e)
    A, B = key((xl - d, 0.0)), key((xr + d, 0.0))         # inlet / outlet apex
    for p in [(xl, s/2), (xl, -s/2)]: edges.add(tuple(sorted([A, key(p)])))
    for p in [(xr, s/2), (xr, -s/2)]: edges.add(tuple(sorted([B, key(p)])))
    pts = sorted({p for e in edges for p in e}, key=lambda p: (p[0], -p[1]))
    idx = {p: i for i, p in enumerate(pts)}
    segs = sorted(((min(idx[a], idx[b]), max(idx[a], idx[b])) for a, b in edges),
                  key=lambda e: ((pts[e[0]][0] + pts[e[1]][0])/2,
                                 -(pts[e[0]][1] + pts[e[1]][1])/2))
    return np.array(pts), segs


# --------------------------------------------------------------------------
# 3. PHYSICS FORMULAS
# --------------------------------------------------------------------------
def resistance(mu, w, h, L):
    """Hydraulic resistance of a rectangular channel (h <= w), Bruus (2008):
       R = 12 mu L / ( w h^3 (1 - 0.630 h/w) )      [Pa s / m^3]
    Works on arrays, so every segment can have its own w, h, L."""
    return 12*mu*L / (w*h**3*(1 - 0.630*h/w))

def hydraulic_diameter(w, h):
    return 2*w*h/(w + h)

def floor_avg_shear(dP_per_L, w, h, nterms=81):
    """Wall shear stress on the channel FLOOR, averaged across the width [Pa].
    Exact series result for a rectangular duct (derived from the exact velocity
    solution, e.g. White, Viscous Fluid Flow):
       tau = (4 h G / pi^2) * sum_{k odd} (1/k^2) * (1 - (2h/(k pi w)) tanh(k pi w/(2h)))
    where G = pressure drop per unit length.   Works on arrays."""
    w = np.atleast_1d(w).astype(float); h = np.atleast_1d(h).astype(float)
    G = np.atleast_1d(dP_per_L).astype(float)
    tau = np.zeros_like(G)
    for k in range(1, nterms + 1, 2):
        a = k*np.pi*w/(2*h)
        tau += (1.0/k**2)*(1.0 - np.tanh(a)/a)
    return 4*h*np.abs(G)/np.pi**2 * tau


# --------------------------------------------------------------------------
# 4. THE SOLVER
# --------------------------------------------------------------------------
def solve_network(segs, n_nodes, w, h, L, q_in=Q_IN, mu=MU, rho=RHO,
                  k_node=K_NODE, minor_losses=True, inlet=0, outlet=None):
    """Solve for node pressures (relative to outlet = 0) and segment flows.

    Unknowns : pressure at each node.
    Equations: for every node, (flow in) - (flow out) = 0, except inlet (+q_in)
               and outlet (-q_in).
    Minor losses make the segment law slightly nonlinear
               dP = R Q + K rho Q|Q| / (2 A^2),
    so we repeat the solve with R_eff = R + K rho |Q| / (2 A^2) until it settles.
    """
    outlet = n_nodes - 1 if outlet is None else outlet
    ns = len(segs)
    n_from = np.array([s[0] for s in segs]); n_to = np.array([s[1] for s in segs])
    area = w*h
    R = resistance(mu, w, h, L)

    # incidence matrix: +1 where a segment leaves a node, -1 where it arrives
    inc = np.zeros((n_nodes, ns))
    inc[n_from, np.arange(ns)] = +1.0
    inc[n_to,   np.arange(ns)] = -1.0
    b = np.zeros(n_nodes); b[inlet] = q_in; b[outlet] = -q_in   # net flow injected

    R_eff = R.copy(); q = np.zeros(ns)
    for it in range(200):
        g = 1.0/R_eff                                   # conductance = 1/resistance
        lap = inc @ np.diag(g) @ inc.T                  # node-by-node system matrix
        keep = [i for i in range(n_nodes) if i != outlet]   # fix p_outlet = 0
        p = np.zeros(n_nodes)
        p[keep] = np.linalg.solve(lap[np.ix_(keep, keep)], b[keep])
        q_new = g*(p[n_from] - p[n_to])                 # flow in each segment
        if not minor_losses:
            q = q_new; break
        # loss applies where flow ENTERS a node (the downstream end), except entering B,
        # where the exit loss is handled separately.
        down = np.where(q_new >= 0, n_to, n_from)
        k_seg = np.where(down == outlet, 0.0, k_node)
        R_eff = R + k_seg*rho*np.abs(q_new)/(2*area**2)
        if it > 0 and np.max(np.abs(q_new - q)) < 1e-15:
            q = q_new; break
        q = q_new
    else:
        raise RuntimeError("minor-loss iteration did not converge")

    # ---- derived results, per segment ----
    v    = np.abs(q)/area                               # average velocity   [m/s]
    Dh   = hydraulic_diameter(w, h)
    Re   = rho*v*Dh/mu
    dP_f = R*q                                          # frictional drop (signed, along from->to)
    down = np.where(q >= 0, n_to, n_from)
    k_seg = np.where(down == outlet, 0.0, k_node) if minor_losses else np.zeros(ns)
    dP_m = k_seg*rho*q*np.abs(q)/(2*area**2)            # minor drop entering downstream node
    tau  = floor_avg_shear(np.abs(dP_f)/L, w, h)        # floor wall shear   [Pa]
    return dict(p=p, q=q, v=v, Re=Re, dP_f=dP_f, dP_m=dP_m, tau=tau,
                n_from=n_from, n_to=n_to, inc=inc, b=b, area=area, R=R)


def inlet_pressure(res, segs, w, h, q_in=Q_IN, rho=RHO, k_exit=K_EXIT, p_out=P_OUT,
                   outlet=None, inlet=0):
    """Absolute pressure at inlet A.  Exit loss uses the outlet-tube velocity, assumed
    to have the same cross-section as the segment that feeds B.  Also splits the drop
    into friction / node losses / exit loss along the main flow path (following the
    largest-flow segment out of every node)."""
    n_nodes = len(res["p"]); outlet = n_nodes - 1 if outlet is None else outlet
    feed = [i for i, s in enumerate(segs) if outlet in s][0]
    v_exit = q_in/(w[feed]*h[feed])
    dP_exit = k_exit*rho*v_exit**2/2
    p_A = p_out + res["p"][inlet] + dP_exit            # pressure at A (absolute)
    # follow the main path
    node, fric, minor, path = inlet, 0.0, 0.0, []
    while node != outlet:
        best, best_q = None, -1.0
        for i, (a, b2) in enumerate(segs):
            if node in (a, b2):
                qdir = res["q"][i] if node == a else -res["q"][i]   # flow leaving 'node'?
                if qdir > best_q: best, best_q = i, qdir
        path.append(best)
        fric  += abs(res["dP_f"][best]); minor += abs(res["dP_m"][best])
        a, b2 = segs[best]; node = b2 if node == a else a
    return dict(p_A=p_A, dP_exit=dP_exit, v_exit=v_exit, path=path,
                dP_fric_path=fric, dP_minor_path=minor,
                dP_total=p_A - p_out)


# --------------------------------------------------------------------------
# 5. VERIFICATION CHECKS  (B3)
# --------------------------------------------------------------------------
def check_mass(res, q_in=Q_IN):
    """(i) Net flow at every node should equal 0 (or +/- Q_in at inlet/outlet)."""
    net = res["inc"] @ res["q"] - res["b"]
    return np.max(np.abs(net))/q_in                      # relative to Q_in

def check_symmetry(res, pts, segs):
    """(ii) Mirror the network top<->bottom and left<->right; mirrored segments must
    carry the same |flow|.  Returns the largest relative difference."""
    mid = np.array([(pts[a] + pts[b])/2 for a, b in segs])
    xsum = pts[:, 0].min() + pts[:, 0].max()          # mirror line is x = xsum/2
    q = np.abs(res["q"]); worst = 0.0
    lookup = {tuple(np.round(m, 3)): i for i, m in enumerate(mid)}
    for i, m in enumerate(mid):
        for mir in [(m[0], -m[1]), (xsum - m[0], m[1]), (xsum - m[0], -m[1])]:
            j = lookup[tuple(np.round(mir, 3))]
            worst = max(worst, abs(q[i] - q[j])/Q_IN)
    return worst


# --------------------------------------------------------------------------
# 6. OUTPUT: TABLE AND COLOR-CODED PLOT
# --------------------------------------------------------------------------
def print_table(res, segs, path="segment_results.csv"):
    hdr = ["seg", "from", "to", "Q [m3/s]", "v [m/s]", "Re [-]", "tau [Pa]",
           "tau [dyn/cm2]", "dP_fric [Pa]", "dP_minor [Pa]"]
    rows = []
    for i, (a, b) in enumerate(segs):
        rows.append([i + 1, a + 1, b + 1, abs(res["q"][i]), res["v"][i], res["Re"][i],
                     res["tau"][i], res["tau"][i]*PA_TO_DYN_CM2,
                     abs(res["dP_f"][i]), abs(res["dP_m"][i])])
    print("{:>4}{:>5}{:>5}{:>12}{:>10}{:>8}{:>10}{:>14}{:>14}{:>14}".format(*hdr))
    for r in rows:
        print("{:>4d}{:>5d}{:>5d}{:>12.3e}{:>10.4f}{:>8.2f}{:>10.4f}{:>14.3f}{:>14.3f}{:>14.4f}".format(*r))
    with open(path, "w", newline="") as f:
        csv.writer(f).writerow(hdr); csv.writer(f).writerows(rows)

def plot_shear(pts, segs, tau, fname="shear_map.png", title="Floor wall shear stress"):
    """Color-coded diagram: each segment is colored by its shear stress [dyn/cm^2]."""
    tau_d = np.asarray(tau)*PA_TO_DYN_CM2
    lines = [[pts[a], pts[b]] for a, b in segs]
    fig, ax = plt.subplots(figsize=(13, 6))
    lc = LineCollection(lines, array=tau_d, cmap="viridis", linewidths=5)
    ax.add_collection(lc); ax.autoscale(); ax.set_aspect("equal"); ax.axis("off")
    cb = fig.colorbar(lc, ax=ax, shrink=0.7); cb.set_label("shear stress [dyn/cm$^2$]")
    ax.text(pts[0][0] - 0.1, pts[0][1] + 0.25, "A", fontsize=14, fontweight="bold")
    ax.text(pts[-1][0] + 0.1, pts[-1][1] + 0.25, "B", fontsize=14, fontweight="bold")
    ax.set_title(title); plt.tight_layout(); plt.savefig(fname, dpi=170); plt.close()


# --------------------------------------------------------------------------
# 7. MAIN PROGRAM
# --------------------------------------------------------------------------
if __name__ == "__main__":
    pts, segs = build_honeycomb()
    n_nodes, ns = len(pts), len(segs)
    # each segment stores its OWN width, height, length (so Part H can change them)
    w = np.full(ns, W0); h = np.full(ns, H0); L = np.full(ns, L0)

    res = solve_network(segs, n_nodes, w, h, L)                       # with minor losses
    res_f = solve_network(segs, n_nodes, w, h, L, minor_losses=False) # friction only
    print_table(res, segs)

    ip = inlet_pressure(res, segs, w, h)
    print("\nnodes = %d, segments = %d" % (n_nodes, ns))
    print("Inlet A pressure (absolute)      : %.1f Pa  (= %.5f bar)" % (ip["p_A"], ip["p_A"]/1e5))
    print("Total drop A -> B                : %.2f Pa" % ip["dP_total"])
    print("  friction along main path       : %.2f Pa" % ip["dP_fric_path"])
    print("  node (minor) losses, main path : %.2f Pa" % ip["dP_minor_path"])
    print("  exit loss at B                 : %.2f Pa  (v_exit = %.4f m/s)" % (ip["dP_exit"], ip["v_exit"]))

    print("\nCHECKS")
    print("  (i)  max node imbalance / Q_in           : %.2e" % check_mass(res))
    print("  (ii) max mirror-segment flow difference/Q : %.2e" % check_symmetry(res, pts, segs))
    plot_shear(pts, segs, res["tau"])

    # (iii) HAND CHECK of segment 1 (A -> node 2).  By symmetry it carries Q/2.
    q1 = Q_IN/2
    R1 = 12*MU*L0/(W0*H0**3*(1 - 0.630*H0/W0))        # same formula, done by hand
    dP1 = R1*q1                                        # friction drop
    v1 = q1/(W0*H0)
    Dh = 2*W0*H0/(W0 + H0)
    Re1 = RHO*v1*Dh/MU
    tau1 = 1.04*6*MU*q1/(W0*H0**2)                    # 1.04 = floor-average correction (h/w=0.4)
    print("\nHAND CHECK, segment 1   (hand | code)")
    print("  Q    [m3/s] : %.4e | %.4e" % (q1, abs(res['q'][0])))
    print("  dP   [Pa]   : %.3f | %.3f" % (dP1, abs(res['dP_f'][0])))
    print("  v    [m/s]  : %.4f | %.4f" % (v1, res['v'][0]))
    print("  Re          : %.2f | %.2f" % (Re1, res['Re'][0]))
    print("  tau  [Pa]   : %.4f | %.4f" % (tau1, res['tau'][0]))
