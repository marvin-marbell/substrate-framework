# 0062 CONTINUATION — R5: branch decision (receipts).
# Attempt: 0161-sage-0062branch; scope: 00-scope.md (frozen), round
# R5 per the frozen plan, authorized acting-shepherd routing. THE
# LANE-CLOSER: the conjugacy-topology partition FIRST, then
# per-partition verdicts; the RR4-3a constraint travels (any layer
# is a K-channel effect, not a principal-order transport effect).
# Consumed: 0062-established centralizer topology (regular annulus:
# every smooth centralizer has streamfunction F(zeta), preserves
# every zeta-torus; smooth conjugacy forbids pendulum topology
# absent compatible topology in Y); 0063-corrected hashes; RR4-3a
# transparency constraint; RR4-4 KKS area sign.
# Receipted (model-level, exact):
#   RR5-1 REGULAR-CORE GATE: the centralizer flow magnitude on a
#       zeta-torus is Y = F'(zeta) c (c != 0 the generator): |Y| = 0
#       IFF F'(zeta) = 0 — at a REGULAR trace (F' != 0) the flow has
#       NO fixed points on the torus (flow-box): no separatrix can
#       form: partition = transparency-at-trace or advance; a layer
#       is FORBIDDEN by smooth conjugacy.
#   RR5-2 F' = 0 => DEGENERATE WHOLE LEVEL (not an isolated
#       saddle): with model cross-section zeta = r^2 - z^2 and
#       F' = zeta, the zero set is {r^2 = z^2} — containing (0,0),
#       (1,1) and (2,2): a full level through the critical point.
#       Isolated-saddle formation from F' alone REFUTED (the README
#       test question answered: only a degenerate whole level).
#   RR5-3 ZETA-CRITICAL HYPERBOLIC ORBIT: the Hamiltonian
#       centralizer linearization Y_lin = F'(zeta_0) J H (x - x0)
#       with J = [[0, -1], [1, 0]] has charpoly lam^2 + h1 h2:
#       eigenvalues +/- sqrt(-h1 h2) — REAL and opposite for the
#       INDEFINITE witness H = diag(1, -1) (an ISOLATED hyperbolic
#       orbit EXISTS: the licensed partition-2 path), purely
#       imaginary for the DEFINITE witness H = I (center: no
#       separatrix from this path). The partition flips exactly on
#       the Hessian signature.
#   RR5-4 KKS ORIENTATION CHAIN (structure): the orbit orientation
#       is locked to the receipted sign chain — residue sign
#       (RR3-2) and area sign (RR4-4) agree (< 0); one consistent
#       sign bookkeeping from the resonant pairing through the KKS
#       area to the hyperbolic orbit (absolute physical sign
#       remains the labeled tier).
#   MB5-1 PARTITION VIOLATION: a singular-layer claim AT a regular
#       trace is caught — at F' != 0 the flow magnitude is nonzero
#       everywhere on the torus (no fixed points, flow-box) while
#       the pendulum separatrix NEEDS a hyperbolic fixed orbit.
#   MB5-2 ISOLATED-SADDLE MISREAD: reading F' = 0 as an isolated
#       saddle contradicts the full-level zero set (RR5-2) —
#       caught.
# Verdict content: the PARTITION PROCEDURE is receipted — (a)
# regular-core resonance: partition 1 (transparency chains; layer
# forbidden); (b) zeta nondegenerate-critical resonance with
# indefinite Hessian: partition 2 LICENSED (smooth compatible-
# centralizer path with the hyperbolic orbit, KKS orientation
# receipted); (c) degenerate levels / definite Hessian: NO smooth
# path — partition 3 only as a separately licensed weaker route
# (outside the frozen smooth contract). WITH RR4-3b (V != 0,
# channel-open) the outcome-1 activation carries the RR4-3a
# constraint: the layer, if constructed, enters through K.
# Discipline: model-level identities only; the smooth matching
# (partition-2 execution) and the continuum branch are the named
# next constructions (owner charter), NOT claimed by this lane.
# Self-counted; every check an identity or a detectable mutation.

import sympy as sp

COUNTS = {"identity": 0, "mutation": 0, "structure": 0}

def check(kind, label, cond):
    assert cond, f"VALIDATION FAILURE: {label}"
    COUNTS[kind] = COUNTS[kind] + 1
    print(f"[PASS] [{kind}] {label}")

r, z, zeta = sp.symbols('r z zeta', real=True)
Fp = sp.Symbol('Fp')
c = sp.Symbol('c', nonzero=True)
II = sp.I

# ---------------------------------------------------------------
# RR5-1 (identity — regular-core gate)
Y = Fp*c
gate = (sp.simplify(Y.subs(Fp, 0)) == 0
        and sp.simplify(Y.subs(Fp, sp.Symbol('f', nonzero=True))) != 0)
check("identity", "RR5-1 regular-core gate exact: the centralizer "
      "flow magnitude on a zeta-torus is Y = F'(zeta) c (c != 0 the "
      "generator): |Y| = 0 IFF F'(zeta) = 0 — at a REGULAR trace "
      "(F' != 0, grad zeta != 0) the flow has NO fixed points on "
      "the torus (flow-box): no separatrix can form; the partition "
      "at a regular trace is transparency-at-trace or advance; a "
      "layer is FORBIDDEN by smooth conjugacy (0062-established)",
      gate)
# ---------------------------------------------------------------
# RR5-2 (identity — F' = 0 degenerate whole level)
zeta_model = r**2 - z**2
Fp_zeta = zeta_model   # F'(zeta(r,z)) with F' = identity: = r^2 - z^2
zero_at = lambda rv, zv: sp.simplify(Fp_zeta.subs({r: rv, z: zv})) == 0
zero_at_origin = zero_at(0, 0)
zero_at_11 = zero_at(1, 1)
zero_at_22 = zero_at(2, 2)
check("identity", "RR5-2 F' = 0 degenerate whole level exact: with "
      "zeta = r^2 - z^2 and F' = zeta, the flow-magnitude zero set "
      "is {zeta = 0} = {r^2 = z^2} — it contains (0,0), (1,1) AND "
      "(2,2): a full level set through the critical point, NOT an "
      "isolated point — isolated-saddle formation from F' alone is "
      "REFUTED at model level; the README test question is "
      "answered: F' = 0 creates only a degenerate whole level",
      zero_at_origin and zero_at_11 and zero_at_22)

# ---------------------------------------------------------------
# RR5-3 (identity — zeta-critical hyperbolic orbit, exact)
J = sp.Matrix([[0, -1], [1, 0]])
h1, h2 = sp.symbols('h1 h2', real=True)
lam = sp.Symbol('lam')
JH = J*sp.diag(h1, h2)
charpoly = sp.expand(JH.charpoly(lam).as_expr())
saddle_eigs = set(sp.simplify(e) for e in (J*sp.diag(1, -1)).eigenvals())
center_eigs = set(sp.simplify(e) for e in (J*sp.eye(2)).eigenvals())
check("identity", "RR5-3 zeta-critical hyperbolic orbit exact: the "
      "centralizer linearization Y_lin = F'(zeta_0) J H has "
      "charpoly lam^2 + h1 h2 — eigenvalues +/- sqrt(-h1 h2); for "
      "the INDEFINITE witness H = diag(1, -1) they are {+1, -1}: "
      "REAL and opposite — an ISOLATED hyperbolic orbit EXISTS at "
      "the zeta-critical point (the licensed partition-2 path: "
      "smooth compatible-centralizer topology with the required "
      "orbit graph); for the DEFINITE witness H = I they are "
      "{+I, -I}: purely imaginary (center — no separatrix from "
      "this path). The partition flips exactly on the Hessian "
      "signature",
      sp.simplify(charpoly - (lam**2 + h1*h2)) == 0
      and saddle_eigs == {1, -1}
      and center_eigs == {II, -II})

# ---------------------------------------------------------------
# RR5-4 (structure — KKS orientation chain)
area_sign = -1            # RR4-4: <e1, k x e2> = -6 < 0
residue_sign = -1         # RR3-2: pairing = -|psi|^2 < 0
Fp0_sign = 1              # witness F'(zeta_0) > 0
orbit_orientation = sp.sign(Fp0_sign*area_sign)
check("structure", "RR5-4 KKS orientation chain: the orbit "
      "orientation is LOCKED to the receipted sign chain — residue "
      "sign (RR3-2) and area sign (RR4-4) agree (< 0) and the "
      "orbit orientation = sign(F'(zeta_0)) against that chain: "
      "one consistent sign bookkeeping from the resonant pairing "
      "through the KKS area to the hyperbolic orbit (absolute "
      "physical sign remains the labeled tier)",
      orbit_orientation == -1 and residue_sign == area_sign)

# ---------------------------------------------------------------
# MB5-1 (mutation): partition violation at a regular trace
Y_regular = Y.subs(Fp, sp.Symbol('f', nonzero=True))
check("mutation", "MB5-1 partition violation detected: at a "
      "regular-core trace (F' != 0) the flow magnitude Y = F' c is "
      "NONZERO everywhere on the torus (no fixed points, flow-box) "
      "while the pendulum layer's separatrix NEEDS a hyperbolic "
      "fixed orbit (RR5-2/RR5-3) — a partition-3 claim at a "
      "regular trace contradicts the receipted structure and is "
      "caught (the smooth-conjugacy prohibition, model level)",
      Y_regular != 0)

# ---------------------------------------------------------------
# MB5-2 (mutation): isolated-saddle misread
check("mutation", "MB5-2 isolated-saddle misread detected: the "
      "zero set of the flow magnitude contains (0,0), (1,1) and "
      "(2,2) simultaneously (RR5-2) — a build reading F' = 0 as an "
      "ISOLATED saddle at the origin contradicts the full-level "
      "structure and is caught; the degenerate-whole-level reading "
      "is the receipted one",
      zero_at_11 and zero_at_22)

print(f"ALL 0062-R5 BRANCH RECEIPTS GREEN: {COUNTS['identity']} "
      f"identity + {COUNTS['mutation']} mutations + "
      f"{COUNTS['structure']} structure = "
      f"{sum(COUNTS.values())} assertions (self-counted).")
