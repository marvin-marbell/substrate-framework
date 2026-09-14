#!/usr/bin/env python3
"""P252 extension (Addendum 3): selected discussion #186 form-level checks.

Scope (owner direction 2026-09-14, issue #211 comment 5659508492): beyond the
original object (comment 18406566, audited in `verify_newton_sign_audit.py`),
this module checks selected load-bearing, self-contained claims surfaced by the
full-thread inventory.  It is not an exhaustive oracle for every claim in the
discussion:

  B14  mjmikulski [36] / report 016: the null-tilt screening family
       N = C - a(r) P, P = l l^T eta with l = (1, n(x)) null:
       F = [d_i N, d_j N] identically zero for every profile a(r)
       (a nilpotency statement), the pinned spectrum at a = a*, the
       potential bound V <= 2 Delta^2 on [0, a*], and the charge-degree
       read on spheres r > R.
  B15  JarekDuda [37]: Koide-surface arithmetic.  Canonical shape
       Q(m) = sum(m) / (sum(sqrt(m)))^2 (Wikipedia anchor: Q_PDG =
       0.66666446(508), range [1/3, 1], Foot cos^2 theta = 1/(3Q)).
  B16  xrodz [35]/[39]: the four posted vacuum curvatures
       371866.88 / 48.02 / 5.229 / 11.52 and 714251 / 79.4 / 6.08 / 11.52,
       the Hessian closed forms, and the reflection probes.
  B17  xrodz [39]/script docstring: Derrick-virial reads:
       E = A/R + B R^3 stationary point has E_curv/V = 3 exactly and
       R* = r (E_curv / 3 V)^(1/4).
  B18  artifact consistency (provenance grade, no proxy verification):
       the vendored R20 results JSON at openwave-labs/openwave pin
       55fcc168b347011ade787493d8aedf2d7bbbb442 carries every number the
       R20 results post [39] quotes.

Conventions (declared per block; B14 uses the source repo's own
eta = diag(1, -1, -1, -1), N = eta M; B16/B17 use the openwave stack
convention eta = diag(-1, 1, 1, 1), N = M eta, code branch s = -1 with
M_vac = diag(8, 1, 3/10, 0) and certified N-spectrum (-8, 1, 3/10, 0)).
Modes mirror the base module:

    python verify_d186_extension.py            # checks AND mutations
    python verify_d186_extension.py --verify   # checks only, must all pass
    python verify_d186_extension.py --mutate   # mutations only, must all break

B18 is provenance grade: it asserts byte-identity of the vendored artifact
(recorded MD5) and consistency of the posted numbers with its contents; it
makes NO claim about the lattice runs themselves (AP-8/AP-14 gate, debt D5
class of the base audit).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys

import mpmath as mp
from mpmath import mpf
import numpy as np
import sympy as sp

HERE = os.path.dirname(os.path.abspath(__file__))
VENDORED_JSON = os.path.join(HERE, "sources", "m5_32_r20_1_axes__55fcc168.json")
VENDORED_MD5 = "954879cf75fbcb89694811d9cb807464"  # md5 of the vendored file at pin 55fcc168b347

ETA_OW = sp.diag(-1, 1, 1, 1)  # openwave stack: N = M eta
ETA_MJ = sp.diag(1, -1, -1, -1)  # mjmikulski repo 016: N = eta M


class Ledger:
    """Collects check and mutation outcomes; a check that cannot fail is not a check."""

    def __init__(self) -> None:
        self.checks: list[tuple[str, bool, str]] = []
        self.mutations: list[tuple[str, bool, str]] = []

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        self.checks.append((name, bool(ok), detail))
        return bool(ok)

    def mutation(self, name: str, broke: bool, detail: str = "") -> bool:
        """A mutation SUCCEEDS when the mutated variant is observed to break."""
        self.mutations.append((name, bool(broke), detail))
        return bool(broke)


# ---------------------------------------------------------------------------
# B14 — mjmikulski report 016: the null-tilt screening family
# ---------------------------------------------------------------------------

def _tilt_structures(eta, lvec):
    """P = l l^T eta and its coordinate derivatives for explicit l(x)."""
    P = lvec * lvec.T * eta
    return P


def block_b14(led: Ledger) -> None:
    x, y, z = sp.symbols("x y z", real=True)
    r = sp.sqrt(x**2 + y**2 + z**2)
    a = sp.symbols("a", real=True)
    eta = ETA_MJ

    # -- symbolic direction components under the null constraint ------------
    l0, l1, l2, l3 = sp.symbols("l0 l1 l2 l3", real=True)
    lsym = sp.Matrix([l0, l1, l2, l3])
    Psym = _tilt_structures(eta, lsym)
    null_cond = (lsym.T * eta * lsym)[0, 0]  # l0^2 - l1^2 - l2^2 - l3^2

    # C14a: P is nilpotent on the null cone: P^2 = (l^T eta l) * P exactly,
    # hence P^2 = 0 whenever l^T eta l = 0.
    P2_closed = sp.expand(Psym * Psym - null_cond * Psym) == sp.zeros(4)
    led.check("B14_C14a_P_nilpotent_on_null_cone",
              bool(P2_closed),
              "P^2 = (l^T eta l) P exactly; zero on the null cone l^T eta l = 0")

    # -- explicit radial null field and its jet ------------------------------
    lvec = sp.Matrix([1, x / r, y / r, z / r])
    P = _tilt_structures(eta, lvec)
    dP = [sp.diff(P, v) for v in (x, y, z)]

    # C14b: the tilt jet is commutative up to P-multiples: P * dP_i =
    # dP_i * P = 0, and the dP_i * dP_j products are P-proportional with a
    # SYMMETRIC coefficient (d_i n . d_j n) - which is what makes the
    # commutators vanish (C14c).  These are verified as coordinate identities
    # on R^3 minus the origin, rather than at one sampled direction.
    pt = {x: 1, y: 2, z: 2}  # r = 3, all entries rational
    Pv = P.subs(pt)
    cross_zero = all(
        all(sp.simplify(entry) == 0 for entry in M * N)
        for M, N in [(P, P)] + [(P, dP[i]) for i in range(3)]
        + [(dP[i], P) for i in range(3)]
    )
    dPdP_sym = all(
        all(sp.simplify(entry) == 0
            for entry in dP[i] * dP[j] - dP[j] * dP[i])
        for i in range(3) for j in range(3)
    )
    led.check("B14_C14b_tilt_jet_commutative_up_to_P",
              cross_zero and dPdP_sym,
              "P*dP_i = dP_i*P = 0; dP_i*dP_j - dP_j*dP_i = 0 "
              "identically on r>0 (P-proportional with symmetric coefficient)")
    # C14c: F = [d_i N, d_j N] identically zero, a and a' symbolic:
    #   d_i N = -a' (x_i/r) P - a dP_i, and by C14b every commutator term
    #   carries a vanishing or symmetric-coefficient product.
    # Expanding dN follows only the four exact identities just checked:
    # [P,P] = [P,dP_i] = [dP_i,P] = [dP_i,dP_j] = 0.  Therefore every
    # coefficient of a^2, a*a', and a'^2 in [d_i N,d_j N] vanishes.
    C_zero = cross_zero and dPdP_sym
    led.check("B14_C14c_F_identity_zero_for_any_profile", C_zero,
              "[d_i N, d_j N] = 0 identically in (a, a') on r>0")

    # C14d: spectrum of N = C - a P: (B, B) plus roots of the pinned
    # quadratic (orientation-independent by the rank-one structure with the
    # (B,B,B) degeneracy; verified at the generic rational direction).
    A_, B_, E0, E1 = sp.symbols("A B E0 E1", real=True)
    C = sp.diag(A_, B_, B_, B_)
    lam = sp.Symbol("lam")
    Nfam_pt = C - a * Pv
    cp = sp.factor(sp.expand(Nfam_pt.charpoly(lam).as_expr()))
    quad = lam**2 - (A_ + B_) * lam + A_ * B_ + a * (A_ - B_)
    A_val, B_val, a_val = sp.Rational(7, 2), sp.Rational(3, 10), sp.Rational(13, 5)
    cp_num = sp.Poly(sp.expand(cp.subs({A_: A_val, B_: B_val, a: a_val})), lam)
    quad_num = sp.Poly(sp.expand((lam - B_val) ** 2 * quad.subs(
        {A_: A_val, B_: B_val, a: a_val})), lam)
    led.check("B14_C14d_spectrum_two_B_plus_pinned_quadratic",
              cp_num == quad_num,
              "charpoly = (lam - B)^2 (lam^2 - (A+B) lam + AB + a(A-B))")

    # C14e: at a* = (E1 - B)(E0 - B)/(E0 + E1 - 2B) the quadratic roots are
    # exactly (E0, E1) with A = E0 + E1 - B.
    a_star = (E1 - B_) * (E0 - B_) / (E0 + E1 - 2 * B_)
    quad_star = sp.expand(quad.subs(a, a_star).subs(A_, E0 + E1 - B_))
    led.check("B14_C14e_spectrum_exactly_vacuum_at_a_star",
              sp.simplify(quad_star - (lam - E0) * (lam - E1)) == 0,
              "quad(a*) - (lam - E0)(lam - E1) simplifies to 0")

    # C14f: V(a) = (lam_+ - E0)^2 + (lam_- - E1)^2 <= 2 Delta^2 on [0, a*].
    # lam_+/lam_- carry the +/− square-root branch: at a = 0 they are
    # (A, B), pairing with (E0, E1); at a = a* they are (E0, E1) exactly.
    E0v, E1v, Bv = sp.Integer(100), sp.Integer(1), sp.Rational(1, 100)
    Dv = E1v - Bv
    Av = E0v + Dv
    a_star_v = (E1v - Bv) * (E0v - Bv) / (E0v + E1v - 2 * Bv)
    disc = sp.sqrt((Av - Bv) ** 2 - 4 * a * (Av - Bv))
    lam_plus = ((Av + Bv) + disc) / 2
    lam_minus = ((Av + Bv) - disc) / 2
    V_of_a = sp.expand((lam_plus - E0v) ** 2 + (lam_minus - E1v) ** 2)
    interior = [s for s in sp.solve(sp.diff(V_of_a, a), a)
                if s.is_real and 0 < s < a_star_v]
    led.check("B14_C14f_potential_bound_2Delta2_on_0_a_star",
              sp.simplify(V_of_a.subs(a, 0)) == 2 * Dv**2
              and sp.simplify(V_of_a.subs(a, a_star_v)) == 0
              and interior == [],
              "V(0) = 2 Delta^2, V(a*) = 0, no interior critical point: "
              "max V = 2 Delta^2 exactly")
    # C14g (exact): at a* an E1 eigenvector is
    # (B-E1, (A-E1)n).  Its nonzero spatial part is a constant multiple of n
    # on every sphere, so the normalized charge-direction map is the identity
    # or antipodal map.  The source fixes the v1 -> n orientation, hence
    # degree +1.
    eigvec = sp.Matrix([
        B_ - E1,
        (A_ - E1) * x / r,
        (A_ - E1) * y / r,
        (A_ - E1) * z / r,
    ])
    Nfam = C - a * P
    eig_residual = (Nfam.subs({A_: E0 + E1 - B_, a: a_star})
                    - E1 * sp.eye(4)) * eigvec.subs(A_, E0 + E1 - B_)
    eig_exact = all(sp.factor(entry) == 0 for entry in eig_residual)
    led.check("B14_C14g_charge_direction_stays_radial_degree_one",
              eig_exact,
              "exact E1 eigenvector has spatial part (A-E1)n; degree +1 "
              "under the stack's v1 -> n orientation convention")

    # Mutations ---------------------------------------------------------------
    # M14a: the radial SPACELIKE tilt s = (0, n(x)) is not null: P_s is not
    # nilpotent and the dP_i dP_j coefficient loses the cancellation, so the
    # F = 0 identity breaks - nullity is load-bearing.
    sv = sp.Matrix([0, x / r, y / r, z / r])
    Ps = sv * sv.T * eta
    Ps_sub = Ps.subs(pt)
    Ps_dP = [sp.diff(Ps, v).subs(pt) for v in (x, y, z)]
    Ps_nilp = sp.simplify(Ps_sub * Ps_sub) == sp.zeros(4)
    dNs = [-(ap * (v / r)) * Ps - a * Pi for v, Pi in zip((x, y, z), Ps_dP)]
    dNs_pt = [D.subs(pt) for D in dNs]
    comm_nonzero = any(
        sp.simplify(dNs_pt[i] * dNs_pt[j] - dNs_pt[j] * dNs_pt[i])
        != sp.zeros(4)
        for i in range(3) for j in range(3)
    )
    led.mutation("B14_M14a_spacelike_tilt_breaks_nilpotency_and_F0",
                 (not Ps_nilp) and comm_nonzero,
                 "s^T eta s = -1: P_s^2 != 0 and [d_i N, d_j N] != 0")

    # M14b: a shifted off a* does not restore the vacuum spectrum.
    a_shift = a_star + sp.Rational(1, 100)
    quad_shift = sp.expand(quad.subs(a, a_shift).subs(A_, E0 + E1 - B_))
    led.mutation("B14_M14b_a_shifted_off_star_loses_vacuum_spectrum",
                 sp.factor(quad_shift) != sp.expand((lam - E0) * (lam - E1)),
                 "quad(a* + 1/100) != (lam - E0)(lam - E1)")


# ---------------------------------------------------------------------------
# B15 — Koide surface arithmetic (comment [37])
# ---------------------------------------------------------------------------

def _koide(masses):
    return sum(masses) / sum(mp.sqrt(m) for m in masses) ** 2


def block_b15(led: Ledger) -> None:
    mp.dps = 40

    q_xrodz = _koide([mpf(1), mpf("4.5"), mpf(162)])
    led.check("B15_C15a_Q_one_4_5_162_is_0_666803",
              abs(q_xrodz - mpf("0.666803")) < mpf("5e-7"),
              f"Q = {mp.nstr(q_xrodz, 9)} vs posted 0.666803")

    q_pdg2024 = _koide([mpf("0.51099895000"), mpf("105.6583755"),
                        mpf("1776.93")])
    led.check("B15_C15b_Q_PDG2024_is_0_66666446",
              abs(q_pdg2024 - mpf("0.66666446")) < mpf("5e-8"),
              f"Q = {mp.nstr(q_pdg2024, 9)} vs Wikipedia 0.66666446(508)")

    q_pdg_old = _koide([mpf("0.51099895"), mpf("105.6583755"),
                        mpf("1776.86")])
    led.check("B15_C15c_Q_PDG_pre2024_is_posted_0_666661",
              abs(q_pdg_old - mpf("0.666661")) < mpf("5e-7"),
              f"Q = {mp.nstr(q_pdg_old, 9)} vs posted 0.666661 (older m_tau)")

    # Q = 2/3 pins m_tau from (m_e, m_mu): Wikipedia's 1776.969 prediction.
    me, mmu = mpf("0.51099895000"), mpf("105.6583755")
    S = mp.sqrt(me) + mp.sqrt(mmu)
    t = (4 * S + mp.sqrt(16 * S**2 - 4 * (3 * (me + mmu) - 2 * S**2))) / 2
    m_tau_pred = t**2
    led.check("B15_C15d_Q_two_thirds_predicts_mtau_1776_969",
              abs(m_tau_pred - mpf("1776.969")) < mpf("5e-4"),
              f"m_tau = {mp.nstr(m_tau_pred, 8)} MeV")

    # the surface claim: Q = 2/3 is one equation on three masses, so the
    # solution set is a two-parameter family; for any (m1, m2) > 0 the right
    # m3 is an explicit root.
    m1, m2, m3 = sp.symbols("m1 m2 m3", positive=True)
    r1, r2 = sp.symbols("r1 r2", positive=True)
    S12 = sp.sqrt(r1) + sp.sqrt(r2)
    # Q = 2/3  =>  3(m1 + m2 + m3) = 2 (sqrt m1 + sqrt m2 + sqrt m3)^2
    #   in t = sqrt m3:  t^2 - 4 S12 t + (3(r1+r2) - 2 S12^2) = 0,
    #   positive root t = 2 S12 + sqrt(6 S12^2 - 3 (r1 + r2)).
    t_root = 2 * S12 + sp.sqrt(6 * S12**2 - 3 * (r1 + r2))
    m3_expr = sp.simplify(t_root**2)
    lhs = 3 * (r1 + r2 + m3_expr)
    rhs = 2 * (S12 + t_root) ** 2
    led.check("B15_C15e_two_parameter_solution_family",
              sp.simplify(lhs - rhs) == 0,
              "m3(m1, m2) explicit: the Q = 2/3 set is parametrised by two "
              "free masses")

    # Mutations ---------------------------------------------------------------
    m_triple = [mpf(1), mpf("4.5"), mpf(162)]
    q_wrong_shape = sum(m_triple) ** 2 / (3 * sum(m ** 2 for m in m_triple))
    led.mutation("B15_M15a_squared_mass_shape_fails_to_reproduce",
                 abs(q_wrong_shape - mpf("0.666803")) > mpf("1e-3"),
                 f"(sum m)^2/(3 sum m^2) = {mp.nstr(q_wrong_shape, 6)}: the "
                 "inverted-algebra shape does NOT reproduce the posted values")

    q_shifted = _koide([mpf(1), mpf("4.5"), mpf(163)])
    led.mutation("B15_M15b_off_surface_triple_leaves_two_thirds",
                 abs(q_shifted - mpf(2) / 3) > mpf("1e-4"),
                 f"Q(1, 4.5, 163) = {mp.nstr(q_shifted, 9)} != 2/3")


# ---------------------------------------------------------------------------
# B16 — posted vacuum curvatures and Hessian closed forms (comments [35]/[39])
# ---------------------------------------------------------------------------

def _sym4_mats():
    """Basis of the symmetric 4x4 matrices as sympy symbols E_ab (a<=b)."""
    idx = [(i, j) for i in range(4) for j in range(i, 4)]
    return idx


def block_b16(led: Ledger) -> None:
    q_code = (sp.Integer(8), sp.Integer(1), sp.Rational(3, 10), sp.Integer(0))
    q_cert = (-q_code[0], q_code[1], q_code[2], q_code[3])  # N-spectrum flips only the g component

    def Pprime(qs, i):
        prod = sp.Integer(1)
        for j, qv in enumerate(qs):
            if j != i:
                prod *= qs[i] - qv
        return sp.simplify(prod)

    # independent route: differentiate V_spec = tr P(N)^2 along diagonal
    # directions at M_vac with gamma = 1, delta = 3/10 exact rationals.
    def hspec_diag(qs):
        """Second diagonal derivatives of V_spec at M_vac, exact."""
        eta = ETA_OW
        vac = sp.diag(*[qs[i] * eta[i, i] for i in range(4)])  # M_vac = diag
        # N = M eta has spectrum qs on the diagonal entries n_i = m_i eta_ii
        d2 = []
        for i in range(4):
            t = sp.Dummy("_t")
            m = sp.Matrix(vac)
            m[i, i] += t
            N = m * eta
            PN = sp.eye(4)
            for qv in qs:
                PN = PN * (N - qv * sp.eye(4))
            V = sp.trace(PN * PN)
            # the stack's Hessian convention: H = coefficient of t s in
            # V(M + t B_a + s B_b), i.e. the raw second derivative V''(0)
            # (twice the t^2 Taylor coefficient)
            d2.append(sp.simplify(sp.diff(V, t, 2).subs(t, 0)))
        return d2

    posted_code = [sp.Rational(9296672, 25), sp.Rational(2401, 50),
                   sp.Rational(2614689, 500000), sp.Rational(288, 25)]
    h_code = hspec_diag(q_code)
    ok_a = all(sp.simplify(h - p) == 0 for h, p in zip(h_code, posted_code))
    led.check("B16_C16a_posted_curvatures_8_1_03_0_exact",
              ok_a,
              "371866.88 / 48.02 / 5.229378 / 11.52 = 2 P'(q_i)^2 exactly")

    h_cert = hspec_diag(q_cert)
    posted_cert = [sp.Rational(17856288, 25), sp.Rational(3969, 50),
                   sp.Rational(3038049, 500000), sp.Rational(288, 25)]
    ok_b = all(sp.simplify(h - p) == 0 for h, p in zip(h_cert, posted_cert))
    led.check("B16_C16b_posted_curvatures_minus8_1_03_0_exact",
              ok_b,
              "714251.52 / 79.38 / 6.076098 / 11.52 exact on the certified "
              "branch (posted rounded/truncated)")

    # C16c: V4 Hessian = D (2 w J^T J) D on the diagonal block, zero on the
    # six Lorentz-conjugation directions and the cross block (w = 1).
    w1 = sp.Integer(1)
    JVtJ = sp.Matrix(4, 4, lambda p, qi: sum(
        (k + 1) * q_code[p] ** k * (k + 1) * q_code[qi] ** k for k in range(4)))
    Dmat = sp.diag(*[ETA_OW[i, i] for i in range(4)])
    H_closed = sp.Rational(2) * w1 * (Dmat * JVtJ * Dmat)

    # exact hessian of V4 at M_vac by symbolic differentiation
    tsym = [sp.Symbol(f"_t{k}") for k in range(10)]
    idx = _sym4_mats()
    M0 = sp.diag(*[q_code[i] * ETA_OW[i, i] for i in range(4)])
    M = sp.Matrix(M0)
    for t, (i, j) in zip(tsym, idx):
        M[i, j] += t
        if i != j:
            M[j, i] += t
    N = M * ETA_OW
    C_p = [sum(qv**k for qv in q_code) for k in range(1, 5)]
    trs = []
    Np = sp.eye(4)
    for p in range(1, 5):
        Np = Np * N
        trs.append(sp.trace(Np) - C_p[p - 1])
    V4 = sum(t**2 for t in trs)
    H = sp.hessian(V4, tsym)
    H_vac = H.subs({t: 0 for t in tsym})
    # diagonal-direction block (incl. cross terms between diagonal
    # directions) equals the coordinate closed form D (2 J^T J) D;
    # every entry with at least one off-diagonal (conjugation) direction
    # vanishes identically.
    diag_idx = [k for k, (i, j) in enumerate(idx) if i == j]
    diag_ok = all(sp.simplify(H_vac[k, k2] - H_closed[idx[k][0], idx[k2][0]]) == 0
                  for k in diag_idx for k2 in diag_idx)
    off_zero = all(sp.simplify(H_vac[k, k2]) == 0
                   for k, (i, j) in enumerate(idx)
                   for k2, (i2, j2) in enumerate(idx)
                   if i != j or i2 != j2)
    led.check("B16_C16c_V4_hessian_D_2JtJ_D_structure",
              diag_ok and off_zero,
              "diagonal-direction block = D (2 J^T J) D at w = 1 "
              f"(H_00 = {sp.simplify(H_closed[0, 0])}); all conjugation-"
              "direction entries 0")

    # C16d: reflection probes distinguish the two root sets.
    pp_code = [Pprime(q_code, i) ** 2 for i in range(4)]
    pp_flip = [Pprime(tuple([-qv if k == 0 else qv for k, qv in enumerate(q_code)]), i) ** 2
               for i in range(4)]
    led.check("B16_C16d_single_root_flip_distinguishes_branches",
              any(a != b for a, b in zip(pp_code, pp_flip)),
              "2 P'(q)^2 changes under q_0 -> -q_0 (the two posted quadruples)")

    # exact V4 Hessian spectrum from the closed form (w = 1); the posted
    # stiffness ratios "7e7" / "7e4" are order-of-magnitude speech for it.
    Hmat = np.array(H_closed.evalf(17).tolist(), dtype=float)
    eigs = sorted(np.linalg.eigvalsh(Hmat), reverse=True)
    ratio_v4 = float(eigs[0] / eigs[-1])
    ratio_vspec = float(posted_cert[0] / posted_cert[3])
    ok_e = 4e7 < ratio_v4 < 1e8 and 5e4 < ratio_vspec < 8e4
    led.check("B16_C16e_ratio_pins_order_of_magnitude_only",
              ok_e,
              f"V4 eig ratio {ratio_v4:.3e} (posted '7e7'); V_spec diag ratio "
              f"{ratio_vspec:.3e} (posted '7e4'): round-speak, not exact pins")

    # Mutations ---------------------------------------------------------------
    led.mutation("B16_M16a_wrong_branch_numbers_do_not_match",
                 not all(abs(float(h) - float(p)) < 1e-9
                         for h, p in zip(h_code, posted_cert)),
                 "code-branch Hessians differ from the certified quadruple")

    H_wrong = sp.Rational(2) * w1 * JVtJ  # eta conjugation dropped
    # the eta conjugation is invisible on the pure diagonal but flips the
    # sign of every cross term involving index 0 (eta_00 = -1):
    sign_flip_seen = any(
        sp.simplify(H_vac[k, k2] - H_wrong[idx[k][0], idx[k2][0]]) != 0
        for k in diag_idx for k2 in diag_idx)
    led.mutation("B16_M16b_eta_conjugation_dropped_breaks_V4_form",
                 sign_flip_seen,
                 "without D = eta the closed form misses the exact Hessian "
                 "on the eta_00 cross terms")


# ---------------------------------------------------------------------------
# B17 — Derrick-virial reads (comment [39] and the R20 script docstring)
# ---------------------------------------------------------------------------

def block_b17(led: Ledger) -> None:
    A, B, R = sp.symbols("A B R", positive=True)
    E = A / R + B * R**3
    Rstar = sp.solve(sp.diff(E, R), R)[0]
    led.check("B17_C17a_stationary_radius_A_over_3B_quarter",
              sp.simplify(Rstar**4 - A / (3 * B)) == 0,
              "R* = (A / 3B)^(1/4)")

    Ecurv = A / Rstar
    Vpot = B * Rstar**3
    led.check("B17_C17b_virial_E_curv_over_V_is_3_at_R_star",
              sp.simplify(Ecurv / Vpot - 3) == 0,
              "E_curv / V = 3 exactly at the quartic + potential equilibrium")

    # the posted estimator: R* = r (E_curv / 3 V)^(1/4) given A = E_curv * r
    # and B = V / r^3 (the local two-term reading at radius r).
    rv, Ecv, Vv = sp.symbols("r Ec V", positive=True)
    Rstar_est = (Ecv / (3 * Vv)) ** sp.Rational(1, 4) * rv
    Rstar_true = ((Ecv * rv) / (3 * (Vv / rv**3))) ** sp.Rational(1, 4)
    led.check("B17_C17c_posted_R_star_estimator_exact_in_two_term_ansatz",
              sp.simplify(Rstar_est - Rstar_true) == 0,
              "R* = r (E_curv / 3 V)^(1/4) is the exact A/3B radius under "
              "A = E_curv r, B = V / r^3")

    # Mutations ---------------------------------------------------------------
    led.mutation("B17_M17a_cubic_exponent_breaks_radius",
                 sp.simplify(Rstar - (A / (3 * B)) ** sp.Rational(1, 3)) != 0,
                 "R* = (A/3B)^(1/3) is not a stationary point")
    led.mutation("B17_M17b_virial_1_breaks_equilibrium_claim",
                 sp.simplify(Ecurv / Vpot - 1) != 0,
                 "E_curv / V = 1 is false at the stationary point")


# ---------------------------------------------------------------------------
# B18 — artifact consistency against the vendored pinned JSON (provenance)
# ---------------------------------------------------------------------------

def _load_json(path):
    with open(path, "rb") as f:
        raw = f.read()
    return raw, json.loads(raw)


def _close(val, posted, abs_tol=None):
    """Posted decimal within half-unit-of-last-digit (or explicit absolute)
    tolerance."""
    posted_value = float(posted)
    if abs_tol is not None:
        return abs(val - posted_value) <= abs_tol
    return abs(val - posted_value) <= 0.5 * 10 ** (-_decimals(posted)) + 1e-12


def _decimals(posted):
    # Callers that need significant trailing zeroes pass a string.  Converting
    # through float here would turn "11.290" into "11.29" and weaken the
    # quoted-digit tolerance by a factor of ten.
    s = str(posted)
    if "e" in s or "E" in s:
        return abs(int(s.split("e")[1]))
    if "." in s:
        return len(s.split(".")[1])
    return 0


def block_b18(led: Ledger) -> None:
    raw, d = _load_json(VENDORED_JSON)
    digest = hashlib.md5(raw).hexdigest()

    led.check("B18_C18a_vendored_json_matches_recorded_md5",
              digest == VENDORED_MD5,
              f"md5 {digest} of the byte-exact vendored copy at pin "
              "55fcc168b347011ade787493d8aedf2d7bbbb442")

    rows = d["rows"]

    def E(tag):
        return float(rows[tag]["E"])

    # posted energies table of comment [39]
    posted = {
        "S1_v4std_n32_g8": "11.290", "Sd_v4std_n32_g8": "63.592",
        "S0_v4std_n32_g8": "30.385",
        "S1_v4std_n32_g8_x4500": "9.781", "Sd_v4std_n32_g8_x4500": "60.876",
        "S0_v4std_n32_g8_x4500": "22.758",
        "S1_v4std_n64_g8": "13.125", "Sd_v4std_n64_g8": "83.733",
        "S0_v4std_n64_g8": "30.922",
        "S1p_v4std_n32_g8": "11.094", "Sdp_v4std_n32_g8": "64.981",
        "S0p_v4std_n32_g8": "26.316",
        "S1p_v4std_n32_g8_x4500": "9.642", "Sdp_v4std_n32_g8_x4500": "61.858",
        "S0p_v4std_n32_g8_x4500": "22.895",
        "S1dd_v4dd_n32_g8": "4.502", "Sddd_v4dd_n32_g8": "17.812",
        "S0dd_v4dd_n32_g8": "18.672",
        "S1dd_v4dd_n32_g8_x4500": "3.993", "Sddd_v4dd_n32_g8_x4500": "16.554",
        "S0dd_v4dd_n32_g8_x4500": "17.213",
        "S1_vspec_n32_g8": "4.885", "Sd_vspec_n32_g8": "55.856",
        "S0_vspec_n32_g8": "16.444",
        "S1_v4std_n32_g32": "18.963",
    }
    bad = {t: (E(t), p) for t, p in posted.items()
           if not _close(E(t), p)}
    led.check("B18_C18b_all_25_posted_energies_in_committed_rows",
              not bad,
              "every posted table energy matches the committed row value" +
              (f"; mismatches {bad}" if bad else ""))

    ctrl = d["results"]["controls"]
    led.check("B18_C18c_g32_control_and_R3_record",
              _close(float(ctrl["S1_g32_vs_R3_record_18.970"]), 18.9631031)
              and _close(float(ctrl["S1_g32_vs_R3_record_18.970"]), 18.963),
              "committed 18.9631031 vs posted '18.963 vs 18.970'")

    trip = d["results"]["axes"]["v4std"]["triple"]
    led.check("B18_C18d_triple_reads_koide_outcome_resolution",
              _close(float(trip["koide_Q"]), 0.38, abs_tol=0.005)
              and trip["outcome"] == "AXES_DEGENERATE",
              f"koide_Q {trip['koide_Q']:.6f} (posted 0.38), outcome "
              f"{trip['outcome']}")
    per = d["results"]["axes"]["v4std"]["per_axis"]
    s1_w = per["S1"]["winding_degree"]["64"]["end"]
    led.check("B18_C18e_S1_winding_degree_1_08_1_06",
              _close(s1_w[0], 1.08, 0.005) and _close(s1_w[1], 1.06, 0.005),
              f"committed end degree {s1_w[0]:.4f} / {s1_w[1]:.4f}")

    sd_tension = float(per["Sd"]["string_tension_read"]["64"])
    s1_tension = [float(per["S1"]["string_tension_read"]["32"]),
                  float(per["S1"]["string_tension_read"]["64"])]
    led.check("B18_C18f_tension_reads_Sd_string_S1_not",
              _close(sd_tension, 0.42, 0.005)
              and _close(s1_tension[0], 0.02, 0.005)
              and _close(s1_tension[1], 0.01, 0.005),
              f"Sd(n64) {sd_tension:.4f} (posted 0.42); S1 "
              f"{s1_tension[0]:.4f}/{s1_tension[1]:.4f} (posted 0.02/0.01)")

    # box increments at equal budget (n64,4500 minus n32,4500)
    inc = {o: E(f"{o}_v4std_n64_g8") - E(f"{o}_v4std_n32_g8")
           for o in ("S1", "Sd", "S0")}
    led.check("B18_C18g_box_increments_1_8_20_1_0_5",
              _close(inc["S1"], 1.8, 0.05) and _close(inc["Sd"], 20.1, 0.05)
              and _close(inc["S0"], 0.5, 0.05),
              f"S1 {inc['S1']:.3f}, Sd {inc['Sd']:.3f}, S0 {inc['S0']:.3f}")

    # virial and R* ranges as posted
    axes_tags = [f"{o}_v4std_{box}" for o in ("S1", "Sd", "S0")
                 for box in ("n32_g8", "n64_g8", "n32_g8_x4500")]
    virials = [float(rows[t]["end_reads"]["derrick"]["virial_E_curv_over_V"])
               for t in axes_tags]
    rstars_s1 = [float(rows[t]["end_reads"]["derrick"]["R_star_from_virial"])
                 for t in axes_tags if t.startswith("S1_")]
    led.check("B18_C18h_virial_14_to_80_and_S1_R_star_13_to_18",
              min(virials) >= 13.9 and max(virials) <= 80.6
              and min(rstars_s1) >= 12.9 and max(rstars_s1) <= 18.1,
              f"virial range [{min(virials):.2f}, {max(virials):.2f}] "
              f"(posted '14 to 80'); S1 R* in "
              f"[{min(rstars_s1):.2f}, {max(rstars_s1):.2f}] (posted '13 to 18')")

    # every undressed minimum is a saddle along the boost dressing
    saddle_ok = all(bool(rows[t]["end_reads"]["frame"]["saddle_along_boost_dressing"])
                    for t in rows)
    led.check("B18_C18i_all_rows_saddle_along_boost_dressing",
              saddle_ok,
              "committed saddle flags true on all 25 rows (posted claim 6)")

    # Mutations ---------------------------------------------------------------
    led.mutation("B18_M18a_recorded_hash_tamper_detected",
                 hashlib.md5(raw + b"\x00").hexdigest() != VENDORED_MD5,
                 "any byte flip breaks the recorded-md5 identity")
    led.mutation("B18_M18b_trailing_zero_precision_is_preserved",
                 not _close(11.294, "11.290"),
                 "11.294 lies outside the half-unit tolerance of posted 11.290; "
                 "the trailing zero remains load-bearing")


BLOCKS = [block_b14, block_b15, block_b16, block_b17, block_b18]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="checks only")
    parser.add_argument("--mutate", action="store_true", help="mutations only")
    args = parser.parse_args(argv)
    if not args.verify and not args.mutate:
        args.verify = args.mutate = True

    led = Ledger()
    for blk in BLOCKS:
        blk(led)

    status = 0
    if args.verify:
        print("== CHECKS ==")
        for name, ok, detail in led.checks:
            print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
        npass = sum(1 for _, ok, _ in led.checks if ok)
        if npass != len(led.checks):
            status = 1
        print(f"CHECKS: {npass}/{len(led.checks)} PASS")
    if args.mutate:
        print("== MUTATIONS ==")
        for name, ok, detail in led.mutations:
            print(f"[{'BROKE' if ok else 'DID-NOT-BREAK'}] {name}" + (f"  ({detail})" if detail else ""))
        nbroke = sum(1 for _, ok, _ in led.mutations if ok)
        if nbroke != len(led.mutations):
            status = 1
        print(f"MUTATIONS: {nbroke}/{len(led.mutations)} BREAK")
    if status == 0:
        print(f"ALL {len(led.checks)} CHECKS PASS; ALL {len(led.mutations)} MUTATIONS BREAK"
              if args.verify and args.mutate else "SELECTED MODES GREEN")
    return status


if __name__ == "__main__":
    sys.exit(main())
