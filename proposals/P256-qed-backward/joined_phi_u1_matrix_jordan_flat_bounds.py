"""NEW-law current-N4 all-flat neutral128 calculation and outward sector bounds.

This executable does not impose Clifford fields, solve a saved root, or adopt a
new source. The mathematical certificate covers every continuous angle and every
joint-character partition through explicit lower bounds; finite searches supply
certified upper bounds, never a dominance proof from absence of a counterexample.
"""
from __future__ import annotations

import argparse
from fractions import Fraction as F
import hashlib
from itertools import product
import json
from math import gcd
from functools import lru_cache, reduce
from pathlib import Path
import sys
import time
from typing import Any
import zipfile

import numpy as np
from scipy.optimize import minimize
import sympy as sp
from threadpoolctl import threadpool_limits

import induced_phi_nonuniform_field_car as field
import joined_phi_u1_stationary_flat_holonomy as old
import joined_phi_u1_stationary_phi_sea_envelope as sea

I = old.Interval
ROOT = Path(__file__).resolve().parent
PREFIX = 'joined_phi_u1_matrix_jordan_flat'
PARTITIONS = ((4,), (3, 1), (2, 2), (2, 1, 1), (1, 1, 1, 1))
# A NEW rational comparison point, not a claim about the saved coupled root.
POINT = (F(289174, 1000000), F(104752, 1000000), F(104752, 1000000), F(104752, 1000000))
PROOF = {
    'law': 'Independent NEWlaw: VJ=Lambda/(16a) sum_x,mu,nu Tr({Phi_mu,Phi_nu}^2), a=1/64,V=64, hence flat quartic/Lambda=256 sum Tr(anticommutator^2). Matrices remain arbitrary globally traceless Hermitian4. No coefficient tuning, Clifford constraint, bareEM, source or root adoption.',
    'flat_classification': old.PROOF['classification'],
    'determinant_AP': old.PROOF['determinant_AP'],
    'components': old.PROOF['centralizer_topology'],
    'occupation': 'For every compatible flat block Phi, native PLUS neutral h_b(k)=sum_mu x_mu(p_b)Phi_mu,b; p_b=(2pi k+pi+theta_b)/4. Physical neutral is -Lambda dGamma(h). TrH=64 A sum_b TrPhi0,b=0 ONLY globally. Its exact neutral128 energy is -Lambda sum GLOBAL top128 of all256 bands. Block scalar parts, unequal block populations, AP, onsite, ordered charged two-link endpoint controls and native roles remain present.',
    'jordan_identity': 'G_mu,nu=TrPhi_muPhi_nu/4. qJ=4096||G||F^2+256 sum_mu,nu ||{Phi_mu,Phi_nu}-2G_mu,nu I4||HS^2. This is a nonnegative identity, not a field constraint. For central holonomies the arbitrary-field minimum at fixed PSD G is the Clifford envelope F_ell=4096||G||F^2-2 sum64 sqrt(xGx), since both the quartic and global half-trace bounds saturate for a same-Gram Clifford realization. Singular G is allowed in the identity and bound.',
    'block_two_exact': 'For any Hermitian2 block Phi_mu=t_mu I2+v_mu.sigma, H_mu,nu=v_mu.v_nu PSD rank<=3. qJ,b=2048[||H||F^2+(t.t)^2+4t^T Ht+2(t.t)TrH]. Scalars are not discarded. In traceless2+2, qJ/4096=(||H1||F^2+||H2||F^2)/2; it is NOT ||(H1+H2)/2||F^2. Only sum_b d_b t_mu,b=0 is required.',
    'rank_saturation': 'qJ=qGram iff every anticommutator equals2G_mu,nu I4. PSD rank4 then yields the irreducible complex Clifford4 representation, so its flat holonomy commutant is scalar and SU4 holonomies are central Z4. This does NOT cover rank3: two2-dimensional Clifford3 summands/copies can saturate with noncentral commuting holonomies. Rank1 commuting sign matrices also saturate; rank0 is harmless. No lower-rank sector is excluded by Schur assertion.',
    'all_angle_covariance': 'For arbitrary shifts and all four samples on each axis, averages sin=cos=sin*cos=0 and sin^2=cos^2=1/2. Therefore mean64 xx^T=C=diag(A^2+2sum c0_j^2,2ci_1^2,2ci_2^2,2ci_3^2), independent of ALL continuous angles, AP shifts and disconnected determinant components.',
    'block_moment_bound': 'Let K_b=Tr(Phi_mu,b Phi_nu,b)/d_b, m_b=Tr(CK_b). Global tracezero gives S128<=Tr|H|/2, and block Cauchy plus momentum Jensen gives S128<=sum_b 32d_b sqrt(m_b). General Jordan trace Cauchy gives qJ,b>=1024d_b||K_b||F^2. A1-dimensional block has rankK<=1, so m_b^2<=D1||K_b||F^2, D1=lambda_max(C)^2. General3/4 blocks use D4=TrC^2.',
    'two_block_improved_bound': 'For the full scalar+Pauli2 block, put s=t.t and Q=||H||F^2+s^2+4tHt+2sTrH. Since rankH<=3, (TrCH)^2<=D3||H||F^2 with D3=sum top3 eigenvalues(C)^2. Also TrCH<=Cmax TrH and tCt<=Cmax s, Cmax^2<=D3. Expanding the square proves m_b^2<=(TrCH+tCt)^2<=D3[||H||F^2+s^2+2sTrH]<=D3 Q. This includes every blockscalar part and rank drop. Dropping global trace and determinant restrictions only relaxes this lower bound.',
    'radial_minimum': 'With the preceding bound, each block contributes at least(d_b/4)e_r, e_r=-3(256D_r)^(1/3), r=1 for d1,r=3 for d2,r=4 for d3/d4. Sum by partition. These are rigorous whole-sector ALL-angle bounds, not just witness penalties. For partition4 use the sharper all64 central convex residual certificate.',
    'central_residual': 'On PSD4, F_ell is8192-strongly convex. At the explicitly declared positive rational POINT, min_G F_ell>=F_ell(POINT)-||8192POINT-sum xx^T/sqrt(xPOINTx)||F^2/16384. This bound remains valid on rank-deficient PSD matrices; no inverseGram or claimed root is used. Actual C is positive definite, so a singular minimizer has a strictly decreasing kernel direction; the unique central minimizing Gram is fullrank. All64 central sectors are enclosed with exact nested radicals.',
    'trial_certification': 'Fields are exact dyadic Hermitian block matrices, projected to GLOBAL tracezero. Angles are rational multiples of pi, with the last block solved by the exact determinant equation and its retained integer sheet. Pi is enclosed by Machin arctan series; trigonometric Taylor remainders and integer dyadic arithmetic enclose exact onebody blocks. Independently rounded eigensystems are certified with rational orthogonality/residual norms, not asserted exact. Sorting all256 lower/upper eigenvalue bounds and summing top128 gives outward globaloccupation sea bounds; quartic is independently exact rational matrix arithmetic.',
    'band_residual_lemma': 'For Hermitian midpoint B, rounded Q,D, eps=||QdagQ-I||F<1/10 and R=BQ-QD, polar U satisfies ||B-UDUdag||F<=2||R||F+(||B||F+2||D||op)eps. Indeed B-QDQdag=RQdag+B(I-QQdag), ||Q||op<2 and ||I-QQdag||F=eps; Q=UP and ||PDP-D||F<=2||D||op eps. Add exact-input midpoint error ||A-B||F, apply Weyl, and sort globally. All square roots and residual sums are outward integer/rational operations.',
    'limits': 'Finite optimizer outputs are exploration only, including failed or nonsmooth runs. Certificates supply explicit sector brackets, not new allflat Clifford dominance unless their lower bounds establish it. Off-flat competitors, fullPhi/SU4 kinetic and Gauss, charged1+63/fullHC/Pi quantum ground, angle minimizing identities, quantization, stationary same-ground photon pole, commoncone and source adoption remain OPEN.'
}


def quartics(phi: np.ndarray) -> tuple[float, float, float]:
    g = sea.gram(phi)
    anti = phi[:, None] @ phi[None, :] + phi[None, :] @ phi[:, None]
    qj = 256 * float(np.sum(abs(anti)**2))
    qo = 4096 * float(np.sum(g*g))
    extra = 256 * float(np.sum(abs(anti - 2*g[:, :, None, None]*np.eye(4))**2))
    return qj, qo, extra


def two_formula(phi: np.ndarray, dims: tuple[int, ...]) -> float:
    total = 0.0
    off = 0
    for db in dims:
        block = phi[:, off:off+db, off:off+db]
        if db == 2:
            t = np.trace(block, axis1=1, axis2=2).real / 2
            v = block-t[:, None, None]*np.eye(2)
            h = np.einsum('mij,nji->mn', v, v).real/2
            total += 2048*(np.sum(h*h)+np.dot(t, t)**2+4*t@h@t+2*np.dot(t, t)*np.trace(h))
        else:
            a = block[:, None]@block[None, :]+block[None, :]@block[:, None]
            total += 256*float(np.sum(abs(a)**2))
        off += db
    return float(total)


def central_certificate(profile: dict[str, Any]) -> dict[str, Any]:
    sectors = []
    gp = [I.of(v) for v in POINT]
    for ell in product(range(4), repeat=3):
        radii = I.of(0)
        m = [[I.of(0) for _ in range(4)] for _ in range(4)]
        for k in product(range(4), repeat=3):
            x = old.exact_x(profile, k, ell)
            r = sum((gp[j]*x[j].square() for j in range(4)), I.of(0)).sqrt()
            radii += r
            for j in range(4):
                for l in range(4):
                    m[j][l] += x[j]*x[l]/r
        f = 4096*sum((v.square() for v in gp), I.of(0))-2*radii
        gradient = [[8192*gp[j]-m[j][l] if j == l else -m[j][l] for l in range(4)] for j in range(4)]
        residual = sum((v.square() for row in gradient for v in row), I.of(0))
        lower = f-residual/16384
        sectors.append(dict(ell=list(ell), point_value=f.public(), gradient_norm_squared=residual.public(), minimum_lower=lower.public()))
    lower = min(F(row['minimum_lower']['lower']) for row in sectors)
    upper = min(F(row['point_value']['upper']) for row in sectors)
    return dict(point_diagonal=[str(z) for z in POINT], point_scope='fresh rational comparison, NOT saved root identity', all64=sectors, lower=str(lower), upper=str(upper), lower_display=float(lower), upper_display=float(upper), no_root_solver=True)


def sector_lower_bounds(profile: dict[str, Any], central: dict[str, Any]) -> dict[str, Any]:
    c = [profile['onsite']**2+2*sum(row['c0']**2 for row in profile['paths']), *[2*next(row['ci'] for row in profile['paths'] if row['axis'] == j)**2 for j in range(3)]]
    ordered = sorted(c, reverse=True)
    e = {r: -3*I.of(256*sum(z*z for z in ordered[:r])).cube_root() for r in (1, 3, 4)}
    bounds = []
    for dims in PARTITIONS:
        b = sum((F(db, 4)*e[1 if db == 1 else 3 if db == 2 else 4] for db in dims), I.of(0))
        if dims == (4,):
            b = I.of(F(central['lower']))
        gap = b-I.of(F(central['upper']))
        bounds.append(dict(partition=list(dims), character_components_per_axis=reduce(gcd, dims), ALL_continuous_angles_lower=b.public(), separation_above_central_trial=gap.public(), certified_excluded_as_global_minimum=bool(gap.lo > 0)))
    return dict(exact_C_diagonal=[str(z) for z in c], rank_moment_radial_bounds={str(r): e[r].public() for r in e}, partitions=bounds, allflat_lower=str(min(F(row['ALL_continuous_angles_lower']['lower']) for row in bounds)))


def basis(dims: tuple[int, ...]) -> np.ndarray:
    rows = []
    for k in range(1, 4):
        d = np.zeros(4)
        d[:k] = 1
        d[k] = -k
        rows.append(np.diag(d)/np.sqrt(k*(k+1)))
    off = 0
    for db in dims:
        for a in range(off, off+db):
            for b in range(a+1, off+db):
                for z in (1, 1j):
                    m = np.zeros((4, 4), complex)
                    m[a, b] = z/np.sqrt(2)
                    m[b, a] = np.conj(z)/np.sqrt(2)
                    rows.append(m)
        off += db
    return np.asarray(rows, complex)


def angle_values(dims: tuple[int, ...], independent: np.ndarray, sheet: tuple[int, ...]) -> np.ndarray:
    if len(dims) == 1:
        return np.pi*np.asarray(sheet)[None]/2
    out = np.zeros((len(dims), 3))
    out[:-1] = independent.reshape(len(dims)-1, 3)
    out[-1] = (2*np.pi*np.asarray(sheet)-np.asarray(dims[:-1])@out[:-1])/dims[-1]
    return out


def objective(params: np.ndarray, dims: tuple[int, ...], sheet: tuple[int, ...], profile: dict[str, Any], generators: np.ndarray) -> tuple[float, np.ndarray]:
    nr = len(generators)
    nf = 4*nr
    phi = np.einsum('ma,aij->mij', params[:nf].reshape(4, nr), generators)
    angles = angle_values(dims, params[nf:], sheet)
    qj, _, _ = quartics(phi)
    anti = phi[:, None]@phi[None, :]+phi[None, :]@phi[:, None]
    dq = 1024*np.sum(phi[None, :]@anti+anti@phi[None, :], axis=1)
    eigs, vectors, xs, dxs = [], [], [], []
    off = 0
    for db, theta in zip(dims, angles, strict=True):
        p, x = old.coefficients(profile, theta)
        h = np.einsum('km,mij->kij', x, phi[:, off:off+db, off:off+db])
        ev, vec = np.linalg.eigh(h)
        eigs.append(ev)
        vectors.append(vec)
        xs.append(x)
        dx = np.zeros((64, 3, 4))
        for row in profile['paths']:
            j = row['axis']
            dx[:, j, 0] = -.5*float(row['c0'])*np.sin(p[:, j])
            dx[:, j, j+1] = -.5*float(row['ci'])*np.cos(p[:, j])
        dxs.append(dx)
        off += db
    allbands = np.concatenate([z.ravel() for z in eigs])
    selected = np.zeros(256)
    selected[np.argsort(allbands)[-128:]] = 1
    dsea = np.zeros((4, 4, 4), complex)
    da = np.zeros((len(dims), 3))
    off, start = 0, 0
    for b, db in enumerate(dims):
        occ = selected[start:start+64*db].reshape(64, db)
        proj = (vectors[b]*occ[:, None, :])@vectors[b].conj().transpose(0, 2, 1)
        dsea[:, off:off+db, off:off+db] = np.einsum('km,kij->mij', xs[b], proj)
        local = np.einsum('kij,mji->km', proj, phi[:, off:off+db, off:off+db]).real
        da[b] = np.einsum('kjm,km->j', dxs[b], local)
        off += db
        start += 64*db
    df = np.einsum('mij,aji->ma', dq-dsea, generators).real.ravel()
    dt = -(da[:-1]-np.asarray(dims[:-1])[:, None]*da[-1]/dims[-1]).ravel()
    return qj-float(allbands@selected), np.concatenate([df, dt])


def search(profile: dict[str, Any], gamma: np.ndarray, iterations: int, starts: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(2221010)
    results = []
    for dims in PARTITIONS:
        generators = basis(dims)
        nf = 4*len(generators)
        sheets = list(product(range(4), repeat=3)) if dims == (4,) else list(product(range(2), repeat=3)) if dims == (2, 2) else [(0, 0, 0)]
        # All64 central sectors have rigorous bounds; fullfield exploration uses
        # two representatives, not a claim to search every continuous minimum.
        if dims == (4,):
            sheets = [(0, 0, 0), (1, 2, 3)]
        candidates = []
        for sheet in sheets:
            for attempt in range(starts):
                if dims == (4,) and attempt == 0:
                    phi = np.sqrt(np.asarray(POINT, float))[:, None, None]*gamma
                elif dims == (2, 2) and attempt == 0:
                    phi = old.witness_phi()
                else:
                    phi = np.einsum('ma,aij->mij', rng.normal(0, .17, (4, len(generators))), generators)
                coords = np.einsum('mij,aji->ma', phi, generators).real.ravel()
                theta = rng.uniform(-np.pi, np.pi, 3*(len(dims)-1))
                if dims == (2, 2) and sheet == (1, 0, 1) and attempt == 0:
                    theta = np.pi*np.array([0., 0., 1.])
                initial = np.concatenate([coords, theta])
                answer = minimize(objective, initial, args=(dims, sheet, profile, generators), method='L-BFGS-B', jac=True, options={'maxiter': iterations, 'ftol': 1e-13, 'gtol': 1e-8, 'maxls': 40})
                phi = np.einsum('ma,aij->mij', answer.x[:nf].reshape(4, len(generators)), generators)
                angles = angle_values(dims, answer.x[nf:], sheet)
                candidates.append(dict(sheet=list(sheet), attempt=attempt, value=float(answer.fun), success=bool(answer.success), message=str(answer.message), iterations=int(answer.nit), gradient_norm=float(np.linalg.norm(answer.jac)), phi_real=phi.real.tolist(), phi_imag=phi.imag.tolist(), angles_radians=angles.tolist(), independent_angles_pi=(answer.x[nf:]/np.pi).tolist()))
        results.append(dict(partition=list(dims), field_parameter_count=nf, angle_parameter_count=3*(len(dims)-1), scalar_parts='all3 global traceless diagonal generators retained; no individual blocktrace constraint', searched_sheets=[list(s) for s in sheets], runs=candidates, best=min(candidates, key=lambda r: r['value'])))
    return results


def gradient_controls(profile: dict[str, Any]) -> list[dict[str, Any]]:
    rng = np.random.default_rng(314222)
    evidence = []
    for dims in PARTITIONS:
        generators = basis(dims)
        params = rng.normal(0, .14, 4*len(generators)+3*(len(dims)-1))
        sheet = (1, 0, 1) if dims == (2, 2) else (0, 0, 0)
        _, gradient = objective(params, dims, sheet, profile, generators)
        directions = [rng.normal(size=len(params)) for _ in range(3)]
        for direction in directions:
            direction /= np.linalg.norm(direction)
            h = 2e-6
            plus, _ = objective(params+h*direction, dims, sheet, profile, generators)
            minus, _ = objective(params-h*direction, dims, sheet, profile, generators)
            fd = (plus-minus)/(2*h)
            analytic = float(gradient@direction)
            error = abs(fd-analytic)
            if error > 1e-4:
                raise ArithmeticError('Actual fullmatrix/globaloccupation gradient failed')
            evidence.append(dict(partition=list(dims), directional_finite_difference=fd, analytic=analytic, error=error))
    return evidence


@lru_cache(maxsize=None)
def pi_interval() -> I:
    def atan(q: int) -> I:
        total = F(0)
        terms = 90
        for n in range(terms):
            total += F((-1)**n, (2*n+1)*q**(2*n+1))
        remainder = F(1, (2*terms+1)*q**(2*terms+1))
        return I.of(total)+I.of(remainder)*I(-old.SCALE, old.SCALE)
    return 16*atan(5)-4*atan(239)


@lru_cache(maxsize=None)
def sincos_pi(q: F) -> tuple[I, I]:
    q = (q+1) % 2-1
    x = I.of(q)*pi_interval()
    x2 = x.square()
    si, co = x, I.of(1)
    st, ct = x, I.of(1)
    for n in range(1, 45):
        st = -st*x2/((2*n)*(2*n+1))
        ct = -ct*x2/((2*n-1)*(2*n))
        si += st
        co += ct
    # |x|<=4. The Lagrange remainders are below 4^91/91!
    # for sin and 4^90/90! for cos, regardless of rounding.
    sin_tail = F(4**91, int(sp.factorial(91)))
    cos_tail = F(4**90, int(sp.factorial(90)))
    return si+I.of(sin_tail)*I(-old.SCALE, old.SCALE), co+I.of(cos_tail)*I(-old.SCALE, old.SCALE)


def rational_phi(phi: np.ndarray) -> tuple[np.ndarray, list[sp.Matrix]]:
    den = 1 << 26
    mats = []
    for mu in range(4):
        m = sp.zeros(4)
        for a in range(4):
            for b in range(a, 4):
                re = sp.Rational(round(float(phi[mu, a, b].real)*den), den)
                im = 0 if a == b else sp.Rational(round(float(phi[mu, a, b].imag)*den), den)
                m[a, b] = re+sp.I*im
                m[b, a] = re-sp.I*im
        m -= sp.trace(m)*sp.eye(4)/4
        mats.append(m)
    return np.asarray([np.asarray(m, complex) for m in mats]), mats


Cpx = tuple[F, F]


def cadd(a: Cpx, b: Cpx) -> Cpx:
    return a[0]+b[0], a[1]+b[1]


def cmul(a: Cpx, b: Cpx) -> Cpx:
    return a[0]*b[0]-a[1]*b[1], a[0]*b[1]+a[1]*b[0]


def cconj(a: Cpx) -> Cpx:
    return a[0], -a[1]


def squared(a: Cpx) -> F:
    return a[0]**2+a[1]**2


def sumc(values: Any) -> Cpx:
    out = (F(0), F(0))
    for v in values:
        out = cadd(out, v)
    return out


def rational_eigen_enclosure(real: list[list[I]], imag: list[list[I]]) -> tuple[list[I], F]:
    db = len(real)
    br = [[F(real[a][b].lo+real[a][b].hi, 2*old.SCALE) for b in range(db)] for a in range(db)]
    bi = [[F(imag[a][b].lo+imag[a][b].hi, 2*old.SCALE) for b in range(db)] for a in range(db)]
    bf = np.asarray([[complex(float(br[a][b]), float(bi[a][b])) for b in range(db)] for a in range(db)])
    ev, vec = np.linalg.eigh(bf)
    b = [[(br[a][c], bi[a][c]) for c in range(db)] for a in range(db)]
    q = [[(F(float(vec[a, c].real)), F(float(vec[a, c].imag))) for c in range(db)] for a in range(db)]
    d = [F(float(z)) for z in ev]
    residual2 = F(0)
    orth2 = F(0)
    error2 = I.of(0)
    for a in range(db):
        for c in range(db):
            r = cadd(sumc(cmul(b[a][k], q[k][c]) for k in range(db)), (-q[a][c][0]*d[c], -q[a][c][1]*d[c]))
            o = sumc(cmul(cconj(q[k][a]), q[k][c]) for k in range(db))
            o = cadd(o, (F(-int(a == c)), F(0)))
            residual2 += squared(r)
            orth2 += squared(o)
            error2 += (real[a][c]-I.of(br[a][c])).square()+(imag[a][c]-I.of(bi[a][c])).square()
    eps = I.of(orth2).sqrt()
    if eps.hi >= old.SCALE//10:
        raise ArithmeticError('Rounded eigensystem failed independent orthogonality bound')
    bnorm = I.of(sum(squared(z) for row in b for z in row)).sqrt()
    delta = 2*I.of(residual2).sqrt()+(bnorm+2*max(abs(z) for z in d))*eps+error2.sqrt()
    radius = F(delta.hi, old.SCALE)
    return [I.of(z)+I.of(radius)*I(-old.SCALE, old.SCALE) for z in d], radius


def certify_trial(profile: dict[str, Any], dims: tuple[int, ...], phi: np.ndarray, independent_pi: list[F], sheet: tuple[int, ...]) -> dict[str, Any]:
    rounded, matrices = rational_phi(phi)
    labels = np.repeat(np.arange(len(dims)), dims)
    for m in matrices:
        if any(m[a, b] != 0 for a in range(4) for b in range(4) if labels[a] != labels[b]):
            raise ArithmeticError('Trial fields do not commute with their declared character blocks')
    if len(dims) == 1:
        angles = [[F(z, 2) for z in sheet]]
    else:
        angles = [independent_pi[3*b:3*b+3] for b in range(len(dims)-1)]
        angles.append([(F(2*sheet[j])-sum((dims[b]*angles[b][j] for b in range(len(dims)-1)), F(0)))/dims[-1] for j in range(3)])
    qj = F(str(256*sum(sp.trace((matrices[m]*matrices[n]+matrices[n]*matrices[m])**2).expand() for m in range(4) for n in range(4))))
    g = sp.Matrix(4, 4, lambda m, n: sp.trace(matrices[m]*matrices[n])/4)
    qo = F(str(4096*sp.trace(g*g).expand()))
    bands = []
    radii = []
    off = 0
    for db, theta in zip(dims, angles, strict=True):
        for k in product(range(4), repeat=3):
            x = [I.of(profile['onsite']), I.of(0), I.of(0), I.of(0)]
            for row in profile['paths']:
                if row['length'] != 1:
                    raise ArithmeticError('Neutral-only certificate consumed charged profile')
                j = row['axis']
                si, co = sincos_pi((2*k[j]+1+theta[j])/4)
                x[0] += 2*row['c0']*co
                x[j+1] -= 2*row['ci']*si
            real = [[sum((x[m]*F(str(sp.re(matrices[m][off+a, off+b]))) for m in range(4)), I.of(0)) for b in range(db)] for a in range(db)]
            imag = [[sum((x[m]*F(str(sp.im(matrices[m][off+a, off+b]))) for m in range(4)), I.of(0)) for b in range(db)] for a in range(db)]
            enclosure, radius = rational_eigen_enclosure(real, imag)
            bands.extend(enclosure)
            radii.append(radius)
        off += db
    if len(bands) != 256:
        raise ArithmeticError('Certificate did not cover every native colour and momentum')
    global_sea = I(sum(sorted(z.lo for z in bands)[-128:]), sum(sorted(z.hi for z in bands)[-128:]))
    energy = I.of(qj)-global_sea
    exact_trace = [str(sum(sp.trace(m[sum(dims[:b]):sum(dims[:b+1]), sum(dims[:b]):sum(dims[:b+1])]) for b in range(len(dims)))) for m in matrices]
    if exact_trace != ['0']*4:
        raise ArithmeticError('Global tracezero was lost by rational trial projection')
    return dict(partition=list(dims), law='NEW Jordan, NOT original Gram', field_exact=[[[str(m[a, b]) for b in range(4)] for a in range(4)] for m in matrices], field_real=np.real(rounded).tolist(), field_imag=np.imag(rounded).tolist(), angles_pi=[[str(z) for z in row] for row in angles], determinant_winding=list(sheet), exact_global_trace=exact_trace, exact_alignment='0: constant block fields commute with each joint character', exact_plaquette='0: commuting seam holonomies', quartic_Jordan_exact=str(qj), quartic_oldGram_comparison_exact=str(qo), extra_exact=str(qj-qo), global_top128_of256_sea=global_sea.public(), new_static_energy_divided_Lambda=energy.public(), maximum_band_enclosure_radius=str(max(radii)), angles_native_radians=(np.asarray(angles, float)*np.pi).tolist(), rounded_trial_scope='actual rational matrices and exact rational-pi determinant equation; NOT optimizer floats')


def native_control(profile: dict[str, Any], native: dict[str, Any], phi: np.ndarray, dims: tuple[int, ...], angles: np.ndarray) -> dict[str, Any]:
    control = old.control(profile, native, phi, list(dims), angles)
    qj, qo, extra = quartics(phi)
    errors = dict(control['errors'])
    errors['new_J_identity'] = abs(qj-qo-extra)
    errors['full_scalar_two_block_formula'] = abs(qj-two_formula(phi, dims))
    if max(errors.values()) > 2e-8:
        raise ArithmeticError(json.dumps(errors))
    return dict(primitive_role=profile['role'], load=profile['load'], scale=profile['scale'], native_oldlaw_control=control, new_Jordan_quartic=qj, exact_identity_extra_float=extra, newlaw_neutral_static_float=qj-control['global_largest128'] if profile['scale'] == 'heavy' else None, errors=errors)


def old_positive_control(profile: dict[str, Any], central: dict[str, Any]) -> dict[str, Any]:
    a, b, c = F(537, 1000), F(229, 500), F(81, 250)
    ci = next(row['ci'] for row in profile['paths'] if row['axis'] == 0)
    s = I.of(0)
    for k in product(range(4), repeat=3):
        x = old.exact_x(profile, k, (0, 0, 2))
        s += 2*(a*a*x[0].square()+2*ci*ci*(b*b+c*c)).sqrt()
    qo = 4096*(a**4+b**4/2+c**4)
    qj = 4096*(a**4+b**4+c**4)
    eold, enew = I.of(qo)-s, I.of(qj)-s
    separation = I.of(F(central['lower']))-eold
    if separation.lo <= 0:
        raise ArithmeticError('Same-engine known OLDlaw counter positive control failed')
    return dict(scope='known OLDlaw positive countercontrol on exactly the same onebody/occupation engine; NOT newlaw exclusion', partition=[2, 2], determinant_component=[1, 0, 1], field_parameters={'a': str(a), 'b': str(b), 'c': str(c)}, old_quartic_exact=str(qo), new_quartic_exact=str(qj), extra_exact=str(qj-qo), same_global_top128_sea=s.public(), OLDlaw_energy=eold.public(), NEWlaw_same_witness_energy=enew.public(), OLDlaw_strict_central_exclusion=separation.public())


def run(output: Path, iterations: int, starts: int) -> dict[str, Any]:
    started = time.monotonic()
    snapshots = dict(field.CAPTURED)
    for module in list(sys.modules.values()):
        path = getattr(module, '__file__', None)
        if path and Path(path).suffix == '.py' and Path(path).parent == ROOT:
            snapshots.setdefault(Path(path).name, Path(path).read_bytes())
    for name, pin in sea.INPUTS.items():
        snapshots[name] = (ROOT/name).read_bytes()
        if hashlib.sha256(snapshots[name]).hexdigest() != pin:
            raise ArithmeticError('Frozen microscopic input changed: '+name)
    data = field.load_mean(g_override=[1, 1, 1, 1], mean_descriptor={'scope': 'fresh unit frame for NEWlaw flat controls, no root or quantum mean solver'})
    profiles, native = data['profiles'], data['native']
    profile = profiles[0]
    central = central_certificate(profile)
    lower = sector_lower_bounds(profile, central)
    positive = old_positive_control(profile, central)
    exploration = search(profile, native['frame'], iterations, starts)
    certificates = []
    controls = []
    for row in exploration:
        dims = tuple(row['partition'])
        best = row['best']
        phi = np.asarray(best['phi_real'])+1j*np.asarray(best['phi_imag'])
        angles_pi = [F(round(z*(1 << 18)), 1 << 18) for z in best['independent_angles_pi']]
        certificate = certify_trial(profile, dims, phi, angles_pi, tuple(best['sheet']))
        exact_phi = np.asarray(certificate['field_real'])+1j*np.asarray(certificate['field_imag'])
        angles = np.asarray(certificate['angles_native_radians'])
        certificates.append(certificate)
        for p in profiles:
            controls.append(native_control(p, native, exact_phi, dims, angles))
        actual = controls[-3]['newlaw_neutral_static_float']
        bound = certificate['new_static_energy_divided_Lambda']
        if not (bound['lower_display']-2e-8 <= actual <= bound['upper_display']+2e-8):
            raise ArithmeticError('Independent native band engine missed rational certificate')
        part = next(p for p in lower['partitions'] if p['partition'] == list(dims))
        if F(bound['upper']) < F(part['ALL_continuous_angles_lower']['lower']):
            raise ArithmeticError('Trial contradicts the declared whole-sector lower bound')
        part['certified_trial_upper'] = bound
        part['sector_bracket'] = {'lower': part['ALL_continuous_angles_lower']['lower'], 'upper': bound['upper']}
        part['bracket_width_upper'] = (I.of(F(bound['upper']))-I.of(F(part['ALL_continuous_angles_lower']['lower']))).public()
    # Native adverse controls intentionally retain unequal block traces/filling.
    adverse = np.zeros((4, 4, 4), complex)
    adverse[0] = np.diag([3, -1, -1, -1])/5
    theta = np.pi*np.asarray([[F(1, 7), F(2, 9), F(3, 11)], [-F(1, 21), -F(2, 27), -F(1, 11)]], float)
    filling = native_control(profile, native, adverse, (1, 3), theta)
    raw = filling['native_oldlaw_control']
    if min(abs(raw['global_largest128']-raw['wrong_fixed_block_half']), abs(raw['global_largest128']-raw['wrong_top2_per_momentum'])) < 1:
        raise ArithmeticError('Adverse GLOBAL occupation control did not reject block-half/per-momentum filling')
    rng = np.random.default_rng(20261010)
    scalar_controls = []
    for dims in PARTITIONS:
        generators = basis(dims)
        phi = np.einsum('ma,aij->mij', rng.normal(0, .11, (4, len(generators))), generators)
        angles = angle_values(dims, rng.uniform(-np.pi, np.pi, 3*(len(dims)-1)), (1, 0, 1) if dims == (2, 2) else (0, 0, 0))
        scalar_controls.append(native_control(profile, native, phi, dims, angles))
    old_witness_controls = [native_control(p, native, old.witness_phi(), (2, 2), np.pi*np.asarray([[0, 0, 1], [1, 0, 0]])) for p in profiles]
    if abs(old_witness_controls[0]['native_oldlaw_control']['energy_static_divided_Lambda']-positive['OLDlaw_energy']['lower_display']) > 2e-8:
        raise ArithmeticError('Native OLDlaw countercontrol missed exact old sea')
    # Positive lower-rank saturation; these configurations may be noncentral.
    rank3 = np.zeros((4, 4, 4), complex)
    sigma = [np.array([[1, 0], [0, -1]]), np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]])]
    for mu in range(3):
        rank3[mu, :2, :2] = sigma[mu]/3
        rank3[mu, 2:, 2:] = sigma[mu]/3
    rank1 = np.zeros((4, 4, 4), complex)
    rank1[0] = np.diag([1, 1, -1, -1])/3
    lowrank = [native_control(profile, native, p, (2, 2), np.pi*np.asarray([[0, 0, 1], [1, 0, 0]])) for p in (rank3, rank1)]
    for row in lowrank:
        if row['exact_identity_extra_float'] > 1e-12:
            raise ArithmeticError('Rankdrop Jordan saturation positive control failed')
    old_band_certificate = certify_trial(profile, (2, 2), old.witness_phi(), [F(0), F(0), F(1)], (1, 0, 1))
    old_bands = old_band_certificate['global_top128_of256_sea']
    same_engine_old_energy = I.of(F(old_band_certificate['quartic_oldGram_comparison_exact']))-I.of(F(old_bands['lower']))
    same_engine_old_gap = I.of(F(central['lower']))-same_engine_old_energy
    if same_engine_old_gap.lo <= 0:
        raise ArithmeticError('Fullmatrix outward-band engine missed the known OLDlaw counter')
    old_band_certificate['OLDlaw_energy_upper'] = same_engine_old_energy.public()
    old_band_certificate['OLDlaw_strict_exclusion_lower'] = same_engine_old_gap.public()
    positive['NEWlaw_same_witness_above_central_trial'] = (I.of(F(positive['NEWlaw_same_witness_energy']['lower']))-I.of(F(central['upper']))).public()
    positive['same_fullmatrix_outward_engine_control'] = old_band_certificate
    derivative_evidence = gradient_controls(profile)
    for row in controls+old_witness_controls:
        if row['native_oldlaw_control']['wrong_AP_omission_matrix_HS'] < 1:
            raise ArithmeticError('Native AP-omission adverse control failed')
    best_upper = min(F(central['upper']), *[F(z['new_static_energy_divided_Lambda']['upper']) for z in certificates])
    counters = [z for z in certificates if F(z['new_static_energy_divided_Lambda']['upper']) < F(central['lower'])]
    # No optimizer absence is a proof: only outward inequalities determine flags.
    dominance = all(F(z['ALL_continuous_angles_lower']['lower']) >= F(central['upper']) for z in lower['partitions'][1:])
    archive = output.with_name(output.stem+'_input_bytes.zip')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for name, raw in sorted(snapshots.items()):
            z.writestr(name, raw)
    before = {name: hashlib.sha256(raw).hexdigest() for name, raw in snapshots.items()}
    after = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in snapshots}
    if before != after:
        raise ArithmeticError('Source/input bytes changed during calculation')
    result = dict(schema=1, branch='DESIGN-EXPLORATION independent NEWlaw, no adoption', scope='currentN4 ALL flat SU4/Phi neutral STATIC sector bounds, NOT full quantum ground', proof=PROOF, exact_central_convex_residual=central, exact_ALLangle_sector_lower_bounds=lower, exact_OLDlaw_positive_countercontrol=positive, fullmatrix_fullscalar_globaloccupation_exploration=exploration, exact_rational_trial_certificates=certificates, all_partition_allrole_native_controls=controls, unconstrained_blockscalar_native_controls=scalar_controls, native_OLDlaw_witness_controls=old_witness_controls, adverse_global_filling=filling, rank3_and_rank1_noncentral_saturation_controls=lowrank, verdict={'new_ALLflat_Clifford_central_dominance_certified': dominance, 'new_actual_certified_lower_counterexample': bool(counters), 'certified_counters': counters, 'honest_result': 'CERTIFIED COUNTER' if counters else 'QUANTITATIVE ALL-ANGLE SECTOR BRACKETS; dominance OPEN', 'allflat_bracket': {'lower': lower['allflat_lower'], 'upper': str(best_upper)}, 'angle_ground_quantization_debts': PROOF['limits']}, primitive_profiles=[dict(role=p['role'], scale=p['scale'], load=p['load'], onsite=str(p['onsite']), paths=[{k: str(v) if isinstance(v, F) else v for k, v in row.items()} for row in p['paths']]) for p in profiles], source_input_sha256_before=before, source_input_sha256_after=after, immutable_input_archive=str(archive), archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(), runtime={'python': sys.version, 'executable': sys.executable, 'numpy': np.__version__, 'sympy': sp.__version__, 'threads': 1, 'optimizer_maxiter': iterations, 'starts_per_sheet': starts}, elapsed_seconds=time.monotonic()-started)
    result['actual_fullmatrix_globaloccupation_gradient_controls'] = derivative_evidence
    output.write_text(json.dumps(result, indent=2)+'\n')
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--iterations', type=int, default=500)
    parser.add_argument('--starts', type=int, default=2)
    args = parser.parse_args()
    if args.iterations < 1 or args.starts < 1:
        parser.error('Positive iterations and starts required')
    with threadpool_limits(limits=1):
        result = run(args.output, args.iterations, args.starts)
    print(json.dumps({'receipt': str(args.output), 'verdict': result['verdict'], 'sector_brackets': [{'partition': row['partition'], 'bracket': row['sector_bracket'], 'excluded': row['certified_excluded_as_global_minimum']} for row in result['exact_ALLangle_sector_lower_bounds']['partitions']]}, indent=2))


if __name__ == '__main__':
    main()
