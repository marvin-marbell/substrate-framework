"""Independent exact finite-CAR first-superadiabatic uniform90 supplier.

The prepared state is W_Kato exp(dGamma(T(eta))) (1-P) b6 f_A, not
an occupied instantaneous eigenstate or a PH-boson Gaussian. The underlying
physical Hamiltonian, all65 species and raw co-local source are unchanged.
The exact coefficient law is a positive mixed-matrix functional; numerical
coefficients below carry residual enclosures, rather than defining that law.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.abc
import importlib.machinery
import importlib.util
import json
import sys
import time
from fractions import Fraction as F
from itertools import product
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
SOURCE_BYTES: dict[str, bytes] = {Path(__file__).name: Path(__file__).read_bytes()}


class Loader(importlib.machinery.SourceFileLoader):
    def get_code(self, fullname: str) -> Any:
        raw = Path(self.path).read_bytes()
        SOURCE_BYTES[Path(self.path).name] = raw
        return self.source_to_code(raw, self.path)


class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        candidate = ROOT / (fullname + '.py')
        if candidate.is_file():
            return importlib.util.spec_from_file_location(fullname, candidate, loader=Loader(fullname, str(candidate)))
        return None


finder = Finder()
sys.meta_path.insert(0, finder)
import induced_phi_full_uniform_colour_fermionic_source_unitary as frozen
sys.meta_path.remove(finder)
SOURCE_BYTES.update(frozen.SOURCE_BYTES)
import mpmath as mp
import numpy as np
from scipy.linalg import expm
from scipy.integrate import solve_ivp

colour, joined = frozen.colour, frozen.joined
full = colour.full
iv: Any = mp.iv
exact, endpoint, cap, pack = frozen.exact, frozen.endpoint, frozen.cap, frozen.pack
BASE_SOURCE_SHA = '8bfeec2518978d9a33e300a342c842755e3f766a5f4c83b4847b7dfda297f91f'
BASE_RECEIPT_SHA = 'e44ea9a373b7d4397ac5357845d5edaf5b5d561aa116d769dbfdb9af93bfdc75'
COLOUR_SHA = '8f50cdc7ab6f5860fde58a0d0607ced4afd6549c6b4207d916b1b14839faf01b'


def scalar(value: Any) -> Any:
    if isinstance(value, F):
        return mp.mpf(value.numerator) / value.denominator
    if hasattr(value, '_mpi_'):
        return scalar((endpoint(value, 0) + endpoint(value, 1)) / 2)
    return mp.mpf(value)


def matrix(values: Any) -> Any:
    return mp.matrix([[scalar(v) for v in row] for row in values])


def norm(value: Any) -> Any:
    return mp.sqrt(sum(abs(v)**2 for v in value))


def display_scalar(value: Any, digits: int = 72) -> str:
    result = mp.nstr(value, digits)
    if result is None:
        raise ArithmeticError('Finite scientific scalar required')
    return result


def upper(value: Any) -> dict[str, str]:
    # Every displayed upper bound is rounded through the frozen outward packer.
    return pack(cap(exact(F(display_scalar(abs(value) * (1 + mp.mpf('1e-60')), 90)))))


def encode(value: Any) -> dict[str, Any]:
    return dict(real=[[display_scalar(mp.re(value[i, j])) for j in range(value.cols)] for i in range(value.rows)],
                imag=[[display_scalar(mp.im(value[i, j])) for j in range(value.cols)] for i in range(value.rows)])


def exact_field_pencil(geom: dict[str, Any], g: list[Any], profiles: list[dict[str, Any]]) -> dict[str, Any]:
    """Reconstruct exact mixed90 K, never read the frozen float matrix display."""
    gamma = full.supplier.primitive.SELECTED
    tensors = full.trace_tensors(geom['M'], gamma)
    rows = [colour.exact_static_species(p, geom, g, tensors) for p in profiles]
    primitive, _ = full.primitive_matrix(geom, g, exact(F(1, 64)))
    Lambda = exact(F(2)**8192)
    rho = exact(F(2)**-2048)
    b = [exact(8)] * 45 + [exact(F(3, 10))] * 45
    sea = [[rows[0]['sea'][i][j] + (rows[1]['sea'][i][j] + 63 * rows[2]['sea'][i][j]) / Lambda for j in range(90)] for i in range(90)]
    H = frozen.resolvent.clifford_primitive_resolvent(geom, g)['H']
    K = [[exact(0) for _ in range(90)] for _ in range(90)]
    for i, j in product(range(90), repeat=2):
        if i < 45 and j < 45:
            K[i][j] = rho * (primitive[i][j] + sea[i][j])
        elif i >= 45 and j >= 45:
            K[i][j] = (H[(i-45) % 15] / 5 if i == j else exact(0)) + rho * sea[i][j]
        else:
            K[i][j] = rho * sea[i][j]
        K[i][j] *= iv.sqrt(b[i] * b[j])
    midpoint = matrix(K)
    input_error = mp.sqrt(sum(scalar((endpoint(x, 1) - endpoint(x, 0)) / 2)**2 for row in K for x in row))
    return dict(F=midpoint, input_error=input_error, interval=K, species=rows, primitive=primitive,
                b=[scalar(v) for v in b], H=H)


def positive_mixed_root(pencil: dict[str, Any], quantum: dict[str, Any]) -> dict[str, Any]:
    """Positive sqrt with a block Sylvester iteration, retaining the PC block.

    In the H0 eigenbasis, D_ab=(E_ab-(D²)_ab)/(h_a+h_b).
    Residual Loewner bounds certify the full matrix function, not ninety scalars.
    """
    Fm = pencil['F']
    pp = mp.matrix([[Fm[i, j] for j in range(45)] for i in range(45)])
    ev, Uphi = mp.eigsy(pp)
    U = mp.eye(90)
    for i, j in product(range(45), repeat=2):
        U[i, j] = Uphi[i, j]
    h = [mp.sqrt(v) for v in ev] + [mp.sqrt(Fm[i, i]) for i in range(45, 90)]
    E = U.T * Fm * U - mp.diag([v*v for v in h])
    D = mp.zeros(90)
    differences = []
    for _ in range(4):
        rhs = E - D*D
        new = mp.matrix([[rhs[i, j] / (h[i] + h[j]) for j in range(90)] for i in range(90)])
        differences.append(norm(new-D))
        D = new
    O = U * (mp.diag(h) + D) * U.T
    residual = O*O-Fm
    omega_min = min(h) - norm(D)
    if omega_min <= 0:
        raise ArithmeticError('Positive mixed90 square-root candidate lost its certified gap')
    # Global absolute error is sufficient here because the internal arithmetic
    # precision exceeds the physical e308 condition number. Input interval width
    # is included separately, as is a generous floating arithmetic envelope.
    roundoff = mp.mpf(10)**(-mp.mp.dps+20) * max(1, norm(O)**2)
    root_error = (norm(residual) + pencil['input_error'] + roundoff) / omega_min
    if root_error >= omega_min / 4:
        raise ArithmeticError('Mixed90 positive matrix-function enclosure failed')
    sqrtb = mp.diag([mp.sqrt(v) for v in pencil['b']])
    inverseb = mp.diag([1/mp.sqrt(v) for v in pencil['b']])
    fast, chi = mp.mpf(2)**3072, mp.mpf(2)**-4096
    A = inverseb * O * inverseb * (fast/chi)
    G = mp.diag([chi*v for v in pencil['b']])
    covariance = mp.inverse(A)/2
    return dict(O=O, A=A, G=G, covariance=covariance, sqrtb=sqrtb, inverseb=inverseb,
                omega_min=omega_min, root_error=root_error, residual=norm(residual),
                iterations=[display_scalar(v) for v in differences],
                actual_nonzero_PC_norm=norm(mp.matrix([[O[i,j] for j in range(45,90)] for i in range(45)])))


def resolve_transition(G: Any, A: Any, physical_gap: Any, kato_coefficients: Any) -> dict[str, Any]:
    """Public all-q/5760 ABI: one positive field pencil, arbitrary CAR columns.

    G must be the positive diagonal field cometric in this Fourier real chart;
    A is the full coupled positive precision. Columns carry actual outgoing
    empty/incoming occupied matrix units. No occupied-occupied denominator.
    Return t and k_eff independently; never obtain k_eff by subtracting k+t.
    """
    n = G.rows
    if A.rows != n or A.cols != n or kato_coefficients.rows != n or physical_gap <= 0:
        raise ArithmeticError('Physical positive-gap transition dimensions invalid')
    sg = mp.diag([mp.sqrt(G[i,i]) for i in range(n)])
    ig = mp.diag([1/mp.sqrt(G[i,i]) for i in range(n)])
    Omega = sg*A*sg
    inverse = mp.inverse(Omega + physical_gap*mp.eye(n))
    y = inverse * (sg*kato_coefficients)
    return dict(t=-ig*Omega*y, effective_connection=ig*y*physical_gap,
                positive_similarity=Omega, gap=physical_gap,
                equation_defect=(physical_gap*mp.eye(n)+A*G)*(-ig*Omega*y)+A*G*kato_coefficients)


def all_q_counterrotations(field_blocks: dict[tuple[int,int,int], dict[str,Any]],
                          transitions: list[dict[str,Any]]) -> list[dict[str,Any]]:
    """Both physical PH sectors in the declared periodic N4 Fourier chart.

    Records supply q_integer,k_in,k_out,role,load,physical_scale, ranks2+2,
    PH_plus[field,empty,occupied], PH_minus[field,occupied,empty] and gaps2x2.
    OUT is k+q, IN is k. Minus(q,k) pairs with plus(-q,k+q), with the two
    band indices transposed, not with plus(q,k). The Kato signs are -Veo/Delta
    and +Voe/Delta. Resolving both independently retains complex coefficients;
    the paired-sector identity is checked, never imposed by dropping imaginary
    parts. Coefficients multiply eta_q in exp(iqx)/8, eta_-q=conj(eta_q).
    """
    def momentum(value: Any) -> Any:
        return tuple(value) if isinstance(value, (list,tuple)) else value

    output = []
    sectors = {}
    for record in transitions:
        qi = record['q_integer']
        q = (int(qi[0]) % 4,int(qi[1]) % 4,int(qi[2]) % 4)
        block = field_blocks[q]
        if record['negative_rank'] != 2 or record['positive_rank'] != 2 or record['load'] not in (1,63):
            raise ArithmeticError('Actual both-band finite-CAR transition contract changed')
        scale = mp.mpf(2)**8192 if record['physical_scale'] == 'heavy' else mp.mpf(1)
        legs = {}
        for name,sign in (('plus',-1),('minus',1)):
            amplitudes = record['PH_'+name]
            gaps = record['gap_'+name]
            solved = []
            for out_band, in_band in product(range(2), repeat=2):
                gap = scalar(gaps[out_band][in_band])*scale
                vertex = mp.matrix([amplitudes[a][out_band][in_band] for a in range(block['G'].rows)])
                k = (sign*scale/gap)*vertex
                solved.append(resolve_transition(block['G'], block['A'], gap, k))
            legs[name] = solved
        key = (q,momentum(record['k_in']),momentum(record['k_out']),record['role'])
        if key in sectors:
            raise ArithmeticError('Duplicate physical Fourier/CAR sector')
        item = dict(q_integer=q,k_in=record['k_in'],k_out=record['k_out'],role=record['role'],load=record['load'],
                    physical_scale=record['physical_scale'],dual_PH=legs)
        output.append(item)
        sectors[key] = item
    for item in output:
        q = item['q_integer']
        mq = ((-q[0])%4,(-q[1])%4,(-q[2])%4)
        paired_key = (mq,momentum(item['k_out']),momentum(item['k_in']),item['role'])
        if paired_key not in sectors:
            raise ArithmeticError('Missing -q reversed-momentum physical adjoint sector')
        partner = sectors[paired_key]
        if partner['load'] != item['load'] or partner['physical_scale'] != item['physical_scale']:
            raise ArithmeticError('Physical adjoint sectors have different species loading/scales')
        defects = []
        for i,j in product(range(2),repeat=2):
            minus,plus = item['dual_PH']['minus'][2*i+j],partner['dual_PH']['plus'][2*j+i]
            for name in ('t','effective_connection'):
                defect = norm(minus[name]+plus[name].conjugate())
                # Frozen providers expose float vertices; this is an observation
                # tolerance, not an exact matrix-function enclosure.
                if defect > mp.mpf('5e-10')*max(1,norm(minus[name]),norm(plus[name])):
                    raise ArithmeticError('Physical q/-q adjoint counterrotation identity failed')
                defects.append(display_scalar(defect))
            if abs(minus['gap']-plus['gap']) > mp.mpf('5e-10')*max(minus['gap'],plus['gap']):
                raise ArithmeticError('Physical adjoint transition gaps differ')
        item.update(adjoint_sector=dict(q_integer=mq,k_in=item['k_out'],k_out=item['k_in']),
                    adjoint_coefficient_defects=defects,
                    anti_H_pairing='T_oe(q,k)=-T_eo(-q,k+q)^dag; full OUT/IN sector sum, eta_-q=conj(eta_q)')
    return output


def point_coefficients(profile: dict[str,Any], k: tuple[int,...], geom: dict[str,Any], g: list[Any]) -> dict[str,Any]:
    """Actual integer Clifford projectors, original ordered one/two-link law."""
    roots = [mp.sqrt(scalar(v)) for v in g]
    half = mp.sqrt(2)/2
    co1, si1, si2 = (1,-1,-1,1),(1,1,-1,-1),(1,-1,1,-1)
    x = [scalar(F(profile['onsite'])),mp.mpf(0),mp.mpf(0),mp.mpf(0)]
    phases = []
    for path in profile['paths']:
        axis, length = path['axis'], path['length']
        co = co1[k[axis]]*half if length == 1 else mp.mpf(0)
        si = si1[k[axis]]*half if length == 1 else mp.mpf(si2[k[axis]])
        c0, ci = scalar(F(path['c0'])), scalar(F(path['ci']))
        x[0] += 2*c0*co
        x[axis+1] -= 2*ci*si
        phases.append((axis,length,c0,ci,co+1j*si))
    clifford = [mp.matrix([[mp.mpc(complex(v)) for v in row] for row in m]) for m in geom['M']]
    gamma = [mp.matrix([[mp.mpc(complex(v)) for v in row] for row in m]) for m in full.supplier.primitive.SELECTED]
    frame = [roots[m]*gamma[m] for m in range(4)]
    sign = -1 if profile['scale'] == 'heavy' else 1
    h = sign*sum((x[m]*frame[m] for m in range(4)),mp.zeros(4))
    energy = mp.sqrt(sum((roots[m]*x[m])**2 for m in range(4)))
    occupied = (mp.eye(4)-h/energy)/2
    empty = mp.eye(4)-occupied
    vertices = []
    for column in geom['columns']:
        vertices.append(sign*sum((scalar(c)*x[slot//15]*clifford[slot%15] for slot,c in column.items()),mp.zeros(4))/(8*mp.sqrt(8)))
    cv = [mp.zeros(4) for _ in range(45)]
    for axis,L,c0,ci,phase in phases:
        hop = c0*frame[0]+1j*ci*frame[axis+1]
        for a in range(15):
            d = 1j*L*(hop*clifford[a]+clifford[a]*hop)/(16*mp.sqrt(8))
            cv[15*axis+a] += sign*(phase*d+mp.conj(phase)*d.H)
    vertices.extend(cv)
    # K=[dP,P], empty/occupied block -V_eo/(2E). Its opposite block
    # is the negative adjoint, so second quantization is exactly anti-H.
    kplus = [-empty*v*occupied/(2*energy) for v in vertices]
    coefficients = mp.matrix([[v[i,j] for i,j in product(range(4),repeat=2)] for v in kplus])
    return dict(h=h,occupied=occupied,empty=empty,vertices=vertices,k=coefficients,E=energy,
                gap=2*energy*(mp.mpf(2)**8192 if sign<0 else 1))


def uniform_transitions(geom: dict[str,Any], g: list[Any], profiles: list[dict[str,Any]], field: dict[str,Any]) -> dict[str,Any]:
    """All64 momentum points, all four PH matrix units, loads1/1/63."""
    O, sb, ib = field['O'],field['sqrtb'],field['inverseb']
    fast = mp.mpf(2)**3072
    inverse_cache: dict[tuple[str,str], Any] = {}
    records, private, per_species = [], [], []
    total_before2 = total_after2 = mp.mpf(0)
    coefficient_uncertainty = mp.mpf(0)
    for profile in profiles:
        Ct = k_norm = t_norm = before2 = after2 = gapmax = mp.mpf(0)
        gapmin = mp.inf
        for index, ki in enumerate(product(range(4),repeat=3)):
            point = point_coefficients(profile,ki,geom,g)
            delta = point['gap']/fast
            cache_key = (profile['role'],display_scalar(point['E'],80))
            if cache_key not in inverse_cache:
                inverse_cache[cache_key] = mp.inverse(O+delta*mp.eye(90))
            inverse = inverse_cache[cache_key]
            y = inverse*(sb*point['k'])
            t = -ib*O*y
            effective = ib*y*delta  # Cancellation-free even when t=-k to >600 decimals.
            linear_before = ib*O*sb*point['k']*fast
            linear_kin = ib*O*sb*effective*fast
            linear_matter = point['gap']*t
            defect = linear_kin+linear_matter
            # L2 of eta_a L_a f: covariance is the complete .5A^-1.
            cov = field['covariance']
            b2 = mp.re(sum((linear_before.H*cov*linear_before)[j,j] for j in range(16)))
            a2 = mp.re(sum((defect.H*cov*defect)[j,j] for j in range(16)))
            before2 += b2
            after2 += max(0,a2)
            k_norm += norm(point['k'])
            t_norm += norm(t)
            Ct += 128*norm(t)  # Finite CAR dGamma norm, rank256 / occupied128; no boson law.
            gapmin, gapmax = min(gapmin,point['gap']),max(gapmax,point['gap'])
            # Exact Omega differs by the certified root enclosure. Resolvent
            # identity controls both separately evaluated coefficient functions.
            err = 2*mp.sqrt(mp.mpf(80)/3)*field['root_error']*norm(point['k'])/(delta+field['omega_min']/2)
            coefficient_uncertainty += profile['load']*err
            if index == 0:
                private.append(dict(profile=profile,point=point,t=t,effective=effective,linear_before=linear_before,
                                    linear_kin=linear_kin,linear_matter=linear_matter,defect=defect))
            records.append(dict(role=profile['role'],load=profile['load'],momentum_integer=list(ki),
                                unscaled_BOTHband_gap=display_scalar(2*point['E']),physical_gap=upper(point['gap']),
                                kato_empty_occupied=encode(point['k']),T_empty_occupied=encode(t),
                                cancellation_free_effective_connection=encode(effective),
                                direct_kin_plus_matter_coefficient_defect=upper(norm(defect))))
        total_before2 += profile['load']*before2
        total_after2 += profile['load']*after2
        per_species.append(dict(role=profile['role'],load=profile['load'],physical_scale=profile['scale'],
                                occupied_negative_rank=128,empty_positive_rank=128,
                                physical_gap_lower=display_scalar(gapmin),physical_gap_upper=display_scalar(gapmax),
                                all64_k_Frobenius_sum=display_scalar(k_norm),all64_t_Frobenius_sum=display_scalar(t_norm),
                                finite_CAR_generator_Lipschitz_upper=display_scalar(Ct),
                                first_linear_action_before=upper(mp.sqrt(before2)),first_linear_kin_matter_sum_after=upper(mp.sqrt(after2))))
    return dict(records=records,private=private,species=per_species,
                before=mp.sqrt(total_before2),after=mp.sqrt(total_after2),coefficient_error=coefficient_uncertainty)


def literal_differential_and_CAR(field: dict[str,Any], transitions: dict[str,Any]) -> dict[str,Any]:
    """Independent jets of -1/2 G_ab d_a d_b and literal finite CAR.

    d_a d_b(eta_c f)=(-delta_ac (A eta)_b-delta_bc (A eta)_a
    +eta_c[(A eta)_a(A eta)_b-A_ab])f. Potential minus E cancels
    the last bracket, leaving eta^T A G t, not eta^T G A t.
    """
    A,G = field['A'],field['G']
    eta = mp.matrix([mp.sqrt(abs(field['covariance'][i,i]))*(1 if i%3 else -1) for i in range(90)])
    w = A*eta
    osc0 = -sum(G[i,i]*(w[i]**2-A[i,i]) for i in range(90))/2
    harmonic = (eta.T*A*G*A*eta)[0]/2-sum(G[i,i]*A[i,i] for i in range(90))/2
    controls = []
    for record in transitions['private']:
        point,t,k = record['point'],record['t'],record['point']['k']
        # Independent product derivatives, not the resolver's AG matrix product.
        kinetic = mp.zeros(1,16)
        for c in range(16):
            polynomial = sum(eta[i]*t[i,c] for i in range(90))
            dd = sum(G[i,i]*(polynomial*(w[i]**2-A[i,i])-2*t[i,c]*w[i]) for i in range(90))
            kinetic[0,c] = -dd/2+(harmonic)*polynomial
        prediction = eta.T*A*G*t
        matter = mp.matrix(4,4)
        t_eta = mp.matrix(4,4)
        for i,j in product(range(4),repeat=2):
            t_eta[i,j] = sum(eta[a]*t[a,4*i+j] for a in range(90))
        matter = point['gap']*t_eta
        kato = eta.T*A*G*k
        cancellation = kinetic+mp.matrix([[matter[i,j] for i,j in product(range(4),repeat=2)]])+kato
        wrong = eta.T*G*A*t
        # Literal 16-dimensional CAR dGamma matrices and its filled two-band
        # vector, independently built from actual h eigenvectors.
        h_np = np.array([[complex(v) for v in row] for row in point['h'].tolist()])
        _,U = np.linalg.eigh(h_np)
        k_probe = np.array([[complex(v) for v in row] for row in point['k'][0,:].tolist()]).reshape(4,4)
        gen = k_probe-k_probe.conj().T
        S = expm(.01*gen)
        wedge = joined.scalar.exterior_unitary(S)
        annih = [joined.scalar.car_annihilator(i,4).toarray() for i in range(4)]
        dGamma = sum((gen[i,j]*annih[i].conj().T@annih[j] for i,j in product(range(4),repeat=2)),np.zeros((16,16),complex))
        direct = expm(.01*dGamma)
        psi = np.zeros(16,complex)
        for i in range(4):
            for j in range(i+1,4):
                psi[(1<<i)+(1<<j)] = np.linalg.det(U[[i,j],:2])
        rotated = direct@psi
        number = np.diag([s.bit_count() for s in range(16)])
        controls.append(dict(role=record['profile']['role'],
            literal_gaussian_product_jet_vs_AG=upper(norm(kinetic-prediction)),
            literal_kin_matter_Kato_sum=upper(norm(cancellation)),
            adverse_GA_instead_of_AG_defect=upper(norm(kinetic-wrong)),
            anti_H_one_particle_defect=float(np.linalg.norm(gen+gen.conj().T)),
            finite_CAR_exponential_vs_exterior_defect=float(np.linalg.norm(direct-wedge)),
            finite_CAR_unitarity_defect=float(np.linalg.norm(direct.conj().T@direct-np.eye(16))),
            literal_particle_number=float(np.vdot(rotated,number@rotated).real),
            literal_finite_CAR_density_trace=float(np.vdot(rotated,rotated).real)))
    if any(v['finite_CAR_exponential_vs_exterior_defect']>2e-12 or abs(v['literal_particle_number']-2)>2e-12 for v in controls):
        raise ArithmeticError('Actual finite-CAR anti-H exponential failed')
    return dict(gaussian_constant_energy_jet_identity=upper(abs(osc0+harmonic)),species=controls,
                derivative_law='Literal second derivatives of eta_c exp(-eta^T A eta/2), actual mixed90 A/G; dGamma(h) commutator gives Delta T_eo. BOTH dual PH blocks make T anti-H.')


def packet_budget(base: dict[str,Any], supplier: dict[str,Any], field: dict[str,Any], transitions: dict[str,Any], geom: dict[str,Any], g: list[Any]) -> dict[str,Any]:
    proof = base['actual_full105_corrected_joint90_quantum_CAR_source_proof']
    prices = proof['nonlinear_and_corrector_prices']
    q = supplier['actual_joint90_quantum']
    def bound(x: dict[str,Any]) -> Any:
        return scalar(F(x['upper']))
    chi = mp.mpf(2)**-4096
    radius = scalar(F(proof['radius_exact']))
    covmax = bound(proof['covariance_operator_upper'])
    normalization = 1/mp.sqrt(scalar(F(proof['compact_corrected_normalizer_lower']['lower'])))
    P4 = bound(proof['corrector_L4_norm_upper'])
    # Radial Gaussian moments dominate the actual mixed90 covariance. Compact
    # multiplication never increases a Gaussian moment before normalization.
    r2 = mp.sqrt(90*covmax)
    r4 = mp.sqrt(90*92)*covmax
    r8 = (90*92*94*96)**mp.mpf('.25')*covmax
    radial_first = r2+mp.sqrt(r4)*P4
    radial_second = r4+r8*P4
    radial_fourth = r8*(1+P4)
    # The score is A eta with C=.5 A^-1. Mixed Phi/colour scales
    # forbid a covariance-independent bound on ||eta|| ||A eta||.
    precision_upper = 1/(2*scalar(F(proof['covariance_operator_lower']['lower'])))
    C1 = bound(proof['global_CAR_first_per_coordinate_upper'])
    C2 = bound(proof['global_CAR_second_per_coordinate_upper'])
    Ct = sum(p['load']*scalar(p['finite_CAR_generator_Lipschitz_upper']) for p in transitions['species'])
    maxG = 8*chi
    generator_norm = normalization*Ct*radial_first
    generator_taylor = normalization*Ct*Ct*radial_fourth/2
    # Anti-H Duhamel gives ||DS||<=Ct and ||D²S||<=Ct²; there
    # is NO exp(||T||) growth or false global-smallness assumption on the ball.
    connection_lipschitz = C2+(C1+Ct)**2+Ct*Ct
    connection_second = normalization*maxG*90*connection_lipschitz
    # Sdag K(eta) S+Sdag DS includes W variation, conjugation by S,
    # and the nonlinear exponential differential. Price all three.
    connection_variation = normalization*maxG*90*connection_lipschitz*precision_upper*radial_second
    matter_quadratic = matter_gap_variation = mp.mpf(0)
    for p,oldp in zip(transitions['species'],proof['actual_all65_gap_and_Kato_prices'],strict=True):
        ct = scalar(p['finite_CAR_generator_Lipschitz_upper'])
        scale = mp.mpf(2)**8192 if p['physical_scale']=='heavy' else 1
        gapmax = scalar(p['physical_gap_upper'])
        D1 = bound(oldp['joint90_first_derivative_upper'])
        matter_quadratic += p['load']*gapmax*ct*ct*radial_fourth*normalization
        matter_gap_variation += p['load']*scale*256*D1*ct*radial_second*normalization
    # Reconstruct actual P derivatives instead of setting the corrector to zero.
    algebra = frozen.resolvent.clifford_primitive_resolvent(geom,g)
    cphi = frozen.exact(F(q['actual_joint_Phi_covariance_operator_upper']['upper']))
    zcov = frozen.cap(1/(1-frozen.exact(F(q['actual_joint_ACC_relative_primitive_alpha0_operator_upper']['upper']))))
    moments = frozen.GaussianRadialBounds(cphi,zcov)
    M = frozen.exact(F(q['actual_joint_conditional_mean_M_operator_upper']['upper']))
    sigmin = frozen.exact(min(endpoint(iv.sqrt(exact(F(2)**-7168)*v),0) for v in algebra['covariance']))
    weighted = sum(v*v for m in algebra['Y'] for row in m for v in row)
    a = cap(iv.sqrt(weighted)); b = cap(3*iv.sqrt(exact(15))*a)
    gradP = frozen.add(frozen.term(a,0,2),frozen.term(b),frozen.term(2*a*(1+M)/sigmin,1,1))
    derivative_corrector = normalization*maxG*mp.sqrt(90)*(C1+2*Ct)*scalar(endpoint(moments.norm(gradP),1))
    derivative_bump = normalization*maxG*mp.sqrt(90)*(C1+2*Ct)*12*radial_first/(radius*radius)
    # Coefficient approximation error is an enclosure on the exact state law.
    # Heavy denominator improves the corresponding generator; the global gap
    # maximum must NOT multiply charged coefficient error. Price each resolvent
    # uncertainty instead by its equation residual bound (Omega+Delta)R.
    coefficient_price = normalization*2*mp.sqrt(mp.mpf(80)/3)*mp.mpf(2)**3072*field['root_error']*sum(p['load']*scalar(p['all64_k_Frobenius_sum']) for p in transitions['species'])*radial_first*128
    inherited = sum(bound(v) for name,v in prices.items() if name not in ('CAR_Kato_first','CAR_Kato_second_PLUS_square'))
    replacement = connection_second+connection_variation+matter_quadratic+matter_gap_variation+derivative_corrector+derivative_bump+coefficient_price
    metric = bound(proof['additional_full105_Haar_Noether_metric_reference_residual_upper'])
    # The previous B1/B2 packet metric estimate remains valid with larger actual
    # first/second derivative bounds: multiply by the explicit derivative ratio.
    metric_inflation = (1+Ct/max(C1,1))**2+Ct*Ct/max(C2+C1*C1,1)
    metric_new = metric*metric_inflation
    new_residual = inherited+replacement+metric_new
    active = transitions['species'][1]
    active_ct = scalar(active['finite_CAR_generator_Lipschitz_upper'])
    source_endpoint = mp.sqrt(mp.mpf('.5'))*generator_norm+normalization*(active_ct/128)*radial_first
    return dict(exact_state='Wtotal=W_Kato exp(dGamma(sum_a eta_a(t_eo,a-t_eo,a^dag))); body=Z^-1/2(1-P)b6f_A. All fields/CAR remain untruncated; compact support unchanged.',
                physical_comparator='U_CAR Wtotal (Ecentre+E90_exact+dGamma(hmean)) Wtotal^dag U_CAR^dag with original scalar sqrtJ map. This reference fixes the compact body; physical H evolves unrestricted. No hmean acting directly on physical rotated state.',
                old_reported_Kato_first_upper=prices['CAR_Kato_first'],observed_actual_linear_Kato_before=upper(transitions['before']),
                observed_actual_linear_kin_plus_matter_after=upper(transitions['after']),
                actual_new_state_reference_residual_upper=upper(new_residual),actual_new_state_Rayleigh_residual_upper=upper(2*new_residual),
                remainder_budget=dict(inherited_physical_nonlinear_fast_corrector_mean_prices=upper(inherited),
                    anti_H_connection_second_derivatives=upper(connection_second),radial_Kato_connection_variation=upper(connection_variation),
                    actual_Delta_T_second_order_CAR=upper(matter_quadratic),actual_hPhiC_gap_variation_T=upper(matter_gap_variation),
                    actual_corrector_derivative_cross=upper(derivative_corrector),actual_compact_bump_derivative_cross=upper(derivative_bump),
                    exact_positive_matrix_function_coefficient_enclosure=upper(coefficient_price),full105_metric_with_rotated_derivatives=upper(metric_new)),
                anti_H_generator_on_compact_packet_L2_upper=upper(generator_norm),generator_exp_T_Taylor_L2_upper=upper(generator_taylor),
                physical_source_endpoint_change_upper=upper(source_endpoint),source_endpoint_derivation='Physical E0=.5 sum4 c_active0^dag R. Ref4 contraction gives ||E0psi||=sqrt(.5); commutator [dGamma(T),E0]=.5 c^dag T R adds one active one-particle generator norm. Bound uses actual active coefficients and occupied CAR load, not ||E0||=sqrt(.5) as an operator fiction.',
                domain='Anti-H T is linear in compact chart coordinates, finite16644-mode CAR; expT is bounded unitary multiplication globally and smooth. Wtotal keeps b6 support and D(H²), no bosonic PH/unbounded generator surrogate.',
                raw_phases_unchanged=dict(primitive=proof['raw_primitive_centre_energy'],occupied_all65=proof['raw_all65_occupied_centre_energy'],joint90=proof['raw_exact_joint90_harmonic_energy_enclosure']),
                no_source_consumer_promotion=True)


def literal_rotated_source(geom: dict[str,Any], g: list[Any], profiles: list[dict[str,Any]], field: dict[str,Any], transitions: dict[str,Any]) -> dict[str,Any]:
    """Actual W_Kato(eta) exp(T(eta)) packet, including nonlinear native links."""
    eta = np.zeros(90)
    eta[0],eta[17],eta[45],eta[58] = 2**-15,2**-16,2**-16,2**-17
    gamma = [mp.matrix([[mp.mpc(complex(v)) for v in row] for row in m]) for m in full.supplier.primitive.SELECTED]
    M = [mp.matrix([[mp.mpc(complex(v)) for v in row] for row in m]) for m in geom['M']]
    frame_mp = [mp.sqrt(scalar(g[m]))*gamma[m] for m in range(4)]
    dframe_mp = [mp.zeros(4) for _ in range(4)]
    for a,column in enumerate(geom['columns']):
        for slot,c in column.items():
            dframe_mp[slot//15] += mp.mpf(float(eta[a]))*scalar(c)*M[slot%15]/(8*mp.sqrt(8))
    X_mp = [sum((mp.mpf(float(eta[45+15*axis+a]))*M[a] for a in range(15)),mp.zeros(4))/(8*mp.sqrt(8)) for axis in range(3)]
    # Exact input intervals and the 80-digit evaluator have an explicit floor.
    # Reject a wider epoch rather than silently spending a too-small input price.
    widths = [abs(scalar(endpoint(iv.sqrt(v),1)-endpoint(iv.sqrt(v),0))) for v in g]
    widths += [abs(scalar(endpoint(c,1)-endpoint(c,0))) for column in geom['columns'] for c in column.values() if hasattr(c,'_mpi_')]
    if max(widths,default=mp.mpf(0)) >= mp.mpf('1e-70'):
        raise ArithmeticError('Physical Kato point input intervals exceed declared numerical price')
    steps = 16
    frame = np.asarray(full.supplier.primitive.SELECTED,dtype=complex)*np.asarray([float(mp.sqrt(scalar(v))) for v in g])[:,None,None]
    dframe = np.einsum('a,amij->mij',eta[:45],geom['deltas'])/8
    X = np.einsum('xa,aij->xij',eta[45:].reshape(3,15),geom['M']/np.sqrt(8))/8
    projectors,undressed,controls = [],[],[]
    anti = unitary = 0.
    for profile in profiles:
        records = [r for r in transitions['records'] if r['role']==profile['role']]
        kato_unitary = transport_error = origin_error = 0.
        ode_price = eigen_price = midpoint_distance = midpoint_variation = density_price = mp.mpf(0)
        ray = mp.mpf(float(np.sum(abs(eta))))
        # Independent actual-ray envelopes: F(s)=F0+s dF and U(s)=exp(i s L X).
        # Original ordered hopping is (hop(s)U(s)+U(s)hop(s))/2 + h.c.
        # Hence ||h'||<=onsite||dF0||+sum2(||dhop||+L||X||||hop||),
        # ||h''||<=sum(4L||X||||dhop||+2L²||X||²||hop||).
        # Frobenius norms dominate operator norms, and ||hop(s)||<=||hop0||+||dhop||.
        D1 = abs(scalar(F(profile['onsite'])))*norm(dframe_mp[0])
        D2 = mp.mpf(0)
        for path in profile['paths']:
            axis,L = path['axis'],path['length']
            c0,ci = scalar(F(path['c0'])),scalar(F(path['ci']))
            hop0 = c0*frame_mp[0]+mp.j*ci*frame_mp[axis+1]
            dhop = c0*dframe_mp[0]+mp.j*ci*dframe_mp[axis+1]
            speed = L*norm(X_mp[axis])
            hop_bound = norm(hop0)+norm(dhop)
            D1 += 2*(norm(dhop)+speed*hop_bound)
            D2 += 4*speed*norm(dhop)+2*speed*speed*hop_bound
        D1 += mp.mpf('1e-60');D2 += mp.mpf('1e-60')
        minimum_ray_gap = mp.inf
        for ki,record in zip(product(range(4),repeat=3),records,strict=True):
            point = point_coefficients(profile,ki,geom,g)
            # Weyl along this literal ray, not any inherited global-normal bound.
            gap_lower = point['E']-D1-mp.mpf('1e-60')
            if gap_lower <= 0:
                raise ArithmeticError('Direct original nonlinear Kato ray lost its positive gap')
            minimum_ray_gap = min(minimum_ray_gap,gap_lower)
            # Riesz/Sylvester differentiation across the occupied/empty gap:
            # ||K'||<=8||h''||/gap+64||h'||²/gap², never within-band division.
            variation = (8*D2/gap_lower+64*D1*D1/(gap_lower*gap_lower))/(4*steps)
            theta = (2*np.pi*np.asarray(ki)+np.pi)/4
            sign = -1 if profile['scale']=='heavy' else 1
            def h_and_derivative(s: float) -> tuple[np.ndarray,np.ndarray]:
                fs = frame+s*dframe
                h = float(F(profile['onsite']))*fs[0]
                derivative = float(F(profile['onsite']))*dframe[0]
                for path in profile['paths']:
                    axis,L = path['axis'],path['length']
                    c0,ci = float(F(path['c0'])),float(F(path['ci']))
                    hop = c0*fs[0]+1j*ci*fs[axis+1]
                    dhop = c0*dframe[0]+1j*ci*dframe[axis+1]
                    link = expm(1j*s*L*X[axis])
                    dlink = 1j*L*X[axis]@link
                    coefficient = (hop@link+link@hop)/2
                    dcoefficient = (dhop@link+link@dhop+hop@dlink+dlink@hop)/2
                    phase = np.exp(1j*L*theta[axis])
                    h = h+phase*coefficient+phase.conjugate()*coefficient.conj().T
                    derivative = derivative+phase*dcoefficient+phase.conjugate()*dcoefficient.conj().T
                return sign*h,sign*derivative
            def exact_pair(s: Any) -> tuple[Any,Any]:
                fs = [frame_mp[m]+s*dframe_mp[m] for m in range(4)]
                h = scalar(F(profile['onsite']))*fs[0]
                derivative = scalar(F(profile['onsite']))*dframe_mp[0]
                for path in profile['paths']:
                    axis,L = path['axis'],path['length']
                    c0,ci = scalar(F(path['c0'])),scalar(F(path['ci']))
                    hop = c0*fs[0]+mp.j*ci*fs[axis+1]
                    dhop = c0*dframe_mp[0]+mp.j*ci*dframe_mp[axis+1]
                    link = mp.expm(mp.j*s*L*X_mp[axis])
                    dlink = mp.j*L*X_mp[axis]*link
                    coefficient = (hop*link+link*hop)/2
                    dcoefficient = (dhop*link+link*dhop+hop*dlink+dlink*hop)/2
                    k = ki[axis]
                    phase = (((1,-1,-1,1)[k]+mp.j*(1,1,-1,-1)[k])/mp.sqrt(2) if L==1 else mp.j*(1,-1,1,-1)[k])
                    h += phase*coefficient+mp.conj(phase)*coefficient.H
                    derivative += phase*dcoefficient+mp.conj(phase)*dcoefficient.H
                return sign*h,sign*derivative
            def rhs(s: float, w: np.ndarray) -> np.ndarray:
                _,connection = joined.projector_connection(*h_and_derivative(s))
                return (connection@w.reshape(4,4)).reshape(16)
            solution = solve_ivp(rhs,(0.,1.),np.eye(4,dtype=complex).reshape(16),rtol=2e-11,atol=2e-13,method='DOP853')
            if not solution.success:
                raise ArithmeticError(solution.message)
            W = solution.y[:,-1].reshape(4,4)
            # A separately computed piecewise-constant midpoint propagator
            # supplies an a priori price on the actual DOP853 result. Unitarity
            # of exact Kato evolution prices its integral defect, without an
            # exp(||K||) loss or treating a step-doubling difference as proof.
            with mp.workdps(80):
                Z = mp.eye(4)
                eval_price = max_eigen = mp.mpf(0)
                for step in range(steps):
                    s = mp.mpf(2*step+1)/(2*steps)
                    hf,dhf = h_and_derivative(float(s))
                    _,Kf = joined.projector_connection(hf,dhf)
                    ev,Uf = np.linalg.eigh(hf)
                    Ump = mp.matrix(Uf.tolist())
                    he,de = exact_pair(s)
                    residual = norm(he*Ump-Ump*mp.diag(ev.tolist()))
                    orthogonal = norm(Ump.H*Ump-mp.eye(4))
                    if orthogonal >= mp.mpf('1e-10'):
                        raise ArithmeticError('Physical midpoint eigenspace is not numerically orthonormal')
                    Pmp = Ump[:,:2]*Ump[:,:2].H
                    bands = Ump.H*de*Ump
                    dp = mp.zeros(4)
                    for i,j in product(range(2),repeat=2):
                        dp[i,2+j] = -bands[i,2+j]/(mp.mpf(float(ev[2+j]))-mp.mpf(float(ev[i])))
                        dp[2+j,i] = mp.conj(dp[i,2+j])
                    dp = Ump*dp*Ump.H
                    Kmp = dp*Pmp-Pmp*dp
                    displayed = mp.matrix(Kf.tolist())
                    # Backward eigenspace residual plus explicit input/arithmetic
                    # floor prices the physical connection, not only P transport.
                    ep = norm(displayed-Kmp)+64*norm(de)*(residual/gap_lower+orthogonal)/gap_lower+mp.mpf('1e-60')
                    eval_price += ep/steps
                    max_eigen = max(max_eigen,ep)
                    Z = mp.expm(displayed/steps)*Z
                distance = norm(mp.matrix(W.tolist())-Z)
                if eval_price+variation >= mp.mpf('.1'):
                    raise ArithmeticError('Physical midpoint numerical error price is vacuous')
                werror = distance+2*(variation+eval_price)+mp.mpf('1e-60')
            ode_price = max(ode_price,werror)
            eigen_price = max(eigen_price,max_eigen)
            midpoint_distance = max(midpoint_distance,distance)
            midpoint_variation = max(midpoint_variation,variation)
            P0,_ = joined.projector_connection(*h_and_derivative(0.))
            P1,_ = joined.projector_connection(*h_and_derivative(1.))
            P = np.asarray(point['occupied'].tolist(),dtype=complex)
            origin_error = max(origin_error,float(np.linalg.norm(P-P0)))
            kato_unitary = max(kato_unitary,float(np.linalg.norm(W.conj().T@W-np.eye(4))))
            transport_error = max(transport_error,float(np.linalg.norm(W@P0@W.conj().T-P1)))
            real,imag = record['T_empty_occupied']['real'],record['T_empty_occupied']['imag']
            t = np.array([[complex(float(a),float(b)) for a,b in zip(ar,br,strict=True)] for ar,br in zip(real,imag,strict=True)])
            plus = (eta@t).reshape(4,4)
            T = plus-plus.conj().T
            S = expm(T)
            with mp.workdps(80):
                plus_mp = mp.matrix(4)
                for i,j in product(range(4),repeat=2):
                    plus_mp[i,j] = sum(mp.mpf(float(eta[a]))*mp.mpc(real[a][4*i+j],imag[a][4*i+j]) for a in range(90))
                Sexact = mp.expm(plus_mp-plus_mp.H)
                serror = norm(mp.matrix(S.tolist())-Sexact)+mp.mpf('1e-60')+transitions['coefficient_error']*ray
                perror = norm(mp.matrix(P.tolist())-point['occupied'])+mp.mpf('1e-60')
                # ||W S P Sdag Wdag - Wnum Snum Pnum Snumdag Wnumdag||F.
                e = werror+serror+werror*serror
                density_price = max(density_price,(2+e)*e+(1+e)**2*perror+mp.mpf('1e-60'))
            density = S@P@S.conj().T
            if profile['role']=='active filled charged':
                projectors.append(W@density@W.conj().T)
                undressed.append(density)
            anti = max(anti,float(np.linalg.norm(T+T.conj().T)))
            unitary = max(unitary,float(np.linalg.norm(S.conj().T@S-np.eye(4))))
        if max(kato_unitary,transport_error,origin_error)>5e-10:
            raise ArithmeticError('Actual original nonlinear uniform Kato/counterrotation consumer failed')
        controls.append(dict(role=profile['role'],load=profile['load'],all64_physical_negative_rank=2,
                             original_mean_projector_defect=origin_error,actual_Kato_unitarity_defect=kato_unitary,
                             actual_nonlinear_projector_transport_defect=transport_error,
                             direct_ray_first_h_derivative_upper=upper(D1),direct_ray_second_h_derivative_upper=upper(D2),
                             direct_ray_distance_to_zero_lower=display_scalar(minimum_ray_gap*(1-mp.mpf('1e-60'))),
                             actual_ODE_operator_error_upper=upper(ode_price),midpoint_steps=steps,
                             observed_DOP853_midpoint_operator_distance_upper=upper(midpoint_distance),
                             exact_midpoint_generator_variation_price_upper=upper(midpoint_variation),
                             explicit_eigenspace_connection_error_upper=upper(eigen_price),
                             physical_density_Frobenius_error_upper=upper(density_price)))
    local = np.mean(projectors,axis=0)
    missing_Kato = float(np.linalg.norm(local-np.mean(undressed,axis=0)))
    source = joined.scalar.literal_car_source(local)
    return dict(coordinate=eta.tolist(),coordinate_norm=float(np.linalg.norm(eta)),actual_counterrotated_density_trace=float(np.trace(local).real),
                actual_unitarity_defect=unitary,actual_anti_H_defect=anti,literal_eight_mode_physical_source=source,
                actual_all65_nonlinear_Kato_consumers=controls,adverse_omitted_physical_Kato_density_defect=missing_Kato,
                numerical_price_law='Actual DOP853 W is compared to an independent16-step midpoint product at80digits. Anti-H Duhamel prices midpoint variation by sup||Kprime||/(4n). D1/D2 are independently derived from actual F0+s dF, exp(i s L X) and original ordered hop/HC; Weyl gap=Emean-D1 on this very ray. NO inherited global-normal derivative bound is used. Each midpoint connection is priced against an80digit original nonlinear h/dh evaluator by backward eigenspace residual and orthonormality; input widths<1e-70 and arithmetic floor1e-60 are explicit. Density price includes W, expT and mean-projector numerical errors. These are point-consumer prices, not extra physical packet remainders or a newly improved source-leg dynamics theorem.',
                actual_state_order='W_Kato(eta) exp(T(eta)) Pmean exp(-T(eta)) W_Kato(eta)^dag; actual original ordered Phi/link law and analytic radial derivative',
                full_particle_loading=dict(heavy=128,active=128,auxiliary=63*128,reference=4,total=8324,source_sector_total=8324),
                exact_probability='Each actual W_Kato expT Bloch projector is rank2; translation averaging yields TrP_site=2. E0/Ref4 hence norm²(4-TrP_site)/4=1/2 exactly for the new state too. Ref4->3 and active128->129 conserve total8324.')


def run(supplier_path: Path, base_path: Path, full_path: Path) -> dict[str,Any]:
    started = time.monotonic()
    inputs = {str(p):p.read_bytes() for p in (supplier_path,base_path,full_path)}
    supplier,base,fullcap = [json.loads(inputs[str(p)]) for p in (supplier_path,base_path,full_path)]
    if hashlib.sha256(SOURCE_BYTES[Path(frozen.__file__).name]).hexdigest()!=BASE_SOURCE_SHA or hashlib.sha256(inputs[str(base_path)]).hexdigest()!=BASE_RECEIPT_SHA:
        raise ArithmeticError('Frozen actual8bfe source/receipt required; no silent epoch replacement')
    if colour.__file__ is None:
        raise ArithmeticError('Frozen colour source has no physical file')
    if hashlib.sha256(inputs[str(full_path)]).hexdigest()!=joined.FULL_RECEIPT_SHA or supplier['source_at_import'][Path(colour.__file__).name]['sha256']!=COLOUR_SHA:
        raise ArithmeticError('Frozen f63/full90 controls required')
    mp.mp.dps=960;iv.dps=1100
    root = supplier['actual_NEW_colour_BO_feedback_mean']
    profiles = supplier['input_descriptor']['species_coefficients']
    if [p['load'] for p in profiles]!=[1,1,63]:
        raise ArithmeticError('All65 physical species loads changed')
    native,geom = joined.literal_geometry(root,fullcap)
    g0,gs = frozen.interval(root['parameters']['g0']),frozen.interval(root['parameters']['gs'])
    g = [g0,gs,gs,gs]
    pencil = exact_field_pencil(geom,g,profiles)
    field = positive_mixed_root(pencil,supplier['actual_joint90_quantum'])
    transitions = uniform_transitions(geom,g,profiles,field)
    literal = literal_differential_and_CAR(field,transitions)
    budget = packet_budget(base,supplier,field,transitions,geom,g)
    source = literal_rotated_source(geom,g,profiles,field,transitions)
    if scalar(F(budget['actual_new_state_reference_residual_upper']['upper'])) >= scalar(F(budget['old_reported_Kato_first_upper']['upper'])):
        raise ArithmeticError('Independent physical state improvement did not beat dominant Kato residual')
    descriptor = dict(law='NEW_EXACT_FINITE_CAR_SUPERADIABATIC_UNIFORM90_TRANSFER',N=4,V=64,a='1/64',Lambda='2^8192',chi='2^-4096',Gphi='8chi',Gc='3chi/10',colour_Casimir='15/8',colour_stdT='B/sqrt8',colour_primitive='.3chi E4²/2',
        all65_loads=[1,1,63],heavy_physical_sign='-Lambda h BOTH occupied bands',Ref=4,particles=8324,clock=4096,
        source='E0=.5 sum4 c_active0alpha^dag Ralpha, raw probability1/2, Ref4->3, no whitening',
        frozen_inputs={name:hashlib.sha256(raw).hexdigest() for name,raw in inputs.items()},
        exact_state=budget['exact_state'],exact_coefficient_law='Omega=G^1/2 A G^1/2=sqrt(G^1/2 K90 G^1/2); t=-G^-1/2 Omega(Delta+Omega)^-1 G^1/2 k; k_eff=G^-1/2 Delta(Delta+Omega)^-1 G^1/2 k. Only empty/occupied gap2E, physicalLambda forheavy. Complete PC/staticPH/contact retained.',
        fullq_ABI='resolve_transition(G,A,physical_gap,k[field,transition]) and all_q_counterrotations(field_blocks,dual_PH_records); field90 eachq, all64 q, total5760; OUTk+q/INk ranks2+2, paired q/-q anti-H matrices, scale/load explicit; no moving imports',
        no_bare_EM_curl=True,omitted_only_this_branch=['Theta/Pi','Coulomb'],underlying_H_unchanged=True)
    return dict(input_descriptor=descriptor,input_sha256=hashlib.sha256(json.dumps(descriptor,sort_keys=True).encode()).hexdigest(),
        actual_mixed90_positive_field_function=dict(root_equation_residual=upper(field['residual']),root_operator_error_upper=upper(field['root_error']),
            full_PC_square_root_norm=upper(field['actual_nonzero_PC_norm']),positive_frequency_lower_divided_2to3072=display_scalar(field['omega_min']),Sylvester_iterations=field['iterations'],
            precision_A=encode(field['A']),field_cometric_G=encode(field['G']),full_covariance=encode(field['covariance'])),
        actual_all65_physical_counterrotation_coefficients=transitions['records'],actual_all65_gap_loading_and_costs=transitions['species'],
        actual_independent_literal_field_CAR_cancellation=literal,actual_new_prepared_state_budget=budget,actual_physical_unitary_density_and_source=source,
        source_at_import={name:dict(path=str(ROOT/name),sha256=hashlib.sha256(raw).hexdigest(),bytes_utf8=raw.decode()) for name,raw in SOURCE_BYTES.items()},
        source_unchanged_since_import={name:(ROOT/name).read_bytes()==raw for name,raw in SOURCE_BYTES.items()},
        input_unchanged={name:Path(name).read_bytes()==raw for name,raw in inputs.items()},
        runtime=dict(executable=sys.executable,python=sys.version,mpmath=mp.__version__,numpy=np.__version__,precision=mp.mp.dps),elapsed_seconds=time.monotonic()-started,
        limitations=['Independent uniform90 physical prepared-state improvement and genuine fullq ABI only; no fullq execution/source consumer evidence claimed.',
            'Not full nonlinear ground/vacuum, QED/commoncone/originalSOURCE or electromagnetic law promotion.',
            'Frozen8bfe/f63/full90 scientific sources and existing source consumer remain unchanged.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--supplier',type=Path,required=True)
    parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--full45',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=run(args.supplier,args.base,args.full45)
    if not all(result['source_unchanged_since_import'].values()) or not all(result['input_unchanged'].values()):
        raise ArithmeticError('Scientific import/input epoch changed during execution')
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(input_sha256=result['input_sha256'],budget=result['actual_new_prepared_state_budget'],literal=result['actual_independent_literal_field_CAR_cancellation'],source=result['actual_physical_unitary_density_and_source'],elapsed_seconds=result['elapsed_seconds']),indent=2))
