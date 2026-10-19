"""Finite-frequency AND nonzero-momentum pure-gauge response of frozen7b GW.

Actual bare-line triangles, both ordered Peierls endpoint contacts, bare CAR
exchange-projector response and full DW phase contact. Not CovariantGF, general
transverse vertex, anomaly, eR or original SOURCE/full conserving QED admission.
"""
from __future__ import annotations
import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any
import numpy as np
import induced_native_charged_green as charged
ROOT = Path(__file__).resolve().parent
OWN = Path(__file__).read_bytes()
CHARGED_SOURCE_SHA = '3a9ffc46f646f52fe9dcdb6a29793c69ea7f9ac93007682174cd27d2bde2fe08'
CHARGED_RECEIPT_SHA = '7b72b6e05f9990699a8bfd425cd0be062564d608b02ba993884332fde87a34d2'
PROBES = [(z, nu) for z in (complex(0, .5), complex(10, .25), complex(30, 1)) for nu in (.17, .61)]


def maxabs(value: np.ndarray) -> float:
    return float(np.max(abs(value)))


def unpack(value: Any) -> np.ndarray:
    return np.asarray(value['real']) + 1j * np.asarray(value['imag'])


def contact_vertices(native: dict[str, Any], profile: dict[str, Any], pout: np.ndarray,
                     pin: np.ndarray, edge: np.ndarray, density: np.ndarray,
                     f: np.ndarray) -> np.ndarray:
    """Literal mixed external-gradient/quantum-A h''; density commutator is zero.

    Complex Fourier f is the linear combination of TWO real physical sources.
    Reverse hopping differentiates exp(-i Theta), not conjugate(complex f).
    """
    M = edge.shape[1]
    result = np.zeros((M + density.shape[1], 4, 4), complex)
    for path in profile['paths']:
        x, y = path['source'], path['target']
        forward = np.exp(1j * (pin @ native['sites'][y] - pout @ native['sites'][x])) / native['V']
        reverse = np.exp(1j * (pin @ native['sites'][x] - pout @ native['sites'][y])) / native['V']
        ell = native['a'] * np.sum(edge[list(path['edges'])], axis=0)
        result[:M] -= (ell * (f[x] - f[y]))[:, None, None] * (forward * path['hop'] + reverse * path['hop'].conj().T)
    return result


def phase_derivative(native: dict[str, Any], profile: dict[str, Any], f: np.ndarray,
                     path_weights: list[float]) -> np.ndarray:
    """Differentiate actual ordered paths, including range2/seams and DW weights."""
    result = np.zeros((4 * native['V'], 4 * native['V']), complex)
    for path, weight in zip(profile['paths'], path_weights, strict=True):
        x, y = path['source'], path['target']
        hop = weight * path['hop']
        result[4*x:4*x+4, 4*y:4*y+4] += 1j * (f[x]-f[y]) * hop
        result[4*y:4*y+4, 4*x:4*x+4] -= 1j * (f[x]-f[y]) * hop.conj().T
    return result


def mode_vertices(O: np.ndarray, amps: np.ndarray) -> np.ndarray:
    return np.einsum('mij,ml->lij', O, amps, optimize=True)


def spectral_triangle(T1: np.ndarray, T0: np.ndarray, e1: np.ndarray, U1: np.ndarray,
                      e0: np.ndarray, U0: np.ndarray, omega: np.ndarray,
                      bare: np.ndarray, z: complex, nu: float, sign: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Perform the actual zero-T spectral convolution of g_a(w+nu) V g_b(w) D.

    Partial fraction denominator is nu+e_b-e_a (BARE electron energies).
    For each boson orientation s, F_e^s = n_e^s/(z-e-s*Omega), where
    n_e^+ = 1-n_e and n_e^- = n_e. Thus opposite occupied/unoccupied
    lines contribute ONE term from EACH boson orientation. A product with
    the same e_b +/- Omega shift on both lines is wrong for these sectors.
    """
    inserted = U1.conj().T @ bare @ U0
    total = np.zeros((4, 4), complex)
    crossing = np.zeros_like(total)
    naive = np.zeros_like(total)
    for a in range(4):
        left = T1 @ U1[:, a]
        na = float((e1[a] > 0) == (sign == 1))
        for b in range(4):
            right = T0 @ U0[:, b]
            nb = float((e0[b] > 0) == (sign == 1))
            delta = nu + e0[b] - e1[a]
            if abs(delta) < 1e-10:
                raise ArithmeticError('Chosen finite frequency meets a bare transition; use nonsingular divided difference')
            weight = (nb/(z-e0[b]-sign*omega) - na/(z+nu-e1[a]-sign*omega)) / delta
            term = np.einsum('li,lj,l->ij', left, right.conj(), inserted[a, b] * weight, optimize=True)
            total += term
            if (e1[a] > 0) != (e0[b] > 0):
                crossing += term
            wrong_weight = nb / ((z+nu-e1[a]-sign*omega) * (z-e0[b]-sign*omega))
            naive += np.einsum('li,lj,l->ij', left, right.conj(), inserted[a, b] * wrong_weight, optimize=True)
    return total, crossing, naive


def endpoint(T: np.ndarray, dT: np.ndarray, eig: np.ndarray, U: np.ndarray,
             omega: np.ndarray, z: complex, sign: int, left: bool) -> np.ndarray:
    result = np.zeros((4, 4), complex)
    for band in range(4):
        if (eig[band] > 0) != (sign == 1):
            continue
        denominator = z-eig[band]-sign*omega
        if left:
            a, b = dT @ U[:, band], T @ U[:, band]
            result += np.einsum('li,lj,l->ij', a, b.conj(), 1/denominator, optimize=True)
        else:
            a = T @ U[:, band]
            # dT is the literal RIGHT vertex (internal -> external), so no
            # Hermitian conjugation of its complex Fourier gauge generator.
            b = np.einsum('i,lij->lj', U[:, band].conj(), dT)
            result += np.einsum('li,lj,l->ij', a, b, 1/denominator, optimize=True)
    return result


def neutral_kernel_closure(supplier: Any, f: np.ndarray, gradient: np.ndarray) -> dict[str, Any]:
    """Actual full-negative-band bubble witness plus exact all-coordinate proof.

    Bare g_alpha(t,t')=U(t)g(t,t')U(t')^-1 follows from its differential
    equation with delta h=i alpha(t)[f,h]-dot(alpha)f and transformed P.
    Equivalently, the inserted bare-line identity is
    g1 delta h g0=i(f g0-g1 f); the triangle above retains BOTH bands.
    Every physical J_mu transforms U J_mu U^-1; density commutes with f.
    Substituting delta g=i(alpha(t)f g-alpha(t')g f) into
    Tr[J_mu(t) g>(t,t') J_nu(t') g<(t',t)] cancels all four endpoint
    phases, with delta J=i alpha[f,J] supplying both ordered contacts.
    Equal-time Tr(P h''_mu,nu) cancels between delta P=i alpha[f,P]
    and the literal ordered-path delta h''=i alpha[f,h''] third contact.
    This holds for EVERY photon/density coordinate pair, all times and
    all 1+63 species, BEFORE PH compression. Thus the unchanged physical
    neutral kernel can retain exactly the SAME fixed256-bin canonical W:
    delta W=0 is a proved pure-gauge identity, not a missing-field fallback.
    Noncollinear quantum q legs (0,pi/2,0) and (pi/2,pi/2,0) allow
    nonzero terms AND x-current phase contacts under external q=pi/2 x.
    """
    native = supplier.native
    s = supplier.s
    V = native['V']
    rho = np.diag(np.repeat(f, 4))
    blocks = [supplier.block(q) for q in ((0, 1, 0), (1, 1, 0))]
    rows: list[dict[str, Any]] = []
    contact_rows = []
    for profile in native['Pprofiles']:
        zero = np.zeros(3*V)
        h = s.primitives.matter_hamiltonian(native, profile, zero)
        ev, U = np.linalg.eigh(h)
        P = U[:, ev < 0] @ U[:, ev < 0].conj().T
        dP = 1j*(rho@P-P@rho)
        operators = []
        for block in blocks:
            legs = []
            for column in range(block.M):
                edge = native['a']*block.data['physical_edgeframe'][:, column]
                J = s.primitives.path_derivative(native, profile, zero, edge)
                dJ = s.primitives.path_derivative(native, profile, zero, np.real(gradient), 2, other=edge) + 1j*s.primitives.path_derivative(native, profile, zero, np.imag(gradient), 2, other=edge)
                legs.append(dict(J=J, dJ=dJ, edge=edge, kind='physical_A', column=column))
            for column in range(block.R):
                density = np.kron(np.diag(block.data['physical_densityframe'][:, column]), np.eye(4))
                legs.append(dict(J=density, dJ=np.zeros_like(density), kind='physical_density', column=column))
            operators.append(legs)
        for leg in operators[0]:
            for other in operators[1]:
                if leg['kind'] != 'physical_A' or other['kind'] != 'physical_A':
                    continue
                edge, edge2 = leg['edge'], other['edge']
                C = s.primitives.path_derivative(native, profile, zero, edge, 2, other=edge2)
                weights = [-float(np.sum(edge[list(path['edges'])]))*float(np.sum(edge2[list(path['edges'])])) for path in profile['paths']]
                dC = phase_derivative(native, profile, f, weights)
                contact_rows.append(dict(species_load=profile['load'], first_A_column=leg['column'], second_A_column=other['column'],
                                         projector_contact_term=np.trace(dP@C), ordered_third_contact_term=np.trace(P@dC),
                                         full_PH_contact_variation_error=float(abs(np.trace(dP@C+P@dC))),
                                         actual_mixed_path_contact_operator_error=maxabs(leg['dJ']-1j*(rho@leg['J']-leg['J']@rho)),
                                         actual_third_path_contact_operator_error=maxabs(dC-1j*(rho@C-C@rho))))
        for nu in (.17, .61):
            t, tp = .37, -.21
            at, ap = np.exp(-1j*nu*t), np.exp(-1j*nu*tp)
            evol = (U*np.exp(-1j*ev*(t-tp))[None, :]) @ U.conj().T
            greater = -1j*evol@(np.eye(4*V)-P)
            lesser_back = 1j*evol.conj().T@P
            dg = 1j*(at*rho@greater-ap*greater@rho)
            dl = 1j*(ap*rho@lesser_back-at*lesser_back@rho)
            for leg in operators[0]:
                J, dJ = leg['J'], leg['dJ']
                for other in operators[1]:
                    K, dK = other['J'], other['dJ']
                    terms = [np.trace((at*dJ)@greater@K@lesser_back), np.trace(J@dg@K@lesser_back),
                             np.trace(J@greater@(ap*dK)@lesser_back), np.trace(J@greater@K@dl)]
                    rows.append(dict(species_load=profile['load'], nu=nu, times=[t, tp], first_leg=leg['kind'], second_leg=other['kind'],
                                     first_q_index=[0, 1, 0], second_q_index=[1, 1, 0], first_column=leg['column'], second_column=other['column'],
                                     full_negative_rank=int(np.sum(ev < 0)), full_positive_rank=int(np.sum(ev > 0)), bubble_variation_terms=terms,
                                     actual_full_two_line_bubble_variation_error=float(abs(sum(terms))),
                                     WRONG_omit_ordered_vertex_contacts_bubble_defect=float(abs(terms[1]+terms[3]))))
    return dict(actual_full_256_matrix_witnesses=rows, actual_full_256_contact_witnesses=contact_rows,
                bubble_variation_max=max(r['actual_full_two_line_bubble_variation_error'] for r in rows),
                PH_contact_variation_max=max(r['full_PH_contact_variation_error'] for r in contact_rows),
                literal_contact_operator_max=max(max(r['actual_mixed_path_contact_operator_error'], r['actual_third_path_contact_operator_error']) for r in contact_rows),
                WRONG_omit_contacts_bubble_defect_max=max(r['WRONG_omit_ordered_vertex_contacts_bubble_defect'] for r in rows),
                all_coordinate_operator_argument=neutral_kernel_closure.__doc__,
                consequence='delta bare full neutral PH/current/contact kernel = 0, hence delta fixed256-bin canonical neutral photon/phi W = 0 for a pure gauge; no missing-deltaW fallback',
                scope='Native frozenPhi Gaussian neutral supplier only, not full microscopic selfconsistent QED')


def construct(supplier: Any, frozen: dict[str, Any]) -> dict[str, Any]:
    native = supplier.native
    s = supplier.s
    V = native['V']
    profile = native['Pprofiles'][0]
    mom = (2*np.pi*native['sites']+np.pi)/native['N']
    idx = [int(np.flatnonzero(np.all(native['sites'] == x, axis=1))[0]) for x in ([0, 0, 0], [1, 0, 0])]
    p0, p1 = mom[idx]
    q = p1-p0
    f = np.exp(1j*native['sites']@q)
    gradient = native['D'].T@f
    rho = np.diag(np.repeat(f, 4))
    allE = [charged.embedding(native, p) for p in mom]
    E0, E1 = (allE[i] for i in idx)
    hfull = s.primitives.matter_hamiltonian(native, profile, np.zeros(3*V))
    bare_gradient = s.primitives.path_derivative(native, profile, np.zeros(3*V), gradient.real) + 1j*s.primitives.path_derivative(native, profile, np.zeros(3*V), gradient.imag)
    barecheck = maxabs(bare_gradient-1j*(rho@hfull-hfull@rho))
    bands = [np.linalg.eigh(E.conj().T@hfull@E) for E in allE]
    ev, U = np.linalg.eigh(hfull)
    P = U[:, ev < 0] @ U[:, ev < 0].conj().T
    u = native['UE']/native['a']
    L = np.repeat(np.repeat(native['Lplus'], 4, axis=0), 4, axis=1)
    Cfull = u*np.kron(np.diag(np.diag(native['Lplus'])), np.eye(4))-2*u*L*P
    deltaP = 1j*(rho@P-P@rho)
    deltaC = E1.conj().T@(-2*u*L*deltaP)@E0
    rec = [dict(p=p, bare=E.conj().T@hfull@E, C=E.conj().T@Cfull@E, B=[], d=[]) for p, E in ((p0, E0), (p1, E1))]
    covariance = np.zeros((3*V, 3*V))
    triangles = np.zeros((len(PROBES), 4, 4), complex)
    crossings = np.zeros_like(triangles)
    naive = np.zeros_like(triangles)
    endpoint_left = np.zeros_like(triangles)
    endpoint_right = np.zeros_like(triangles)
    qrows = []
    contact_errors = []
    internal_bare_errors = []
    countA = countR = countOmega = pairs = opposite_pairs = 0
    index = {tuple(site): i for i, site in enumerate(native['sites'])}
    for site in native['sites']:
        qb = tuple(int(i) for i in site)
        if qb > tuple((-i)%native['N'] for i in qb):
            continue
        block = supplier.block(qb)
        edge, density = block.data['physical_edgeframe'], block.data['physical_densityframe']
        amp, omega = block.amplitudes, block.omega
        covariance += native['a']**2*edge@np.real(amp[:block.M]@amp[:block.M].conj().T)@edge.T
        countA += block.M
        countR += block.R
        countOmega += len(omega)
        allowed = sorted({index[tuple((native['sites'][idx[0]]+sign*np.asarray(qb))%native['N'])] for sign in (-1, 1)})
        for k0 in allowed:
            k1 = index[tuple((native['sites'][k0]+[1, 0, 0])%native['N'])]
            e0, U0 = bands[k0]
            e1, U1 = bands[k1]
            O0 = charged.vertices(native, profile, p0, mom[k0], edge, density)
            O1 = charged.vertices(native, profile, p1, mom[k1], edge, density)
            dl = contact_vertices(native, profile, p1, mom[k0], edge, density, f)
            dr = contact_vertices(native, profile, mom[k1], p0, edge, density, f)
            contact_errors.extend([maxabs(dl-1j*(O0-O1)), maxabs(dr-1j*(O0.conj().transpose(0, 2, 1)-O1.conj().transpose(0, 2, 1)))])
            for sign in (1, -1):
                amplitudes = amp if sign == 1 else amp.conj()
                T0, T1 = mode_vertices(O0, amplitudes), mode_vertices(O1, amplitudes)
                dleft = mode_vertices(dl, amplitudes)
                dright = mode_vertices(dr, amplitudes.conj())
                for r, eig, Ub, T in ((rec[0], e0, U0, T0), (rec[1], e1, U1, T1)):
                    for column in range(4):
                        if (eig[column] > 0) == (sign == 1):
                            r['B'].append((T@Ub[:, column]).T)
                            r['d'].append(eig[column]+sign*omega)
                for probe, (z, nu) in enumerate(PROBES):
                    bare = allE[k1].conj().T@bare_gradient@allE[k0]+1j*nu*np.eye(4)
                    internal_bare_errors.append(maxabs(bare-1j*(nu*np.eye(4)+allE[k0].conj().T@hfull@allE[k0]-allE[k1].conj().T@hfull@allE[k1])))
                    tri, cross, wrong = spectral_triangle(T1, T0, e1, U1, e0, U0, omega, bare, z, nu, sign)
                    triangles[probe] += tri
                    crossings[probe] += cross
                    naive[probe] += wrong
                    endpoint_left[probe] += endpoint(T0, dleft, e0, U0, omega, z, sign, True)
                    endpoint_right[probe] += endpoint(T1, dright, e1, U1, omega, z+nu, sign, False)
            pairs += 16*len(omega)
            opposite_pairs += 8*len(omega)
        qrows.append(dict(q=qb, physical_A_coordinates=block.M, physical_phi_coordinates=block.R,
                          positive_boson_modes=len(omega), actual_internal_momentum_pairs=[[mom[k], mom[index[tuple((native['sites'][k]+[1, 0, 0])%native['N'])]]] for k in allowed]))
    if (len(qrows), countA, countR, countOmega) != (36, 129, 63, 3865):
        raise ArithmeticError('Incomplete actual physical photon/phi spectrum')
    DWfull = np.zeros_like(hfull)
    weights = []
    for path in profile['paths']:
        edges = list(path['edges'])
        weight = -.5*float(np.sum(covariance[np.ix_(edges, edges)]))
        weights.append(weight)
        x, y = path['source'], path['target']
        DWfull[4*x:4*x+4, 4*y:4*y+4] += weight*path['hop']
        DWfull[4*y:4*y+4, 4*x:4*x+4] += weight*path['hop'].conj().T
    deltaDW = E1.conj().T@phase_derivative(native, profile, f, weights)@E0
    for r, E in zip(rec, (E0, E1), strict=True):
        r['DW'] = E.conj().T@DWfull@E
        r['h'] = r['bare']+r['C']+r['DW']
        r['B'], r['d'] = np.concatenate(r['B'], axis=1), np.concatenate(r['d'])
    frozen_checks = []
    for i, r in enumerate(rec):
        for old in frozen['charged_receivers'][i]['retarded_probes']:
            z = complex(old['z']['real'], old['z']['imag'])
            Sigma = charged.self_energy(r['B'], r['d'], z)
            G = np.linalg.solve(z*np.eye(4)-r['h']-Sigma, np.eye(4))
            frozen_checks.append(dict(receiver=i, z=z, frozen7b_Sigma_error=maxabs(Sigma-unpack(old['Sigma'])), frozen7b_raw_G_error=maxabs(G-unpack(old['G']))))
    responses = []
    for probe, (z, nu) in enumerate(PROBES):
        Sigma0 = charged.self_energy(rec[0]['B'], rec[0]['d'], z)
        Sigma1 = charged.self_energy(rec[1]['B'], rec[1]['d'], z+nu)
        F0 = z*np.eye(4)-rec[0]['h']-Sigma0
        F1 = (z+nu)*np.eye(4)-rec[1]['h']-Sigma1
        G0, G1 = np.linalg.solve(F0, np.eye(4)), np.linalg.solve(F1, np.eye(4))
        Vbare = E1.conj().T@bare_gradient@E0+1j*nu*np.eye(4)
        deltaSigma = triangles[probe]+endpoint_left[probe]+endpoint_right[probe]
        vertex = Vbare+deltaC+deltaDW+deltaSigma
        response_terms = {name: G1@value@G0 for name, value in dict(bare=Vbare, bare_C_exchange=deltaC, full_DW_phase=deltaDW,
                         bare_line_triangle=triangles[probe], ordered_J_left=endpoint_left[probe], ordered_J_right=endpoint_right[probe]).items()}
        response = sum(response_terms.values())
        expected = 1j*(G0-G1)
        controls = dict(WRONG_omit_all_ordered_J_and_DW_contacts=maxabs(G1@(Vbare+deltaC+triangles[probe])@G0-expected),
                        WRONG_omit_opposite_occupied_unoccupied_triangle=maxabs(G1@(vertex-crossings[probe])@G0-expected),
                        WRONG_naive_same_shift_double_denominator=maxabs(G1@(Vbare+deltaC+deltaDW+endpoint_left[probe]+endpoint_right[probe]+naive[probe])@G0-expected),
                        WRONG_bare_vertex_only=maxabs(G1@Vbare@G0-expected),
                        WRONG_omit_DW_phase_contact=maxabs(G1@(vertex-deltaDW)@G0-expected),
                        WRONG_omit_C_exchange_projector=maxabs(G1@(vertex-deltaC)@G0-expected))
        responses.append(dict(z_in=z, z_out=z+nu, genuine_external_nu=nu, external_q=q, external_physical_q=q/native['a'],
                              bare_charge_current_puregauge_insertion=Vbare, actual_bare_C_exchange_projector_insertion=deltaC,
                              actual_FULL_DW_phase_contact=deltaDW, actual_triangle=triangles[probe], actual_opposite_occupation_triangle=crossings[probe],
                              actual_left_ordered_J_contact=endpoint_left[probe], actual_right_ordered_J_contact=endpoint_right[probe],
                              actual_loopSigma_variation_from_diagrams=deltaSigma, actual_dressed_puregauge_contraction=vertex/1j,
                              actual_raw_Green_response_terms=response_terms, actual_raw_Green_response=response, actual_raw_G0=G0, actual_raw_G1=G1,
                              loopSigma_gauge_covariance_error=maxabs(deltaSigma-1j*(Sigma0-Sigma1)),
                              dressed_inverseGreen_difference_CONSUMER_ONLY_error=maxabs(vertex-1j*(F1-F0)),
                              raw_Green_response_puregauge_error=maxabs(response-expected), counterfactual_consumer_defects=controls))
    neutral = neutral_kernel_closure(supplier, f, gradient)
    return dict(actual_p0=p0, actual_p1=p1, actual_q=q, actual_nu=[.17, .61], gauge_convention='alpha_x(t)=alpha exp(i q.x-i nu.t), U=exp(i alpha_x); deltaTheta=D.T alpha, deltaA=deltaTheta/a, deltaPhi=-dot(alpha)=i nu alpha; delta h=JTheta[D.T alpha]+i nu rho',
                physical_external_gradient=gradient, physical_external_scalar_coefficients=[1j*nu*f for nu in (.17, .61)],
                bare_full256_gradient_Ward_error=barecheck, internal_bare_charge_current_Ward_max=max(internal_bare_errors),
                ordered_J_contact_path_max=max(contact_errors), equal_time_projector_density_variation_max=maxabs(np.diag(deltaP)),
                FULL_DW_third_phase_operator_error=maxabs(phase_derivative(native, profile, f, weights)-1j*(rho@DWfull-DWfull@rho)),
                all_qgroups=qrows, ALL_qgroups=36, ALL_A_coordinates=129, ALL_phi_coordinates=63, ALL_positive_boson_modes=3865,
                actual_spectral_triangle_band_mode_pairs=pairs, actual_opposite_occupation_band_mode_pairs=opposite_pairs,
                actual_internal_bands='ALL four spin columns: two occupied negative plus two unoccupied positive, with both boson orientations in crossing triangles',
                frozen7b_consumer_checks=frozen_checks, actual_responses=responses, neutral_puregauge_kernel_closure=neutral,
                actual_constructed_raw_bath=[{key: r[key] for key in ('p', 'bare', 'C', 'DW', 'h', 'B', 'd')} for r in rec],
                spectral_triangle_derivation=spectral_triangle.__doc__)


def run(path: Path) -> dict[str, Any]:
    started = time.monotonic()
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != CHARGED_RECEIPT_SHA:
        raise RuntimeError('Frozen7b actual charged receipt required')
    charged_source = charged.SOURCES[str(Path(charged.__file__).resolve())]
    if hashlib.sha256(charged_source).hexdigest() != CHARGED_SOURCE_SHA:
        raise RuntimeError('Frozen3a9 actual charged source required')
    photon_raw = charged.RECEIPT.read_bytes()
    if hashlib.sha256(photon_raw).hexdigest() != charged.RECEIPT_SHA:
        raise RuntimeError('Frozen actual photon receipt required')
    frozen = json.loads(raw)
    supplier = charged.load_supplier()
    captures = {str(Path(__file__).resolve()): OWN, **charged.SOURCES}
    mean_path = Path(supplier.s.primitives.MEAN)
    mean_raw = mean_path.read_bytes()
    descriptor = charged.packed(dict(plan_id='issue-222-induced-photon-delta13-continuation-20261008', governing_execution_revision='d86132736ac1f3d1592b01da60e6e981c5005500265f0d012e979610896318c7',
        source_hashes_at_IMPORT={name: hashlib.sha256(data).hexdigest() for name, data in captures.items()},
        charged_source_sha256=CHARGED_SOURCE_SHA, charged_receipt=dict(path=str(path), sha256=CHARGED_RECEIPT_SHA),
        photon_source_sha256=charged.PHOTON_SHA, photon_receipt=dict(path=str(charged.RECEIPT), sha256=charged.RECEIPT_SHA),
        actual_charged_input_descriptor=frozen['input_descriptor'], mean_input=dict(path=str(mean_path), sha256=hashlib.sha256(mean_raw).hexdigest(), bytes_utf8=mean_raw.decode()),
        probes=PROBES, momenta_dimensionless=[[np.pi/4]*3, [3*np.pi/4, np.pi/4, np.pi/4]],
        external_gauge='exp(i q.x-i nu.t), q=pi/2 x; actual charge/current diagrams not inverseG-defined',
        original_Lambda_UNCHANGED='2^8192', original_chi_UNCHANGED='2^-4096', original_clock_UNCHANGED='4096',
        binding_phase='after actual frozen supplier import, before physical calculation'))
    input_sha = hashlib.sha256(json.dumps(descriptor, sort_keys=True).encode()).hexdigest()
    result = construct(supplier, frozen)
    supplier.s.assert_frozen()
    responses = result['actual_responses']
    neutral = result['neutral_puregauge_kernel_closure']
    errors = dict(bare=result['bare_full256_gradient_Ward_error'], internal_bare=result['internal_bare_charge_current_Ward_max'], contacts=result['ordered_J_contact_path_max'],
                  DW=result['FULL_DW_third_phase_operator_error'], frozen7b=max(max(r['frozen7b_Sigma_error'], r['frozen7b_raw_G_error']) for r in result['frozen7b_consumer_checks']),
                  loopSigma=max(r['loopSigma_gauge_covariance_error'] for r in responses), dressed_vertex=max(r['dressed_inverseGreen_difference_CONSUMER_ONLY_error'] for r in responses),
                  raw_Green=max(r['raw_Green_response_puregauge_error'] for r in responses),
                  neutral_bubble=neutral['bubble_variation_max'], neutral_PH_contact=neutral['PH_contact_variation_max'],
                  neutral_literal_contacts=neutral['literal_contact_operator_max'])
    acceptance = max(errors.values()) < 1e-8 and all(min(r['counterfactual_consumer_defects'].values()) > 1e-9 for r in responses)
    return dict(status='actual_finite_nu_AND_nonzero_q_puregauge_diagram_response' if acceptance else 'physical_diagram_failure_preserved', acceptance=acceptance,
                input_descriptor=descriptor, input_sha256=input_sha, errors=errors, actual_finite_frequency_nonzero_momentum_Ward=result,
                source_at_import={Path(name).name: dict(path=name, sha256=hashlib.sha256(data).hexdigest(), bytes_utf8=data.decode()) for name, data in captures.items()},
                source_unchanged_since_import={Path(name).name: Path(name).read_bytes() == data for name, data in captures.items()},
                input_unchanged=path.read_bytes() == raw and charged.RECEIPT.read_bytes() == photon_raw and mean_path.read_bytes() == mean_raw,
                runtime=dict(executable=sys.executable, python=sys.version, numpy=np.__version__), elapsed_seconds=time.monotonic()-started,
                scope=['Same frozen7b positive Gaussian one-loop/GW law, native N4 AP, held Phi, all1+63 filled bands, all36 real photon/phi pole groups including q0 holonomy; full realPeierls ordered paths and instantaneous bareCAR C are retained.',
                       'Finite external nu=.17/.61 AND q=pi/2 x. Pure gauge longitudinal contraction and raw two-frequency Green response only; no general transverse vertex, anomaly, renormalized eR, full selfconsistent conserving microscopic QED, fullPhi/Noether/original SOURCE or vacuum claim.',
                       'Large actual photon phase fluctuations are unchanged; exact gauge covariance of this declared Gaussian diagram law is not a small-coupling truncation or full interacting particle proof. Quantum law and parameters remain declared inputs.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--charged', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = run(args.charged)
    args.output.write_text(json.dumps(charged.packed(report), indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(status=report['status'], input_sha256=report['input_sha256'], errors=report['errors'],
                         counterfactuals=[r['counterfactual_consumer_defects'] for r in report['actual_finite_frequency_nonzero_momentum_Ward']['actual_responses']], elapsed_seconds=report['elapsed_seconds']), indent=2), flush=True)
    if not report['acceptance'] or not all(report['source_unchanged_since_import'].values()) or not report['input_unchanged']:
        raise ArithmeticError('Actual finite-frequency nonzero-q diagram acceptance failed; receipt preserved')
