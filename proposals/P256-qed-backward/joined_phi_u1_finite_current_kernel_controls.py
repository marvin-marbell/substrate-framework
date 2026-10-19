"""Independent finite-CAR/Hermite same-law current consumer, not all129 proof.

This module does not import the accepted current helper or forward its values.
Its two-site, two-orbital, two-Ref model is explicitly a finite control, not the
physical fullN4/fourRef input. Both charged profiles share the oscillator.
"""
from __future__ import annotations

from itertools import combinations
import math
from typing import Any

import numpy as np
from scipy.linalg import eigh
from scipy.special import eval_hermitenorm, roots_hermitenorm

G = 64.
S = 2.


def sector(n: int, count: int) -> list[tuple[int, ...]]:
    return list(combinations(range(n), count))


def lift(a: np.ndarray, count: int) -> np.ndarray:
    states = sector(len(a), count)
    index = {s:i for i,s in enumerate(states)}
    out = np.zeros((len(states),len(states)),complex)
    for col, state in enumerate(states):
        for j in state:
            reduced = tuple(k for k in state if k != j)
            sj = (-1)**state.index(j)
            for i in range(len(a)):
                if i in reduced:
                    continue
                target = tuple(sorted((*reduced,i)))
                si = (-1)**target.index(i)
                out[index[target],col] += si*sj*a[i,j]
    return out


def slater(w: np.ndarray) -> np.ndarray:
    return np.array([np.linalg.det(w[list(s)]) for s in sector(len(w),w.shape[1])])


def create(i: int, count: int) -> np.ndarray:
    low, high = sector(4,count), sector(4,count+1)
    idx = {s:k for k,s in enumerate(high)}
    out = np.zeros((len(high),len(low)),complex)
    for j,state in enumerate(low):
        if i not in state:
            target = tuple(sorted((*state,i)))
            out[idx[target],j] = (-1)**target.index(i)
    return out


def microscopic(omit_right_endpoint: bool = False) -> tuple[list[list[tuple[float,np.ndarray]]], list[np.ndarray]]:
    # Literal both-endpoint ordered scalar-U1 factors. The second path has
    # AP seam -1; its charge is one, not a length-dependent charge.
    fx = np.array([[.7,.19+.12j],[.19-.12j,-.5]])
    fy = np.array([[.4,-.11+.08j],[-.11-.08j,-.8]])
    onsite = [np.diag([-.13,.21,.17,-.09]).astype(complex),
              np.diag([-.2,.11,.16,-.04]).astype(complex)]
    terms: list[list[tuple[float,np.ndarray]]] = []
    for scale in (1.,.83):
        row = []
        for q, seam, bare in ((1.,1.,np.array([[.31,.14j],[-.07j,.23]])),
                              (2.,-1.,np.array([[.075,.021],[-.017,.064]]))):
            block = seam*scale*(fx@bare+(0 if omit_right_endpoint else bare@fy))/2
            a = np.zeros((4,4),complex)
            a[:2,2:] = block
            row.extend(((q,a),(-q,a.conj().T)))
        terms.append(row)
    return terms, onsite


def build(cutoff: int, source: bool, omission: str = '') -> dict[str, Any]:
    terms, onsite = microscopic()
    na = 3 if source else 2
    nb = 2
    aa, bb = sector(4,na), sector(4,nb)
    da, db = len(aa), len(bb)
    fdim = da*db
    eye = np.eye(fdim)
    # Full density in this finite geometry, active AND auxiliary; source has
    # its Ref hole at site zero. Lplus is the exact two-site pseudoinverse.
    lp = np.array([[.25,-.25],[-.25,.25]])
    density = []
    for a in aa:
        for b in bb:
            rho = np.array([sum(i<2 for i in a)+sum(i<2 for i in b)-2,
                            sum(i>=2 for i in a)+sum(i>=2 for i in b)-2],float)
            if source and omission != 'Ref_density':
                rho[0] -= 1
            if omission == 'aux_density':
                rho -= np.array([sum(i<2 for i in b)-1,sum(i>=2 for i in b)-1])
            density.append(G*rho@lp@rho)
    hc = np.diag(density)
    z,weight = roots_hermitenorm(max(100,4*cutoff))
    basis = np.stack([eval_hermitenorm(n,z)/math.sqrt(math.factorial(n)) for n in range(cutoff)])
    weight = weight/math.sqrt(2*math.pi)
    def character(q: float) -> np.ndarray:
        return (basis*(weight*np.exp(1j*q*math.sqrt(S)*z)))@basis.T
    p2 = np.diag((2*np.arange(cutoff)+1)/(4*S))
    for n in range(cutoff-2):
        p2[n,n+2] = p2[n+2,n] = -math.sqrt((n+1)*(n+2))/(4*S)
    kinetic = np.kron(G*p2,eye)
    h = np.zeros_like(kinetic,dtype=complex)
    if omission != 'Pi':
        h += kinetic
    if omission != 'HC':
        h += np.kron(np.eye(cutoff),hc)
    j = np.zeros_like(h)
    contact = np.zeros_like(h)
    flavour_currents = []
    single = []
    for flavour,count,dimension in ((0,na,da),(1,nb,db)):
        os = lift(onsite[flavour],count)
        lifted = np.kron(os,np.eye(db)) if flavour == 0 else np.kron(np.eye(da),os)
        h += np.kron(np.eye(cutoff),lifted)
        flavour_j = np.zeros_like(j)
        small_h = onsite[flavour].copy()
        for q,a in terms[flavour]:
            small_h += a
            if omission == 'two_link' and abs(q) == 2:
                continue
            lifted0 = lift(a,count)
            lifted = np.kron(lifted0,np.eye(db)) if flavour == 0 else np.kron(np.eye(da),lifted0)
            em = character(q)
            h += np.kron(em,lifted)
            contribution = np.kron(-1j*q*em,lifted)
            j += contribution
            flavour_j += contribution
            contact += np.kron(-q*q*em,lifted)
        _,w = eigh(small_h)
        single.append(slater(w[:,:2]))
        flavour_currents.append(flavour_j)
    oscillator = np.zeros(cutoff,complex)
    oscillator[0] = 1
    vectors = []
    if source:
        for alpha in range(2):
            # Independent two-Ref CAR oracle: coefficient1/sqrt2 gives raw
            # norm around .5, rather than silently normalizing the source.
            vectors.append(np.kron(oscillator,np.kron(create(alpha,2)@single[0],single[1]))/math.sqrt(2))
    else:
        vectors.append(np.kron(oscillator,np.kron(single[0],single[1])))
    psi = np.stack(vectors,axis=1)
    for name,matrix in (('H',h),('J',j),('C',contact)):
        if np.linalg.norm(matrix-matrix.conj().T)>1e-9:
            raise ArithmeticError('Independent Hermiticity failed: '+name)
    return dict(H=h,J=j,C=contact,psi=psi,HC=hc,kinetic=kinetic,
                microscopic_terms=terms,cutoff=cutoff,flavour_currents=flavour_currents)


def observed(model: dict[str, Any], clocks: list[float]) -> dict[str, Any]:
    h,j,psi = model['H'],model['J'],model['psi']
    a = h@j-j@h
    b = h@a-a@h
    f = float(np.trace(psi.conj().T@(a@j-j@a)@psi).real)
    second = float((1j*np.trace(psi.conj().T@(b@j-j@b)@psi)).real)
    eig,w = eigh(h)
    xp = w.conj().T@psi
    xj = w.conj().T@(j@psi)
    mass = float(np.linalg.norm(psi)**2)
    b0 = float(np.linalg.norm(j@psi))
    bnorm = float(np.max(np.abs(eigh(b,eigvals_only=True))))
    # A real finite-matrix proof, not a norm for unbounded physical Pi.
    coefficient = bnorm*math.sqrt(mass)*b0
    rows = []
    for t in clocks:
        phase = np.exp(-1j*t*eig)
        up = w@(phase[:,None]*xp)
        uj = w@(phase[:,None]*xj)
        response = float(2*np.trace(up.conj().T@j@uj).imag)
        residual = abs(response-t*f)
        bound = t*t*coefficient
        if residual > bound+1e-10:
            raise ArithmeticError('Independent finite proven remainder was violated')
        rows.append(dict(t=t,response=response,linear=t*f,residual=residual,
                         finite_proven_remainder_upper=bound,signal_to_bound=abs(t*f)/bound,
                         wrong_sign_residual=abs(response+t*f)))
    comm_state = h@psi@psi.conj().T-psi@psi.conj().T@h
    return dict(raw_mass=mass,F=f,Rdoubleprime0=second,finite_quadratic_coefficient=coefficient,
                nonstationarity_density_commutator=float(np.linalg.norm(comm_state)),
                rows=rows)


def run() -> dict[str, Any]:
    clocks = [2.**-20,2.**-21]
    output: dict[str,Any] = {}
    base_models = {}
    for source in (False,True):
        name = 'raw_source' if source else 'background'
        low = build(12,source)
        high = build(18,source)
        base_models[name] = high
        lo = observed(low,clocks)
        hi = observed(high,clocks)
        if hi['rows'][0]['signal_to_bound'] <= 1:
            raise ArithmeticError('Finite same-H control clock was vacuous')
        hi['cutoff12_F'] = lo['F']
        hi['cutoff12_to18_F_absolute_change'] = abs(lo['F']-hi['F'])
        hi['cutoff12_to18_response_change'] = abs(lo['rows'][0]['response']-hi['rows'][0]['response'])
        hi['contact_frobenius'] = float(np.linalg.norm(high['C']))
        output[name] = hi
    defects = {}
    for omission, source in (('Pi',False),('HC',False),('aux_density',False),('Ref_density',True),('two_link',False)):
        wrong = observed(build(18,source,omission),clocks)
        correct = output['raw_source' if source else 'background']
        defect = abs(wrong['F']-correct['F'])
        defects[omission+'_F_discriminator'] = defect
        if defect < 1e-7:
            raise ArithmeticError('Independent adverse omission was blind: '+omission)
    # The source is tested RAW. A normalized-source response differs by its
    # actual small-model mass, independently determined by CAR creation.
    mass = output['raw_source']['raw_mass']
    defects['incorrect_source_normalization_F_discriminator'] = abs(output['raw_source']['F']*(1/mass-1))
    # A deliberately nonstationary SAME-H preparation rejects an odd-only
    # stationary/pole expansion. It is not substituted for actual all129.
    generic = dict(base_models['background'])
    psi = generic['psi'].copy()
    psi[36:72] = .23j*psi[:36]
    psi /= np.linalg.norm(psi)
    generic['psi'] = psi
    adverse = observed(generic,[2.**-19,2.**-20])
    if abs(adverse['Rdoubleprime0']) < 1e-7:
        raise ArithmeticError('Nonstationary second-order discriminator was blind')
    residuals = [r['residual'] for r in adverse['rows']]
    adverse['observed_half_clock_remainder_ratio'] = residuals[0]/residuals[1]
    adverse['second_order_predicted_at_first_clock'] = .5*adverse['Rdoubleprime0']*(2.**-19)**2
    wrong_cubic = adverse['finite_quadratic_coefficient']*(2.**-19)**3
    adverse['forbidden_cubic_bound'] = wrong_cubic
    adverse['forbidden_cubic_bound_violation_factor'] = residuals[0]/wrong_cubic
    if residuals[0] <= wrong_cubic:
        raise ArithmeticError('False cubic remainder was not independently rejected')
    bg = base_models['background']
    pbg = bg['psi']
    ja,jb = bg['flavour_currents']
    shared_cross = complex(np.trace(pbg.conj().T@ja@jb@pbg))
    independent_cross = complex(np.trace(pbg.conj().T@ja@pbg)*np.trace(pbg.conj().T@jb@pbg))
    defects['independent_Theta_current_second_discriminator'] = abs(shared_cross-independent_cross)
    if defects['independent_Theta_current_second_discriminator'] < 1e-8:
        raise ArithmeticError('Shared-Theta control was blind')
    # Independent microscopic controls for AP, both endpoints and exact
    # charge/length signs use literal small blocks, not forwarded jets.
    terms,_ = microscopic()
    theta = .37
    jliteral = sum(-1j*q*np.exp(1j*q*theta)*a for q,a in terms[0])
    wrong_ap = sum(-1j*q*np.exp(1j*q*theta)*(-a if abs(q)==2 else a) for q,a in terms[0])
    wrong_charge = sum(-1j*q*q*np.exp(1j*q*q*theta)*a for q,a in terms[0])
    wrong_terms,_ = microscopic(omit_right_endpoint=True)
    wrong_endpoint = sum(-1j*q*np.exp(1j*q*theta)*a for q,a in wrong_terms[0])
    defects['right_Phi_endpoint_omission_current_discriminator'] = float(np.linalg.norm(jliteral-wrong_endpoint))
    defects['AP_seam_current_discriminator'] = float(np.linalg.norm(jliteral-wrong_ap))
    defects['length_as_charge_current_discriminator'] = float(np.linalg.norm(jliteral-wrong_charge))
    # Exact neutral spectator commutation and current response cancellation.
    hn = np.array([[3.,.4],[.4,-2.]])*1e18
    spectator_j = np.kron(np.eye(2),jliteral)
    neutral_comm = np.kron(hn,np.eye(4))@spectator_j-spectator_j@np.kron(hn,np.eye(4))
    defects['neutralheavy_current_commutator_exact_float_zero'] = float(np.linalg.norm(neutral_comm))
    if defects['neutralheavy_current_commutator_exact_float_zero'] != 0:
        raise ArithmeticError('Neutral spectator did not commute')
    return dict(scope='Independent two-site/two-orbital active+aux/twoRef Hermite-Galerkin SAME Hamiltonian law; not the fullN4/fourRef global proof',
        coefficients=dict(HE=64,HC=64,density_variance=2,shared_oscillator=True),
        finite_controls=output,adverse_omissions=defects,nonstationary_order_control=adverse,
        exact_finite_proof='Finite Hermitian B=ad_H^2J has norm from complete eigenspectrum, so |R(t)-tF|<=t^2 ||B|| sqrt(rawmass)||Jpsi||. Natural actual exp(-itH) evolution is evaluated independently.',
        neutral_exact='Kronecker identity [I tensor Hneutral,Jcharged tensor I]=0; huge neutral coefficient cancels before evolution or energy pricing.',
        limitations=['Galerkin cutoff12/18 comparison is an observation, not a continuum/unboundedPi error bound','Finite control uses twoRef; actual all129 source retains all four raw accepted channels','No first-moment values copied from accepted helper or global producer'])
