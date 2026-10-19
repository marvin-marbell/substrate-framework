"""Full-Phi same-H charged decrement against the SAME neutral-circuited source.

This new scientist CLI imports (but never runs or changes) the actual fullGauss
source/profile supplier. Physical arithmetic is rational. Finite signed-CAR
controls are separate diagnostic fields, not a Gaussian integral or all-N proof.
Run through physical_charged_descent_execute_20261011.py for an actual natural
exit and exclusive immutable before/after byte archives. No ground/phase claim.
"""
from __future__ import annotations

import argparse
from fractions import Fraction as Q
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time
import zipfile
from typing import Any

import numpy as np
from scipy.linalg import expm
from scipy.sparse import csr_matrix, diags, eye, kron
from scipy.sparse.linalg import expm_multiply

import physical_singlet_circuit_source_20261011 as source

ROOT = Path(__file__).resolve().parent
SOURCE_SHA = 'dea3025ae8509cd6669ac26df9fcb1a5d7b416aee0642778d86f2be7cb8aca03'
G0 = Q(2**32)
N0 = Q(2**10)
MATTER0 = Q(2**20)
M2, M4, M6 = Q(120), Q(120*122), Q(120*122*124)


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def exclusive_json(path: Path, value: Any) -> None:
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def immutable_archive(path: Path, files: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, 'x', zipfile.ZIP_DEFLATED) as archive:
        for name, raw in files.items():
            archive.writestr(name, raw)
    path.chmod(0o444)


def actual_profiles(inputs: dict[str, bytes]) -> tuple[Any, Any, Any, Any]:
    primitive = source.primitive
    frame, scale = primitive.frozen.exact_frame(inputs[primitive.INPUTS[0]])
    cfg = json.loads(inputs[primitive.INPUTS[1]])['input_descriptor']['original_micro_config_UNCHANGED']
    assert cfg['a'] == 1/64 and cfg['auxiliary_count'] == 63
    profiles = primitive.frozen.profiles(cfg, frame, scale)
    band = primitive.frozen.RotationBand(a=cfg['a'], m=cfg['heavy_m'], r=cfg['heavy_r'], v=cfg['heavy_v'], improved=False)
    onsite_n, paths_n = primitive.frozen.scalar_paths(band)
    cp = next(p['coefficients'] for p in profiles[0]['paths'] if p['axis'] == 0 and p['length'] == 1)
    # Every role's selected hop is checked, not extrapolated from two copies.
    assert sum(p['count'] for p in profiles) == 64
    assert [(p['role'], p['count']) for p in profiles] == [('active', 1), ('auxiliary', 63)]
    for p in profiles:
        assert abs(p['onsite']/scale) < 64 and len(p['paths']) == 6
        for path in p['paths']:
            assert all(abs(Q(r)) < 64 and abs(Q(i)) < 64 for r, i in path['coefficients'])
        assert next(x['coefficients'] for x in p['paths'] if x['axis'] == 0 and x['length'] == 1) == cp
    assert not band.improved and len(paths_n) == 3
    nc = next(c for disp, c in paths_n if disp[0] == 1)
    assert sum(Q(float(z.real))**2+Q(float(z.imag))**2 for z in nc) < 2**8
    charged = np.array([complex(float(Q(r)), float(Q(i))) for r, i in cp])
    return profiles, scale, charged, nc


def rational_increment(profiles: Any, scale: Q, charged: Any, request: dict[str, Any]) -> dict[str, Any]:
    eta = Q(1, 2**request['eta_exponent'])
    sqrteta = Q(1, 2**(request['eta_exponent']//2))
    assert request['eta_exponent'] % 2 == 0
    chi = Q(1, 2**4096)
    epsilon_n = Q(1, 2**request['neutral_epsilon_exponent'])
    kappa = Q(2**8192)*epsilon_n
    tau = Q(request['tau'])
    assert tau < 0 and kappa*sqrteta == Q(1, 2**20)
    literal = next(p['coefficients'] for p in profiles[0]['paths'] if p['axis'] == 0 and p['length'] == 1)
    csum = sum(Q(r)**2+Q(i)**2 for r, i in literal)
    # One copy: E||Hhop psi||² = (1/2) E||T||HS².
    # 64 orthogonal individual-copy defects; V/2 disjoint matching edges.
    w_per_edge = 120*csum*eta
    w_per_site = w_per_edge/2
    leading_matter = 4*tau*w_per_site
    leading_full_e = 256*tau*tau*w_per_site
    prices = {
        'matter_all_onsites_and_matching_hop_remainder': Q(2, 3)*abs(tau)**3*MATTER0*G0**3*M4*eta**2,
        'fullE64_finite_jet_remainder': 32*(2*tau**4*G0**4*M4*eta**2 + abs(tau)**5*G0**5*M6*eta**2*sqrteta + tau**6*G0**6*M6*eta**3/4),
        'Phi_charged_derivative': 240*chi*tau*tau*G0**2,
        'Phi_neutral_charged_Berry_cross': 480*chi*kappa*kappa*tau*tau*N0**2*G0**2*M2*eta,
        'colour_charged_derivative': Q(9, 8)*chi*tau*tau*G0**2*M2*eta,
        'colour_neutral_charged_Berry_cross': Q(9, 4)*chi*kappa*kappa*tau*tau*N0**2*G0**2*M4*eta**2,
    }
    leading = leading_matter+leading_full_e
    increment_upper = leading+sum(prices.values())
    rows = []
    for n in (4, 6):
        v = n**3
        match = [(x,y,z) for x in range(n) for y in range(n) for z in range(n) if (x+y+z)%2 == 0]
        target = [((x+1)%n,y,z) for x,y,z in match]
        gate = {point: i for i, pair in enumerate(zip(match, target)) for point in pair}
        assert len(gate) == v and len(match) == v//2
        matching_directed = 0
        nonmatching_directed = 0
        for p in profiles:
            for path in p['paths']:
                axis, length = path['axis'], path['length']
                for point in gate:
                    end = list(point); end[axis] = (end[axis]+length)%n
                    if gate[point] == gate[(end[0],end[1],end[2])]:
                        assert axis == 0 and length == 1 and point in match
                        matching_directed += p['count']
                    else:
                        nonmatching_directed += p['count']
        assert matching_directed == 64*v//2
        assert matching_directed+nonmatching_directed == 64*6*v
        rows.append({'N':n,'V':v,'matching_edges':v//2,'charged_copies':64,'particles_per_charged_copy':2*v,'neutral_particles':2*v,'matching_directed_paths':matching_directed,'zero_by_gate_number_nonmatching_one_two_paths':nonmatching_directed,'increment_upper_total_exact':str(v*increment_upper)})
    if tau == -Q(1, 128):
        assert leading == -w_per_site/64
    assert leading < 0 and increment_upper < leading/2
    return {
        'kind':'EXACT_RATIONAL_ALL_EVEN_N_GE4_SAME_H_FINITE_STEP_INCREMENT_BOUND',
        'eta_exact':str(eta),'epsilon_neutral_exact':str(epsilon_n),'kappa_exact':str(kappa),'neutral_effective_delta_exact':str(kappa*sqrteta),'tau_exact':str(tau),
        'microscopic_parameters_unchanged':{'Lambda':'2^8192','chi':'2^-4096','F':'2^10240','a':'1/64','law':'NEW Jordan primitive','charged_roles':'actual 1+63'},
        'charged_selected_coefficient_square_sum_exact':str(csum),
        'W_per_matching_edge_exact':str(w_per_edge),'W_per_site_exact':str(w_per_site),
        'leading_matter_increment_per_site_exact':str(leading_matter),'leading_fullE64_increment_per_site_exact':str(leading_full_e),'leading_net_increment_per_site_exact':str(leading),
        'positive_remainder_prices_per_site_exact':{k:str(v) for k,v in prices.items()},
        'price_over_gain_exact':{k:str(v/(-leading)) for k,v in prices.items()},
        'increment_upper_per_site_exact':str(increment_upper),'increment_upper_over_gain_exact':str(increment_upper/(-leading)),
        'strict_half_gain_bound':'Delta<H> < (4tau+256tau²)W_per_site/2 < 0; total bound V times per-site bound',
        'all_even_N_ge4':True,'finite_step_not_initial_derivative':True,'finite_inventory_cases':rows,
        'charged_onsites':[{'role':p['role'],'count':p['count'],'coefficient_exact':str(p['onsite']/scale)} for p in profiles],
        'bosonic_potentials_increment_exact':'0 by pointwise matter norm, not by freezing Phi',
        'neutral_H_increment_exact':'0 because charged gate acts on a distinct CAR species tensor factor and preserves charged pointnorm',
        'nonmatching_one_two_hopping_increment_exact':'0 by per-species canonical number of each disjoint matching pair, not by a C0 estimate',
        'HC_policy':'HC is the Gauss-longitudinal part of fullE64 and is NOT added again',
        'numerical_full_field_integral':False,
    }


def absolute_source_budget(neutral_coeff: Any, request: dict[str, Any], increment: dict[str, Any]) -> dict[str, Any]:
    eta=Q(1,2**request['eta_exponent'])
    sqrteta=Q(1,2**(request['eta_exponent']//2))
    chi,lam,bigf,t=Q(1,2**4096),Q(2**8192),Q(2**10240),Q(2**7168)
    delta=Q(1,2**20)
    kappa=lam/Q(2**request['neutral_epsilon_exponent'])
    mn2=Q(15,4)*sum(Q(float(z.real))**2+Q(float(z.imag))**2 for z in neutral_coeff)
    mn4=Q(2**24)
    assert 3*mn2**2 < mn4
    neutral_upper=lam*sqrteta*(-2*delta*mn2+Q(4,3)*delta**3*mn4)
    neutral_lower=-2*delta*mn2*lam*sqrteta
    prices={
        'Phi_scalar_Gaussian':60*chi/eta,
        'Jordan_exact':3000*lam*eta**2,
        'alignment_exact':Q(9,2)*bigf*eta,
        'colour_scalar_upper':Q(27,4)*chi*t,
        'colour_potential_upper':Q(7,5)*bigf/t*(450+48*7168+24),
        'neutral_Phi_circuit_upper':240*chi*kappa**2*N0**2,
        'neutral_colour_circuit_upper':Q(9,8)*chi*kappa**2*N0**2*M2*eta,
    }
    baseline_upper=neutral_upper+sum(prices.values())
    after_upper=baseline_upper+Q(increment['increment_upper_per_site_exact'])
    # Alignment is an EXACT expectation on the unchanged Gaussian; the positive
    # lower is not obtained by misusing the variational upper as a lower.
    after_lower=prices['alignment_exact']+neutral_lower-Q(2**24)*sqrteta
    if request['eta_exponent'] == 4000:
        assert after_lower > Q(2**6240)
        regime='HIGH_ABSOLUTE_ENERGY_REJECTED_BY_GENUINE_POSITIVE_LOWER; charged decrement alone is not a repair of accepted source'
    elif request['eta_exponent'] == 4200:
        assert baseline_upper < -Q(2**6070) and after_upper < baseline_upper
        regime='NEGATIVE_FULLGAUSS_BASELINE_AND_STRICTLY_LOWER_CHARGED_SOURCE'
    else:
        regime='exact lower and upper recorded without a ground claim'
    return {
        'kind':'ABSOLUTE_FULL_H_SOURCE_ENERGY_BUDGET_SEPARATE_FROM_SAME_H_INCREMENT',
        'regime':regime,
        'neutral_M2_exact':str(mn2),'neutral_M4_upper_exact':str(mn4),
        'neutral_energy_upper_per_site_exact':str(neutral_upper),'neutral_energy_lower_per_site_exact':str(neutral_lower),
        'positive_baseline_prices_per_site_exact':{k:str(v) for k,v in prices.items()},
        'neutral_circuited_baseline_energy_upper_per_site_exact':str(baseline_upper),
        'charged_after_energy_upper_per_site_exact':str(after_upper),
        'charged_after_energy_lower_per_site_exact':str(after_lower),
        'per_site_and_total':'Each displayed energy is per site; total finite-N bound is V times it.',
        'lower_derivation':'E alignment=(9/2)F eta exactly: 3V edges,4 matrices/edge and independent zero-mean endpoints give180etaV before /40; neutral sin(x)<=x gives lower -2delta M2 Lambda sqrteta, all other bosonic and kinetic terms nonnegative; charged matter absolute<=2^24 sqrteta/site including all actual onsites and primitive paths.',
        'baseline_charged_matter':'exact0 by one-RDM I4/2, traceless Phi, fixed vertex occupation and initial product sites',
        'neutral_upper_derivation':'sin(x)>=x-x³/6, Wick M4<2^24; unmatched neutral paths0 and neutral onsite odd under global centered inversion',
        'campaign_complete':False,
    }

def finite_controls(charged: Any, neutral_coeff: Any, onsite: list[float]) -> tuple[dict[str, Any], dict[str, Any]]:
    rng = np.random.default_rng(222101101)
    basis = source.su4_basis()
    fields = np.einsum('xma,aij->xmij', .17*rng.normal(size=(2,4,15)), basis)
    u = expm(1j*np.einsum('a,aij->ij', .31*rng.normal(size=15), basis))
    masks = [sum(1<<j for j in c) for c in itertools.combinations(range(8),4)]
    d = len(masks); ident = eye(d, format='csr')
    def both(ff: Any, uu: Any) -> Any:
        h = source.lift(source.hop(ff,uu,charged),masks)
        return kron(h,ident,format='csr')+kron(ident,h,format='csr')
    h = both(fields,u)
    single_rho = np.array([(m&15).bit_count()-2 for m in masks])
    rho = (single_rho[:,None]+single_rho[None,:]).ravel()
    density = diags(2*rho*rho)
    g = density@h-h@density
    index = {m:i for i,m in enumerate(masks)}
    v = np.zeros(d*d,complex)
    for left in itertools.combinations(range(4),2):
        lc=[i for i in range(4) if i not in left]
        for right in itertools.combinations(range(4),2):
            rc=[i for i in range(4) if i not in right]
            ma=sum(1<<i for i in left)+sum(1<<(i+4) for i in right)
            mb=sum(1<<i for i in lc)+sum(1<<(i+4) for i in rc)
            v[index[ma]*d+index[mb]]=source.epsilon_sign(list(left)+lc)*source.epsilon_sign(list(right)+rc)/6
    tau = -1/128
    cv = expm_multiply(tau*g,v)
    wrong_d = kron(diags(2*single_rho**2),ident)+kron(ident,diags(2*single_rho**2))
    wg=wrong_d@h-h@wrong_d
    wrong_v=expm_multiply(tau*wg,v)
    wrong_gap=float(np.linalg.norm(cv-wrong_v))
    # FullE=-i dtheta, full electric energy 64||dtheta psi||².
    hc=source.hop(fields,u,charged)
    t=hc[:4,4:]
    hp=source.lift(np.block([[np.zeros((4,4)),1j*t],[-1j*t.conj().T,np.zeros((4,4))]]),masks)
    hp=kron(hp,ident)+kron(ident,hp)
    gp=density@hp-hp@density
    ev,ej1,ej2=source.jet_exponential(tau*g,tau*gp,-tau*g,v)
    matter=float(np.vdot(cv,h@cv).real)
    full_e=64*float(np.vdot(ej1,ej1).real)
    e_action=-64*ej2
    # Actual unequal charged role onsite remains present, not averaged away.
    ao=source.lift(np.block([[fields[0,0],np.zeros((4,4))],[np.zeros((4,4)),fields[1,0]]]),masks)
    actual_o=onsite[0]*kron(ao,ident)+onsite[1]*kron(ident,ao)
    onsite_increment=float((np.vdot(cv,actual_o@cv)-np.vdot(v,actual_o@v)).real)
    df=np.zeros_like(fields);df[0,1]=basis[0]
    dh=both(df,u); dg=density@dh-dh@density
    fv,fj1,fj2=source.jet_exponential(tau*g,tau*dg,csr_matrix(g.shape),v)
    # Neutral full256 CAR, with the same field coordinate and colour U.
    allm=list(range(256)); vn=np.eye(256,dtype=complex)[:,15]
    pn=np.diag([1.]*4+[0.]*4)
    nh=-source.hop(fields,u,neutral_coeff); ndh=-source.hop(df,u,neutral_coeff)
    ns=source.lift(nh@pn-pn@nh,allm)
    nds=source.lift(ndh@pn-pn@ndh,allm)
    kappa=.025
    nv,nj1,nj2=source.jet_exponential(-kappa*ns,-kappa*nds,csr_matrix((256,256)),vn)
    neutral_initial_berry=complex(np.vdot(vn,nds@vn))
    neutral_berry=complex(np.vdot(nv,nj1));charged_berry=complex(np.vdot(fv,fj1))
    berry_cross=2*float((neutral_berry.conjugate()*charged_berry).real)
    # The complete second derivative has 2 nc' tensor c'; norm is exact by tensor factorization.
    mixed_second_norm=2*float(np.linalg.norm(nj1)*np.linalg.norm(fj1))
    # Nonzero neutral Berry at finite step rules out a neutrality/C0 argument.
    assert neutral_initial_berry == 0 and abs(neutral_berry) > 1e-9
    assert mixed_second_norm > 1e-5
    assert wrong_gap > 1e-5 and np.linalg.norm((g-wg).data) > 1
    assert full_e > 1e-4 and np.linalg.norm(e_action) > 1e-3
    assert np.linalg.norm(g@v-2*h@v) < 1e-11
    assert abs(np.vdot(cv,cv)-1) < 1e-12 and np.linalg.norm(ev-cv) < 1e-12
    # Beta is electric/Bloch momentum, not a theta-coordinate bump.
    boost = 1/8
    shifted_action = e_action-128j*boost*ej1+64*boost*boost*ev
    omitted_twist_error = float(np.linalg.norm(shifted_action-e_action))
    even_boost_energy = 32*(float(np.vdot(-1j*ej1+boost*ev,-1j*ej1+boost*ev).real)+float(np.vdot(-1j*ej1-boost*ev,-1j*ej1-boost*ev).real))
    even_boost_defect = abs(even_boost_energy-full_e-64*boost*boost*float(np.vdot(ev,ev).real))
    assert omitted_twist_error > 1e-2 and even_boost_defect < 1e-10
    packet_width = Q(1,64)
    coordinate_packet_lower = 16/packet_width**2
    momentum_packet_upper = 64*packet_width**2
    assert coordinate_packet_lower > momentum_packet_upper
    # SU4 and U1 covariance on the actual CAR source, not the two-copy inequality.
    gr=expm(.11j*basis[2]);gc=expm(-.13j*basis[7])
    tf=np.array([[gr@fields[0,m]@gr.conj().T for m in range(4)],[gc@fields[1,m]@gc.conj().T for m in range(4)]])
    bg=np.block([[gr,np.zeros((4,4))],[np.zeros((4,4)),gc]])
    covariance_error=float(np.linalg.norm(source.hop(tf,gr@u@gc.conj().T,charged)-bg@hc@bg.conj().T))
    gauss_error=float(np.linalg.norm(ej1-1j*rho*ev))
    assert covariance_error < 1e-12 and gauss_error < 1e-10
    control={
        'kind':'NEW_SIGNED_CAR_FINITE_DIAGNOSTICS_NOT_PHYSICAL_FULL_GAUSSIAN_ENERGY',
        'charged_dimension':d*d,'neutral_fullCAR_dimension':256,'tau':tau,'neutral_diagnostic_kappa':kappa,
        'wrong_percopy_density_operator_difference':float(np.linalg.norm((g-wg).data)),
        'wrong_percopy_density_finite_source_gap':wrong_gap,
        'correct_Gpsi_equals_2Hpsi_residual':float(np.linalg.norm(g@v-2*h@v)),
        'charged_hop_increment':matter,'actual_unequal_role_onsite_increment':onsite_increment,
        'fullE64_energy':full_e,'wrong_omittedE_energy_error':full_e,'fullE64_complete_action_norm':float(np.linalg.norm(e_action)),
        'finite_matter_plus_fullE_increment':matter+onsite_increment+full_e,
        'neutral_initial_Berry_exact':str(neutral_initial_berry),'neutral_finite_Berry':[neutral_berry.real,neutral_berry.imag],
        'charged_finite_Berry':[charged_berry.real,charged_berry.imag],'actual_Phi_Berry_energy_cross':berry_cross,
        'wrong_omitted_neutral_charged_second_Phi_jet_action_norm':mixed_second_norm,
        'actual_cross_never_inferred_zero_from_C0':True,'Berry_cross_bound_used_in_allN_even_if_this_control_small':True,
        'SU4_covariance_error':covariance_error,'U1_Gauss_first_jet_error':gauss_error,
        'source_norm':float(np.vdot(cv,cv).real),'complete_jets':'block Duhamel; no numerical variance or derivative extrapolation',
        'Bloch_beta_is_momentum_not_theta':True,
        'even_Bloch_boost_energy_defect':even_boost_defect,
        'omitted_Bloch_twist_complete_action_error':omitted_twist_error,
        'coordinate_circle_packet_energy_lower_exact':str(coordinate_packet_lower),
        'Bloch_circle_packet_energy_upper_exact':str(momentum_packet_upper),
        'coordinate_wrongcase_not_the_actual_N4_cycle_metric':True,
    }
    arrays={'fields':fields,'colour':u,'charged_state':cv,'wrong_density_state':wrong_v,'charged_theta_first_jet':ej1,'charged_fullE_action':e_action,'charged_Phi_first_jet':fj1,'charged_Phi_second_jet':fj2,'neutral_state':nv,'neutral_Phi_first_jet':nj1,'neutral_Phi_second_jet':nj2}
    return control,arrays


def derivation(request: dict[str, Any]) -> dict[str, Any]:
    return {
        'source_and_baseline':[
            'Reuse physical_singlet_circuit_source_20261011.construction: full realCart Gaussian variance eta, tr(TaTb)=delta_ab/2, gauge-invariant colour Haar density Z(t)^-1 exp(-t Ccolour), t=2^7168; neutral parity canonical2V, 32 local epsilon wedge2 pairs, actual active1+aux63, inert Ref4, native compact beta0. Only eta, epsilon_n and charged step change. Baseline contains this SAME new neutral-circuited source with charged step exactly0.',
            f"eta=2^-{request['eta_exponent']}; epsilon_n=2^-{request['neutral_epsilon_exponent']}; kappa=Lambda epsilon_n=2^{8192-request['neutral_epsilon_exponent']}; kappa sqrteta=2^-20. Charged Uc=product_matching exp(tau [rho_r²+rho_c²,Hhop]); tau={request['tau']}. rho_y=sum64(n_s,y-2), never a per-copy rho² sum. All local occupancies and the entire exact finite-CAR exponential are retained.",
            'Pointwise normalization makes both sources have the identical Gaussian/Haar normalization, not two independent fields or independent seas. Endpoint adjoint fields and ordered one/two-link Wilson products are the unchanged actual primitive law. The archived frozen frame supplies scalar primitive coefficients only; Phi is integrated over all60V real coordinates.',
        ],
        'norm_caps_and_Wick':[
            'Let R=|q_r,q_c| in120 realCart coordinates. For every U, ||T||HS <= sqrt(sum|c_mu|²) R/2. Finite-CAR ||dGamma(offdiag T)||<=4||T||HS; sum64 gives <=128 sqrt(sum|c|²)R <2^16 R. Density ||rho_r²+rho_c²||<=32768, so commutator <=65536 times hopping norm, hence ||G||<=2^32 R. Each Cart derivative cap2^32, and each colour/theta first derivative cap2^32 R. Neutral [h,P] cap2^10 R and derivative2^10 (colour derivative2^10 R). Actual checked coefficients and onsite absolute values<64 imply complete local matching charged matter cap2^20 R.',
            'Conditional on arbitrary interacting colour U, endpoint q are independent centered isotropic real Gaussians. E||T||HS²=(15/4)eta sum|c|² independent of U. The conditional real covariance satisfies Wick E||T||HS4=(tr Sigma)²+2tr Sigma²<=3(tr Sigma)². R moments are exactly E R²=120eta, E R4=14640eta², E R6=1815360eta³. Global inversion q->-q preserves both Gaussian and Haar weights, with no unjustified independence of Wilson products.',
            'Each charged copy has local wedge2 one-RDM I4/2 but is not a Wick state. For Hhop on one copy, opposite endpoints are independent and its two charge orientations have norms summing (1/2)||T||HS². Distinct-copy hopping defects have different individual-copy occupation labels, hence are orthogonal even within the coherent epsilon pair. Therefore W_edge=64*(1/2)*(15/4)eta sum|c|²=120eta sum|c|². No two-copy extrapolation supplies an inequality.',
        ],
        'finite_matter':[
            'Uc conserves n_s,r+n_s,c separately for every species in every disjoint matching pair. Initial value4 is exact. Every nonmatching one/two primitive hop changes this label in two different gates, giving pointwise expectation0 at ALL steps. For evenN>=4 only selected axis0,length1 edges and their adjoints lie within one gate; all primitive paths are inventoried. Onsites are NOT canceled: active and aux roles differ and onsite action is nonzero.',
            'On the initial local charged source rho_r=rho_c=0; one Hhop defect has density2, so Gpsi=2Hhop psi. Complete charged onsite O preserves local occupations, so <[O,G]>=0 by orthogonality, not equality of masses. Thus the whole charged matter derivative is4W per matching edge.',
            'For M=complete charged onsite plus matching Hhop, e^-tauG M e^tauG has Taylor expansion through second order. Its second coefficient has field degree3 and integrates exactly0 by global centered Gaussian inversion. Taylor with unitary conjugation bounds the remaining finite difference by |tau|³||ad_G³ M||/6 <=(4/3)|tau|³ MATTER0 G0³ R4. Per site divide by2 matching edges. This is a finite-step estimate on unbounded real fields, not an initial-derivative inference.',
        ],
        'finite_fullE':[
            'Baseline neutral circuit has no theta, and beta0 has E=0. FullE coefficient64 is used once; longitudinal HC is not a second Hamiltonian addend. Optional original-cover beta is cycle Bloch MOMENTUM, with a common normalized even compact-smooth packet h(beta), independent of q,U and the circuits. FullE_beta=FullE0+B beta; integrated twist/source cross is0 because mean beta=0, not because the charged electric mean vanishes. The identical packet energy cancels the SAME-H increment; this is not beta minimization.',
            'Jtheta e^tauG psi=tau a+tau² b+r, a=Gtheta psi, b=(G Gtheta+Gtheta G)psi/2, ||r||<=|tau|³||G||²||Gtheta||/2. This follows by Taylor of the unitary Duhamel integrand e^(tau(1-s)G) Gtheta e^(tausG), whose second tau derivative norm is at most ||G||²||Gtheta||. It is valid for every field, without expanding e^(||G||).',
            '||a||²=4||Htheta psi||²=4W. Theta multiplies each opposite charge orientation by +/-i; individual occupation defects retain the same norms. The cubic cross <a,b> integrates0 by Gaussian inversion. With g=G0 R and gtheta=G0 R, the remaining squared-jet price is 2tau4 g²gtheta²+|tau|5 g³gtheta²+tau6 g4gtheta²/4. E R5<=M6 eta^(5/2) by Cauchy and sqrt(M4 M6)<=M6. Multiply64 and divide2 edges/site to obtain the displayed exact price.',
            'Leading charged matter4tau W plus fullE256tau²W is (4tau+256tau²)W, equal to -W/64 ONLY at tau=-1/128. The negative low-energy-width branch uses a separately priced smaller finite step; the coefficient is NOT copied from the first lead. Only after paying every remainder and field kinetic cross does the exact rational inequality certify strict decrease.',
        ],
        'field_cross_derivation':[
            'For U=exp(tA) and initial psi, <psi,A_d psi>=0 when A_d changes the original local occupation labels. This holds for neutral parity and charged density gate, independently of C0 or neutrality. Udag dU=t integral_0^1 exp(-stA) A_d exp(stA) ds. Subtract the zero first term and use ||[A_d,A]||<=2||A_d||||A||: |<Upsi,dUpsi>|<=t²||A||||A_d||. No small pointwise |q| assumption is made.',
            'Thus neutral Cart Berry is bounded by kappa² N0² R, charged Cart Berry by tau² G0² R. Their energy cross is <=2kappa²tau²N0²G0²R² per coordinate; this is quadratic in EACH circuit strength, not the invalid kappa*tau bound and not a claimed zero cross. Sum120 coordinates per matching pair, coefficient4chi, V/2 pairs; E R² supplies the displayed Phi price. Charged derivative self price is4chi*(120/2)tau²G0².',
            'For colour, replace both derivative caps by the same constant times R. Berry cross uses E R4, charged self uses E R². Sum15 left SU4 generators per matching edge with Hamiltonian coefficient3chi/20. Real positive colour Gibbs amplitude and real Gaussian amplitude have exactly zero scalar-gradient real cross with unitary fibre derivatives. Unchanged neutral self derivative cancels in comparison because the charged tensor factor has norm1.',
            'All scalar multiplication potentials (including Jordan and alignment) cancel by pointnorm. Neutral H cancels by separate species/commutation. Charged onsites do not commute with the charged gate and were paid above. No coupling, Cov, bareB2, root, source, or Phi-freeze adoption occurs.',
        ],
        'fullGauss_and_domain':[
            'The initial wedge2 epsilon pairs and neutral wedge4/0 are local SU4 singlets; charged initial rho0 and gauge-invariant scalar field amplitudes satisfy Gauss. T transforms gr T gc^dag, ordered two-link products use the actual endpoints, and G is gauge covariant with U1 charge phases. Multiplying by both exact covariant circuits preserves full SU4 and U1 Gauss, canonical particles and the same normalization. Z_N(t)>0 is the exact compact Haar integral, never assigned a numerical value.',
            'For every coordinate D, complete first jet of e^A is integral exp((1-s)A)(DA)exp(sA)ds. Complete second jet is the D²A insertion plus the two ordered DA insertions. Cart D²A=0; colour/theta D²A is retained. Product jets include D²(Un tensor Uc)=Unsecond tensor Uc+2 Unfirst tensor Ucfirst+Un tensor Ucsecond. Within-copy/coherent and all64 cross terms are already in the full generator, not a tangent PH truncation.',
            'H(f g U psi0)=-4chi sum_Cart[(D²f)gU+2(Df)gDU+fgD²U]psi0 -(3chi/20)sum_colour[f(X²g)U+2f(Xg)XU+fgX²U]psi0 -64sum_theta fg Dtheta²U psi0 + [VJordan+F Walign+F Wcolour+Hcharged-Lambda Hneutral]f g U psi0. This is the complete analytic D(H) action, including the neutral/charged second jets. Native compact beta0 is smooth; the optional original-cover Bloch momentum packet is C-infinity compactly supported, while its coordinate envelope is real Schwartz with all derivatives square integrable.',
            'For Cart: ||DU||<=kappa N0+|tau|G0 and ||D²U||<=(kappa N0+|tau|G0)². For colour/theta: first cap L R, second cap L R+L²R² with L=kappa N0+|tau|G0 (neutral theta part exactly0). For Gaussian df/f=-q/(2eta), d²f/f=q²/(4eta²)-1/(2eta). Colour ratios are bounded on finite compact configuration space by24tV and48tV+576t²V², as in the reused source construction.',
            'All complete-action terms are bounded by finite polynomials in R_all in60V Gaussian coordinates: matter degree1, Jordan degree4, alignment degree2, compact colour potential degree0, circuit jets at most degree2. E R_all^(2k)=eta^k product_j(60V+2j). The triangle sum of their L2 norms is finite for each finiteN, proving D(H). The energy INCREMENT bound is N-uniform per site; the complete-action norm and its square are NOT a per-site variance or an N-uniform total variance. No numerical full-H residual/variance is computed.',
            'For the optional original cover, choose integral cycle B with DB=0, d=2V+1 and G=B^T B. The common even normalized momentum packet h_w(beta) has |beta_j|<w=1/(64dTrG). In its direct-integral fibres fullE_beta=fullE0+Bbeta: retain the native second action, the 128 sum_e (Bbeta)_e fullE0_e first-action term, and64|Bbeta|^2 U. No theta derivative acts on h_w(beta). Its electric price <=1/(64dTrG) is TOTAL; centered beta cancels the integrated twist/source cross and preserves the exact charged increment. In physical transverse x=Q^T theta coordinates the equivalent real Schwartz envelope is det(G)^(1/4) inverseFourier[h_w](B^TQ x); its complete coordinate Leibniz terms are an alternative representation, not derivatives of the momentum bump. Finite bounded-beta shifts and polynomial field jets establish D(H) only. Unequal or noncentered packets require their own cross price; no original-cover L2 vacuum or numerical variance is inferred.',
        ],
        'debts':['Parent independent scientific review and final selected checker remain necessary.','Strict finite SAME-H variational gain is not a stationary/fullPhi or lattice ground, local SO5 cancellation, phase/stiffness/IR atom, photon pole, commoncone, originalSOURCE admission or campaign closure.','Exact integral-defined Haar normalization is not numerically evaluated.','No full-H numerical variance; complete analytic action/domain proof only.'],
        'campaign_complete':False,
    }


def main() -> None:
    # Exact dyadic physical budgets exceed Python's default decimal I/O cap.
    sys.set_int_max_str_digits(0)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    args=parser.parse_args()
    request=json.loads(args.request.read_text())
    prefix=request['prefix']
    assert prefix.startswith('physical_charged_descent_')
    started=time.monotonic()
    inputs={name:(ROOT/name).read_bytes() for name in source.PINS}
    assert all(sha(inputs[n]) == pin for n,pin in source.PINS.items())
    assert sha((ROOT/Path(source.__file__).name).read_bytes()) == SOURCE_SHA
    captured={Path(__file__).name:Path(__file__).read_bytes(),Path(source.__file__).name:Path(source.__file__).read_bytes(),args.request.name:args.request.read_bytes()}
    captured.update(source.primitive.snapshot())
    for module in tuple(sys.modules.values()):
        path=getattr(module,'__file__',None)
        if path and Path(path).parent == ROOT and str(path).endswith('.py'):
            captured[Path(path).name]=Path(path).read_bytes()
    captured.update(inputs)
    captured['physical_charged_descent_execute_20261011.py']=(ROOT/'physical_charged_descent_execute_20261011.py').read_bytes()
    immutable_archive(ROOT/(prefix+'preexecution.zip'),captured)
    profiles,scale,charged,nc=actual_profiles(inputs)
    increment=rational_increment(profiles,scale,charged,request)
    measured,arrays=finite_controls(charged,nc,[float(p['onsite']/scale) for p in profiles])
    proof=derivation(request)
    proof.update(request=request,rational_increment=increment,absolute_source_budget=absolute_source_budget(nc,request,increment),observed_controls=measured)
    npz=ROOT/(prefix+'controls.npz')
    with npz.open('xb') as stream:
        np.savez_compressed(stream,**arrays)
    output=ROOT/(prefix+'proof.json')
    exclusive_json(output,proof)
    unchanged=all((Path('/'+n.removeprefix('runtime_imports/')) if n.startswith('runtime_imports/') else ROOT/n).read_bytes() == raw for n,raw in captured.items())
    assert unchanged
    receipt={'kind':'NEW_FULLGAUSS_SAME_H_CHARGED_FINITE_STEP_CLI','command':[sys.executable,Path(__file__).name,'--request',args.request.name],'source_input_hashes':{n:sha(raw) for n,raw in captured.items()},'source_inputs_unchanged':unchanged,'outputs':{p.name:sha(p.read_bytes()) for p in (npz,output)},'runtime_seconds':time.monotonic()-started,'python':sys.version,'stage':'strict rational finite-step charged gain; independent parent review pending','actual_exit':'external runner only','no_fullH_numeric_variance':True}
    exclusive_json(ROOT/(prefix+'receipt.json'),receipt)
    print(json.dumps({'strict_half_gain_bound':increment['strict_half_gain_bound'],'price_over_gain_log2_upper':{k:Q(v).numerator.bit_length()-Q(v).denominator.bit_length()+1 for k,v in increment['price_over_gain_exact'].items()},'finite_controls':measured,'source_inputs_unchanged':unchanged}))


if __name__ == '__main__':
    main()
