#!/usr/bin/env python3
"""Exact canonical charge law, primitive arcs and cold-clock reference bridge.

The free positive-T comparison is NOT fullGauss K**Nt. Every native canonical
occupation is retained. Ground vectors are comparison objects with an explicit
thermal remainder, never an interacting Slater replacement.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
ROOT=Path(__file__).resolve().parent
OWN_BYTES=Path(__file__).read_bytes()
LOADED=sys._getframe().f_code
MATCH=LOADED==compile(OWN_BYTES,LOADED.co_filename,'exec',dont_inherit=True,optimize=sys.flags.optimize)
if not MATCH:raise ArithmeticError('Loaded clock bridge differs from captured bytes')
import numpy as np
from scipy.linalg import eigh
import covariant_relative_ring_boundary as ring
PLAN_REVISION='7650a3636f3d39a74243a986afb1683be8fae895c3f5afb77720b05330363a51'


def chiral_coordinates(gamma: np.ndarray,sites: int) -> tuple[np.ndarray,np.ndarray]:
    values,frame=eigh(gamma[0]);plus=frame[:,values>0]
    gamma5=gamma[0]@gamma[1]@gamma[2]@gamma[3];minus=gamma5@plus
    local=np.column_stack((plus,minus))
    localJ=plus@minus.conj().T-minus@plus.conj().T
    return np.kron(np.eye(sites),local),np.kron(np.eye(sites),localJ)


def joint_law(matrix: np.ndarray,states: np.ndarray,rho: np.ndarray) -> tuple[dict[str,Any],np.ndarray]:
    """Actually evaluate EVERY exterior minor, including thermal occupations."""
    count=len(states);dimension=rho.shape[1]
    total=0.;first=np.zeros(2*dimension);second=np.zeros((2*dimension,2*dimension))
    fourth=0.;zero=0.;occupation_probability=np.zeros(count)
    for start in range(0,count,16):
        rows=states[start:start+16]
        minors=matrix[rows[:,None,:,None],states[None,:,None,:]]
        weights=abs(np.linalg.det(minors))**2
        total+=float(np.sum(weights));occupation_probability[start:start+len(rows)]=weights.sum(axis=1)
        r=rho[start:start+len(rows)]
        left=weights.sum(axis=1);right=weights.sum(axis=0)
        first[:dimension]+=r.T@left;first[dimension:]+=rho.T@right
        second[:dimension,:dimension]+=r.T@(left[:,None]*r)
        second[dimension:,dimension:]+=rho.T@(right[:,None]*rho)
        cross=r.T@weights@rho
        second[:dimension,dimension:]+=cross;second[dimension:,:dimension]+=cross.T
        absolute=np.sum(abs(r),axis=1)[:,None]+np.sum(abs(rho),axis=1)[None,:]
        fourth+=float(np.sum(weights*absolute**4))
        zero+=float(np.sum(weights[np.all(r==0,axis=1)][:,np.all(rho==0,axis=1)]))
    mean=first/total;covariance=second/total-np.outer(mean,mean)
    return dict(full_canonical_components=count,actual_joint_normalization=total,mean=mean,covariance=covariance,
        covariance_spectrum=eigh(covariance,eigvals_only=True),joint_charge_zero_probability=zero/total,
        actual_joint_L1_fourth_moment=fourth/total),occupation_probability/total


def primitive_arcs(A: np.ndarray,states: np.ndarray,normalizer: float,sites: int,kappa: float) -> dict[str,Any]:
    minus=np.array([4*x+s for x in range(sites) for s in (2,3)])
    plus=np.array([4*x+s for x in range(sites) for s in (0,1)])
    D=A[np.ix_(minus,minus)];R=A[np.ix_(plus,minus)]@np.linalg.inv(D)
    vacuum=abs(np.linalg.det(D))**2/normalizer
    arcs=[]
    for x in range(sites):
        y=(x+1)%sites
        block=R[2*x:2*x+2,2*y:2*y+2]
        probability=vacuum*float(np.sum(abs(block)**2))
        arcs.append(dict(edge=[x,y],actual_total_two_spin_primitive_probability=probability,
            actual_probability_over_kappa4=probability/kappa**4))
    return dict(chiral_reference_vacuum_joint_probability=vacuum,arcs=arcs,
        minimum_two_spin_arc_over_kappa4=min(a['actual_probability_over_kappa4'] for a in arcs),
        meaning='Exact ALL-canonical denominator. Arc is summed over BOTH physical spin components, not a selected invariant spin plane.')


def case(gamma: np.ndarray,sites: int,kappa: float,Nt: int) -> dict[str,Any]:
    _,_,physical=ring.ring_half(gamma,sites,kappa,.31,0.)
    O,J=chiral_coordinates(gamma,sites)
    dual=float(np.linalg.norm(J@physical@J.conj().T-np.linalg.inv(physical),2)/np.linalg.norm(np.linalg.inv(physical),2))
    polynomial_dual=float(np.linalg.norm(physical@J@physical-J,2)/(1+np.linalg.norm(physical,2)**2))
    ring.require(polynomial_dual<1e-12,'Actual physical T must satisfy the stable chirality polynomial identity T J T=J; no generic Cartesian assumption')
    T=O.conj().T@physical@O;values,frame=eigh(T);number=2*sites
    logs=np.log(values);vacuum_log=float(np.sum(logs[-number:]))
    eigen_states=np.array(ring.supplier.occupations(4*sites,number))
    weights=np.exp(Nt*(logs[eigen_states].sum(axis=1)-vacuum_log))
    vacuum_index=np.all(eigen_states==np.arange(4*sites-number,4*sites),axis=1)
    thermal=float(np.sum(weights[~vacuum_index]))
    # Common NEUTRAL vacuum scalar arithmetic; no charged own-max reference.
    exponent=(Nt/2)*(logs-vacuum_log/number)
    A=(frame*np.exp(exponent))@frame.conj().T
    rho=np.array([[sum(4*x<=i<4*x+4 for i in state)-2 for x in range(1,sites)] for state in eigen_states])
    law,probability=joint_law(A,eigen_states,rho)
    spectral_normalizer=float(np.sum(weights))
    ring.require(abs(law['actual_joint_normalization']-spectral_normalizer)<1e-8*spectral_normalizer,
        'Dense ALL-exterior norm must equal exact fixed-N spectral coefficient')
    ground=frame[:,-number:];P=ground@ground.conj().T
    ground_cov=np.zeros((sites-1,sites-1))
    for x in range(1,sites):
        for y in range(1,sites):
            px=slice(4*x,4*x+4);py=slice(4*y,4*y+4)
            ground_cov[x-1,y-1]=(sum(float(np.sum(abs(P[px,4*z:4*z+4])**2)) for z in range(sites) if z!=x)
                if x==y else -float(np.sum(abs(P[px,py])**2)))
    upper=np.array([4*x+s for x in range(sites) for s in (0,1)])
    lower=np.array([4*x+s for x in range(sites) for s in (2,3)])
    X=ground[upper]@np.linalg.inv(ground[lower])
    graph_hermitian=float(np.linalg.norm(X-X.conj().T,2));ring.require(graph_hermitian<1e-10,'Actual inverse-dual ground graph must be Hermitian')
    rng=np.random.default_rng(222);angles=rng.uniform(-math.pi,math.pi,(32,sites));angles[:,0]=0
    characters=[];negative_characters=[]
    for chi in angles:
        g=np.repeat(np.exp(1j*chi),4)
        value=np.linalg.det(ground.conj().T@(g[:,None]*ground))*np.exp(-2j*sum(chi))
        opposite=np.linalg.det(ground.conj().T@(g.conj()[:,None]*ground))*np.exp(2j*sum(chi))
        characters.append(value);negative_characters.append(opposite)
    characters=np.array(characters);negative_characters=np.array(negative_characters)
    ring.require(bool(np.min(np.real(characters))>0 and np.max(abs(np.imag(characters)))<1e-9),
        'Genuine mean-Clifford reference character must be positive in its actual background phase')
    arcs=primitive_arcs(A,eigen_states,law['actual_joint_normalization'],sites,kappa)
    return dict(sites=sites,kappa=kappa,whole_unprojected_comparison_clocks=Nt,
        actual_stable_chirality_polynomial_residual=polynomial_dual,
        actual_chirality_duality_relative_error=dual,actual_graph_Hermitian_error=graph_hermitian,
        actual_graph_operator_norm=float(np.linalg.norm(X,2)),graph_norm_over_kappa2=float(np.linalg.norm(X,2)/kappa**2),
        full_ALL_occupation_law=law,actual_canonical_thermal_excitation_relative_trace=thermal,
        reference_ground_covariance=ground_cov,primitive_arcs=arcs,
        positive_ground_character_minimum=float(np.min(np.real(characters))),
        positive_ground_character_evenness_error=float(np.max(abs(characters-negative_characters))),
        actual_joint_variance_over_kappa4=float(law['covariance'][0,0]/kappa**4),
        actual_joint_cross_over_kappa4=float(law['covariance'][0,sites-1]/kappa**4),
        reference_ground_variance_over_kappa4=float(ground_cov[0,0]/kappa**4),
        actual_fourth_over_V2_kappa4=float(law['actual_joint_L1_fourth_moment']/(sites**2*kappa**4)),
        scope='Actual unprojected positive-T canonical comparison ONLY. Nt counts T powers; this is not swapping Gamma(T)^Nt past fullPhi/P_Gauss in the REAL K^Nt.')


def mixed_ground_boundary(gamma: np.ndarray,sites: int) -> dict[str,Any]:
    """Falsify mixed-character evenness using two actual Root central fields."""
    O,_=chiral_coordinates(gamma,sites);number=2*sites
    upper=np.array([4*x+s for x in range(sites) for s in (0,1)])
    lower=np.array([4*x+s for x in range(sites) for s in (2,3)])
    frames=[]
    for theta in (.31,1.07):
        _,_,physical=ring.ring_half(gamma,sites,.04,theta,0.)
        _,vectors=eigh(O.conj().T@physical@O);ground=vectors[:,-number:]
        X=ground[upper]@np.linalg.inv(ground[lower])
        values,basis=eigh(np.eye(number)+X@X)
        normal=(basis/np.sqrt(values))@basis.conj().T
        frame=np.zeros((4*sites,number),complex);frame[upper]=X@normal;frame[lower]=normal
        ring.require(bool(np.linalg.norm(frame.conj().T@frame-np.eye(number))<1e-10),
            'Mixed comparison uses normalized ACTUAL ground frames in their true background phase')
        frames.append(frame)
    rng=np.random.default_rng(22213);angles=rng.uniform(-math.pi,math.pi,(64,sites));angles[:,0]=0
    chars=[];opposites=[]
    for chi in angles:
        g=np.repeat(np.exp(1j*chi),4)
        chars.append(np.linalg.det(frames[1].conj().T@(g[:,None]*frames[0]))*np.exp(-2j*sum(chi)))
        opposites.append(np.linalg.det(frames[1].conj().T@(g.conj()[:,None]*frames[0]))*np.exp(2j*sum(chi)))
    chars=np.asarray(chars,dtype=np.complex128);opposites=np.asarray(opposites,dtype=np.complex128)
    parity=float(np.max(abs(chars-opposites)))
    ring.require(bool(np.min(np.real(chars))>0 and np.max(abs(np.imag(chars)))<1e-10),
        'Inverse-dual mixed occupied minor is pointwise positive in the true canonical phase')
    ring.require(parity>1e-10,'Different central fields must falsify inherited single-field evenness')
    return dict(sites=sites,kappa=.04,actual_field_thetas=[.31,1.07],samples=len(angles),
        positive_mixed_character_minimum=float(np.min(np.real(chars))),
        mixed_character_imaginary_error=float(np.max(abs(np.imag(chars)))),
        mixed_character_evenness_defect=parity,
        scope='Actual mean-Clifford central-U1 comparison. Mixed positivity is real; mixed evenness is FALSE. No full-Phi or source admission.')


def mixed_relative_fourier_bound(V: Any,Nt: Any,k: Any,N: Any) -> dict[str,Any]:
    """Conditional scalar reference proof, not an operator/conditioned-Gram gate."""
    mp=ring.supplier.mp;beta=16*N*k**4;delta=(N*k**4)**(-mp.mpf(39)/100)
    chart=mp.mpf(10)**-50;boundaries=Nt+3;dimension=boundaries*(V-1)
    nonlinear=beta*V*(4*chart+2*delta)**4/8
    static=mp.mpf(2)**20*N*V*(64*k)**6/(1-64*k)
    core=mp.mpf(2)**48*N*k**4*V**2*delta**4
    phase=(4*V+1)**2*delta**2/2
    entropy=dimension*mp.log(24*mp.pi*beta)/2
    tail_exponent=beta*delta**2/(4*mp.pi**2)
    log_tail=mp.log(boundaries)+entropy-tail_exponent
    combined=4*boundaries*(nonlinear+static+core+phase)
    ring.require(log_tail<0,'Pay the actual whole-clock Gaussian normalization before claiming a relative tail')
    ring.require(4*V*(4*chart)**2<mp.mpf('1e-60'),
        'Large-angle tail uses the paid gauge-transported temporal increment, not small absolute torons')
    return dict(beta_physical_exact='16*Naux*kappa4=1e150',coexact_chart_radius='1e-50',
        gauge_transported_temporal_increment_bound='4e-50',absolute_flat_holonomies_restricted=False,
        clock_boundaries=int(boundaries),Gaussian_dimension_exact=str(int(dimension)),
        single_boundary_mixed_cosine_remainder_log10=float(mp.log10(nonlinear)),
        single_boundary_static_kappa6_remainder_log10=float(mp.log10(static)),
        all_boundaries_relative_scalar_charge_price_log10=float(mp.log10(combined)),
        actual_global_Gaussian_normalization_entropy=str(entropy),
        actual_global_normalized_tail_natural_log_upper=str(log_tail),
        primitive_graph_proof='Write a=B/(2κ), ||B-I||<=6κ, F=Fdag, ||F||<=3, t=||a^-1||<=3κ. The graph Riccati map is X=-a^-1 F a^-1+a^-1 X a^-1-X F a^-1 F a^-1+X F a^-1 X a^-1. At ||X||<=32κ², κ<1/128, it maps the ball into itself and has Lipschitz bound90κ²+1728κ4<1/2. Its leading remainder from -4κ²F is bounded by199κ³+2880κ4+27648κ6<=2048κ³/(1-64κ). This proves the constant used in the primitive-edge bound, rather than postulating a Riesz coefficient.',
        actual_mixed_kappa4_identity='For U(X)=[X;I](1+X²)^-1/2, log det(U(Xprime)dag g U(X)) after background2/site is -||Xprime-gXgdag||F²/2+O(Vκ6) on the EVEN bipartite macro box. BOTH physical spins give log occupied_minor=-16κ4 Σ_e[1-cos(Aprime_e-A_e-gradχ_e)]+O(Vκ6). This is the original positive mixed kernel, not a substituted Gaussian model.',
        mixed_parity_repair='Require actual Coulomb representatives D(Aprime-A)=0 and gauge-transported total temporal increments |Aprime-A|infinity<=4chart, exactly the frozen Source2a scalar block. Retain arbitrary absolute flat holonomies and nonstationary harmonic increments. Then Σ(Aprime-A)gradχ=0 exactly. Gaussian χ core is even; the ACTUAL mixed cosine need not be. Its entire core discrepancy, including odd terms, is paid by beta*V*(4chart+2delta)^4/8. No stationary toron or individual-character positivity is inferred.',
        normalized_tail_proof='For 3D periodic L>=6, shell bound26r² and sin(pi r/L)>=2r/L give Green00<=26/32+pi²/(48L)<1 and resistance<=4. Thus E=Σ|exp(iχx)-exp(iχy)|²>=||χ||infinity²/pi² in rooted principal angles. For ||χ||infinity<=pi/4, Coulomb cancellation leaves |sin dA-dA|<=|dA|³/6 and |sin gradχ-gradχ|<=|gradχ|³/6; the paid transported increment makes their energy error <E/4. For larger angles, Cauchy gives the cross error <=4chart sqrt(6V E)<<E/4. The mixed cosine action relative to χ0 is >=E/4 outside the core. Pay normalized Haar volume and Jlower=(24pi beta)^(-dimension/2) for ALL Nt+3 boundaries before dividing. This is an explicit relative tail, not absolute/Z or RN-to-Loewner.',
        scalar_charge_ratio='On the Gaussian core, odd charge phase cancels in the EVEN REFERENCE, while loss<=((4V+1)delta)^2/2 for every active canonical occupation and all Ref4 states. Exact mixed oddness and source-preserved phases are handled by the uniform nonlinear price, not deleted. C0(Aprime,A), detB, normal heat and actual coexact/harmonic field density remain in BOTH numerators and denominators.',
        CONDITIONAL=True,
        required_join='Use the ACTUAL frozen Source2a Coulomb/periodic scalar reference with its long-increment price, preserving exact integer large-gauge seams and source-string/frame transport. This is not an absolute-toron chart or state gate. Physical Dminus/fullPhi/Gauss statistical transport must still supply these actual pairs in the all-species conditioned kernel. Scalar per-charge Fourier estimates are not operator Loewner bounds or normalized photon/charged receiver-Gram admission. Even-Nt mixed-SVD/Schatten and odd-source-clock reference trace ratios remain separate.',
        full_physical_conditioned_source_admitted=False)

def macro_bound() -> dict[str,Any]:
    mp=ring.supplier.mp
    with mp.workdps(110):
        V=mp.mpf(2)**72;Nt=mp.mpf(2)**20;k=mp.mpf(10)**-400;N=mp.mpf(10**1750//16)
        graph_mean=2*V*(32*k*k)**2
        r=16*k*k;a=(2*V)**2*r**Nt
        # Complete canonical counts sum_j binomial(2V,j)^2 r^(j Nt).
        # For a<1/64 its relative excitation sum and j^4 moment <=2a.
        thermal_log=mp.log(2)+mp.log(a)
        char_bridge_log=mp.log(16*N)+thermal_log/2
        cold_fourth=mp.mpf(2)**30*(V*k**4+V**2*r**Nt)
        warm_fourth=mp.mpf(2)**30*(V*k**4+V**2*r**2)
        beta=N*k**4;delta=beta**(-mp.mpf(39)/100)
        core=mp.mpf(2)**48*beta*V**2*delta**4
        charge=4*V+1;phase=charge**2*delta**2/2
        phi_parameters=ring.supplier.Parameters(kappa_power=400)
        phi=ring.supplier.macro_phi_budget(phi_parameters)
        mixed=ring.supplier.frozen.auxiliary_canonical_kernel_budget(phi_parameters)
        scalar=ring.supplier.frozen.compact_photon_kernel_budget(phi_parameters)
        integer_gauss=ring.supplier.frozen.canonical_gaussian_Gauss_budget(phi_parameters)
        return dict(actual_parameter_epoch='Existing Root-compatible Sourceκ400β150, unchanged',kappa='1e-400',
            Naux_exact=str(10**1750//16),beta_induced='1e150',actual_Naux_kappa4_log10=float(mp.log10(beta)),
            NOT_applicable_to_Naux_1e1000=True,wrong_epoch_Naux1e1000_kappa4_log10=-600,
            volume_exact=str(int(V)),Nt_exact=str(int(Nt)),
            actual_reference_ground_plus_occupation_mean_log10=float(mp.log10(graph_mean)),
            canonical_thermal_excitation_relative_trace_log_upper=str(thermal_log),
            ALL_Naux_relative_charge_character_cold_bridge_log10_upper=float(char_bridge_log/mp.log(10)),
            cold_joint_fourth_moment_upper_log10=float(mp.log10(cold_fourth)),
            two_clock_joint_fourth_moment_upper_log10=float(mp.log10(warm_fourth)),
            primitive_edge_probability_lower_over_kappa4=2**-16,
            primitive_edge_proof='In gamma5-aligned physical chirality, Fxy=(plusdag gamma_j minus) Uxy/2 has BOTH singular values1/2 for every simple scalar-U1 nearest edge, L>=3. Analytic Riesz graph has X=-4κ²F+O(κ³), with operator remainder<=2048κ³/(1-64κ); atκ400 this gives ||Xxy||F>=κ². Filled-minus overlap probability is det(1+X²)^-1>=exp(-2048Vκ4). The cold amplitude Γ(T^(Nt/2)) differs from its ACTUAL leading eigenvector amplitude in Hilbert-Schmidt norm<=sqrt(2a); here sqrt(2a)<<κ², so summing BOTH spin arcs and dividing by1+2a gives probability>=κ4/2^16. No primitive-arc claim is transferred to L2 coincident forward/back edges or generic noncentral Cartesian links.',
            exact_positive_ground_character_floor='>=1-4*(2V)*(32kappa²)^2 >1/2',
            core_fourth_remainder_log10=float(mp.log10(core)),all_active_Ref_charge_shift_loss_log10=float(mp.log10(phase)),
            mixed_SVD_all_occupation_actual_Nt_budget=mixed,physical_Phi_projective_actual_budget=phi,
            actual_frozen_Source2a_positive_scalar_reference_budget=scalar,
            actual_frozen_Source2a_integer_Gauss_source_string_budget=integer_gauss,
            conditional_mixed_positive_boundary_relative_Haar=mixed_relative_fourier_bound(V,Nt,k,N),
            proof='J T Jdag=T^-1 at the actual frozen mean Clifford/central-U1 reference. X=Xdag and ||X||<=32κ². Ground character det(1+gdag X g X)/det(1+X²) is strictly positive and even, with true background2/site. Γ_N(T^(Nt/2)) retains every occupation. Its excited Hilbert-Schmidt ratio is sum_j binom(2V,j)^2 r^(jNt)<=2a. Joint-law TV from the ground amplitude reference is<=2sqrt(2a)+4a; the positive reference character floor gives an explicitly RELATIVE all-Naux power error, not absolute/Z. For ALL thermal eigenoccupation Slaters, plus-chirality number mean<=j_e+μgraph; its fourth moment is the Poisson-binomial polynomial μ+7μ²+6μ³+μ⁴. Summing the exact canonical excitation counts yields the quoted full joint fourth bound. These are reference decomposition identities, never an interacting Slater assertion.',
            actual_boundary_bridge='The real K contains normal heat and a Haar projection EACH step. Only the frozen mixed-SVD uniform pair-kernel plus its canonical Schatten_Nt remainder may transfer this reference lemma to that boundary. Phi projective ledger is separately executable; actual active/source/globalGauss conditioned Gram join is NOT proved by the scalar reference.',
            full_physical_conditioned_source_admitted=False)


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--sites',type=int,default=3);parser.add_argument('--kappas',type=float,nargs='+',default=[.04,.004,.0004])
    parser.add_argument('--clocks',type=int,nargs='+',default=[2,4,8]);args=parser.parse_args();start=time.monotonic()
    gamma,_=ring.supplier.phi_frame(ring.supplier.Parameters())
    rows=[case(gamma,args.sites,k,Nt) for k in args.kappas for Nt in args.clocks]
    mixed_parity=mixed_ground_boundary(gamma,args.sites)
    captured=dict(ring.supplier.frozen.SNAPSHOT)
    captured['covariant_canonical_gauss_boundary_kernel']=ring.supplier.OWN_BYTES
    captured['covariant_relative_ring_boundary']=ring.OWN_BYTES;captured[Path(__file__).stem]=OWN_BYTES
    hashes={name:hashlib.sha256(data).hexdigest() for name,data in captured.items()}
    unchanged=all((ROOT/(name+'.py')).read_bytes()==data for name,data in captured.items())
    ring.require(unchanged,'Clock scientific sources changed during calculation')
    result=dict(plan=ring.PLAN,governing_revision=PLAN_REVISION,parameters=vars(args)|{'output':str(args.output)},
        native_rows=rows,mixed_ground_character_negative_control=mixed_parity,
        actual_macro_reference_bound=macro_bound(),compiled_source_hashes=hashes,
        captured_source_bytes={name:data.decode() for name,data in captured.items()},compiled_source_unchanged=unchanged,
        main_code_matches_captured_source=MATCH,elapsed_seconds=time.monotonic()-start,
        macro_source_admitted=False,meaning='Exact all-occupation free canonical clock reference and primitive-edge proof control; not full physical K power or source admission.')
    args.output.write_text(json.dumps(ring.supplier.portable(result),indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(args.output),source_sha256=hashes[Path(__file__).stem],elapsed_seconds=result['elapsed_seconds'],
        rows=[dict(k=r['kappa'],Nt=r['whole_unprojected_comparison_clocks'],variance_over_kappa4=r['actual_joint_variance_over_kappa4'],
            cross_over_kappa4=r['actual_joint_cross_over_kappa4'],arc_min_over_kappa4=r['primitive_arcs']['minimum_two_spin_arc_over_kappa4']) for r in rows])))
    print('Natural exit: all canonical occupations retained; κ4 charge arcs and explicit cold-clock reference bridge, no macro source admission')

if __name__=='__main__':main()
