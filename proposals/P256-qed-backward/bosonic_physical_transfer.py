#!/usr/bin/env python3
"""Field-Noether/current transfer with finite loop dynamics and a physical cone fork."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from fractions import Fraction
from pathlib import Path
from typing import Any, cast

if __name__ == '__main__':
    for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = '1'

import numpy as np
from scipy import sparse
from scipy.linalg import expm
from scipy.optimize import brentq
from scipy.sparse.linalg import expm_multiply

from bulk_quantum_selection import BulkQuantumLaw
from noncompact_screened_gauss_dressing import (
    frame_chart, centre_scalar, mechanical_core_noether_consumer,
    same_neutral_photon_commutator_consumer,
    orientation_core_halfspin_bound, physical_orientation_lie_majorant,
    physical_raw_noether_geometry_majorant, relative_photon_noether_graph,
)
from original_charged_principal import original_frame, portable
from rotation_band import RotationBand
from sea_selection import SELECTED, reframe
from unrestricted_field_minimum import BARE, local_potential, coordinates

ROOT = Path(__file__).resolve().parent
C = Fraction(1130202988249777, 9007199254740992)
NSTAR = 5 * 2**96
UE = Fraction.from_float(1.5429442570866347e-148) / 2**4096


def accept(error: float, label: str, tolerance: float = 3e-7) -> float:
    value = float(error)
    if not math.isfinite(value) or value > tolerance:
        raise AssertionError(f'{label}: {value} > {tolerance}')
    return value


def adverse(error: float, label: str, tolerance: float = 3e-7) -> dict[str, Any]:
    try:
        accept(error, label, tolerance)
    except AssertionError as failure:
        return dict(defect=float(error), rejected=True, assertion=str(failure))
    raise AssertionError(f'adverse control passed: {label}')


def exterior(matrix: np.ndarray, masks: tuple[int, ...]) -> np.ndarray:
    lookup = {m: k for k, m in enumerate(masks)}
    result = np.zeros((len(masks), len(masks)), complex)
    for column, mask in enumerate(masks):
        for source in range(8):
            if not mask >> source & 1:
                continue
            rest = mask ^ (1 << source)
            first = (-1)**((mask & ((1 << source)-1)).bit_count())
            for target in range(8):
                if rest >> target & 1:
                    continue
                sign = first*(-1)**((rest & ((1 << target)-1)).bit_count())
                result[lookup[rest | (1 << target)], column] += sign*matrix[target, source]
    return result


def field_generators(frame: np.ndarray, basis: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Solve full15 original mass/Gauss action on Cartesian Phi, not end spin."""
    generators = basis/math.sqrt(8)
    D = np.asarray([coordinates(1j*(t@frame-frame@t), basis).reshape(60)
                    for t in generators]).T
    mass = np.r_[np.ones(60), np.full(45, 1/.6)]
    G = np.vstack((D, np.zeros((45, 15))))
    inertia = G.T@(mass[:, None]*G)
    solved = []
    errors = []
    for j, k in ((2, 3), (3, 1), (1, 2)):
        tangent = np.zeros_like(frame)
        tangent[j] = -frame[k]
        tangent[k] = frame[j]
        y = np.r_[coordinates(tangent, basis).reshape(60), np.zeros(45)]
        coeff = np.linalg.solve(inertia, G.T@(mass*y))
        s = np.einsum('a,aij->ij', coeff, generators)
        errors.append(float(np.linalg.norm(1j*(s@frame-frame@s)-tangent)))
        solved.append(s)
    return np.asarray(solved), dict(all15=True, original_mass_field=1.,
        original_mass_link=1/.6, inertia_min=float(np.linalg.eigvalsh(inertia)[0]),
        physical_vector_errors=errors)


def native_transfer(frame: np.ndarray, law: Any) -> dict[str, Any]:
    """Finite coupled Cartesian-Phi/real-photon field slice, complete Pauli sectors.

    Two oscillator DVR nodes per field are a finite witness, not the microscopic
    ground. The original Cartesian kinetic, quartic and real quadratic curl
    coefficients are retained on this declared one-bond coordinate slice.
    """
    from bosonic_original_law_join import NativeOriginalModel
    model = NativeOriginalModel()
    masks, W = model.code_isometry(4)
    projector = W@W.conj().T
    loop = model.stabilizers[0]
    I = model.identity
    rng = np.random.default_rng(22210691)
    normal = law.fields(rng.normal(size=(law.volume, 4, 15)))[0]
    fields0 = np.asarray([frame+.002*normal, frame-.001*normal])
    spins, physical = field_generators(frame, law.basis)
    s = spins[2]
    tangent = np.zeros_like(frame); tangent[1] = -fields0[0, 2]; tangent[2] = fields0[0, 1]
    scale = .02
    theta = np.asarray([-scale, scale])
    oscillator_x = np.asarray([[0., 1.], [1., 0.]])/math.sqrt(2)
    _, node_modes = np.linalg.eigh(oscillator_x)
    # Full p^2 compression diag(1/2,3/2), NOT square of compressed p.
    p2 = node_modes.T@np.diag([.5, 1.5])@node_modes
    inertia = float(np.sum(coordinates(tangent, law.basis)**2))
    chi = .004
    field_kinetic = chi/(4*scale**2*inertia)*p2
    # Literal local real photon H=(UE/2)p^2+K (curl A)^2, curl singular value2.
    ue = .0025; photon_speed = float(C); K = photon_speed**2/(2*ue)
    omega = 2*photon_speed
    amplitude = math.sqrt(ue/omega)
    avec = np.asarray([-amplitude/math.sqrt(2), amplitude/math.sqrt(2)])
    photon_kinetic = ue/(2*amplitude**2)*p2
    photon_potential = K*4*avec**2
    onsite = (law.band.m+3*law.band.r/law.band.a)
    link = expm(.025j*law.generators[3])
    labels = np.asarray([np.eye(4), *(SELECTED[j]@SELECTED[k] for j,k in ((2,3),(3,1),(1,2)))])/2
    beta = np.asarray([1., .3j, -.2+.4j, .6]); beta /= np.linalg.norm(beta)
    Hblocks = []; code_blocks = []; source_blocks = [[], []]; physical_jets = []
    current_blocks = []; contact_blocks = []
    derivative_error = 0.; nonclifford = []
    for angle in theta:
        fields = fields0.copy()
        fields[0,1] = math.cos(angle)*fields0[0,1]-math.sin(angle)*fields0[0,2]
        fields[0,2] = math.sin(angle)*fields0[0,1]+math.cos(angle)*fields0[0,2]
        tangent = np.zeros_like(frame)
        tangent[1] = -fields[0,2]; tangent[2] = fields[0,1]
        charts = [frame_chart(fields[k])[0] for k in range(2)]
        nonclifford.append(frame_chart(fields[0])[1]['nonClifford_residual'])
        scalar = centre_scalar(charts[0].conj().T@link@charts[1], 4., law.generators)[0]
        mark = scalar*(charts[0]@labels@charts[1].conj().T)
        step = 2e-6
        def marks(t: float) -> np.ndarray:
            chart = frame_chart(fields[0]+t*tangent)[0]
            ss = centre_scalar(chart.conj().T@link@charts[1],4.,law.generators)[0]
            return ss*(chart@labels@charts[1].conj().T)
        jet = (marks(step)-marks(-step))/(2*step)
        physical_jets.append(float(np.linalg.norm(jet)))
        # Off-Clifford horizontal tangent is RETAINED, not forced to i[S,Phi].
        vertical = 1j*(s@fields[0]-fields[0]@s)
        horizontal = tangent-vertical
        def directional(direction: np.ndarray) -> np.ndarray:
            values=[]
            for sign in (-1,1):
                chart=frame_chart(fields[0]+sign*step*direction)[0]
                ss=centre_scalar(chart.conj().T@link@charts[1],4.,law.generators)[0]
                values.append(ss*(chart@labels@charts[1].conj().T))
            return (values[1]-values[0])/(2*step)
        derivative_error = max(derivative_error,float(np.linalg.norm(jet-directional(vertical)-directional(horizontal))))
        potential = sum(local_potential(fields,[(0,1)],[link]))
        for a in avec:
            h = np.zeros((8,8),complex)
            h[:4,:4] = onsite*fields[0,0]
            f0 = -(law.band.r*fields[0,0]+1j*law.band.v*fields[0,1])/(2*law.band.a)
            f1 = -(law.band.r*fields[1,0]+1j*law.band.v*fields[1,1])/(2*law.band.a)
            cross = np.exp(1j*a)*(f0@link+link@f1)/2
            # Fold an actual spatial bond into the bulk native Galerkin mode.
            # The opposite reference is STATIC and never a propagation site.
            off = np.zeros_like(h); off[:4,:4] = cross+cross.conj().T
            h += off
            encoded = model.encode_even(h-off)+loop@model.encode_even(off)
            j = np.zeros_like(h); j[:4,:4]=1j*(cross-cross.conj().T)
            contact = -off
            encoded_j = loop@model.encode_even(j)
            encoded_contact = loop@model.encode_even(contact)
            accept(np.linalg.norm(encoded@W-W@exterior(h,masks)), 'actual Phi EM code Hamiltonian')
            accept(np.linalg.norm(encoded_j@W-W@exterior(j,masks)), 'same physical link current code transfer')
            accept(np.linalg.norm(encoded_contact@W-W@exterior(contact,masks)), 'same physical contact code transfer')
            coulomb = ue/3*(sum((model.n(k) for k in range(4)),np.zeros_like(I))-2*I)@(
                sum((model.n(k) for k in range(4)),np.zeros_like(I))-2*I)
            Hblocks.append(sparse.csr_matrix(encoded+coulomb+potential*I))
            code_blocks.append(sparse.csr_matrix(W.conj().T@(encoded+coulomb+potential*I)@W))
            current_blocks.append(sparse.csr_matrix(encoded_j)); contact_blocks.append(sparse.csr_matrix(encoded_contact))
            for label,m in enumerate((np.einsum('a,aij->ij',beta,mark),mark[0]+.4j*mark[2])):
                source = np.zeros((8,8),complex); source[:4,4:]=np.exp(-1j*a)*m
                source_blocks[label].append(sparse.csr_matrix(model.encode_even(source)))
    accept(derivative_error,'actual nonClifford physical Gauss plus horizontal source derivative')
    field = sparse.kron(sparse.csr_matrix(field_kinetic),sparse.eye(2))
    photon = sparse.kron(sparse.eye(2),sparse.csr_matrix(photon_kinetic+np.diag(photon_potential)))
    field_operator = field+photon
    base = sparse.block_diag(Hblocks)+sparse.kron(field_operator,sparse.eye(model.dimension))
    base_code = sparse.block_diag(code_blocks)+sparse.kron(field_operator,sparse.eye(W.shape[1]))
    B = [sparse.block_diag(values,format='csr') for values in source_blocks]
    seed_matter = model.state((4,5,6,7))
    seed_field = np.kron(node_modes.T[:,0],node_modes.T[:,0])
    seed = np.kron(seed_field,seed_matter)
    seed_code = np.kron(seed_field,W.conj().T@seed_matter)
    time = .7
    neutral = expm_multiply(-.23j*base,seed)
    neutral_code = expm_multiply(-.23j*base_code,seed_code)
    U = sparse.kron(sparse.eye(4),sparse.csr_matrix(W))
    accept(np.linalg.norm(neutral-U@neutral_code),'actual SAME neutral field history exact code')
    raw = np.column_stack([b@neutral for b in B])
    gram = raw.conj().T@raw
    values,modes=np.linalg.eigh(gram)
    if values[0] <= 1e-8: raise AssertionError('two fixed joint fullRef4 vectors lost actual global2x2 Gram')
    inverse=(modes/np.sqrt(values))@modes.conj().T
    outputs=raw@inverse
    # One unwhitened input is normalized ONCE; detector vectors remain fixed.
    input_raw=raw[:,0]+(.3+.2j)*raw[:,1]; input_raw/=np.linalg.norm(input_raw)
    input_code=U.conj().T@input_raw
    baseline=expm_multiply(-1j*time*base,input_raw)
    comparison=U@expm_multiply(-1j*time*base_code,input_code)
    accept(np.linalg.norm(baseline-comparison),'physical source current allRef4 fullflight exact code')
    jtotal=sparse.block_diag(current_blocks,format='csr')
    ctotal=sparse.block_diag(contact_blocks,format='csr')
    fullflight=[]
    beta_defect=.07
    # Z on non-tree wrap edge is charge preserving and full onsite-SU4 scalar.
    from bosonic_statistics import tensor_pauli
    word=['I']*len(model.edges); word[model.edge_index[(0,7)]]='Z'
    defect=tensor_pauli(word)
    for colour in law.generators:
        g=np.zeros((8,8),complex);g[:4,:4]=colour;g[4:,4:]=colour
        encoded_g=model.encode_even(g)
        accept(np.linalg.norm(defect@encoded_g-encoded_g@defect),'finite defect actual onsite fullSU4 gauge action')
    leakage_op=sparse.kron(sparse.eye(4),sparse.csr_matrix(I-projector))
    _, Qbasis = model.occupation_basis(4, signs=(-1,))
    Qembed = sparse.kron(sparse.eye(4),sparse.csr_matrix(Qbasis))
    Qmin = float(np.linalg.eigvalsh((Qembed.conj().T@base@Qembed).toarray())[0])
    Pmax = float(np.linalg.eigvalsh(base_code.toarray())[-1])
    for penalty in (12.,48.,192.):
        perturb=model.loop_penalty(penalty)+beta_defect*defect
        finite=base+sparse.kron(sparse.eye(4),sparse.csr_matrix(perturb))
        state=expm_multiply(-1j*time*finite,input_raw)
        probabilities=abs(outputs.conj().T@state)**2
        control=abs(outputs.conj().T@baseline)**2
        gap = penalty+Qmin-Pmax-beta_defect
        certified = None
        if gap > 0:
            # Four mixing costs cover undressed source and output histories.
            certified = 4*beta_defect/gap+time*beta_defect**2/gap
            accept(max(0.,float(np.linalg.norm(state-baseline))-certified),
                   'actual finite-gap physical source flight bound')
        fullflight.append(dict(penalty=penalty,nonzero_defect=beta_defect,
            source_state_distance=float(np.linalg.norm(state-baseline)),
            loop_leakage=float(np.vdot(state,leakage_op@state).real),
            two_fixed_joint_probabilities=probabilities,
            fixed_receiver_probability_difference=float(np.max(abs(probabilities-control))),
            full_band_gap_lower=gap,certified_state_transfer_upper=certified,
            same_current_expectation=float(np.vdot(state,jtotal@state).real),
            same_contact_expectation=float(np.vdot(state,ctotal@state).real)))
    if fullflight[0]['loop_leakage'] <= 1e-12:raise AssertionError('finite flux dynamics did not reach the physical receiver')
    # Physical source Noether acts on BOTH actual charged and SAME neutral histories.
    oscillator_p=np.asarray([[0.,-1j],[1j,0.]])/math.sqrt(2)
    P=node_modes.conj().T@oscillator_p@node_modes/(math.sqrt(2)*scale)
    J=sparse.kron(sparse.kron(sparse.csr_matrix(P),sparse.eye(2)),sparse.eye(model.dimension))
    relative=J@(B[0]@neutral)-B[0]@(J@neutral)
    target=(J@B[0]-B[0]@J)@neutral
    noise=float(np.linalg.norm(B[0]@(J@neutral)))
    relative_error=accept(np.linalg.norm(relative-target),'coupled Phi source RAW SAME neutral Noether commutator')
    wrong_neutral=adverse(np.linalg.norm(J@(B[0]@neutral)-target),'coupled Phi source RAW SAME neutral Noether commutator')
    # Resolve the actual wedge3 colour channels instead of deleting an arbitrary row.
    one_colour=model.n(4)@model.n(5)@model.n(6)@(I-model.n(7))
    wrong_output=sparse.kron(sparse.eye(4),sparse.csr_matrix(one_colour))@raw[:,0]
    wrong_ref=adverse(abs(np.linalg.norm(wrong_output)-np.linalg.norm(raw[:,0])), 'actual fullRef4 source norm')
    return dict(scope='finite original-coefficient folded spatial-bond Cartesian-Phi/noncompact-photon Galerkin witness with STATIC actual reference; no microscopic ground claim',
        modes=8,edge_qubits=len(model.edges),field_photon_nodes=4,full_reference=4,
        charged_compensated_reference='wedge3 from actual nref4 via light-create/ref-annihilate; totalN4',
        same_neutral='actual coupled field/photon/Peierls evolution, not discarded vacuum phase',
        physical_Noether=physical,nonClifford_residuals=nonclifford,
        source_Lie_jet_norms=physical_jets,physical_Gauss_horizontal_jet_error=derivative_error,
        original_coefficients=dict(chi=chi,quartic_stiffness=2.,alignment=.2,UE=ue,K=K,c=photon_speed),
        exact_source_current_fullflight_error=float(np.linalg.norm(baseline-comparison)),
        fixed_output_Gram=gram,fixed_output_global2x2_inverse=inverse,
        complex_quartet_coefficients=beta,input_column_normalization=False,
        finite_defect_fullflight=fullflight,relative_Noether_error=relative_error,
        retained_neutral_Noether_noise=noise,
        adverse=dict(wrong_same_neutral=wrong_neutral,missing_reference=wrong_ref),
        source_chart_used_only_for_native_witness=True,
        selected_large_family_source='entire full-Haar eta18 alpha2^371, no polar chart imposed')


def exact_rational(value: Any) -> Fraction:
    if isinstance(value, dict):
        return Fraction(int(value['exact_numerator_hex'],16),
                        int(value['exact_denominator_hex'],16))
    return Fraction(value)


def selected_weighted_budget(receipt_path: Path) -> dict[str, Any]:
    """Actual selected form graph; complementary rho image costs G^-3/4."""
    receipt_bytes=receipt_path.read_bytes()
    receipt=json.loads(receipt_bytes)
    if receipt['N'] != NSTAR or receipt['selected_total_exponent'] != 4096:
        raise ValueError('physical transfer requires the SAME selected b4096 Nstar receipt')
    if exact_rational(receipt['exact_family']['UE']) != UE:
        raise ValueError('normal receipt has a different original photon coupling')
    V=NSTAR**3; target=UE/Fraction(V*2**40)
    constants=receipt['full_Phi_and_photon_graph_capability']
    A4=exact_rational(constants['A4']); CD=exact_rational(constants['CD'])
    Hvac=exact_rational(receipt['exact_normal_constants']['Hvac'])
    from bosonic_original_law_join import quartic_vacuum_lower
    vacuum_advantage=-quartic_vacuum_lower(V,1,RotationBand(**BARE))
    hvac_log=Hvac.numerator.bit_length()-Hvac.denominator.bit_length()+1
    selected_log=int(receipt['time6_operator_capability']['local_log2_Lambda'])
    if 8*hvac_log >= 5*selected_log:
        raise AssertionError('selected normal coefficient has unpaid Hvac/Lambda term')
    # Hvac Lambda^-5/8, original weak photon trial, and residual are <=1 each.
    rho4=2/A4*(CD+vacuum_advantage+3)
    rho_bits=(rho4.numerator.bit_length()-rho4.denominator.bit_length()+4)//4
    rho_upper=Fraction(2**max(0,rho_bits))
    if rho_upper**4 < rho4:raise AssertionError('outward actual quartic graph root failed')
    metric=exact_rational(receipt['actual_positive_metric_source_multiplier'])
    source_value=Fraction(2**4); source_jet=Fraction(2**427)
    field_coefficient=4*source_jet*NSTAR*(1+rho_upper)
    current_coefficient=64*V*(1+rho_upper)
    prepared=receipt['actual_graph_packet']['data']
    half_photon=exact_rational(prepared['actual_Gamma_prepared_photon_graph'][
        'normalized_original_input_Hph_half_norm_upper'])
    harmonic_jet=exact_rational(prepared['normal_metric_source_input']['source_harmonic_first_upper'])
    # Outward real-curl scalar boundary, retaining Coulomb/harmonic coefficient.
    # sqrt(UE)<=1 is an upper bound, NOT a changed photon Hamiltonian.
    photon_coefficient=16*NSTAR**2*harmonic_jet*(1+half_photon)/C
    graph_weight=metric*(source_value+field_coefficient+current_coefficient+photon_coefficient+1)
    beta=Fraction(1,16)
    base_bits=(target.denominator.bit_length()-target.numerator.bit_length()
               +graph_weight.numerator.bit_length()-graph_weight.denominator.bit_length()+24)
    k=4*((max(1,base_bits)+2)//3)
    # ||z||<=beta/G, <z,Qz><=beta²/G and the quartic FORM imply
    # ||rho z||<=C beta/G^(3/4); do not invent a rho4 operator norm.
    weighted_image=graph_weight*beta/Fraction(2**(3*k//4))
    ordinary=graph_weight*(4*beta+6*beta*beta)/Fraction(2**k)
    em_gap=5*180375//4
    em_image=8*graph_weight*beta/Fraction(2**(em_gap//2))
    bound=8*weighted_image+ordinary+em_image
    if bound>target:raise AssertionError('absolute physical source graph selector misses UE/V')
    return dict(N=NSTAR,V=V,eta=18,alpha='2^371',source_sites=19,tree_word=NSTAR//2+2,
        harmonic_domain='kappa0 fibre or explicit kappa/minus-kappa paired histories',
        source_jet_upper_exponents=[4,427,850,1274,1698,2122,2547],
        all64_flat_holonomies_retained=True,no_nonlocal_topological_penalty=True,
        gap_log2=k,finite_positive_defect='1/16',time=6,
        target='UE/(V*2^40)',absolute_graph_error_over_target=float(bound/target),
        actual_rho_fourth_root_upper_log2=max(0,rho_bits),
        absolute_source_current_graph_log2_upper=graph_weight.numerator.bit_length()-graph_weight.denominator.bit_length()+1,
        finite_penalty='BN+Utrial+window+beta+max(2^gap_log2,BN+Utrial+window+beta+1)',
        gap_is_floor_not_absolute=True,retained_EM_gap_log2_lower=em_gap,
        retained_EM_gap_image_power='-1/2',
        photon_boundary_not_discarded=True,
        photon_graph_scope='actual prepared Gamma graph with separate microscopic normal/source transfer; no invented interacting-ground graph',
        no_neutral_momentum_variance_needed=True,
        projected_state_is_quasimode=True,source_receipt_sha256=hashlib.sha256(receipt_bytes).hexdigest(),
        actual_normal_supplier='selected-normal-PH packet7336bc084e247ffdb6864493e59da350d78525541675347bd5bc0590c9c6bca2',
        source_value_Lambda_log2_requirement=180375)


def original_microscopic_graph_budget(graph_path: Path) -> dict[str, Any]:
    """Whole ORIGINAL-microscopic graph, including actual chart prefactor."""
    content=graph_path.read_bytes(); packet=json.loads(content)
    if packet['N'] != NSTAR or exact_rational(packet['actual_family']['UE']) != UE:
        raise ValueError('original microscopic graph has a different Nstar/UE family')
    jets=physical_raw_noether_geometry_majorant(64.,1.)
    graph=packet['actual_microscopic_U']['log2_upper']
    mixed=int(graph['two_current_two_angular_without_CY_squared'])+2*jets['selected_CY_log2']
    current=exact_rational(packet['current_coefficient'])
    current_bits=current.numerator.bit_length()-current.denominator.bit_length()+1
    # Reconstruct source19 from its OWN normalized profile and source jets.
    # U_g bounds are universal in unit-l2 coexact g; no RAW radial Leta copied.
    cy=jets['selected_CY_log2']
    source19_angular1=max(427+graph['rho'],4+graph['n']+8)+cy+4
    source19_angular2=max(850+graph['rho2'],4+graph['n']+16)+2*cy+8
    source19_mixed=max(graph['two_current_two_angular_without_CY_squared'],
        850+graph['rho4']+2*current_bits)+2*cy+8
    chart=int(packet['normal_core_tube_log2_prefactor'])
    # Explicitly retain tube and normalization; they are not the mixed graph.
    needed=max(mixed,source19_mixed)+chart+2856+427+32
    envelope=65536
    if needed>envelope:raise AssertionError('actual microscopic graph exceeds finite selected envelope')
    gap=131072; beta=Fraction(1,16); target=UE/Fraction(NSTAR**3*2**40)
    # This is a GAP FLOOR, not absolute dominance of the full Phi law.
    # Actual G>=BN+Utrial+window+beta+1 and Utrial>Lambda^(5/4).
    # The unchanged selected Lambda has log2 Lambda>=180375.
    em_gap=5*180375//4
    phi_bound=Fraction(2**envelope)*(8*beta/Fraction(2**(3*gap//4))
        +(4*beta+6*beta*beta)/Fraction(2**gap))
    # Retained coexact A/source boundary images cost G^-1/2, not G^-3/4.
    em_bound=Fraction(2**envelope)*8*beta/Fraction(2**(em_gap//2))
    bound=phi_bound+em_bound
    if bound>target:raise AssertionError('original microscopic weighted transfer misses UE/V')
    ratio=bound/target
    return dict(receipt_sha256=hashlib.sha256(content).hexdigest(),
        source_profile=packet['source_profile'],source_family_not_relabelled=True,
        source19_domination=dict(normalized_profile_L2_upper=2,
            actual_source_count=19,actual_tree_word=NSTAR//2+2,
            dominating_tree_word=NSTAR//2+60,source_q6_ceils=[4,427,850,1274,1698,2122,2547],
            own_scalar_orbital_coefficient_upper=192,
            angular1_log2=source19_angular1,angular2_log2=source19_angular2,
            two_current_two_angular_log2=source19_mixed,
            U_g_definition='-sum gk/sqrt(omega_k)*(Q0+omega_k)^-1 Jk Omega; all2(V-1) modes',
            U_g_is_unit_l2_complex_g_uniform=True,
            actual_input='B a_dagger[g] Omega, NOT U_g and not Gamma response',
            own_charged_input_norm_lower='1/sqrt8',
            own_normalized_charged_photon_half_log2_upper=600,
            RAW_profile_orbital_RB_half_graph_copied=False),
        state_domain=packet['state_domain'],proof_scope=packet['proof_scope'],
        original_microscopic_U_graph_log2=graph,actual_CY_log2=jets['selected_CY_log2'],
        two_current_two_angular_graph_log2=mixed,actual_chart_prefactor_log2=chart,
        finite_combined_envelope_log2=envelope,finite_positive_defect='1/16',
        finite_auxiliary_gap_log2=gap,
        finite_auxiliary_gap_is_floor_not_absolute=True,
        actual_gap='max(2^131072,BN(Lambda)+Utrial(Lambda)+window+beta+1)',
        actual_penalty='BN(Lambda)+Utrial(Lambda)+window+beta+actual_gap',
        full_Gauss_trial='V*(300*Lambda+(132/5)*Lambda^(5/4)+15*Lambda^(-1/2)+4*c)',
        retained_EM_image_gap_power='-1/2',
        actual_EM_gap_log2_lower=em_gap,
        retained_EM_image_error_log2_upper=envelope+3-4-em_gap//2,
        error_over_UE_V_2power40_log2_upper=ratio.numerator.bit_length()-ratio.denominator.bit_length()+1,
        original_photon_half_graph_log2=packet['actual_microscopic_photon_created_Omega']['log2_upper']['positive_photon_half'],
        Gamma_ground_substituted=False,total_Noether_variance_deleted=False,
        source_RB_energy_half_norm_claimed=False,all_flat_auxiliary_sectors=8,
        original_colour_histories=64,
        fullflight_probability_claimed=False)


def selected_physical_source_bound(receipt_path: Path, graph_path: Path) -> dict[str, Any]:
    """Literal whole-Haar source and ORIGINAL microscopic photon graph."""
    receipt=json.loads(receipt_path.read_bytes())
    packet=json.loads(graph_path.read_bytes())
    if receipt['N'] != packet['N'] or packet['N'] != NSTAR:
        raise ValueError('physical source and actual microscopic photon graph differ')
    # Original creation: ||a_g Omega||<=sqrtUE*2^800, BEFORE applying B.
    u_bits=int(packet['actual_microscopic_U']['log2_upper']['n'])
    if UE*Fraction(2**(2*u_bits)) > Fraction(1,2**20):
        raise AssertionError('original microscopic photon creation graph is unpaid')
    core=orientation_core_halfspin_bound(NSTAR//2+2,18.,.025,2.,.001)
    if core['alpha_expression'] != '2^371':
        raise AssertionError('physical source bound silently changed selected alpha')
    lie=physical_orientation_lie_majorant(core,2.,.25,1025/1024)
    geometry=physical_raw_noether_geometry_majorant(64.,1.)
    if geometry['selected_CY_log2'] > 1027:
        raise AssertionError('literal original physical vector jets exceed normal CY cap')
    log_ue=math.log2(UE.numerator)-math.log2(UE.denominator)
    # AFTER applying B and normalization Hph-half<=2^600, not the bare150.
    # Factor it logarithmically; never allocate floating energy graph2^1200.
    photon=relative_photon_noether_graph(log_ue,float(C),1.,
        64.,1.,float(NSTAR**2),.25,coexact_singular_lower=4/float(NSTAR))
    photon['relative_photon_Noether_upper_log2']+=600
    photon['actual_normalized_charged_photon_half_log2_upper']=600
    photon['coefficient_scope']='scalar coefficient is before explicit 2^600 half-graph factor'
    return dict(source='SAME scattering eta18/source19, entire whole-Haar alpha2^371, fullRef4',
        core=core,literal_physical_source_Lie=lie,actual_vector_jet_geometry=geometry,
        relative_scalar_EM_and_coexact_orbital_A=photon,
        physical_statement='J_relative = S_core + actual scalar orbital + retained fixed-reference/scalar-EM boundary + bounded source/normal/loop remainder',
        intrinsic_halfspin_core_generator_error=core['physical_core_halfspin_generator_error'],
        ordinary_closed_source_loop_not_claimed_minus=True,
        source19_scalar_orbital_graph_not_relabelled_RAW_nativeN41=True,
        normal_bad_graph='(1+sqrt_upper(2*(CD+3)/A4))*d/G outside full W support; CY2^64 below supplied cap2^1027',
        normal_input_state_is_quasimode_not_exact_Gamma_ground=True,
        actual_photon_graph_scope='ORIGINAL microscopic B a_dagger[g] Omega; normalized charged Hph half<=2^600, norm lower1/sqrt8',
        total_J_squared_halfspin_for_arbitrary_source_not_claimed=True)


def spacetime_fork() -> dict[str, Any]:
    """Construct/reselect an improved local Phi branch at the same photon c."""
    speed=float(C)
    def candidate(v: float) -> tuple[RotationBand,np.ndarray,np.ndarray]:
        b=RotationBand(a=1.,m=.2,r=.12,v=v,improved=True)
        selected,frame=original_frame(b,9,np.zeros(3),False)
        return b,np.asarray(selected['G']),frame
    def equation(v: float) -> float:
        _,G,_=candidate(v)
        return v*math.sqrt(float(G[1,1]))-speed
    v=cast(float,brentq(equation,.05,1.,xtol=2e-13))
    band,G,frame=candidate(v)
    g0=float(G[0,0]); gs=float(G[1,1]); mass=band.m*math.sqrt(g0)
    cone=band.v*math.sqrt(gs)
    accept(abs(cone-speed),'reselected physical Phi charged photon low-energy cone')
    rows=[]
    for radius in (.2,.1,.05):
        hbound=cone*radius**3/6+math.sqrt(g0)*band.r*radius**4/4
        current_bound=cone*radius**2/2+math.sqrt(g0)*band.r*radius**3
        energy_error=0.;current_error=0.;ward_error=0.;flight_error=0.
        for direction in (np.array([1.,0.,0.]),np.ones(3)/math.sqrt(3),np.array([2.,-1.,1.])/math.sqrt(6)):
            p=radius*direction
            h=np.einsum('m,mij->ij',np.r_[band.mass(p),band.v*np.sin(p)],frame)
            cont=band.m*frame[0]+np.einsum('j,jik->ik',band.v*p,frame[1:])
            error=float(np.linalg.norm(h-cont,ord=2));energy_error=max(energy_error,error)
            accept(max(0.,error-hbound),'improved actual-Phi massive dispersion bound')
            cur=np.asarray([reframe(j,frame) for j in band.current(p,p)])
            continuum=band.v*frame[1:]
            current_error=max(current_error,max(float(np.linalg.norm(x-y,ord=2)) for x,y in zip(cur,continuum)))
            q=np.array([.011,-.007,.005]);out=p+q;inn=p-q
            difference=(np.einsum('m,mij->ij',np.r_[band.mass(out),band.v*np.sin(out)],frame)
                        -np.einsum('m,mij->ij',np.r_[band.mass(inn),band.v*np.sin(inn)],frame))
            vertex=np.asarray([reframe(j,frame) for j in band.current(out,inn)])
            ward_error=max(ward_error,float(np.linalg.norm(difference-np.einsum('j,jik->ik',2*np.sin(q),vertex))))
            flight_error=max(flight_error,float(np.linalg.norm(expm(-6j*h)-expm(-6j*cont),ord=2)))
        accept(max(0.,current_error-current_bound),'improved actual-Phi current cone bound')
        accept(ward_error,'same improved-hop exact finite-transfer Ward identity')
        accept(max(0.,flight_error-6*hbound),'actual massive spacetime propagator Duhamel bound')
        rows.append(dict(physical_cutoff=radius,H_Dirac_error=energy_error,H_upper=hbound,
            current_error=current_error,current_upper=current_bound,
            T6_massive_propagation_error=flight_error,T6_upper=6*hbound,Ward_error=ward_error))
    bare=RotationBand(**BARE)
    selected,_=original_frame(bare,9,np.zeros(3),False)
    bg=np.asarray(selected['G']); bv=bare.v*math.sqrt(float(bg[1,1]));bm=bare.m*math.sqrt(float(bg[0,0]))
    bare_g=2*bv*bv/(bv*bv+float(bg[0,0])*bare.m*bare.r*bare.a)
    bad=adverse(abs(bare_g-2),'bare physical-Phi Wilson gyromagnetic versus Dirac',1e-4)
    return dict(original_bare_retained=dict(parameters=BARE,actual_G=bg,mass=bm,
            linear_velocity=bv,photon_c=speed,physical_g_eff=bare_g,adverse=bad),
        constructive_fork=dict(parameters=dict(a=1.,m=.2,r=.12,v=v,improved=True),
            full10_actual_Phi_Gram=G,Gram_sampling=9,charged_cone=cone,photon_c=speed,
            mass=mass,positive_gap_lower=mass,rest_low_energy_g=2.,
            g_scope='continuum O(p^2) Foldy coefficient; finite-spacing current/flight debts bounded, not exact Lorentz',
            rows=rows,photon_UE_and_source_normalization_retuned=False,
            v_is_explicit_parameter_calibration_not_generated=True,
            improvement='range2 covariant two-link Wilson mass, original first/second derivatives from same hops',
            microscopic_original_bare_ground_not_relabelled=True,
            all_momenta_exact_Lorentz=False))


def run(normal_receipt: Path, microscopic_graph: Path) -> dict[str, Any]:
    law=BulkQuantumLaw(3,band=RotationBand(**BARE))
    _,frame=original_frame(law.band,3,np.zeros(3),False)
    native=native_transfer(frame,law)
    core=mechanical_core_noether_consumer(law,frame,4.)
    photon=same_neutral_photon_commutator_consumer(law,frame,4.)
    paths=['bosonic_physical_transfer.py','bosonic_physical_transfer.md',
           'bosonic_original_law_join.py','bosonic_original_law_join.md',
           'noncompact_screened_gauss_dressing.py','noncompact_screened_gauss_dressing.md',
           'original_charged_principal.py','unrestricted_field_minimum.py','rotation_band.py']
    hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in paths}
    return dict(achievement='actual physical Phi-Gauss tangent, same-current fullRef4 source and finite flux dynamics joined at two fixed global2x2 receiver vectors',
        native=native,actual_full15_Noether_core=core,actual_RAW_same_neutral_photon=photon,
        selected_actual_field_Noether_source=selected_physical_source_bound(normal_receipt,microscopic_graph),
        selected_weighted_loop_transfer=selected_weighted_budget(normal_receipt),spacetime=spacetime_fork(),
        original_microscopic_weighted_transfer=original_microscopic_graph_budget(microscopic_graph),
        source_sha256=hashes,derivation_scope='field-independent code isometry intertwines original Phi/EM derivatives exactly; weighted finite defect graph and improved-hop Duhamel bounds',
        finite_witness_scope='native N3 physical supplier and one-bond coupled field/photon Pauli consumer; not full Nstar dense allocation or ground solution',
        limits=['Original microscopic cubic law is not exact SO3/Lorentz.',
                'Prepared Gamma photon graph composes separate microscopic normal/source transfer; this loop allocation alone is not full-flight admission.',
                'Reselected improved fork is constructive low-energy branch, not a claim the original bare branch acquired g=2.',
                'Actual projected neutral is a quasimode in kappa0/paired harmonic domain, never an exact effective ground.'])


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--normal-receipt',type=Path,required=True,
                        help='actual selected b4096 far normal/current source receipt')
    parser.add_argument('--microscopic-graph',type=Path,required=True,
                        help='actual original microscopic Phi/current/source graph receipt')
    args=parser.parse_args()
    print(json.dumps(portable(run(args.normal_receipt,args.microscopic_graph)),indent=2,sort_keys=True))
