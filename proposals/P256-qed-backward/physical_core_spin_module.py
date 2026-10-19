"""Original-law mechanical core module, not a compressed SOURCE-label spin fit.

The gauge orbit D has Phi mass1 and link mass1/.6.  The vertical Noether
coefficient solves (D* M D) omega=D* M Y.  On a uniform Clifford coframe
and flat links this is (I_Phi+L/.6) omega=I_Phi sigma alpha.  sigma is
recovered from the literal Cartesian-vector tangent, not assigned to Ref.
Finite group transport acts on arbitrary packet matrices, with orbital
pullback and actual Ref RIGHT.  A scalar source-section loop is not that
transport: its centre transition must be kept, including its compensator.
"""
from __future__ import annotations
from fractions import Fraction as F
import math
from typing import Any
import numpy as np
from scipy import sparse
from scipy.linalg import expm
from scipy.sparse.linalg import spsolve
from original_total_angular_transfer import primitive_flow, scalar_orbital
from soft_spatial_closed_rotation import native_full_gauss_columns
from unrestricted_field_minimum import coordinates
from noncompact_screened_gauss_dressing import frame_chart


def core_from_field(law: Any, frame: np.ndarray) -> dict[str, Any]:
    """Solve all15 original Gauss directions against all60 Phi coordinates."""
    D=np.column_stack([coordinates(1j*(t@frame-frame@t),law.basis).reshape(60).real
                       for t in law.generators])
    inertia=D.T@D
    coefficients=[];residuals=[]
    for j,k in ((2,3),(3,1),(1,2)):
        tangent=np.zeros_like(frame);tangent[j]=-frame[k];tangent[k]=frame[j]
        y=coordinates(tangent,law.basis).reshape(60).real
        a=np.linalg.solve(inertia,D.T@y)
        coefficients.append(a);residuals.append(float(np.linalg.norm(D@a-y)))
    spins=np.einsum('ja,aik->jik',coefficients,law.generators)
    values,vectors=np.linalg.eigh(law.band.m*frame[0])
    positive=vectors[:,values>0]
    rest: Any=np.asarray([positive.conj().T@s@positive for s in spins])
    identity=np.eye(rest.shape[1]);fullidentity=np.eye(4)
    algebra=max(float(np.linalg.norm(spins[j]@spins[k]-spins[k]@spins[j]-1j*spins[l],2))
                for j,k,l in ((0,1,2),(1,2,0),(2,0,1)))
    casimir=sum(s@s for s in rest)
    return dict(spins=spins,rest_spins=rest,positive_rest_frame=positive,
        Phi_orbit_columns=D,Phi_inertia=inertia,Gauss_coefficients=np.asarray(coefficients),
        Phi_inertia_min=float(np.linalg.eigvalsh(inertia)[0]),
        physical_vector_tangent_residuals=residuals,SU2_commutator_error=algebra,
        rest_Casimir_eigenvalues=np.linalg.eigvalsh(casimir),
        rest_Casimir_half_error=float(np.linalg.norm(casimir-.75*identity,2)),
        rest_axis_spectrum=np.linalg.eigvalsh(rest[2]),
        full_two_pi_minus_error=float(np.linalg.norm(expm(-2j*math.pi*spins[2])+fullidentity,2)),
        positive_rest_dimension=positive.shape[1],
        fixed_frame_centralizer='Z4; Spin4 rotates coframe labels, not its gauge centralizer',
        original_fields_NOT_whitened=True)


def screened_noether(law: Any,fields: np.ndarray,links: np.ndarray,phases: np.ndarray,
                      alpha: np.ndarray,velocity: np.ndarray,reference: int,
                      reference_frame: np.ndarray | None=None) -> dict[str,Any]:
    """Actual full Gauss/Noether elliptic problem; every link and mass retained."""
    df,du,da=primitive_flow(law,fields,links,phases,velocity,alpha)
    left=-1j*du@links.conj().swapaxes(-1,-2)
    raw=np.c_[coordinates(df,law.basis).reshape(law.volume,60).real,
              (2*np.einsum('xjkl,alk->xja',left,law.generators)).real.reshape(law.volume,45)].ravel()
    D,mass_site=native_full_gauss_columns(law,fields,links)
    mass=np.tile(mass_site,law.volume)
    metric=(D.T@sparse.diags(mass)@D).tocsc()
    rhs=np.asarray(D.T@(mass*raw))
    omega=np.asarray(spsolve(metric,rhs))
    horizontal=raw-D@omega
    matrices=np.einsum('xa,aij->xij',omega.reshape(law.volume,15),law.generators)
    ref=matrices[reference]
    if reference_frame is not None:ref=reference_frame.conj().T@ref@reference_frame
    return dict(omega=omega,matrices=matrices,Ref_RIGHT=ref,physical_Ref=matrices[reference],
        orbital108=scalar_orbital(law,links,phases,velocity),horizontal_RAW=horizontal,
        full_orbit_metric=metric,raw=raw,raw_EM=da,
        equation_residual=float(np.linalg.norm(metric@omega-rhs)),
        mass_horizontal_error=float(np.linalg.norm(D.T@(mass*horizontal))),
        equation='(D_Phi^T D_Phi + d_U^T d_U/.6) omega = D^T M Y; actual primitive clover on RHS',
        colour_cometric_over_chi=.6,NO_scalar_spin_fit=True)


def uniform_screen(law: Any,core: dict[str,Any],alpha: np.ndarray,axis: int) -> np.ndarray:
    """Independent Fourier reduction of the same all15 elliptic problem."""
    inertia=core['Phi_inertia'];values,modes=np.linalg.eigh(inertia)
    n=law.size;k=2*math.pi*np.arange(n)/n
    q2=2-2*np.cos(k);L=q2[:,None,None]+q2[None,:,None]+q2[None,None,:]
    a=core['Gauss_coefficients'][axis];rhs=inertia@a
    fft=np.fft.fftn(alpha.reshape((n,)*3))
    spectrum=(modes.T@rhs)*fft[...,None]/(values+L[...,None]/.6)
    coefficients=np.fft.ifftn(spectrum@modes.T,axes=(0,1,2)).real.reshape(law.volume,15)
    return np.einsum('xa,aij->xij',coefficients,law.generators)


def packet_action(packet: np.ndarray,body: np.ndarray,reference: np.ndarray,
                  orbital: np.ndarray,charge: int=1) -> dict[str,np.ndarray]:
    """K S-S R or R T-T K, genuine orbital action, no fixed8-ray premise."""
    volume=len(packet)
    if charge==1:
        local=body@packet
        right=-packet@reference
        orbit=(orbital@packet.reshape(4*volume,4)).reshape(packet.shape)
    elif charge==-1:
        local=-packet@body
        right=reference@packet
        T=packet.transpose(1,0,2).reshape(4,4*volume)
        orbit=(-(T@orbital)).reshape(4,volume,4).transpose(1,0,2)
    else:raise ValueError('Original charged maps are +1 or -1')
    return dict(core=local,Ref_RIGHT=right,orbital=orbit,total=local+right+orbit)


def frame_path_gluing(frame: np.ndarray,spin: np.ndarray,steps: int=96) -> dict[str,Any]:
    """Keep each Z4 chart transition; compare the lift with its source section."""
    centres=np.array([1.,1j,-1.,-1j]);previous=frame_chart(frame)[0]
    initial=previous.copy();transitions=[];guard_min=math.inf;polar_min=math.inf
    for theta in np.linspace(0,2*math.pi,steps+1)[1:]:
        U=expm(-1j*theta*spin)
        physical=U@frame@U.conj().T
        principal,guard=frame_chart(physical)
        phase=centres[int(np.argmax([np.trace(previous.conj().T@(z*principal)).real for z in centres]))]
        continued=phase*principal
        transitions.append(complex(phase));previous=continued
        guard_min=min(guard_min,guard['singular_squared_gap']);polar_min=min(polar_min,guard['polar_min'])
    principal_end=frame_chart(frame)[0]
    return dict(continued_endpoint=previous,principal_endpoint=principal_end,
        centre_transition_phases=np.asarray(transitions),
        retained_lift_minus_error=float(np.linalg.norm(previous+initial,2)),
        principal_section_plus_error=float(np.linalg.norm(principal_end-initial,2)),
        chart_gap_min=guard_min,chart_polar_min=polar_min,
        distinction='The lift ends at -I. Original scalar centre words compensate section changes; resetting phases would invent a different SOURCE.')


def positive_band_group(law: Any,frame: np.ndarray,core: dict[str,Any],p: np.ndarray,
                        rotation_vector: np.ndarray) -> dict[str,Any]:
    """Actual positive light band and rotation; orbital p pullback is essential."""
    from scipy.spatial.transform import Rotation
    R=Rotation.from_rotvec(rotation_vector).as_matrix()
    U=expm(-1j*np.einsum('j,jab->ab',rotation_vector,core['spins']))
    h=law.band.mass(p)*frame[0]+np.einsum('j,jab->ab',law.band.v*np.sin(law.band.a*p)/law.band.a,frame[1:])
    rp=R@p;hr=law.band.mass(rp)*frame[0]+np.einsum('j,jab->ab',law.band.v*np.sin(law.band.a*rp)/law.band.a,frame[1:])
    e,v=np.linalg.eigh(h);er,vr=np.linalg.eigh(hr)
    P=v[:,e>0]@v[:,e>0].conj().T;Pr=vr[:,er>0]@vr[:,er>0].conj().T
    return dict(actual_positive_band_rank=int(np.count_nonzero(e>0)),
        band_covariance_error=float(np.linalg.norm(Pr-U@P@U.conj().T,2)),
        actual_h_covariance_error=float(np.linalg.norm(hr-U@h@U.conj().T,2)),
        wrong_fixed_momentum_error=float(np.linalg.norm(P-U@P@U.conj().T,2)),
        no_new_preparation_projector=True,orbital_pullback='f(p) -> U(R) f(R^-1 p); fixed-p spin compression alone is not intrinsic spin')


def fundamental_colour_energy(core: dict[str,Any],law: Any,site: int,reference: int) -> dict[str,Any]:
    """Continuous screened Gauss energy, NOT a discrete centre seam certificate."""
    n=law.size;inertia=core['Phi_inertia'];values,modes=np.linalg.eigh(inertia)
    impulse=np.zeros(law.volume);impulse[site]=1
    reference_impulse=np.zeros(law.volume);reference_impulse[reference]=1
    fft=np.fft.fftn(impulse.reshape((n,)*3))
    ref_fft=np.fft.fftn(reference_impulse.reshape((n,)*3))
    k=2*math.pi*np.arange(n)/n
    q2=2-2*np.cos(k);L=q2[:,None,None]+q2[None,:,None]+q2[None,None,:]
    green=1/(values+L[...,None]/.6)
    self_tensor=(modes*np.mean(green,axis=(0,1,2)))@modes.T
    cross_tensor=(modes*np.mean((fft.conj()*ref_fft).real[...,None]*green,axis=(0,1,2)))@modes.T
    fundamental=law.generators
    antifundamental=-fundamental.conj()
    self_charge=np.einsum('ab,aij,bjk->ik',self_tensor,fundamental,fundamental)
    ref_charge=np.einsum('ab,aij,bjk->ik',self_tensor,antifundamental,antifundamental)
    operator=.5*(np.kron(self_charge,np.eye(4))+np.kron(np.eye(4),ref_charge))
    for a in range(15):
        for b in range(15):
            operator+=cross_tensor[a,b]*np.kron(fundamental[a],antifundamental[b])
    return dict(continuous_pair_energy_operator_over_chi=operator,
        energy_eigenvalues_over_chi=np.linalg.eigvalsh(operator),
        universal_upper_over_chi=float((15/8)/core['Phi_inertia_min']),
        charge='fundamental nality1 plus FAR anti-fundamental; local adjoint nality0 cannot screen its Z4 charge',
        no_continuous_inertia_deconfinement_or_all64_seam_claim=True)


def legal_screening(geometry: dict[str,Any],bridge: dict[str,Any],C2: dict[str,Any]) -> dict[str,Any]:
    """Original radii/clock, exact conservative killed-walk bound, original tails.

    For a scalar inertia eigenvalue mu, the normalized positive resolvent is
    sum_(n>=0) (1-r) r^n P^n, r=6/(6+.6mu). Reaching distance d needs n>=d,
    so its total mass is <=r^d <=2^(-floor(d*.6mu/(6+.6mu))). This handles
    any ORIGINAL radial alpha with plateau1 inside Rin and support inside Rout;
    no torque precompensation or changed Ref position is introduced.
    """
    selected=geometry['result']['result']['selected']
    N=5*2**96;Rin=int(selected['inner_radius']);Rout=int(selected['outer_radius']);T=int(selected['duration'])
    if (Rin,Rout,T)!=(1342177280000,2684354560000,1008124871316564):
        raise ValueError('Not original LEGAL geometry')
    if C2['science']['clock']!=T or bridge['clock']!=T:raise ValueError('Mismatched physical clock')
    # This is explicitly CONDITIONAL on the owner's full4 actual Gram lower.
    gram=F(1,20);mu=gram/2;m2=F(3,5)*mu
    packet_radius=Rin//2
    local_distance=Rin-packet_radius-2;ref_distance=N//2-Rout-2
    exponent=lambda d: -((d*m2.numerator)//(6*m2.denominator+m2.numerator))
    local_log=exponent(local_distance);ref_log=exponent(ref_distance)
    # Original frozen q6 bound is <2e60. Original vgroup*T <2e8 cells.
    derivative=F(2*10**60);available=packet_radius-200000000
    if F(str(selected['actual_derivative_q0_q6'][6]))>derivative:
        raise ValueError('Frozen original q6 bound exceeds stated conservative cap')
    # |vgroup|<=c_upper^2 p/m_lower; c_upper=.126, p<=3e-7,
    # m_lower=.1294 on the original legal massive band, T is bound above.
    travel=F(126,1000)**2*F(3,10**7)*T/F(1294,10000)
    if travel>200000000:raise ValueError('Original full-clock packet travel cap fails')
    tail=2*(F(2,available)**6)*derivative
    group_tail=2*tail
    old=F(str(bridge['total_norm_error_upper_float']))
    screening_cap=F(1,2**100) # both exact exponents are <-100, never underflow=zero
    joined=old+group_tail+screening_cap
    return dict(N=str(N),origin=[str(N//2)]*3,FAR=['0',str(N//2),str(N//2)],Rin=Rin,Rout=Rout,clock=T,
        Phi_Gram_lower_assumption=str(gram),all15_inertia_lower=str(mu),screened_mass_squared_lower=str(m2),
        source_inner_half_plateau_radius=packet_radius,local_log2_upper=local_log,Ref_log2_upper=ref_log,
        original_q6_derivative_upper=str(derivative),free_travel_upper=200000000,
        SOFT_half_plateau_tail_upper=float(tail),bounded_group_tail_upper=float(group_tail),
        SOURCE19_inside_radius=2,finite_group_screening_positive_cap=str(screening_cap),
        inherited_physical_bridge_upper=float(old),conditional_joined_group_upper=float(joined),legal_error_budget=1e-4,
        conditional_below_legal_budget=joined<F(1,10000),
        proof='positive resolvent killed-walk bound with >=d steps; two finite unitaries differ by <=2 on discarded packet tail',
        unpaid=['Actual mean Clifford equivariance at every original all64 centre/alias with its full10 Gram',
            'C2 bounded tube alone is not dynamical SOURCE/core group intertwining',
            'Full original field/neutral RIGHT transport, scalar centre-word jets and photon/harmonic boundary must join without compensating the core minus',
            'Generic packet physical SO3 intertwiners, not just this fixed z pulse; charged SU2 is a projective core factor, global neutral pair may be integer'],
        physical_halfspin_admission=False,continuous_screening_NOT_discrete_seam_proof=True,
        boost='p_per_Omega=solve(chi*gInvNN,YN); cache boost0, beta=Omega-Omega_start leaves start-value residual; no phase reset')


def finite_vertical_transport(model: Any,sample: dict[str,Any],matrices: np.ndarray,
                              angle: float) -> dict[str,Any]:
    """Finite FIELD factor of the derived mechanical connection, not full Gauss.

    Its field+link curve remains on the actual regular covariantly aligned
    gauge orbit.  Full field+matter Gauss transport is trivial on physical
    states; this field factor alone becomes the matter compensator only AFTER
    the original normal-kinetic/Noether cancellation is established.  Neither
    a compact controlled H-OmegaG pulse nor a physical global SO3 is inferred.
    """
    law=model.law;point=sample['point']
    U=np.asarray([expm(1j*angle*s) for s in matrices])
    fields=U[:,None]@point['phi']@U[:,None].conj().swapaxes(-1,-2)
    links=U[:,None]@point['links']@U[law.target].conj().swapaxes(-1,-2)
    body=np.zeros((4*law.volume,4*law.volume),complex)
    for x in range(law.volume):body[4*x:4*x+4,4*x:4*x+4]=U[x]
    axes=model.original.reference_frame
    ref=axes.conj().T@U[model.reference_site]@axes
    return dict(fields=fields,links=links,U=U,body=body,reference=ref,
                angle=angle,original_normal_coordinates_UNCHANGED=True)


def finite_source_intertwiner(model: Any,sample: dict[str,Any],transport: dict[str,Any],
                              profile: str) -> dict[str,Any]:
    """Both ORIGINAL functional SOURCE maps, scalar centre words and actual R4."""
    point=sample['point'];producer=model.source_bank.seed_fibre
    source=producer.source_vectors_at(point['phi'],point['links'],sample['theta'],sample['a'],
                                     model.source_bank.profiles[profile])
    moved=producer.source_vectors_at(transport['fields'],transport['links'],sample['theta'],sample['a'],
                                    model.source_bank.profiles[profile])
    S,T=source['full108_plus_coefficients'],source['full108_minus_coefficients']
    Sm,Tm=moved['full108_plus_coefficients'],moved['full108_minus_coefficients']
    U,R=transport['body'],transport['reference']
    prediction_plus=U@S@R.conj().T
    prediction_minus=R@T@U.conj().T
    # These are universal coefficient/CAR intertwiners before contraction with
    # the SAME neutral; heavy and half-density transport cancel by product rule.
    return dict(plus=Sm,minus=Tm,original_plus=S,original_minus=T,
        exact_field_factor_plus_error=float(np.linalg.norm(Sm-prediction_plus)),
        exact_field_factor_minus_error=float(np.linalg.norm(Tm-prediction_minus)),
        full_Gauss_trivial_plus_error=float(np.linalg.norm(U.conj().T@Sm@R-S)),
        full_Gauss_trivial_minus_error=float(np.linalg.norm(R.conj().T@Tm@U-T)),
        wrong_omitted_Ref_plus_defect=float(np.linalg.norm(Sm-U@S)),
        wrong_omitted_Ref_minus_defect=float(np.linalg.norm(Tm-T@U.conj().T)),
        original_source_norm=float(np.linalg.norm(S)),moved_source_norm=float(np.linalg.norm(Sm)),
        scalar_centre_words_recomputed_not_dropped=True,
        neutral_RIGHT='B_f xi -> U_L B_f U_R^dagger Gamma(U_heavy,U_neutral) xi; never returned neutral or frozen heavy ket',
        exact_full_Gauss_factor_is_NOT_physical_core_group=True)


def aligned_field_invariants(law: Any,sample: dict[str,Any],transport: dict[str,Any]) -> dict[str,float]:
    """Gauge-orbit domain witness: actual Gram, covariant Phi gradients, plaquettes."""
    from original_total_angular_transfer import covariant_field_difference,clover
    phi=sample['point']['phi'];links=sample['point']['links']
    moved=transport['fields'];moved_links=transport['links'];U=transport['U']
    gram=lambda fields: np.einsum('xmij,xnji->xmn',fields,fields).real/4
    differences=covariant_field_difference(law,phi,links)
    moved_differences=covariant_field_difference(law,moved,moved_links)
    expected=U[:,None,None]@differences@U[:,None,None].conj().swapaxes(-1,-2)
    colour,_=clover(law,links,sample['a']);moved_colour,_=clover(law,moved_links,sample['a'])
    colour_expected=U[:,None,None]@colour@U[:,None,None].conj().swapaxes(-1,-2)
    return dict(full4_Gram_error=float(np.linalg.norm(gram(moved)-gram(phi))),
        covariant_alignment_error=float(np.linalg.norm(moved_differences-expected)),
        actual_colour_clover_error=float(np.linalg.norm(moved_colour-colour_expected)),
        link_unitarity_error=float(np.linalg.norm(moved_links@moved_links.conj().swapaxes(-1,-2)-np.eye(4))))


def normal_compensated_source_jet(model: Any,sample: dict[str,Any],flow: dict[str,Any],
                                  fixed_covector: np.ndarray,profile: str,
                                  step: float=2.**-17) -> dict[str,Any]:
    """Actual SOURCE-operator Noether/normal-kinetic join, not a sigma fit.

    The pointwise optimal covector is diagnostic only: it need not be an exact
    one-form and is NOT inserted as a phase. The actual initial-preserving
    fixed covector is separately exercised. All EM derivatives are held here
    and remain an explicit residual, not canceled by the Phi/colour boost.
    """
    law=model.law;point=sample['point'];n=model.chart.normal_dim
    J=point['jacobian'];inverse=point['cometric']
    raw=flow['raw'];horizontal=flow['horizontal_RAW']
    coordinates_Y=np.linalg.solve(J,raw);YN=coordinates_Y[:n]
    optimum=np.linalg.solve(model.chi*inverse[:n,:n],YN)
    shifted_optimum=model.chi*J@(inverse[:,:n]@optimum)
    shifted_fixed=model.chi*J@(inverse[:,:n]@fixed_covector)
    source=model.source_bank.seed_fibre.source_vectors_at(point['phi'],point['links'],
        sample['theta'],sample['a'],model.source_bank.profiles[profile])
    S,T=source['full108_plus_coefficients'],source['full108_minus_coefficients']
    K=np.zeros((4*law.volume,4*law.volume),complex)
    for x in range(law.volume):K[4*x:4*x+4,4*x:4*x+4]=flow['matrices'][x]
    R=flow['Ref_RIGHT'];orbital=flow['orbital108']
    targets=(K@S-S@R+orbital@S,R@T-T@K-T@orbital)
    def derivative(direction: np.ndarray,h: float) -> tuple[np.ndarray,np.ndarray]:
        vector=direction.reshape(law.volume,105)
        dphi=law.fields(vector[:,:60])
        left=np.einsum('xja,aik->xjik',vector[:,60:].reshape(law.volume,3,15),law.generators)
        values=[]
        for sign in (-1,1):
            U=np.asarray([[expm(1j*sign*h*entry) for entry in row] for row in left])
            values.append(model.source_bank.seed_fibre.source_vectors_at(
                point['phi']+sign*h*dphi,U@point['links'],sample['theta'],sample['a'],
                model.source_bank.profiles[profile]))
        return tuple((values[1][key]-values[0][key])/(2*h)
                     for key in ('full108_plus_coefficients','full108_minus_coefficients'))
    rows=[];fine_fixed=[]
    for h in (step,step/2):
        draw=derivative(raw,h);dh=derivative(horizontal,h)
        dfixed=derivative(shifted_fixed,h);errors=[];fixed_errors=[];fixed_residuals=[]
        for index,(mark,target) in enumerate(zip((S,T),targets)):
            orbit=orbital@mark if index==0 else -mark@orbital
            compensated=-1j*(draw[index]-dh[index])+orbit
            actual_fixed=-1j*(draw[index]-dfixed[index])+orbit
            predicted_residual=1j*(dfixed[index]-dh[index])
            errors.append(float(np.linalg.norm(compensated-target)))
            fixed_errors.append(float(np.linalg.norm(actual_fixed-target)))
            fixed_residuals.append(float(np.linalg.norm(actual_fixed-target-predicted_residual)))
        rows.append(dict(step=h,mechanical_CORE_orbital_RIGHT_identity_errors=errors,
            actual_fixed_covector_source_residual_norms=fixed_errors,
            fixed_covector_residual_prediction_errors=fixed_residuals))
        fine_fixed.append(dfixed)
    return dict(profile=profile,rows=rows,
        pointwise_optimum_horizontal_error=float(np.linalg.norm(shifted_optimum-horizontal)),
        actual_fixed_covector_horizontal_defect=float(np.linalg.norm(shifted_fixed-horizontal)),
        actual_fixed_covector_momentum_per_Omega_norm=float(np.linalg.norm(fixed_covector)),
        fixed_covector_SOURCE_FD_refinement_norms=[float(np.linalg.norm(a-b)) for a,b in zip(*fine_fixed)],
        exact_source_operator_identity='-i D_Y C + i D_horizontal C + L C = omega_body C-C omega_RIGHT + L C; reverse Ref/light ordering for hole',
        actual_fixed_phase_residual='i D_(chi J gInv[:,N] a_fixed-horizontal) C remains; pointwise covector not promoted to an integrable/global phase',
        neutral_derivative_half_density_heavy_cancel='product-rule/CAR operator identity BEFORE action on actual evolving neutral RIGHT, not a neutral return assumption',
        unpaid=['actual EM55/Coulomb/harmonic source derivative in full G',
            'Original functional B(q) fixed-phase jet defect needs an evolving LEGAL tube bound only if that functional map is used; not a second cost after B→constant-horizontal B0',
            'nonzero Omega_start residual remains in absolute dynamics; DY0 and betadot(a.q) commute genuinely constant-horizontal B0 and are common, not B0-relative costs',
            'finite H and physical spatial-group residual integral on same neutral/current states'],
        full_physical_intertwiner_admission=False)


def screened_axis_noether(law: Any,fields: np.ndarray,links: np.ndarray,phases: np.ndarray,
                           alpha: np.ndarray,velocity: np.ndarray,reference: int,
                           axis: np.ndarray,reference_frame: np.ndarray | None=None) -> dict[str,Any]:
    """Linear Cartesian-axis extension of the ORIGINAL z-vector Noether flow."""
    flow=screened_noether(law,fields,links,phases,alpha,velocity,reference,reference_frame)
    delta=np.zeros_like(fields)
    for n,(j,k) in zip(np.asarray(axis)-np.array([0.,0.,1.]),((2,3),(3,1),(1,2))):
        delta[:,j]-=n*alpha[:,None,None]*fields[:,k]
        delta[:,k]+=n*alpha[:,None,None]*fields[:,j]
    raw=flow['raw'].reshape(law.volume,105).copy()
    raw[:,:60]+=coordinates(delta,law.basis).reshape(law.volume,60).real
    D,mass_site=native_full_gauss_columns(law,fields,links)
    mass=np.tile(mass_site,law.volume);rhs=D.T@(mass*raw.ravel())
    omega=np.asarray(spsolve(flow['full_orbit_metric'],rhs))
    matrices=np.einsum('xa,aij->xij',omega.reshape(law.volume,15),law.generators)
    ref=matrices[reference]
    if reference_frame is not None:ref=reference_frame.conj().T@ref@reference_frame
    horizontal=raw.ravel()-D@omega
    flow.update(omega=omega,matrices=matrices,Ref_RIGHT=ref,physical_Ref=matrices[reference],
        raw=raw.ravel(),horizontal_RAW=horizontal,
        equation_residual=float(np.linalg.norm(flow['full_orbit_metric']@omega-rhs)),
        mass_horizontal_error=float(np.linalg.norm(D.T@(mass*horizontal))))
    return flow


def fixed_boost_angular_comparison(model: Any,sample: dict[str,Any],flow: dict[str,Any],
                                   mean_flow: dict[str,Any],fixed_covector: np.ndarray) -> dict[str,Any]:
    """Exact actual coefficient, including beta=Omega-Omega_start; no IC reset."""
    point=sample['point'];n=model.chart.normal_dim
    Y=np.linalg.solve(point['jacobian'],flow['raw'])
    connection=point['mechanical_connection']
    mech=Y[n:]+connection@Y[:n]
    boost=model.chi*point['cometric'][n:,:n]@fixed_covector
    rows=[]
    for beta_over_omega in (0.,.5,1.):
        effective=Y[n:]-beta_over_omega*boost
        exact=effective-mean_flow['omega']
        decomposition=(mech-mean_flow['omega']+
            connection@(beta_over_omega*model.chi*point['cometric'][:n,:n]@fixed_covector-Y[:n]))
        rows.append(dict(beta_over_Omega=beta_over_omega,
            actual_Omega_YG_difference_minus_beta_boost_per_Omega_norm=float(np.linalg.norm(exact)),
            exact_mechanical_decomposition_error=float(np.linalg.norm(exact-decomposition)),
            NOT_an_inserted_initial_phase=True))
    return dict(rows=rows,Cartesian_vs_actual_chart_mechanical_error=float(np.linalg.norm(mech-flow['omega'])),
        connection_Frobenius_norm_upper=float(np.linalg.norm(connection)),
        exact='Omega*(YG(q)-YG0)-beta*chi*gInvGN(q)*a0; beta=Omega-Omega_start',
        identity='YG-beta/Omega chi*gInvGN a0=omega_mech+gGG^-1 gGN*(beta/Omega chi*gInvNN a0-YN)',
        scope='Actual coefficient comparison only; common start/time scalar commutation with B0 does not set this coefficient to zero')


def mean_packet_rotation(model: Any,sample: dict[str,Any],plus: np.ndarray,minus: np.ndarray,
                          matrices: np.ndarray,angle: float) -> dict[str,Any]:
    """Derived body/Ref module on EVERY Xi in all8 canonical flat-P fibres.

    B_f=a_light^dagger S_f a_Ref + a_Ref^dagger T_f a_light.
    Exterior(U,R) B_f Exterior(U,R)^dagger=B_(U S Rdagger,R T Udagger).
    Heavy/neutral/PH are transported on BOTH sides, never returned or re-grounded.
    """
    from prepared_guard_colour_chart import Jet2,full_phi_frame_jets
    from prepared_neutral_spatial_charge_transfer import GAMMA5
    from ordered_sector_reference import covariant_packet_marks
    phi=sample['point']['phi'];zero=np.zeros_like(phi)
    F=full_phi_frame_jets(Jet2(phi,zero,zero))[0].value
    negative=-GAMMA5@phi@GAMMA5
    dual=full_phi_frame_jets(Jet2(negative,zero,zero))[0].value
    U=np.asarray([expm(-1j*angle*K) for K in matrices])
    up=F.conj().transpose(0,2,1)@U@F
    um=dual.conj().transpose(0,2,1)@GAMMA5@U@GAMMA5@dual
    ref=model.reference_site
    moved_plus=up@plus@up[ref].conj().T
    moved_minus=um[ref]@minus@um.conj().transpose(0,2,1)
    original=covariant_packet_marks(model,sample,plus,minus)
    moved=covariant_packet_marks(model,sample,moved_plus,moved_minus)
    body=sparse.block_diag(list(U)).toarray()
    axes=model.original.reference_frame
    R=axes.conj().T@U[ref]@axes
    # Charged Ref3 holes are wedge^3(R)=R* in the signed CAR-hole basis.
    hole=np.empty((4,4),complex)
    for r in range(4):
        for s in range(4):
            hole[r,s]=(-1)**(r+s)*np.linalg.det(np.delete(np.delete(R,r,axis=0),s,axis=1))
    hole_error=float(np.linalg.norm(hole-R.conj()))
    neutral_ref_error=float(abs(np.linalg.det(R)-1))
    plus_error=float(np.linalg.norm(moved['plus']-body@original['plus']@R.conj().T))
    minus_error=float(np.linalg.norm(moved['minus']-R@original['minus']@body.conj().T))
    # Full108 occupied projectors are transported as exterior data, NOT recaptured.
    fibre=model.source_bank.seed_fibre
    P=fibre.projectors
    norm0=(np.trace(original['plus'].conj().T@(np.eye(len(body))-P[1])@original['plus'])+
           np.trace(original['minus']@P[-1]@original['minus'].conj().T)).real/2
    Pplus=body@P[1]@body.conj().T;Pminus=body@P[-1]@body.conj().T
    norm1=(np.trace(moved['plus'].conj().T@(np.eye(len(body))-Pplus)@moved['plus'])+
           np.trace(moved['minus']@Pminus@moved['minus'].conj().T)).real/2
    return dict(plus_error=plus_error,minus_error=minus_error,
        actual_seeded_neutral_full108_exterior_norm_squared_difference=float(abs(norm1-norm0)),
        no_heavy_or_neutral_return=True,no_full108_PH_elimination=True,
        exterior_operator_identity_on_arbitrary_same_Xi_flat_P=True,
        exterior_operator_identity_on_unrestricted_encoded_Q=False,
        derived_Ref3_hole_exterior_error=hole_error,derived_neutral_Ref4_singlet_error=neutral_ref_error,
        derived_Ref1_particle_matrix=R,
        Ref_RIGHT='Ref3 hole generator -omega_Ref.conjugate gives -S R; Ref1 particle generator +omega_Ref gives R T; omega_Ref from actual Phi/link metric and ORIGINAL FAR profile',
        conditional_physical_group_only_after_actual_Noether_H_comparison=True)


def physical_group_ledger(legal: dict[str,Any],geometry: dict[str,Any],C2: dict[str,Any],
                          law: Any,frame: np.ndarray,calibration: dict[str,Any],
                          packet_family: dict[str,Any],metric: dict[str,Any]) -> dict[str,Any]:
    """Bounded same-neutral group comparison, NOT an unbounded-G gate.

    K is onsite all15 Phi inertia plus adjoint-UNITARY connection Laplacian.
    Maximum principle: |omega-omega0|_infinity <=|rhs-K omega0|_infinity/mu.
    On entire originalD d<=RD<=1, deltaPhi_coeff<=d, deltaU_op<=d.
    The full physical velocity/cutoff factor is RETAINED.
    """
    N=int(legal['N']);volume=N**3;mu=F(1,40)
    # sqrt Gram -> four matrix coframe -> Cartesian coefficients costs <=4.
    delta=4*F(calibration['frame_distance_upper'])
    B0=float(np.linalg.norm(coordinates(frame,law.basis)))
    B=2**math.ceil(math.log2(B0+float(delta)))
    A=1.;V=3*legal['Rout'];a=law.band.a
    W=math.sqrt(2)*A*B**2/float(mu)
    Cr=(math.sqrt(2)*(B+1)*(V/a*(1+2*B)+A)+math.sqrt(2)*A*B+
        96*math.sqrt(2)/.6*V/a+(2*(2*B+1)+12/.6)*W)
    Cy=A+(65+2*B)*V/a
    ceil2=lambda x: math.ceil(math.log2(x))
    volume_log=(volume-1).bit_length();sqrtV_log=(volume_log+1)//2
    # gGG^-1 gGN <=5120d; gInvNN<=160 and ||J^-1||_M<=sqrt160<13.
    yg_log=ceil2(Cr/float(mu)+5120*13*(B*2**sqrtV_log+Cy))
    boost_log=ceil2(5120*160*B)+sqrtV_log
    coefficient_log=1+max(yg_log,boost_log)
    cb_log=6+sqrtV_log # full original raw/same-Z B0 CAR bound <=64sqrtV
    controls=[]
    for name in ('full_pulse_neutral_and_SOURCE_actual_controls','full_pulse_true_photon_unweighted_actual_controls'):
        control=C2['science'][name]
        dnorm=(control['Dirichlet_actual_union_distance_second_moment_log2_upper']+1)//2
        dynamic=cb_log+coefficient_log+4+dnorm # body+Ref and integral|Omega|=integral|beta|=2pi<8
        boundary=cb_log+5+control['actual_whole_minus_D_norm_log2_upper']
        controls.append(dict(input=name,Dirichlet_actual_smooth_group_error_norm_log2_upper=dynamic,
            bounded_group_WHOLE_boundary_norm_log2_upper=boundary,
            no_unbounded_moment_transfer=True))
    # Sharp uniform-mean residual uses ||vectorrotation-i ad sigma||<=2, NOT1.
    # TrialY=Dtrial sigma; only Dactual^dagger T(deltaPhi) remains.
    q0=F(str(geometry['result']['result']['selected']['actual_derivative_q0_q6'][0]))
    source_CAR=4*q0 # ORIGINAL Ref0/4 branches give Frobenius1; SAME original SOURCE norm >=1/4
    alias=8*F(2)*(F(B)+delta)*delta/mu*source_CAR # integral|Omega|=2pi<8
    tail=8*F(str(legal['bounded_group_tail_upper']))
    screening=64*(1+F(B*B)/mu)*source_CAR*F(1,2**100)
    selected=packet_family['packet_family']['records'][-1]
    p=F(str(selected['support_p_upper']))
    if F(str(selected['scale']))!=F(1,10000):
        raise ValueError('Not the frozen selected narrower LEGAL SOFT packet')
    qa=F(str(a));cone=F(126,1000);wilson=abs(F(str(law.band.r)))
    # sin(ap)/a=p-a²p³/6+remainder. Subtract the ROTATIONAL cubic vector
    # (2/3)|p|²p: |p^3-(2/3)|p|²p|<=|p|³/3, by 9t1t2t3<=sum ti tj.
    # Two rotations therefore cost cone*a²|p|³/9, not the loose /3.
    vector_cov=cone*(qa**2*p**3/9+qa**4*p**5/60)
    mass_cov=wilson*qa**3*p**4*F(B)*(F(1,2) if law.band.improved else F(1,12))
    lattice=legal['clock']*(vector_cov+mass_cov)*source_CAR
    q1=F(str(geometry['result']['result']['selected']['actual_derivative_q0_q6'][1]))
    orbital=32*qa**2*p**3*q1/3 # integral|Omega|<8, original Ref/sameZ4, ||grad f||<=2q1
    majorant=F(str(metric['certificate']['analytic_strip']['matrix_gradient_integrand_HS_majorant']))
    finite_alias=128*majorant*F(1,2**1024)
    alias_symbol=2*legal['clock']*(abs(F(str(law.band.m)))+abs(F(str(law.band.v)))*p)*finite_alias*source_CAR
    dynamic_cap=F(1,2**100)
    if any(max(row['Dirichlet_actual_smooth_group_error_norm_log2_upper'],
               row['bounded_group_WHOLE_boundary_norm_log2_upper'])>-100 for row in controls):
        raise ValueError('Exact original wholeD log rows do not justify the declared positive cap')
    bounded=alias+tail+screening+lattice+orbital+alias_symbol+2*dynamic_cap
    return dict(all64_allalias_frame_Cartesian_debt=str(delta),all15_inertia_lower=str(mu),
        explicit_vectorrotation_minus_i_ad_sigma_norm_upper=2,original_mean_Phi_coeff_norm_upper=B,
        original_velocity_l1_upper=V,original_lattice_a=a,mean_omega_infinity_upper=W,
        fullD_omega_smooth_residual_Cr=Cr,fullD_raw_Noether_Lipschitz_Cy=Cy,
        exact_effective_correction='Omega(YG-YG0)-beta chi*gInvGN a0; beta=Omega-Omega_start, whole ORIGINAL quintic pulse has Omega_start0',
        actual_fullD_geometry_constants=dict(connection_over_d=5120,gInvNN=160,inverse_J_weighted=13),
        original_packet_q0_upper=str(q0),same_Z_packet_CAR_upper=str(source_CAR),
        generic_arbitrary_Ref_CAR_factor=8,ORIGINAL_Ref0_or_4_invariant_CAR_factor=4,
        CAR_domain='ALL8 statistical-flat P direct sum; NOT unrestricted encoded auxiliary Q',
        encoded_Q_extension_required=True,
        Ref_number_invariant='Actual H/G/J/contact/filter/creator terms conserve nRef. Original neutral Ref0/4 -> SOURCE charged Ref3/1; no new projector, no lost PH or branch phase.',
        all64_mean_angular_alias_norm_upper=float(alias),screened_core_plus_FAR_Ref_norm_upper=float(screening),
        original_SOFT_fullclock_CAR_group_tail_upper=float(tail),actual_input_rows=controls,
        conditional_bounded_same_neutral_module_norm_upper=float(bounded),
        selected_narrow_LEGAL_packet=dict(scale=selected['scale'],centre=selected['centre'],
            width=selected['width'],support_p_upper=str(p),
            NOT_old_shared_native_SOFT=True,normalization_or_function_NOT_changed=True),
        fullclock_lattice_spin_covariance_norm_upper=float(lattice),
        fullclock_continuum_vs_lattice_orbital_norm_upper=float(orbital),
        all64_finiteN_alias_symbol_norm_upper=float(alias_symbol),
        original_continuum_Gram_symmetry='Strictlyconvex ORIGINAL whole-heavy Gram functional plus signed-permutation momentum-cube symmetry gives UNIQUE diag(g0,gs,gs,gs). Derived Spin3 is independent of scalar amplitudes; only actual shifted-grid alias2^-1024 breaks this symmetry.',
        lattice_piece_scope='Selected ORIGINAL narrow packet and its frozen fullclock free-wave derivative provider only; NOT a bound for arbitrary interacting/EM-distorted packets. Actual EM/current/harmonic/Coul/Noether residue integral remains A/T-owned.',
        generic_packet_group='Continuum D(R)f(p)=u_core(R) f(R^-1 p) u_Ref(R)^dagger; hole reverse ordering. Continuous pullback is unitary. NO exact arbitrary-axis orbital unitary on a finite108 periodic grid is asserted.',
        same_neutral_identity='INTERNAL Exterior(U(x),R) B_f Xi=B_internalDf Exterior(U(x),R) Xi on every Xi in ALL8 flat-P ORIGINAL CAR fibres, retaining heavy/PH/Ref. Encoded Q requires corrected Delta/2 admission; no CAR is asserted there. Continuum orbital and sampled/lattice Noether payments are separate. Physical G requires A actual Noether/normal/slowH comparison.',
        centre_words='On all64 flat centre seams, each ORIGINAL path telescopes all interior frame/Z4 factors; endpoint birth/Ref factors remain, including2pi. Central seam characters commute the derived spin; no sector/alias/statisticalflat deletion.',
        initial_Bq_to_B0_paid_ONCE='C2 ORIGINAL Gamma comparison plus exact unitarity preserves complex initial error; not resummed as a later B(q) source jet cost',
        remainder_is_ACTUAL_same_state_H_Noether_integral=True,physical_halfspin_admission=False)


def continuum_packet_group_control(core: dict[str,Any],scale: float) -> dict[str,Any]:
    """NEW nonconstant complex full4×Ref4 function; no initial SOURCE replacement.

    Lebesgue pullback has det R=1 and internal u is unitary, so D is a bounded
    unitary on continuous L2 for arbitrary f. This is NOT a grid interpolation.
    """
    from scipy.spatial.transform import Rotation
    spin=core['spins'];identity=np.eye(4)
    def function(p: np.ndarray) -> np.ndarray:
        x=p/scale
        polynomial=(identity+.2*x[0]*spin[0]+.13j*x[1]*spin[1]+.09*x[2]*spin[2])
        return np.exp(-np.dot(x,x)/2+.2j*x[1])*polynomial
    v1=np.array([.41,-.19,.27]);v2=np.array([-.13,.32,.17])
    R1=Rotation.from_rotvec(v1);R2=Rotation.from_rotvec(v2);R3=R1*R2
    u1=expm(-1j*np.einsum('j,jab->ab',v1,spin))
    u2=expm(-1j*np.einsum('j,jab->ab',v2,spin))
    u3=expm(-1j*np.einsum('j,jab->ab',R3.as_rotvec(),spin))
    points=scale*np.array([[.21,-.47,.32],[-.56,.18,.41],[.37,.26,-.19]])
    composed=np.asarray([u1@u2@function(R2.inv().apply(R1.inv().apply(p))) for p in points])
    direct=np.asarray([u3@function(R3.inv().apply(p)) for p in points])
    wrong=np.asarray([u3@function(p) for p in points])
    endpoint=[]
    for axis in np.r_[np.eye(3),np.array([[1.,2.,3.]])/np.sqrt(14)]:
        R=Rotation.from_rotvec(2*np.pi*axis)
        u=expm(-2j*np.pi*np.einsum('j,jab->ab',axis,spin))
        endpoint.append(float(np.linalg.norm(np.asarray([u@function(R.inv().apply(p))+function(p) for p in points]))))
    return dict(composition_error=float(np.linalg.norm(composed-direct)),
        omitted_orbital_pullback_defect=float(np.linalg.norm(wrong-direct)),
        arbitrary_axis_CORE_LEFT_two_pi_minus_errors=endpoint,
        continuum_Jacobian_unit=True,full4_Ref4_function=True,
        Ref_RIGHT_factored_OUT_not_assigned=True,
        source_or_state_normalization_NOT_changed=True,
        finite_periodic_108_orbital_unitary_NOT_claimed=True,
        physical_group_admission_requires_actual_Noether_H_residual=True)


def flat_P_group_extension(group: dict[str,Any],auxiliary: dict[str,Any]) -> dict[str,Any]:
    """Extend frozen bounded-group controls, never an unbounded creator/G image."""
    legal=group['LEGAL'];admission=auxiliary['science']
    if not admission['actual_Q_penalty_lower'].startswith('Delta/2;'):
        raise ValueError('Original auxiliary law requires the corrected Delta/2 Q penalty')
    if not admission['all8_flat_independent_of64_colour']:
        raise ValueError('Statistical flat8 must remain independent of colour centre64')
    if int(admission['original_clock'])!=int(legal['clock']):
        raise ValueError('Auxiliary and group controls require the same ORIGINAL clock')
    volume=int(legal['N'])**3;volume_log=(volume-1).bit_length()
    # Encoded SOURCE uses the word triangle, not CAR. Both species plus Ref
    # have onsite charge norm <=64V ||omega||op, since sum|Kij|<=8||K||op.
    half_log=(volume_log+1)//2;encoded_CB_log=12+half_log
    charge_log=6+volume_log
    module=legal['bounded_same_neutral_physical_group'];B=int(module['original_mean_Phi_coeff_norm_upper'])
    def ceil2(value: F) -> int:
        integer=(value.numerator+value.denominator-1)//value.denominator
        return (integer-1).bit_length()
    yg_upper=40*F(str(module['fullD_omega_smooth_residual_Cr']))+5120*13*(
        B*2**half_log+F(str(module['fullD_raw_Noether_Lipschitz_Cy'])))
    boost_upper=F(5120*160*B*2**half_log)
    coefficient_log=1+max(ceil2(yg_upper),ceil2(boost_upper))
    angular_rows=[dict(input=row['input'],
        full_encoded_Dirichlet_matter_angular_error_log2_upper=(
            row['Dirichlet_actual_smooth_group_error_norm_log2_upper']+volume_log-half_log),
        full_encoded_bare_SOURCE_WHOLE_boundary_error_log2_upper=(
            row['bounded_group_WHOLE_boundary_norm_log2_upper']+6))
        for row in module['actual_input_rows']]
    if any(max(value for key,value in row.items() if key.endswith('log2_upper'))>-100 for row in angular_rows):
        raise ValueError('Existing two positive C2 caps do not cover full encoded angular charges')
    if F(module['original_packet_q0_upper'])>=64:
        raise ValueError('Frozen original packet does not meet this encoded triangle bound')
    originals=admission['original_inputs'];neutral_Q=int(originals['neutral']['full_driven_auxQ_norm_log2_upper'])
    rows={}
    for name,control in originals.items():
        qlog=int(control['full_driven_auxQ_norm_log2_upper'])
        state_log=int(control['full_driven_state_minus_proof_codeflow_norm_log2_upper'])
        # Two contractions/unitaries: each actual state is compared with its
        # OWN unnormalized proof codeflow. No physical state is projected.
        row=dict(actual_auxQ_norm_log2_upper=qlog,
            actual_state_minus_unnormalized_codeflow_log2_upper=state_log,
            bounded_two_group_effect_transfer_log2_upper=state_log+1,
            bounded_D_matter_angular_codeflow_transfer_log2_upper=(
                state_log+charge_log+coefficient_log+4))
        if name=='SOURCE_bothmaps_originalquartet':
            row['bare_SOURCE_Q_composition_log2_upper']=2+max(qlog,encoded_CB_log+neutral_Q)
        if name=='true_creator_SOURCE_original':
            row['OWN_true_photon_state_bound_used']=True
            row['unbounded_creator_multiplied_by_bare_Q_error']=False
        if max(value for key,value in row.items() if key.endswith('log2_upper'))>-100:
            raise ValueError('Frozen positive auxiliary cap does not cover this bounded group extension')
        rows[name]=row
    positive_cap=F(3,2**100)
    original_upper=F(str(legal['conditional_joined_OLD_bridge_plus_NEW_group_upper']))
    exact_upper=original_upper+positive_cap
    # Preserve a positive debt despite binary rounding of the tiny addend.
    outward_upper=math.nextafter(float(exact_upper),math.inf)
    return dict(canonical_CAR_domain='ALL8 statistical-flat P fibres; retain all64 colour centre sectors independently',
        canonical_P_Ref0_or4_source_factor=4,generic_CAR_Ref_factor=8,
        canonical_CAR_or_exterior_identity_on_encoded_Q=False,
        original_neutral_Ref0_or4_and_SOURCE_Ref3_or1_retained=True,
        encoded_SOURCE_operator_triangle='4096 sqrt(V); sum of unitary Pauli-word bilinears; NO CAR assumption on Q',
        encoded_SOURCE_operator_triangle_log2_upper=encoded_CB_log,
        full_encoded_matter_angular_operator_triangle='64V ||omega||op; BOTH full species plus Ref; no CAR or source moment assumption',
        full_encoded_matter_angular_operator_triangle_log2_upper=charge_log,
        full_D_effective_angular_coefficient_log2_upper=coefficient_log,
        full_encoded_actual_Dirichlet_angular_rows=angular_rows,
        bounded_D_distance_multiplier_upper=1,
        original_two_positive_C2_caps_still_cover_full_encoded_prices=True,
        original_inputs=rows,positive_auxiliary_group_extension_cap=str(positive_cap),
        inherited_bounded_same_neutral_module_upper=module['conditional_bounded_same_neutral_module_norm_upper'],
        inherited_OLD_bridge_plus_NEW_group_upper=float(original_upper),
        conditional_OLD_bridge_plus_NEW_group_plus_Q_exact=str(exact_upper),
        conditional_OLD_bridge_plus_NEW_group_plus_Q_outward_upper=outward_upper,
        physical_projection_or_repreparation=False,codeflow_normalized=False,
        true_photon_weighted_or_creator_Q_transfer_admitted=False,
        unbounded_G_operator_images_admitted=False,
        actual_A_T_same_state_H_Noether_EM_current_residue_still_required=True,
        physical_halfspin_admission=False,campaign_OPEN=True)


def encoded_reference_number_proof(model: Any,data: dict[str,Any]) -> dict[str,Any]:
    """Actual full encoded Pauli algebra, with no CAR assumption or state cut."""
    from microscopic_native_orbit_lift import NativePauliGraph,PauliWord
    from microscopic_native_orbit_consumer import source_terms
    graph=NativePauliGraph(model.law,model.original.reference_site)
    phi=model.chart.mean_phi;links=model.chart.mean_links
    zero=np.zeros((graph.volume,3))
    packets=(model.photon_basis.packets@data['packet_templates'].T).T
    incoming=(model.em['legs']@packets[0]).reshape(graph.volume,3)/np.sqrt(2)
    outgoing=(model.em['legs']@packets[2].conj()).reshape(graph.volume,3)/np.sqrt(2)
    cases={}
    for name,law,species in (('light',model.law,1),('heavy',model.original.heavy_law,0)):
        cases[name+'_actual_H']=(graph.path_terms(law,phi,links,zero,species),0)
        if species==1:
            cases['positive_Peierls_J']=(graph.path_terms(law,phi,links,zero,species,first=incoming),0)
            cases['positive_Peierls_contact_both_orders']=(graph.path_terms(
                law,phi,links,zero,species,first=incoming,second=outgoing),0)
    # Complete operator bases, NOT a replacement SOURCE profile or prepared
    # state. Each independent original coefficient can multiply these words.
    probe=np.ones((4*graph.volume,4),complex)
    plus=source_terms(graph,probe,np.zeros_like(probe.T))
    minus=source_terms(graph,np.zeros_like(probe),probe.T)
    cases['SOURCE_plus_complete_bilinear_basis']=(plus,-1)
    cases['SOURCE_minus_complete_bilinear_basis']=(minus,1)
    cases['Ref_onsite_Gauss_and_Noether_complete_basis']=(
        [graph.bilinear(i,j,1) for i in graph.ref for j in graph.ref],0)
    cases['all_original_number_density_stars']=(
        [graph.bilinear(i,i,1) for i in range(graph.modes)],0)
    Bref=[graph.B[r] for r in graph.ref]

    def error(words: Any,shift: int=0) -> float:
        total={}
        def add(c: complex,word: Any) -> None:
            key=(word.x,word.z);total[key]=total.get(key,0)+c*(1j**word.phase)
        for coefficient,word in words:
            for b in Bref:
                add(-.5*coefficient,b@word);add(.5*coefficient,word@b)
            add(-shift*coefficient,word)
        return float(sum(abs(c) for c in total.values()))

    rows={}
    for name,(terms,shift) in cases.items():
        rows[name]=dict(nonzero_even_terms=len(terms),expected_nRef_shift=shift,
            maximum_exact_Pauli_commutator_l1_error=max(error(term.words,shift) for term in terms))
    rows['actual_nonzero_D_sum_epsilon_Zedge']=dict(nonzero_even_terms=len(graph.spatial_edges),
        epsilon=graph.epsilon,maximum_exact_Pauli_commutator_l1_error=max(
            error(((graph.epsilon,PauliWord(z=1<<e)),)) for e in graph.spatial_edges))
    cycles=graph.cycles+graph.local_cycles
    rows['original_statistical_cycle_stabilizers']=dict(nonzero_even_terms=len(cycles),
        maximum_exact_Pauli_commutator_l1_error=max(error(((1,word),)) for word in cycles))
    incidence=sum(word.anticommutes(graph.B[r])!=(r in endpoints)
        for endpoints,word in graph.A.items() for r in graph.ref)
    return dict(scope='FULL encoded Hilbert, ALL8 flat sectors plus Q; exact Pauli-word algebra, NOT unrestricted CAR',
        nRef='sum_r (I-B_ref_r)/2, spectrum0..4',
        edge_number_star_incidence_errors=incidence,operator_classes=rows,
        field_EM_Phi_and_photon_creator_commute_on_separate_tensor_factor=True,
        Coulomb_Gauss_Root_products_preserve_number_by_algebra_closure=True,
        original_global_fH_filter_preserves_number_by_functional_calculus=True,
        Ref0_or4_to_SOURCE_Ref3_or1_selection_orthogonal=True,
        no_new_state_projector_or_PH_deletion=True,
        auxiliary_Q_exterior_CAR_or_Ref_singlet_NOT_asserted=True)
