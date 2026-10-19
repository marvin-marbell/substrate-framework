"""Original ordinary-T6 photon/current contour and required OUT creator transfer.

All photon Weyl products act on the whole supplied mode space. Full light
matrices are never declared invariant SOURCE subspaces. The long Noether
rotation is a different experiment; its literal orbital current is exposed
only as a derivative/continuity control, not substituted in ordinary T6.
"""
from __future__ import annotations
from dataclasses import dataclass
from fractions import Fraction
import math
from typing import Any
import numpy as np
from flint import arb,ctx
from original_total_angular_transfer import scalar_orbital
from functional_source_photon_bank import Full108ReferenceMarks


def directed_currents(law: Any,fields: np.ndarray,links: np.ndarray,A: np.ndarray,
                      velocity: np.ndarray,Omega: float=0.) -> dict[str,Any]:
    """Positive link derivative of actual H-Omega scalar orbital, before P_T.

    With D=target-source and div=-D*, i[H,n_x]=-div(partial_A H).
    Every range-two path contributes at BOTH edges and telescopes internally.
    The body/Ref mechanical matrix has zero A derivative. EM differential
    Noether and rho.lambda_C are distinct frame terms, not erased here.
    """
    n=4*law.volume;edges=3*law.volume
    light=np.zeros((edges,n,n),complex);orbital=np.zeros_like(light)
    for x,y,block,path in law.terms(fields,links,A):
        sx,sy=slice(4*x,4*x+4),slice(4*y,4*y+4)
        for site,axis in path:
            j=3*site+axis
            light[j,sx,sy]+=1j*block;light[j,sy,sx]-=1j*block.conj().T
    for x in range(law.volume):
        for axis in range(3):
            y=int(law.target[x,axis]);sx,sy=slice(4*x,4*x+4),slice(4*y,4*y+4)
            b=(velocity[x,axis]+velocity[y,axis])*links[x,axis]*np.exp(1j*A[x,axis])/(4j*law.band.a)
            j=3*x+axis;orbital[j,sx,sy]+=1j*b;orbital[j,sy,sx]-=1j*b.conj().T
    h=law.onebody(fields,links,A);o=scalar_orbital(law,links,A,velocity)
    K=h-Omega*o;current=light-Omega*orbital
    div=np.zeros((law.volume,n,n),complex)
    for x in range(law.volume):
        for axis in range(3):
            y=int(law.target[x,axis]);div[x]+=current[3*x+axis];div[y]-=current[3*x+axis]
    errors=[]
    for x in range(law.volume):
        density=np.zeros((n,n),complex);density[4*x:4*x+4,4*x:4*x+4]=np.eye(4)
        errors.append(float(np.linalg.norm(1j*(K@density-density@K)+div[x])))
    return dict(H=h,scalar_orbital=o,J_original=light,J_orbital=orbital,
                J_pulse=current,continuity_errors=errors,
                rho='n_light-2+(Nref-4)delta_FAR; Ref occupation conserved by ordinary H',
                convention='D=target-source; div=-Dstar=outgoing-incoming; rho_dot=-div Jpositive',
                photon_J_not_replaced=True,full_EM_Noether_frame_terms_not_erased=True)


@dataclass(frozen=True)
class WeylContour:
    """Exact free-HEM contour characteristic in all represented coexact modes.

    A=legs*y, y=(a+a†)/sqrt2. A vertex phase is exp(i phase.y).
    W(z)W(w)=exp(-i Im(z†w)) W(z+w), z=phase/sqrt2.
    The external legs are literal creation/annihilation vectors, not scalars.
    """
    frequencies: np.ndarray
    incoming: np.ndarray
    outgoing: np.ndarray

    def product(self,vertices: list[tuple[np.ndarray,float]]) -> dict[str,complex|np.ndarray]:
        z=np.zeros_like(self.incoming,dtype=complex);phase=0.
        for frequency,time in vertices:
            w=np.asarray(frequency,dtype=complex)*np.exp(1j*self.frequencies*time)/math.sqrt(2)
            phase-=float(np.vdot(z,w).imag);z+=w
        vacuum=complex(np.exp(-.5*float(np.vdot(z,z).real)+1j*phase))
        overlap=complex(np.vdot(self.outgoing,self.incoming))
        return dict(vacuum=vacuum,one_photon=vacuum*(overlap-np.vdot(self.outgoing,z)*np.vdot(z,self.incoming)),
                    net_displacement=z,external_overlap=overlap)

    def split(self,left_vertices: list[tuple[np.ndarray,float]],
              right_vertices: list[tuple[np.ndarray,float]]) -> dict[str,complex|np.ndarray]:
        """Exact <0|WL a(gout) WR a†(gin)|0>; creator is NOT transported.

        Only the right displacement contracts with the outgoing annihilator.
        The combined displacement contracts with the incoming creator.
        Thus neutral-backward/charged-forward cross terms survive even when
        incoming/outgoing modes are orthogonal.
        """
        left=self.product(left_vertices);right=self.product(right_vertices)
        zl=np.asarray(left['net_displacement']);zr=np.asarray(right['net_displacement'])
        vacuum=complex(left['vacuum'])*complex(right['vacuum'])*complex(np.exp(-complex(np.vdot(zl,zr))))
        value=vacuum*(complex(np.vdot(self.outgoing,self.incoming))-complex(np.vdot(self.outgoing,zr))*complex(np.vdot(zl+zr,self.incoming)))
        return dict(vacuum=vacuum,one_photon=complex(value),left_displacement=zl,right_displacement=zr)

    def leading_two_leg(self,vertices: list[tuple[np.ndarray,float]]) -> complex:
        z=sum((np.asarray(k,dtype=complex)*np.exp(1j*self.frequencies*t)/math.sqrt(2)
               for k,t in vertices),np.zeros_like(self.incoming,dtype=complex))
        return complex(-np.vdot(self.outgoing,z)*np.vdot(z,self.incoming))


def full_car_transition(left: np.ndarray,right: np.ndarray,
                        first: np.ndarray,second: np.ndarray|None=None) -> complex:
    """Full Slater transition of one/two bilinears, with evolving neutral bra.

    Both complete occupied frames enter the transition determinant. No
    neutral return, rank-four particle closure or frozen occupied projector.
    Invertible overlap is required by this evaluation, not an invariance claim.
    """
    M=left.conj().T@right;det=complex(np.linalg.det(M))
    R=right@np.linalg.solve(M,left.conj().T)
    a=complex(np.trace(R@first))
    if second is None:return det*a
    b=complex(np.trace(R@second))
    return det*(a*b+np.trace(R@first@second)-np.trace(R@first@R@second))


def distinct_block_car_Gram(sources: list[np.ndarray],P_L: np.ndarray,
                            P_R: np.ndarray) -> dict[str,Any]:
    """Reference-only Wick contraction for B=a_L† S a_R on disjoint blocks.

    P_L/P_R are the declared non-ground Slater covariances. This does not
    assert that interacting Coulomb/particle-hole evolution stays Slater.
    """
    direct=np.array([[np.trace(P_R@a.conj().T@b) for b in sources] for a in sources])
    blocked=np.array([[np.trace(P_R@a.conj().T@P_L@b) for b in sources] for a in sources])
    return dict(D=direct-blocked,unblocked=direct,Pauli_blocking=blocked,
        wrong_plus_adverse_Gram=direct+blocked,
        formula='Tr[P_R S_j† (1-P_L) S_k]',
        disconnected_LR_mean_zero=True,interacting_state_not_assumed_Slater=True)


def joint_receiver_Gram(D: np.ndarray,N: np.ndarray,
                        photon_overlap: np.ndarray) -> np.ndarray:
    """Exact full-state CCR split, including off-diagonal photon coherence.

    D_jk=<Psi|B_j† B_k|Psi>;
    N_jk=<Psi|B_j† a†(gout_k) a(gout_j) B_k|Psi>.
    No vacuum cumulant subtraction, Gaussian-after-U or matter closure.
    """
    return photon_overlap*D+N


def complex_function_derivative(function: Any,direction: np.ndarray) -> Any:
    """Complex derivative of the Gaussian-DIVIDED factor, including Weyl phase.

    The supplier wave derivative also differentiates exp(-y²/2). Its -g.y
    cancels the oscillator's +g.y; it must NOT be counted as photon number.
    """
    from prepared_neutral_noether_gaussian_partners import FiniteGaussianFunction
    from prepared_neutral_weyl_hard_fock import GaussianFunction,ProjectionPolynomial
    direction=np.asarray(direction,complex);terms=[]
    for term in function.terms:
        coefficients=dict(term.polynomial.directional_derivative(direction).terms)
        phase=1j*(direction@term.phase)
        for key,value in term.polynomial.terms.items():
            coefficients[key]=coefficients.get(key,0j)+phase*value
        terms.append(GaussianFunction(ProjectionPolynomial(term.polynomial.projections,coefficients),term.phase))
    return FiniteGaussianFunction(tuple(terms))


def gamma_annihilator_Gram(bank: Any,packets: np.ndarray) -> np.ndarray:
    """Closed Gaussian moments of a(g)Gamma, with cancellation-safe sums.

    Gamma=i sum_r[(F_r.y)sin(P_r.y)-.5(F_r.P_r)cos(P_r.y)].
    Uses its actual binary64 coefficients, not 162² expanded polynomials.
    """
    import mpmath as mp
    with mp.workdps(60):
        P=[[mp.mpf(float(x)) for x in row] for row in bank.coefficients['phases']]
        F=[[mp.mpf(float(x)) for x in row] for row in bank.coefficients['coexact']]
        G=[[mp.mpc(complex(x).conjugate()) for x in row] for row in packets]
        dot=lambda a,b:mp.fsum(x*y for x,y in zip(a,b))
        size=len(P)
        PP=[[dot(a,b) for b in P] for a in P]
        FP=[[dot(a,b) for b in P] for a in F]
        FF=[[dot(a,b) for b in F] for a in F]
        # The constant in the supplied Gamma is the original computed IEEE dot.
        original_dot=[mp.mpf(float(a@b)) for a,b in zip(bank.coefficients['coexact'],bank.coefficients['phases'])]
        slopes=[[dot(p,g) for p in P] for g in G]
        alpha=[[dot(f,g)+original_dot[r]*slopes[j][r]/2 for r,f in enumerate(F)] for j,g in enumerate(G)]
        out=np.empty((2,2),complex)
        for j in range(2):
            for k in range(j,2):
                terms=[]
                for r in range(size):
                    for t in range(size):
                        E=mp.exp(-(PP[r][r]+PP[t][t])/4)
                        sh=mp.sinh(PP[r][t]/2);ch=mp.cosh(PP[r][t]/2)
                        ys=(FP[t][r]*ch-FP[t][t]*sh)/2
                        sy=(FP[r][t]*ch-FP[r][r]*sh)/2
                        yy=FF[r][t]*ch/2-(FP[r][r]*FP[t][r]+FP[r][t]*FP[t][t])*ch/4+(FP[r][r]*FP[t][t]+FP[r][t]*FP[t][r])*sh/4
                        terms.append(E*(alpha[k][r].conjugate()*alpha[j][t]*sh+
                            alpha[k][r].conjugate()*slopes[j][t]*ys+
                            slopes[k][r].conjugate()*alpha[j][t]*sy+
                            slopes[k][r].conjugate()*slopes[j][t]*yy)/2)
                out[j,k]=complex(mp.fsum(terms));out[k,j]=out[j,k].conjugate()
        return out


def gamma_receiver_cross(bank: Any,packets: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    """<Gamma|a(g) f_n> and <f_m|a(g) Gamma>, without expanded cancellation."""
    import mpmath as mp
    hard=bank.hard;size=len(bank.coefficients['phases']);count=hard.dimension+1+size
    left=np.zeros((2,count),complex);right=np.zeros_like(left)
    with mp.workdps(60):
        P=[[mp.mpf(float(x)) for x in row] for row in bank.coefficients['phases']]
        F=[[mp.mpf(float(x)) for x in row] for row in bank.coefficients['coexact']]
        G=[[mp.mpc(complex(x).conjugate()) for x in row] for row in packets]
        U=[[mp.mpc(complex(x).conjugate()) for x in row] for row in hard.packets.T]
        dot=lambda a,b:mp.fsum(x*y for x,y in zip(a,b))
        PP=[[dot(a,b) for b in P] for a in P];FP=[[dot(a,b) for b in P] for a in F]
        original_dot=[mp.mpf(float(a@b)) for a,b in zip(bank.coefficients['coexact'],bank.coefficients['phases'])]
        mean=1j*mp.fsum((FP[r][r]-original_dot[r])*mp.exp(-PP[r][r]/4)/2 for r in range(size))
        for j,g in enumerate(G):
            pg=[dot(p,g) for p in P];fg=[dot(f,g) for f in F]
            for t in range(size):
                sums_left=[];sums_right=[]
                for r in range(size):
                    E=mp.exp(-(PP[r][r]+PP[t][t])/4)
                    sh=mp.sinh(PP[r][t]/2);ch=mp.cosh(PP[r][t]/2)
                    sums_left.append(E*((FP[r][r]-original_dot[r])*ch-FP[r][t]*sh))
                    sums_right.append(E*(fg[r]*sh+pg[r]*(FP[r][t]*ch+(original_dot[r]-FP[r][r])*sh)/2))
                left[j,hard.dimension+1+t]=complex(-1j*pg[t]*mp.fsum(sums_left)/(2*mp.sqrt(2)))
                right[j,hard.dimension+1+t]=complex(1j*mp.fsum(sums_right)/mp.sqrt(2))
            for index,label in enumerate(hard.labels):
                if sum(label)!=1:continue
                u=U[label.index(1)]
                terms=[mp.exp(-PP[r][r]/4)*(fg[r]*dot(u,P[r])+pg[r]*dot(u,F[r])+
                    pg[r]*dot(u,P[r])*(original_dot[r]-FP[r][r])/2) for r in range(size)]
                right[j,index]=complex(1j*mp.fsum(terms)/2)
                left[j,index]=complex(mean.conjugate()*dot(g,[x.conjugate() for x in u]))
    return left,right


def native_annihilator_cross(model: Any,data: dict[str,Any],packets: np.ndarray,
                              gamma_left: np.ndarray,gamma_right: np.ndarray) -> np.ndarray:
    """Full <f_m|a(g_j)f_n> for the unchanged hard/Gamma/81-Weyl library."""
    hard=model.noether_bank.hard;h=hard.dimension;phases=model.noether_bank.coefficients['phases']
    count=h+1+len(phases);cross=np.zeros((2,count,count),complex)
    coefficients=np.asarray(data['packet_templates'][1:3])
    for j in range(2):
        cross[j,:h,:h]=np.einsum('i,ijk->jk',coefficients[j].conj(),hard.creation.conj().transpose(0,2,1))
        for r,p in enumerate(phases):
            exponential=math.exp(-float(p@p)/4);slope=p@packets[j].conj()
            projections=hard.packets.conj().T@p
            for m,label in enumerate(hard.labels):
                degree=sum(label)
                if degree%2==0:
                    moment=complex(exponential)
                    for mode,n in enumerate(label):moment*=(1j*projections[mode]/math.sqrt(2))**n/math.sqrt(math.factorial(n))
                    cross[j,m,h+1+r]=slope*moment/math.sqrt(2)
                if degree!=2:continue
                value=0j
                for mode,n in enumerate(label):
                    if not n:continue
                    remaining=list(label);remaining[mode]-=1;other=remaining.index(1)
                    value+=coefficients[j,mode].conjugate()*math.sqrt(n)*(hard.packets[:,other]@p)*exponential/math.sqrt(2)
                cross[j,h+1+r,m]=value
        cross[j,h,:]=gamma_left[j];cross[j,:,h]=gamma_right[j]
    return cross


def full_state_receiver_Born(raw_CTC: np.ndarray,G: np.ndarray,
                              original_input_normalizer: float) -> dict[str,Any]:
    """r_j=<a†gout_j B_j U_N Xi|U_C a†gin B Xi>; SAME-state G required.

    The caller supplies the original fixed input normalizer, never a new
    bank/window/receiver Z. Valid for full interacting kernels, not just Slater.
    """
    values,vectors=np.linalg.eigh(G)
    inverse_root=(vectors*values**(-.5))@vectors.conj().T
    amplitudes=inverse_root@raw_CTC/original_input_normalizer
    probabilities=abs(amplitudes)**2
    return dict(raw_joint_CTC_numerator=raw_CTC,one_GLOBAL_joint_inverse_root=inverse_root,
        joint_Gram_eigenvalues=values,
        normalized_joint_receiver_amplitudes=amplitudes,
        forward_effect_probability=float(probabilities[0]),nonforward_effect_probability=float(probabilities[1]),
        two_receiver_effect_sum=float(np.sum(probabilities)),
        original_input_normalizer=original_input_normalizer,
        formula='a=G^(-1/2) r / Z_original; P2=|a2|^2; E2=|Y G^(-1/2) e2><Y G^(-1/2) e2|',
        numerator_Gram_same_Psi_B1_B2_gout_required=True,no_per_bank_or_new_input_normalization=True)


def native_joint_receiver_Gram(model: Any,data: dict[str,Any],bank: Any,
                               duration: float=6.,legal_size: int=5*2**96) -> dict[str,Any]:
    """Same cached prepared state/H, true uncompressed two-creator Gram.

    The cached native weak Hamiltonian has its declared quadrature debt.
    Its evolved functional wave is NOT promoted to LEGAL interacting Psi.
    Its exact full52 scalar creator/annihilator graphs include finite Weyl
    tails outside the hard15 bank. No new state/filter/receiver is installed.
    """
    from scipy.linalg import eigh
    from prepared_neutral_shared_photon_state import frozen_nodes,conditional_values
    from prepared_neutral_noether_gaussian_partners import (
        creator_function,function_kernel)
    energies,vectors=eigh(data['neutral_H'],data['neutral_Gram'])
    initial=np.asarray(data['neutral_state'])
    evolved=vectors@(np.exp(-1j*duration*energies)*(vectors.conj().T@data['neutral_Gram']@initial))
    # Original SOURCE19 action, not recapture into a newly fitted packet.
    charged=data['source19_map']@evolved
    coefficients=data['neutral_coordinates']@evolved
    original_profile=model.source_bank.profiles['source19']
    recoil=2*np.pi/legal_size*np.array([1.,-1.,0.])
    output_profile=original_profile*np.exp(1j*((model.law.sites-1)@recoil))
    profiles=[original_profile,output_profile]
    from prepared_neutral_phi_word_channels import coefficient_channels
    _,normalizer_metadata=coefficient_channels(data,'charged','source19')
    original_Z=float(normalizer_metadata['global_SOURCE_normalizer'])
    raw_source=data['charged_coordinates']@data['source19_raw_source_state']
    creation=np.einsum('i,ijk->jk',data['packet_templates'][0],model.photon_basis.creation)
    raw_incoming=np.zeros_like(raw_source)
    raw_incoming[:1680]=(raw_source[:1680].reshape(112,15)@creation.T).reshape(-1)
    # This is the already supplied cached input, not a vacuum reset. Its
    # original native SOURCE image has only hard-vacuum support; a†gin is
    # therefore represented exactly in the declared one-photon columns.
    photon_degree=np.sum(np.asarray(model.photon_basis.labels),axis=1)
    support_defect=float(np.linalg.norm(raw_source[1680:])+
        np.linalg.norm(raw_source[:1680].reshape(112,15)[:,photon_degree!=0]))
    if support_defect>1e-12:
        raise ValueError('Original native true incoming creator cannot be exactly represented; do not change its preparation')
    incoming=np.linalg.solve(data['charged_coordinates'],raw_incoming)
    charged_energies,charged_vectors=eigh(data['charged_H'],data['charged_Gram'])
    charged_evolved=charged_vectors@(np.exp(-1j*duration*charged_energies)*
        (charged_vectors.conj().T@data['charged_Gram']@incoming))
    charged_coefficients=data['charged_coordinates']@charged_evolved
    functions=model.noether_bank.gaussian_functions()
    packets=np.asarray(bank.packets[1:3])  # original forward/nonforward OUT
    native_overlap=packets.conj()@packets.T
    # Original hard supports {+x,-x} and {+y,-y} are disjoint; each has
    # exact (16+1)/17 norm. Binary64 Fourier-frame roundoff is not a new N.
    overlap=np.eye(2,dtype=complex)
    created=[[creator_function(f,g) for f in functions[:15]] for g in packets]
    annihilated=[[complex_function_derivative(f,g.conj()) for f in functions] for g in packets]
    count=len(functions);hard=model.noether_bank.hard.dimension
    parity=np.r_[[sum(n)%2 for n in model.noether_bank.hard.labels],0,np.ones(count-hard-1,int)]
    scalar_gram=model.noether_bank.scalar_gram()
    phases=model.noether_bank.coefficients['phases']
    norm=np.einsum('ri,ri->r',phases,phases)
    cosgram=np.exp(-(norm[:,None]+norm[None,:])/4)*np.cosh(phases@phases.T/2)
    slopes=phases@packets.T
    number_kernel=np.zeros((2,2,count,count),complex)
    hard_creator_kernel=np.empty((2,2,hard,hard),complex)
    gamma_number=gamma_annihilator_Gram(model.noether_bank,packets)
    gamma_left,gamma_right=gamma_receiver_cross(model.noether_bank,packets)
    annihilator_cross=native_annihilator_cross(model,data,packets,gamma_left,gamma_right)
    # Independent creator and annihilator graphs; stabilize the sine/sine
    # block by its exact cosh Gaussian integral, not near-equal Weyl subtraction.
    # Parity-zero moments are algebraically zero, never numerically clipped.
    for j in range(2):
        for k in range(2):
            for m in range(count):
                for n in range(count):
                    if parity[m]!=parity[n]:continue
                    if m==hard and n==hard:
                        number_kernel[j,k,m,n]=gamma_number[j,k]
                    elif m>hard and n>hard:
                        number_kernel[j,k,m,n]=.5*slopes[m-hard-1,k]*slopes[n-hard-1,j].conjugate()*cosgram[m-hard-1,n-hard-1]
                    elif m==hard:
                        number_kernel[j,k,m,n]=np.dot(gamma_right[k,:hard].conj(),annihilator_cross[j,:hard,n])
                    elif n==hard:
                        number_kernel[j,k,m,n]=np.dot(annihilator_cross[k,:hard,m].conj(),gamma_right[j,:hard])
                    else:
                        number_kernel[j,k,m,n]=function_kernel(annihilated[k][m],annihilated[j][n])/2
            for m in range(hard):
                for n in range(hard):
                    hard_creator_kernel[j,k,m,n]=function_kernel(created[j][m],created[k][n])
    matter_metric=model.fibre.overlap('charged')
    D=np.zeros((2,2),complex);N=np.zeros_like(D)
    hard_true_G=np.zeros_like(D);hard_split_G=np.zeros_like(D)
    raw_CTC=np.zeros(2,complex);charged_functional_norm=0j
    for record in frozen_nodes(model):
        sample=record['sample'];base,extra=conditional_values(model,sample)
        neutral_wave=np.empty((base['neutral'].shape[0],count),complex)
        neutral_wave[:,:15]=base['neutral']@coefficients[:210].reshape(14,15)
        neutral_wave[:,15:]=np.einsum('aen,e->an',extra['neutral'],coefficients[210:],optimize=True)
        charged_wave=np.empty((base['charged'].shape[0],count),complex)
        charged_wave[:,:15]=base['charged']@charged_coefficients[:1680].reshape(112,15)
        charged_wave[:,15:]=np.einsum('aen,e->an',extra['charged'],charged_coefficients[1680:],optimize=True)
        # Apply BOTH original physical atomic templates on the actual RIGHT
        # neutral wave; do not equate B1/B2 or recapture its occupied orbitals.
        waves=[]
        for profile in profiles:
            source=model.fibre.source_vectors_at(sample['point']['phi'],sample['point']['links'],
                sample['theta'],sample['a'],profile)['source_matrix']
            waves.append(source@neutral_wave)
        weight=sample['importance_weight']/14
        charged_density=charged_wave.conj().T@matter_metric@charged_wave
        charged_functional_norm+=weight*np.einsum('mn,mn->',charged_density,scalar_gram)
        for j in range(2):
            CTC_density=waves[j].conj().T@matter_metric@charged_wave
            raw_CTC[j]+=weight*np.einsum('mn,mn->',CTC_density,annihilator_cross[j])
            for k in range(2):
                density=waves[j].conj().T@matter_metric@waves[k]
                D[j,k]+=weight*np.einsum('mn,mn->',density,scalar_gram)
                N[j,k]+=weight*np.einsum('mn,mn->',density,number_kernel[j,k])
                hard_density=density[:hard,:hard]
                hard_true_G[j,k]+=weight*np.einsum('mn,mn->',hard_density,hard_creator_kernel[j,k])
                hard_D=weight*np.einsum('mn,mn->',hard_density,scalar_gram[:hard,:hard])
                hard_split_G[j,k]+=overlap[j,k]*hard_D+weight*np.einsum('mn,mn->',hard_density,number_kernel[j,k,:hard,:hard])
    true_G=joint_receiver_Gram(D,N,overlap)
    receiver_Born=full_state_receiver_Born(raw_CTC,true_G,original_Z)
    eigenvalues=receiver_Born['joint_Gram_eigenvalues']
    inverse_root=receiver_Born['one_GLOBAL_joint_inverse_root']
    normalized_Gram=inverse_root.conj().T@true_G@inverse_root
    omitted_inverse_root=np.diag(np.real(np.diag(overlap*D))**(-.5))
    defect=float(np.linalg.norm(hard_true_G-hard_split_G))
    source_metric_defect=abs(D[0,0]-np.vdot(charged,data['charged_Gram']@charged))
    # Compare kernels directly: G-D can round away a genuine tiny N.
    omitted=float(np.linalg.norm(N))
    scale=max(1.,float(np.linalg.norm(true_G)))
    if defect>2e-9*scale or source_metric_defect>2e-9*scale:
        raise ValueError('True full52 receiver CCR/source functional contraction failed')
    if not omitted>0 or not abs(N[0,1])>0:
        raise ValueError('Selected actual evolving neutral did not expose omitted/offdiagonal N adverse control')
    nonreturn=evolved-initial*np.vdot(initial,data['neutral_Gram']@evolved)
    omitted_Born=full_state_receiver_Born(raw_CTC,overlap*D,original_Z)
    incoming_norm=float(np.vdot(incoming,data['charged_Gram']@incoming).real)
    final_norm=float(np.vdot(charged_evolved,data['charged_Gram']@charged_evolved).real)
    receiver_Born.update(omitted_N_adverse_Born=omitted_Born,
        omitted_N_adverse_nonforward_probability_difference=abs(
            receiver_Born['nonforward_effect_probability']-omitted_Born['nonforward_effect_probability']),
        raw_incoming_metric_norm_squared=incoming_norm,raw_final_metric_norm_squared=final_norm,
        frozen_Z_squared=original_Z**2,initial_true_creator_support_defect=support_defect,
        input_preparation_and_global_Z_NOT_changed=True,
        charged_evolution_metric_norm_defect=abs(final_norm-incoming_norm),
        charged_wave_vs_same_cached_metric_defect=float(abs(charged_functional_norm-final_norm)),
        native_neutral_dimension=len(initial),native_charged_dimension=len(incoming),
        full52_uncompressed_CTC_cross_graph=True,
        flatCAR_point_CTC_NOT_used_as_this_numerator=True,
        same_actual_native_Psi_SOURCE_B1_B2_gout_as_G=True,
        ordinary_Omega=0,duration=duration,
        full_interacting_formula='r_j=<Xi|U_N† B_j† a(gout_j) U_C a†gin B|Xi>; a=G^(-1/2)r/Z_original',
        LEGAL_whole_H_probability_certified=False,
        scope='compatible original cached weak-H CTC/Born with its native quadrature/finite functional-bank debts; LEGAL interacting numerator remains unexpanded')
    return dict(D=D,N=N,true_OUT_Gram=true_G,
        compatible_CTC_Born=receiver_Born,
        one_GLOBAL_joint_inverse_root=inverse_root,
        normalized_joint_receiver_Gram=normalized_Gram,
        native_joint_Gram_eigenvalues=eigenvalues,
        one_GLOBAL_whitening_defect=float(np.linalg.norm(normalized_Gram-np.eye(2))),
        omitted_N_adverse_inverse_root_defect=float(np.linalg.norm(inverse_root-omitted_inverse_root)),
        original_distinct_B1_B2_atomic_templates_retained=True,
        original_output_atomic_recoil=[float(x) for x in recoil],
        LEGAL_size_for_original_output_template=legal_size,
        no_per_branch_or_per_node_whitening=True,
        photon_overlap=overlap,native_binary64_photon_overlap=native_overlap,
        original_disjoint_hard_support_exact_Kronecker_overlap=True,
        independent_uncompressed_hard_creator_CCR_defect=defect,
        original_source_map_metric_defect=float(source_metric_defect),
        omitted_N_adverse_Gram=overlap*D,omitted_N_adverse_defect=omitted,
        offdiagonal_neutral_photon_coherence=complex(N[0,1]),
        actual_neutral_nonreturning_metric_norm=float(np.sqrt(np.vdot(nonreturn,data['neutral_Gram']@nonreturn).real)),
        evolved_neutral_metric_norm=float(np.vdot(evolved,data['neutral_Gram']@evolved).real),
        selected_SOURCE='SOURCE19',RAW41_distinct=True,original_eta18_Mc_not_replaced=True,
        selected_OUT_packet_indices=[1,2],ordinary_Omega=0,duration=duration,
        literal_global_SOURCE_normalizer_not_changed=True,
        exact_full52_uncompressed_creator_graph=True,hard15_NOT_physical_cutoff=True,
        offdiagonal_N_retained=True,vacuum_subtraction=False,
        full_interacting_D_N_kernels_retained_not_replaced_by_Wick=True,
        scope='native cached weak-H evolving functional diagnostic; not LEGAL whole-H Born/Maxwell closure')


def packet_outside_action(fibre: Any,marks: Full108ReferenceMarks,
                          current: np.ndarray) -> dict[str,Any]:
    """Relative packet image plus full before-crop PH controls, not closure."""
    from functional_source_photon_bank import full_packet_mark_action
    zero=Full108ReferenceMarks(np.zeros_like(marks.plus),np.zeros_like(marks.minus))
    result=full_packet_mark_action(fibre,marks.plus,marks.minus,zero.plus,zero.minus,
                                   current,np.zeros((4,4),complex))
    return result


def fraction(value: Any) -> Fraction:
    if isinstance(value,dict):
        return Fraction(int(value['exact_numerator_hex'],16),int(value['exact_denominator_hex'],16))
    return Fraction(value)


def uniform_photon_number_bound(V: int,initial_factorials: list[Any],
                                UE: Fraction,c: Fraction,coefficient_log: int,
                                duration: int=6) -> dict[str,Any]:
    """Ordinary whole-H second moment, independent of Lambda.

    rho=||Phi||HS. Original stiffness2 gives Lambda*rho^4/(64V).
    The literal heavy+light Pauli triangle b<=2^16 and Young give the
    global quartic moment <rho^4><=2^30 V^2 at absolute energy<=16V Lambda.
    Positive alignment/kinetic/Coul/Root penalties are retained. Actual
    initial/conserved E<=8V Lambda; bounded mean-light replacement leaves
    the reference below16V Lambda at the selected hierarchy, not a new state.

    For z<=1, the displacement conjugation of M=N+1 bounds its fractional
    graph norms. Duhamel [M^2,W] has relative form order3/2 and coefficient
    <=256|z|. Thus M2'<=512 C|z| <rho M^3/2>
    <=512 C|z| <rho^4>^1/4 M2^3/4. Use H_R=Hrest+P_R Hlight P_R
    on the finite-photon/compact-Phi form core. Quartic domination is uniform
    in R; strong-form/resolvent convergence and lower semicontinuity of
    ||(N+1)Psi|| remove the proof cutoffs. Evolving cutoff states are NOT
    monotone. Only ordinary Peierls fails to commute N; no driven assertion.
    """
    if V<=0 or duration!=6:raise ValueError('Original ordinary T6 only')
    ctx.prec=192
    a=[fraction(v) for v in initial_factorials]
    initial=arb(a[0].numerator)/a[0].denominator
    initial=initial**2+3*(arb(a[1].numerator)/a[1].denominator)**2+(arb(a[2].numerator)/a[2].denominator)**2
    C=arb(2)**coefficient_log
    phi4=arb(2)**30*V**2
    displacement=(arb(16)*arb(UE.numerator)/UE.denominator/(arb(c.numerator)/c.denominator)).sqrt()
    if not displacement<1:raise ValueError('Allmode original Weyl displacement premise fails')
    quarter=initial.sqrt().sqrt()+128*C*displacement*phi4.sqrt().sqrt()*duration
    M2=quarter**4
    if not M2<16:raise ValueError('Uniform whole-H moment ceiling16 not established')
    return dict(initial_Nplus1_squared_upper=str(initial.upper()),Phi_fourth_moment_upper=str(phi4.upper()),
        allmode_path_displacement_upper=str(displacement.upper()),Nplus1_squared_upper=16,
        evaluated_Nplus1_squared_upper=str(M2.upper()),
        exact_original_quartic_coefficient='Lambda/(64V)',
        coercivity_remainder='global heavy+light b<=2^16: <||Phi||HS^4><=2^30 V^2; betaD/Lambda>=-V for Lambda>=1, positive penalties retained',
        actual_energy_premise='actual ordinary conserved E<=8V Lambda; mean-light reference <=16V Lambda',
        proof_scope='global second number MOMENT; no global fourth-moment graph or driven Noether assertion',
        no_photon_mode_truncation_in_final=True)


def physical_OUT_transfer(neutral_error_log: int,late_source_error_log: int,
                          lambda_log: int,V: int,reference_Gram_lower: Fraction=Fraction(1,2)) -> dict[str,Any]:
    """Required a†OUT comparison + ONE joint physical-Gram inverse-root.

    B commutes all coexact a and ||B||<=16sqrtV on the original SOURCE
    coefficient domain/full encoded code. The bound is not a compact jet.
    For BOTH actual/ref M2<=16, ||a† delta||²<=8||delta||.
    Late (B-B0)Psi has weighted graph<=2CB*4; use interpolation separately.
    Raw two-column norm errors pay the fixed prescribed Loewdin comparison;
    the comparison receiver is never installed in place of the actual E2.
    """
    if not 0<reference_Gram_lower<=1:raise ValueError('Positive literal reference Gram margin required')
    vceil=(V-1).bit_length();CBlog=4+(vceil+1)//2
    # both moment norms <=4, source+neutral weighted terms and TWO columns
    col_log=CBlog+4+(max(neutral_error_log,late_source_error_log)+1)//2-lambda_log//16
    yref_log=CBlog+3
    gram_log=2+yref_log+col_log
    if gram_log>=-((reference_Gram_lower.denominator-1).bit_length()+1):
        raise ValueError('Prescribed physical joint Gram comparison not positive')
    # Functional calculus on Gram>=s/2: inverse-root <=sqrt(2/s),
    # derivative <=(1/2)(2/s)^(3/2), not the old static s=1/2 constant.
    inverse_squared_log=(reference_Gram_lower.denominator-1).bit_length()+1
    inverse_log=(inverse_squared_log+1)//2
    derivative_log=(3*inverse_squared_log+1)//2
    receiver_log=3+derivative_log+2*yref_log+col_log
    return dict(global_SOURCE_CB_log2_upper=CBlog,
        raw_OUT_two_column_error_log2_upper=col_log,actual_joint_Gram_error_log2_upper=gram_log,
        actual_Gram_lower=str(reference_Gram_lower/2),reference_Gram_lower=str(reference_Gram_lower),
        physical_receiver_error_log2_upper=receiver_log,
        inverse_root_log2_upper=inverse_log,inverse_root_derivative_log2_upper=derivative_log,
        rate='Lambda^-1/16 from NEW ordinary unweighted Lambda^-1/8, not driven647/C2',
        late_B_to_B0_paid_separately=True,source_global_bound_requires_original_domain_binding=True,
        actual_OUT='a†gout (Bplus+Bminus) U_N(6) Xi; unchanged gout and SAME global physical Gram',
        no_NEW_receiver_or_state_normalization=True)


def ordinary_actual_graphs(weighted: dict[str,Any],V: int) -> dict[str,Any]:
    """NEW Omega=0 actual-side IMS, not a relabelled driven C2 estimate.

    Every strong graph below is on the ORIGINAL regular Dirichlet union D.
    No zero-extended state is asserted to lie in D(Hwhole). Whole evolution
    is compared separately by the original form/Agmon boundary and global M2.
    On D, put beta I with the potential: h0+beta=Kfast+Wtilde+positive
    commuting EM/HC/Root/harm, Wtilde>=0. Covariant integration by parts gives
      Re<Kpsi,Wtilde psi> >= -chi <covLap Wtilde>/4.
    Original quartic/colour/alignment second derivatives scale at most
    Lambda^5/4, chi=Lambda^-1/2. Thus ||Kfast psi||² is bounded by the
    conserved Dirichlet strong graph squared plus C_N Lambda^3/4.

    C_N^128 V^64 pays regular jets through order2, inverse metric products,
    encoded charge row squares and channel sums: at most16 regular jet
    factors of size C_N^8 and at most90V coordinates. Root/D field
    derivatives vanish. This scalar count alone is NOT a graph argument.

    The original cutoff neutral f(h/L) retains ||hXi||<=L. Its D-cutoff
    commutator is the original regular cutoff row, not global SOURCE jets.
    For true creator use g=x f, Cf1(g)<2^512 by original Cf2/3 and support;
    a RAWtau=0 gives ||a hXi||<=32 CL Cf1(g). CCR pays [h,a†].
    Multiplication by the ORIGINAL bounded C2 SOURCE preserves D(H_D)
    using regular jets<=2 and even heavy/Root endpoint commutation.
    beta[D,B]P costs Lambda^1/2 in the STRONG source graph; do not call
    this O(L). The separate conserved source FORM expectation is O(L):
    BP lies in P, each periodic defect edge has PZedgeP=0, and initial
    Q<=C_N Lambda^-1/8 pays beta cross terms. Compact kinetic product
    rules use E<=8V Lambda; their chi M1 Pi term is at most C_N Lambda^1/4.
    The same form bound applies to the original true-photon SOURCE via
    CCR. At later times actual Q/PDQ are separately priced, not discarded.
    """
    ledger=weighted['ledger'];common=int(ledger['actual_common_coefficient_log2_upper'])
    lam=int(ledger['selected_log2_lambda_lower']);v=(V-1).bit_length()
    window=common+2
    covariance=int(ledger['covariance_log2_upper'])
    source_first=int(ledger['global_union']['source_first_log2'])
    if common<max(source_first,covariance,1231):raise ValueError('Original universal regular coefficient not bound')
    derivative=128*common+64*v+4096
    hgraph=window+5*common+16*v+2048
    kgraph=max(hgraph,(derivative+1)//2)+4
    distance=hgraph+199+8
    mixed_squared=10+(distance+1)//2+kgraph+1
    mixed=(mixed_squared+1)//2
    cross=3+34+v+mixed
    potential=3+304+(distance+1)//2
    remainder=3+64+2*v+8*common
    coefficient=max(cross,potential,remainder)+4
    late=source_first+(distance+1)//2+4
    initial=8*common+64*v+1231+4096
    q_coefficient=(common+2)//2
    # Q norm uses original absolute8V Lambda and G>=Utrial>
    # (132/5)V Lambda^5/4. Rest >=-C_N V Lambda, so Q²<=2C_N Lambda^-1/4.
    auxiliary=q_coefficient+3+4+6+(v+1)//2
    barrier=int(ledger['global_union']['barrier_log2_coefficient_lower'])
    action=int(ledger['global_union']['action_log2_lower'])-2
    # SAME spatial D and physical f/L/Z. Ecut=onequarter original barrier
    # is only an auxiliary spectral comparison, not a new preparation.
    # Compact initial chi B a†Xi has a whole-H domain proof by regular C2
    # product rule on its interior support. The evolved Dirichlet zero
    # extension is NEVER asserted H2. PHP strong source graph is O(L):
    # at Ecut~Lambda^1/2 its high tail is O(Lambda^-1/8).
    # Low side uses the original Agmon action, weakened by TWO bits.
    # Full-H compact-source high tail at G/2 is O(Lambda^-3/4).
    # Qlow solves (QHQ-Hlow)X=-QHP: <=2 beta||D||/G. Its codeflow
    # integral <=2T beta²||D||²/G=O(Lambda^-1/4), plus initial Q.
    # Do not integrate beta times the crude Q norm.
    if lam//4+action < lam.bit_length()+8:
        raise ValueError('Whole Agmon action does not dominate comparison powers')
    boundary=hgraph-barrier+8+source_first
    auxiliary=max(auxiliary,derivative+16*v+common+64)
    coefficient=max(coefficient,auxiliary,boundary)+2
    return dict(protocol='ORIGINAL ordinary Omega=0/T6',window_coefficient_log2=window,
        original_regular_C_N_log2=common,covariant_potential_second_jet_coefficient_log2_upper=derivative,
        conserved_Dirichlet_neutral_h_graph_coefficient_log2_upper=hgraph,
        conserved_Dirichlet_neutral_h_graph_Lambda_power='3/8',
        conserved_Dirichlet_SOURCE_h_graph_coefficient_log2_upper=hgraph,
        conserved_Dirichlet_SOURCE_h_graph_Lambda_power='1/2',
        conserved_Dirichlet_SOURCE_form_coefficient_log2_upper=hgraph,
        conserved_Dirichlet_SOURCE_form_Lambda_power='3/8',
        conserved_PHP_SOURCE_h_graph_Lambda_power='3/8',
        PHP_graph_does_NOT_discard_actual_beta_PD_Q=True,
        original_spatial_barrier_coefficient_log2=barrier,
        auxiliary_spectral_comparison_threshold_Lambda_power='1/2',
        auxiliary_spectral_comparison_threshold_coefficient_log2=barrier-2,
        ordinary_whole_to_D_error_coefficient_log2_upper=boundary,
        ordinary_whole_to_D_error_log2_upper=boundary-lam//8,
        compact_initial_comparison_domain_proved=True,
        whole_actual_SOURCE_strong_H_domain_asserted=False,
        strong_Kfast_coefficient_log2_upper=kgraph,strong_Kfast_Lambda_power_upper='1/2',
        actual_distance_squared_coefficient_log2_upper=distance,actual_distance_squared_Lambda_power='-5/8',
        NEW_ordinary_full_D_IMS_squared_coefficient_log2_upper=mixed_squared,
        NEW_ordinary_full_D_IMS_squared_Lambda_power='-1/4',
        NEW_ordinary_unweighted_reference_coefficient_log2_upper=coefficient,
        NEW_ordinary_unweighted_reference_error_log2_upper=coefficient-lam//8,
        initial_CAR_comparator_coefficient_log2_upper=initial,
        initial_CAR_comparator_norm_error_log2_upper=initial-lam//4,
        initial_CAR_scope='short SAME f/floor/Z filter-time Duhamel + original RAW full-cluster Kato; conditional non-ground Slater banks, full Theta',
        late_SOURCE_norm_coefficient_log2_upper=late,
        late_SOURCE_norm_error_log2_upper=late-5*lam//16,
        initial_true_SOURCE_CCR_norm_error_log2_upper=max(late-5*lam//16,152+512+304+32-window-3*lam//8)+4,
        mean_Gauss_norm_error_log2_upper=4+16+v-lam//2,
        Q_norm_log2_upper=q_coefficient-lam//8,
        Q_AND_beta_PD_Q_log2_upper=auxiliary-lam//8,
        no_driven_or_oldshadow_rate_retag=True,no_reference_graph_gate=True,
        mean_Gauss_is_charge_square_NOT_Gaussian=True,
        strong_graph_scope='ORIGINAL regular Dirichlet D only; never global SOURCE derivatives or zero-extension H2',
        whole_boundary_scope='original Omega=0 whole-to-D form Agmon; source bound globally, jets only on regular cutoff support',
        mean_Gauss_convention='Hfast retains heavy gGG(q); B0=chi/2[Qt gGG(0)^-1 Qt-Qh gGG(0)^-1 Qh]; full-mean supplier discrepancy separately paid',
        actual_state_not_projected_or_refiltered=True)


def initial_photon_cloud(weighted: dict[str,Any],V: int) -> dict[str,Any]:
    """Only a paid unnormalized comparator; original state/OUT stays mixed."""
    ledger=weighted['ledger'];lam=int(ledger['selected_log2_lambda_lower'])
    window=int(ledger['actual_common_coefficient_log2_upper'])+2
    cb=4+(((V-1).bit_length()+1)//2)
    v=(V-1).bit_length()
    neutral=16+v+117+304-window-3*lam//8
    # Second Fourier pullthrough has the first second-commutator plus the
    # two first-commutator Touchard term. Since displacement<=1 and L>=1,
    # Cf2*CL²*(1+Phi²)/L dominates both; original global quartic moment
    # pays <=2^16 V. This is a factorial norm, not the moment of N².
    second=32+2*v+199+2*304-window-3*lam//8
    return dict(neutral_A1_log2_upper=neutral,
        neutral_A2_log2_upper=second,
        SOURCE_true_creator_vacuum_comparator_error_log2_upper=cb+4+neutral,
        original_Cf1_log2_upper=117,original_CL_log2_upper=304,
        original_L_coefficient_log2=window,rate='Lambda^-3/8 before endpoint interpolation',
        original_initial_photon_vacuum_not_asserted=True,
        comparator_Pvac_not_installed_or_normalized=True,
        photon_CCR_not_norm_error_forwarding=True)


def reference_Gram_admission(V: int,UE: Fraction,c: Fraction,
                             initial_error_log: int,lambda_log: int) -> dict[str,Any]:
    """Historical conditional margin; NOT a corrected full-state D/N admission.

    The original Cbeta has duplicated singular values:
    sum sigma²=33/40, product sigma²=13481/160000.
    Hence sigma_min²>=product/sum>1/16. At each actual mean branch,
    positive/negative Bloch projectors have rank2. Ref4plus uses empty
    positive electrons; Ref0minus uses the occupied NEGATIVE sea, with
    four genuine positive Spect spectators giving a positive increment.
    No interacting neutral return is assumed. Thus SOURCE norm² is at
    least 2 sigma_min² ||eta_eff||². All actual19 source paths
    have length<D+2<2^98; tanh36>1-2^-102, so their signed screening
    product magnitude>15/16. The tiny recoil changes no eta norm.
    The unnormalized CAR comparator has norm>=1/2, so free Gram>=1/64;
    hard forward/outgoing photon IDs make its offdiagonal EXACTLY zero.

    Hmean Peierls W-I on vacuum costs <=6*2^304*d. FullHC and harmonic
    occupation/source graph cost <=2^1024 UE V^4; they are retained, not
    Gaussianized. MeanGauss costs <=6*2^16 V Lambda^-1/2.
    Number interpolation pays a† on the neutral difference. This proves
    Gref>=1/128 on the actual mixed neutral endpoint.
    """
    ctx.prec=256
    if initial_error_log>=-2:raise ValueError('CAR comparator norm margin unproved')
    v=(V-1).bit_length();cb=4+(v+1)//2
    displacement=(arb(16)*arb(UE.numerator)/UE.denominator/(arb(c.numerator)/c.denominator)).sqrt()
    ph=arb(6)*arb(2)**304*displacement
    hc_harm=arb(2)**1024*(arb(UE.numerator)/UE.denominator)*V**4
    ordinary_small=ph+hc_harm
    neutral_log=math.ceil(float((ordinary_small.log()/arb(2).log()).upper()))
    neutral_log=max(neutral_log,3+16+v-lambda_log//2,initial_error_log)+3
    column_log=cb+4+(neutral_log+1)//2
    gram_log=2+cb+3+column_log
    if gram_log>=-7:raise ValueError('Literal mixed-grade reference Gram margin not proven')
    determinant=Fraction(13481,160000);singular_sum=Fraction(33,40)
    if determinant/singular_sum<=Fraction(1,16):raise ValueError('Original complex quaternion invertibility bound failed')
    return dict(historical_conditional_epoch=True,corrected_full_D_N_Gram_admitted=False,
        free_raw_joint_Gram_lower='1/64',reference_raw_joint_Gram_lower='1/128',
        actual_raw_joint_Gram_lower_after_transfer='1/256',
        literal_reference_neutral_difference_log2_upper=neutral_log,
        literal_reference_OUT_column_error_log2_upper=column_log,
        literal_reference_Gram_perturbation_log2_upper=gram_log,
        exact_complex_quaternion_min_singular_squared_lower=str(determinant/singular_sum),
        original_screened_path_magnitude_lower='15/16',
        BOTH_templates_and_Ref_banks_before_ONE_Gram=True,
        raw_Gram_not_native_response_metric=True,
        actual_reference_OUT_mixed_photon_grades_retained=True)


def leading_reference_factorization(descriptor: dict[str,Any],paired: dict[str,Any],
                                    screen: dict[str,Any]) -> dict[str,Any]:
    """Exact leading SOURCE factor, not a lower-Gram replacement.

    At the point Clifford reference LABELS=[I,G2G3,G3G1,G1G2]/2,
    M=sum beta LABELS. Trace(M†M)=sum|beta|², and both sea projectors
    give Tr(M†P±M)=Tr(M†M)/2. With the ORIGINAL Mc and harmonic
    birth phase, Parseval gives s=(Tr M†M/2)||Mc eta||². Ref0 adds
    the positive Spect insertion R; its separately enclosed correction
    is NOT omitted. Original fork amplitudes/fn/Z become actual wq,
    not two re-normalized state preparations.

    For prescribed forward/outgoing photon IDs G0=s I and C0_2=0.
    Write C2_q=s (UE/V)cq, G1_21,q=s (UE/V)gq, M0_1,q=s mq.
    Differentiating the ONE joint inverse root at sI yields
      Aq/(UE/V)=sqrt(wq)[cq-.5(sum_r wr gr)mq].
    s cancels against original input sqrt(s) and output sqrt(s);
    no SOURCE/column rescaling is installed. Actual Phi*_ms, theta,
    P0->Utheta P0 Utheta†, finite Spect R and GLOBAL G0-sI perturbations
    are the literal C/G/m enclosures below. The coarse1/128 only
    stabilizes later remainder/actual-OUT transfer, never this identity.
    """
    ctx.prec=256
    if int(screen['literal_LEGAL19_geometry']['N'])!=int(descriptor['N']):
        raise ValueError('Leading SOURCE screen geometry differs')
    from prepared_neutral_full_field import BETA
    beta_squared=sum((Fraction.from_float(float(z.real))**2+
                      Fraction.from_float(float(z.imag))**2 for z in BETA),Fraction(0))
    eta_screen=arb(screen['numerical_Arb_witness']['screened_eta_squared'])
    common=eta_screen*(arb(beta_squared.numerator)/beta_squared.denominator)/2
    harmonic=paired['ordinary_all512_fullGaussian_harmonic']
    spect=paired['new_R4_source_slack_enclosure']
    # Actual leading departures from scalar point G0, not a N(k) floor.
    gram_error=arb(harmonic['zeroth_GLOBAL_Gram_operator_error'])+arb(
        spect['source_and_zeroth_GLOBAL_Gram_relative_increment_upper'])
    leading_error=arb(harmonic['GLOBAL_leading_relative_error'])+arb(spect['GLOBAL_leading_relative_error'])
    if not gram_error<arb('1e-20'):raise ValueError('Actual mean leading Gram correspondence unproved')
    gcap=arb(paired['joint_raw_G_centre_cap'])
    rows={}
    for bank in ('plus','minus'):
        raw=paired['result'][bank];errors={k:arb(v) for k,v in raw['raw_component_errors'].items()}
        m=raw['frozen_raw_m'];mabs=(arb(m['real'])**2+arb(m['imag'])**2).sqrt()
        # Retain Gram-coherence cross, including its error-product term.
        disk=errors['C']+(errors['G']*mabs+gcap*errors['m']+errors['G']*errors['m'])/2
        # GLOBAL leading amplitude payment is normalized to UE/V already:
        # subtract OUTSIDE sqrt(w), separately from the raw-component disk.
        coefficient=arb('0.49').sqrt()*(arb(raw['frozen_raw_C']['real'])-gcap*mabs/2-disk)-8*leading_error
        rows[bank]=dict(reconstructed_leading_lower=str(coefficient.lower()),
            current_C_error=str(errors['C']),joint_Gram_G_error=str(errors['G']),
            forward_m_error=str(errors['m']),reconstructed_C_G_m_disk=str(disk.upper()),
            GLOBAL_leading_structured_payment=str((8*leading_error).upper()),
            actual_joint_photon_coherence_term_retained=True)
    return dict(point_common_SOURCE_norm_squared=str(common),
        beta_squared_exact=str(beta_squared),actual_screened_eta_squared=str(eta_screen),
        point_identity='s=.5 Tr(Mdag M)||Mc eta||²; G0=sI; C0_nonforward=0',
        actual_non_scalar_leading_Gram_relative_error=str(gram_error.upper()),
        actual_harmonic_Spect_finiteN_leading_error=str(leading_error.upper()),
        normalized_C_G_m_definition='C2=s(UE/V)c, G1_21=s(UE/V)g, M0_1=s m',
        one_GLOBAL_inverse_root_derivative='-.5 s^-3/2 G1, including offdiagonal photon coherence',
        fork_interval=paired['actual_bank_probability_interval'],
        branch_response='muq=sqrt(wq)[cq-.5 bar_g mq] is <F2GLOBAL|U|Psi_input_q>, not the paired-input amplitude',
        paired_actual_input='Apaired/(UE/V)=sum_q a_q muq, a_q=||Pi_Ref3or1 Psi_in||>=0, sum a_q²=1; principal a_q=sqrt(wq)',
        original_relative_phase='each normalized branch keeps its ORIGINAL fork phase; same phase in Y and input, no rephasing or state change',
        positive_paired_lower_reason='Re muq>=b>0 BOTHq; sum a_q>=1 implies Re sum a_q muq>=b, including unequal actual source norms',
        physical_input_normalization='a_q=sqrt(wq)n_q/sqrt(sum_r wr nr²); finiteR/initialcloud/norm corrections separately priced',
        fork_and_Z='original neutral fn/floor/Gamma/Z; actualw after one normalization, not reset1/2',
        harmonic_birth_phase=harmonic['common_birth_phase'],
        actual_mean_Gram_enclosure=harmonic,actual_R4_Spect_enclosure=spect,
        ordinary_all512_fullGaussian_scope=True,
        coarse_invertibility_floor_used_as_leading_Gram=False,
        reconstructed_from_raw_C_G_m_NOT_copied_mu_lower=True,banks=rows)


def joined_physical_Born(descriptor: dict[str,Any],weighted: dict[str,Any],
                         paired: dict[str,Any],ordinary: dict[str,Any],
                         gram: dict[str,Any],OUT: dict[str,Any],
                         factorization: dict[str,Any]) -> dict[str,Any]:
    """Historical conditional arithmetic from the pre-delta13 Gram epoch.

    Kept to reproduce its ledger, NOT to certify the corrected twoY receiver.
    The full-state D/N Gram must be used before any physical probability claim.
    Coulomb alone has zero nonforward first e² term by exact hard-mode
    orthogonality. Its nonzero mixed correction is already retained/priced.
    The mean Gauss charge square is a bounded retained interaction and gets
    its own absolute state/OUT error, never a Gaussian determinant.
    """
    V=int(descriptor['N'])**3;UE=Fraction(descriptor['physical_family']['UE'])
    if gram['reference_raw_joint_Gram_lower']!='1/128':raise ValueError('Bind literal reference Gram')
    ctx.prec=256
    lower=min(arb(factorization['banks'][bank]['reconstructed_leading_lower']) for bank in ('plus','minus'))
    if not lower>arb('0.6'):raise ValueError('Reconstructed actual SOURCE/current/leading Gram coefficient not positive')
    relative=arb(paired['mean_allmode_Born_remainder']['leading_UE_over_V_relative_remainder'])*arb(2)**128
    # Ratio between old1/2 and literal1/128 is64. Inverse-root derivatives
    # through third order cost <=64^(7/2), input/source norms and raw columns
    # cost <2^32; 2^128 pays these without assuming the old Gram margin.
    lam=int(weighted['ledger']['selected_log2_lambda_lower'])
    cb=4+(((V-1).bit_length()+1)//2)
    initial=ordinary['initial_CAR_comparator_norm_error_log2_upper']
    initial_input=cb+5+initial
    initial_receiver=3+12+2*(cb+3)+cb+4+(initial+1)//2
    logs=[ordinary['NEW_ordinary_unweighted_reference_error_log2_upper'],
          ordinary['initial_true_SOURCE_CCR_norm_error_log2_upper'],
          ordinary['mean_Gauss_norm_error_log2_upper'],
          ordinary['Q_AND_beta_PD_Q_log2_upper'],OUT['physical_receiver_error_log2_upper'],
          initial_input,initial_receiver]
    if max(logs)+8>=-12000:raise ValueError('Whole actual state/creator/Gram errors not below absolute Born budget')
    scale=(arb(UE.numerator)/UE.denominator)/V
    amplitude=(lower-relative)*scale-arb(2)**-12000
    if not amplitude>scale/2:raise ValueError('New actual nonforward amplitude lower not positive')
    return dict(physical_probability_certified=False,historical_conditional_epoch=True,
        corrected_full_D_N_receiver_probability_certified=False,clock=6,Omega=0,
        actual_normalized_true_photon_SOURCE_input=True,
        actual_unconditional_binary_effect='(I-E2,E2), E2 from SAME TWO true Y and ONE joint physical Gram',
        conservative_actual_amplitude_lower=str((scale/2).lower()),
        physical_probability_lower=str(((scale/2)**2).lower()),
        stronger_evaluated_amplitude_lower=str(amplitude.lower()),
        existing_mean_leading_coefficient_lower=str(lower.lower()),
        leading_reference_factorization=factorization,
        newly_repriced_allmode_relative_remainder_upper=str(relative.upper()),
        additional_absolute_actual_field_CODE_initialCAR_OUT_budget='2^-12000',
        actual_errors_log2_upper=logs,initial_Bq_to_B0_counted_once=True,
        fullHC_Gauss_harm_nonGaussian_interactions_retained_and_priced=True,
        Born_is_NOT_native_receiver_or_Lambda_OUT_rate=True,
        actual_phi_mean='finite-N Phi*_ms/all8 twists/all64 aliases; G_infinity NOT assigned exact',
        ordinary_E0_kappa_all64_and_harmonic_boundary_preserved=True,
        no_NEW_source_filter_receiver_Z_or_photon_transport=True)
