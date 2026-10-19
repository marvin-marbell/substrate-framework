"""Original SOURCE-relative G/J/contact functions and exact all52 creator images.

Values are divided by the ORIGINAL common Gaussian/Haar/cutoff factor.  The
caller owns that factor and its physical jets ONCE.  This is an operator-family
representation, not a particle/spin/mass/scattering or invariant-subspace claim.
"""
from __future__ import annotations
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, cast
import numpy as np
from original_total_angular_transfer import scalar_orbital
from native_closed_rotation_quantum import physical_chart_pullback
from prepared_neutral_source_relative_embedding import relative_source
from prepared_neutral_phi_word_channels import coefficient_channels
from prepared_neutral_full_field import BETA
from prepared_neutral_weyl_hard_fock import GaussianFunction, ProjectionPolynomial
from prepared_neutral_quartet_functions import full_slater_relative_spin


@dataclass
class Full108ReferenceMarks:
    """Arbitrary packet f(x,spin;Ref4), NOT eight fitted profile/word rays.

    These are physical even-CAR SOURCE coefficients. Light/current/orbital
    action changes the packet on all108 modes. Ref action is independently
    supplied by the actual Noether producer, never assigned a Pauli sigma.
    Relative actions do NOT replace absolute H/neutral RIGHT/PH products.
    """
    plus: np.ndarray
    minus: np.ndarray

    def light_action(self, body: np.ndarray) -> Full108ReferenceMarks:
        return Full108ReferenceMarks(body@self.plus,-self.minus@body)

    def reference_action(self, reference: np.ndarray) -> Full108ReferenceMarks:
        return Full108ReferenceMarks(-self.plus@reference,reference@self.minus)

    def noether_relative(self, body: np.ndarray,reference: np.ndarray,
                         derivative: Full108ReferenceMarks) -> Full108ReferenceMarks:
        return Full108ReferenceMarks(-1j*derivative.plus+body@self.plus-self.plus@reference,
                                    -1j*derivative.minus+reference@self.minus-self.minus@body)

    def physical_source(self,fibre: Any) -> dict[str,Any]:
        volume=fibre.law.volume
        return fibre.source_vectors_from_marks({1:self.plus.reshape(volume,4,4),
            -1:self.minus.reshape(4,volume,4).transpose(1,0,2)})


@dataclass
class FullPacketWeylTerm:
    """Scalar field function times arbitrary full108xRef physical marks."""
    function: GaussianFunction
    marks: Full108ReferenceMarks

    def incoming_creator(self,packet: np.ndarray) -> FullPacketWeylTerm:
        return FullPacketWeylTerm(creator_function(self.function,packet),self.marks)

    def light_product(self,body: np.ndarray,phase52: np.ndarray) -> FullPacketWeylTerm:
        """Actual Fourier-body convolution: new packet, no projection/recapture."""
        return FullPacketWeylTerm(GaussianFunction(self.function.polynomial,self.function.phase+phase52),
                                 self.marks.light_action(body))


def full_packet_mark_action(fibre: Any,plus: np.ndarray,minus: np.ndarray,
                            dplus: np.ndarray,dminus: np.ndarray,
                            body: np.ndarray,reference: np.ndarray) -> dict[str,Any]:
    """Actual full-packet relative Noether action WITH independent RIGHT/PH.

    All marks use full108x4 / 4x108. Original initial packet and its single
    globalZ remain the caller's once-only state preparation. Absolute H is NOT
    replaced by this commutator; the original neutral RIGHT/PH stays exported.
    """
    marks=Full108ReferenceMarks(plus,minus);source=marks.physical_source(fibre)
    derivative=Full108ReferenceMarks(dplus,dminus)
    relative=marks.noether_relative(body,reference,derivative).physical_source(fibre)
    comm=marks.noether_relative(body,reference,Full108ReferenceMarks(np.zeros_like(plus),np.zeros_like(minus))).physical_source(fibre)['source_matrix']
    b=source['source_matrix'];qc=fibre.onebody(body,'charged')+fibre.reference(reference,'charged')
    qn=fibre.onebody(body,'neutral')+fibre.reference(reference,'neutral')
    right=qc@b-b@qn-comm
    control=full_slater_relative_spin(fibre,source,body)
    return dict(source=source,relative=relative,relative_matrix=relative['source_matrix'],
        light_packet_action=marks.light_action(body),explicit_Ref_action=marks.reference_action(reference),
        actual_neutral_RIGHT_PH=right,actual_neutral_RIGHT_projected=b@qn,
        original_full_Fock_CAR_and_beforecompression_PH_controls=control,
        no_packet_profilefit_recapture_or_intrinsic_spin_assignment=True,
        absolute_H_neutral_RIGHT_not_replaced_by_relative_action=True)


@dataclass
class OperatorWeylTerm:
    """Physical coefficient = fibre_coefficients @ channel_coefficients.

    ``channels`` names columns in the whole bank; no Gaussian is in coefficients
    or polynomial. GaussianFunction only supplies the established scalar ABI.
    """
    function: GaussianFunction
    fibre_coefficients: np.ndarray
    channel_coefficients: np.ndarray
    channels: np.ndarray

    def coefficient(self, rows: slice | None = None) -> np.ndarray:
        values=self.fibre_coefficients if rows is None else self.fibre_coefficients[rows]
        return values@self.channel_coefficients


@dataclass
class CoexactOperatorPack:
    terms: tuple[OperatorWeylTerm, ...]
    fibre_dimension: int
    channel_count: int
    base_channel_count: int
    packet: np.ndarray
    metadata: dict[str,Any]

    def values(self, y: np.ndarray) -> np.ndarray:
        result=np.zeros((self.fibre_dimension,self.channel_count),complex)
        for term in self.terms:
            polynomial,_=term.function.polynomial.evaluate(y)
            scalar=polynomial*np.exp(1j*(term.function.phase@y))
            result[:,term.channels]+=scalar*term.coefficient()
        return result

    def coexact_directional(self, y: np.ndarray, direction: np.ndarray) -> np.ndarray:
        """Derivative of FACTORED functions, NOT a Gaussian wave derivative."""
        result=np.zeros((self.fibre_dimension,self.channel_count),complex)
        for term in self.terms:
            value,gradient=term.function.polynomial.evaluate(y)
            scalar=np.exp(1j*(term.function.phase@y))*(gradient@direction+1j*(term.function.phase@direction)*value)
            result[:,term.channels]+=scalar*term.coefficient()
        return result

    def export(self) -> dict[str,Any]:
        return dict(terms=self.terms,phases52=np.asarray([term.function.phase for term in self.terms]),
            polynomials=tuple(term.function.polynomial for term in self.terms),
            fibre_coefficients=tuple(term.fibre_coefficients for term in self.terms),
            channel_coefficients=tuple(term.channel_coefficients for term in self.terms),
            channels=tuple(term.channels for term in self.terms),channel_count=self.channel_count,
            coefficient_semantics='fibre_coefficients @ channel_coefficients at named columns; no Gaussian/Haar/cutoff',
            polynomial_semantics='ProjectionPolynomial: bilinear projection dot y52, monomial index tuples, exp(i real phase52.y52)',
            physical_Gaussian_jets_NOT_preapplied=True,metadata=self.metadata)


def creator_function(function: GaussianFunction, packet: np.ndarray) -> GaussianFunction:
    """Exact canonical a_g^dagger, including Weyl AND polynomial derivatives."""
    polynomial=function.polynomial
    multiplier=ProjectionPolynomial(packet[None,:],{(0,):math.sqrt(2),():cast(complex,-1j*(function.phase@packet)/math.sqrt(2))})
    product=polynomial.multiply(multiplier)
    derivative=polynomial.directional_derivative(packet)
    terms=dict(product.terms)
    for factors,coefficient in derivative.terms.items():
        terms[factors]=terms.get(factors,0j)-coefficient/math.sqrt(2)
    return GaussianFunction(ProjectionPolynomial(product.projections,terms),function.phase)


class FunctionalSourcePhotonBank:
    """C48 original functional operator bank + incoming48; N14 + incoming14.

    Coexact dependence is compiled from literal original primitive producers.
    The fixed-seed LAB fibre is intentional; an ordered/field reference must
    transform these physical functions and photon operator, not freeze point
    arrays or drop its Udagger D_g U / outside-PH connection.
    """
    def __init__(self,model: Any,data: dict[str,Any],source_step: float=2.**-18):
        self.model=model;self.data=data;self.source_step=source_step
        argv=data['manifest']['argv'];self.guard_path=argv[argv.index('--noether-guard')+1]
        self.guard=json.loads(Path(self.guard_path).read_bytes())
        self.velocity=np.asarray(self.guard['velocity']);self.alpha=np.asarray(self.guard['alpha'])
        self.packets=(model.photon_basis.packets@data['packet_templates'].T).T
        self.packet=np.asarray(self.packets[0]);self.fibre=model.source_bank.seed_fibre
        self.transform=np.zeros((8,8),complex)
        for word in range(4):
            self.transform[2*word,0]=BETA[word];self.transform[2*word+1,0]=np.conj(BETA[word])
        order=[(charge,word) for charge in (1,-1) for word in range(4)]
        for column,(charge,word) in enumerate(order[1:],1):self.transform[2*word+(charge==-1),column]=1
        self._compiled: dict[tuple[str,bytes,bytes],CoexactOperatorPack]={}

    def initial_coefficients(self,profile: str,incoming: bool=True) -> np.ndarray:
        if profile=='neutral':
            result=np.zeros(28,complex);result[14 if incoming else 0:28 if incoming else 14]=self.data['preparation_state14']
        else:
            if profile not in ('source19','SOFT'):raise ValueError('Original neutral/source19/SOFT only')
            result=np.zeros(96,complex);result[48 if incoming else 0]=1
        return result

    def _common(self,sample: dict[str,Any]) -> tuple[dict[str,Any],np.ndarray,complex]:
        em=np.r_[np.zeros(52),sample['em'][52:]]
        original=self.model.original.basis_at(sample['q'],em,sample['point'])
        phase=np.exp(1j*self.model.boost['Omega']*(self.model.boost['Y0']@sample['q'])/self.model.chi)
        psi=original['neutral']@self.data['preparation_state14']
        return original,psi,complex(phase)

    def compile(self,sample: dict[str,Any],profile: str) -> CoexactOperatorPack:
        key=(profile,np.ascontiguousarray(sample['q']).tobytes(),np.ascontiguousarray(sample['em'][52:]).tobytes())
        if key in self._compiled:return self._compiled[key]
        model=self.model;original,psi,phase=self._common(sample);constant=ProjectionPolynomial.constant(52)
        zero=np.zeros(52);terms=[]
        if profile=='neutral':
            values=phase*original['neutral'];base=14
            terms.append(OperatorWeylTerm(GaussianFunction(constant,zero),values,np.eye(14),np.arange(14)))
            metadata=dict(neutral_basis='original14 raw normal-polynomial functions, no new preparation/conditioning',
                original_prepared_state_norm_squared=float(np.vdot(self.data['preparation_state14'],self.data['preparation_state14']).real))
            dimension=2
        else:
            if profile not in ('source19','SOFT'):raise ValueError('Original neutral/source19/SOFT only')
            C,metadata=coefficient_channels(self.data,'charged',profile);z=metadata['global_SOURCE_normalizer'];base=48;dimension=self.fibre.charged_dimension
            point=sample['point'];theta=original['theta'];a=original['a'];sources=[]
            for word in range(4):sources.append(self.fibre.source_vectors_at(point['phi'],point['links'],theta,a,model.source_bank.profiles[profile],np.eye(4)[word]))
            def columns(matrices: list[np.ndarray]) -> np.ndarray:
                return phase*np.column_stack([matrix[:,bank]*psi[bank]/z for matrix in matrices for bank in range(2)])
            def marked(plus: list[np.ndarray],minus: list[np.ndarray]) -> np.ndarray:
                matrices=[self.fibre.source_vectors_from_marks({1:s.reshape(model.law.volume,4,4),-1:t.reshape(4,model.law.volume,4).transpose(1,0,2)})['source_matrix'] for s,t in zip(plus,minus)]
                return columns(matrices)
            S=[source['full108_plus_coefficients'] for source in sources];T=[source['full108_minus_coefficients'] for source in sources]
            def block_images(x: int,y: int,block: np.ndarray) -> np.ndarray:
                sx,sy=slice(4*x,4*x+4),slice(4*y,4*y+4);plus=[];minus=[]
                for s,t in zip(S,T):
                    p=np.zeros_like(s);m=np.zeros_like(t);p[sx]=block@s[sy];m[:,sy]=-t[:,sx]@block;plus.append(p);minus.append(m)
                return marked(plus,minus)
            unit=columns([source['source_matrix'] for source in sources]);source_values=unit@self.transform
            source_error=float(np.linalg.norm(source_values-sample['charged']@C))
            if source_error>1e-11:raise ValueError('Functional LAB seed SOURCE/word/boost representation mismatch')
            terms.append(OperatorWeylTerm(GaussianFunction(constant,zero),unit,self.transform,np.arange(8)))
            pull=physical_chart_pullback(model,sample,self.alpha,self.velocity);gauge=model.gauge_operators(pull['gauge'],'charged');K,R=gauge['body'],gauge['reference']
            geometry=[relative_source(model,dict(sample,em=original['em'],a=a,theta=theta),profile,np.eye(4)[word],K,R,pull['normal'],np.zeros(55),self.source_step) for word in range(4)]
            terms.append(OperatorWeylTerm(GaussianFunction(constant,zero),columns([row['relative_matrix'] for row in geometry]),np.eye(8),np.arange(8,16)))
            orbital=scalar_orbital(model.law,point['links'],a,self.velocity)
            for x in range(model.law.volume):
                for axis in range(3):
                    y=int(model.law.target[x,axis]);block=orbital[4*x:4*x+4,4*y:4*y+4];k=model.em['legs'][3*x+axis]
                    for xx,yy,b,frequency in ((x,y,block,k),(y,x,block.conj().T,-k)):
                        terms.append(OperatorWeylTerm(GaussianFunction(constant,frequency),block_images(xx,yy,b),np.eye(8),np.arange(8,16)))
            legs=[(model.em['legs']@packet).reshape(model.law.volume,3)/math.sqrt(2) for packet in (self.packets[0],self.packets[1],self.packets[2].conj())]
            for x,y,block,path in model.law.terms(point['phi'],point['links'],a):
                k=sum((model.em['legs'][3*site+axis] for site,axis in path),np.zeros(52));slopes=np.asarray([sum(leg[edge] for edge in path) for leg in legs])
                for xx,yy,b,frequency,sign in ((x,y,block,k,1),(y,x,block.conj().T,-k,-1)):
                    maps=np.column_stack([1j*sign*slopes[j]*self.transform for j in range(3)]+[-slopes[0]*slopes[2]*self.transform])
                    terms.append(OperatorWeylTerm(GaussianFunction(constant,frequency),block_images(xx,yy,b),maps,np.arange(16,48)))
            noether=model.noether_bank.coefficients;displacement=(model.law.sites-model.law.sites[model.reference_site])/model.law.size
            relative_scalar=noether['lambda_C']-noether['lambda_C'][:,model.reference_site,None]-noether['harmonic']@displacement.T
            for frequency,scalar in zip(noether['phases'],relative_scalar):
                component=np.repeat(scalar,4);value=marked([component[:,None]*s for s in S],[-t*component[None,:] for t in T])
                for sign in (-1,1):terms.append(OperatorWeylTerm(GaussianFunction(constant,sign*frequency),value,sign/(2j)*np.eye(8),np.arange(8,16)))
            metadata=dict(metadata,source_function_error=source_error,physical_chart_reconstruction_error=pull['reconstruction_error'],
                geometry_light_matrix=K,geometry_Ref_matrix=R,normal_Noether_direction=pull['normal'],
                geometry_derivative_marks=tuple(row['derivative_marks'] for row in geometry),
                core_and_explicit_Ref_Noether_NOT_assigned_spinor=True,
                column_order='SOURCE8(originalcoherent,+word1..3,-word0..3); G8(wordouter,+/-inner); Jin8; Jforward8; Jout-conjugate8; Cin,out-conjugate8')
        base_terms=tuple(terms)
        terms.extend(OperatorWeylTerm(creator_function(term.function,self.packet),term.fibre_coefficients,term.channel_coefficients,term.channels+base) for term in base_terms)
        pack=CoexactOperatorPack(tuple(terms),dimension,2*base,base,self.packet,dict(metadata,
            functional_at_actual_q_theta=True,photon_derivative_FULL_not_multiplication_only=True,
            all52_exact_finite_Weyl_and_original55_harmonic_Gaussian=True,
            physical_Gaussian_Haar_cutoff_jets_NOT_preapplied=True,original_static_boost_in_values=True,
            original_SOURCE_globalZ_once=True,no_state_effect_reprep_or_renormalization=True,
            invariant_H_Q_G_J_particle_spin_mass_scattering_NOT_claimed=True))
        # Only current-q/previous-q packs are useful to directional consumers.
        if len(self._compiled)>=8:self._compiled.clear()
        self._compiled[key]=pack
        return pack

    def values(self,sample: dict[str,Any],profile: str) -> np.ndarray:
        return self.compile(sample,profile).values(sample['em'][:52])

    def directional_values(self,sample: dict[str,Any],profile: str,dq: np.ndarray,dem: np.ndarray,h: float) -> np.ndarray:
        """Factor jets: EXACT coexact plus centered q/theta jet; NO Gaussian jet."""
        pack=self.compile(sample,profile);y=sample['em'][:52]
        derivative=pack.coexact_directional(y,dem[:52])
        if np.any(dq) or np.any(dem[52:]):
            scale=max(1.,float(np.linalg.norm(dq)),float(np.linalg.norm(dem[52:])));step=h/scale
            shifted=[]
            for sign in (-1,1):
                q=sample['q']+sign*step*dq;em=sample['em'].copy();em[52:]+=sign*step*dem[52:]
                shifted.append(self.values(self.model.basis_at(q,em),profile))
            derivative+=(shifted[1]-shifted[0])/(2*step)
        return derivative

    def coexact_export(self,sample: dict[str,Any],profile: str) -> dict[str,Any]:
        return self.compile(sample,profile).export()


def coefficient_block(pack: CoexactOperatorPack,rows: slice,base_only: bool=True) -> np.ndarray:
    count=len(pack.terms)//2 if base_only else len(pack.terms);width=pack.base_channel_count if base_only else pack.channel_count
    n=len(range(pack.fibre_dimension)[rows]);out=np.zeros((count,n,width),complex)
    for i,term in enumerate(pack.terms[:count]):out[i,:,term.channels]=term.coefficient(rows).T
    return out


def coexact_metric(pack: CoexactOperatorPack,block_rows: int=24) -> np.ndarray:
    """Exact all52 Weyl moments, including the derivative part of a_gdagger.

    Blocks avoid materializing termcount*fibre*channels*fullpolynomialdegree.
    Roundoff and q/theta integration remain caller-visible, not PSD-clipped.
    """
    count=len(pack.terms)//2;phases=np.asarray([term.function.phase for term in pack.terms[:count]])
    norms=np.sum(phases*phases,axis=1);kernel=np.exp(-np.maximum(norms[:,None]+norms[None,:]-2*phases@phases.T,0)/4)
    dots=phases@pack.packet;cross=(-1j*dots[:,None]/math.sqrt(2))*kernel
    photon=(float(np.vdot(pack.packet,pack.packet).real)+dots[:,None]*dots.conj()[None,:]/2)*kernel
    n=pack.base_channel_count;result=np.zeros((2*n,2*n),complex)
    for start in range(0,pack.fibre_dimension,block_rows):
        c=coefficient_block(pack,slice(start,min(start+block_rows,pack.fibre_dimension)))
        for matrix,i,j in ((kernel,0,0),(cross,0,1),(cross.conj().T,1,0),(photon,1,1)):
            applied=(matrix@c.reshape(count,-1)).reshape(c.shape)
            result[i*n:(i+1)*n,j*n:(j+1)*n]+=np.einsum('tfk,tfl->kl',c.conj(),applied,optimize=True)
    return result


def original_LAB_effect_coefficients(model: Any,data: dict[str,Any],sample: dict[str,Any],frozen_E2_coefficients: np.ndarray) -> np.ndarray:
    """Coefficient rows in EXISTING hard15/gamma/sin scalar library, fixed YLAB."""
    em=np.r_[np.zeros(52),sample['em'][52:]];old=model.original.basis_at(sample['q'],em,sample['point'])
    raw=model.charged_coordinates@frozen_E2_coefficients
    source=model.source_bank.source_values(old['point']['phi'],old['point']['links'],old['theta'],old['a'],old['neutral'])
    extra=model.noether_bank.conditional_coefficients(old,model.source_bank,model.prepared14)['charged']
    coefficient=np.einsum('fks,k->fs',extra,raw[1680:],optimize=True)
    coefficient[:,:15]+=source@raw[:1680].reshape(112,15)
    phase=np.exp(1j*model.boost['Omega']*(model.boost['Y0']@sample['q'])/model.chi)
    return phase*coefficient


def LAB_scalar_plane_kernels(model: Any,phases: np.ndarray,packet: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    """Existing hard15/gamma/sin against plane and FULL creator-plane, stably.

    Actual finite Gaussian/Wick identities; consumer compares to the original
    GaussianFunction/gaussian_kernel ABI, not a projected oscillator fallback.
    """
    hard=model.noether_bank.hard;noether=model.noether_bank.coefficients
    weight=np.exp(-np.sum(phases*phases,axis=1)/4);alpha=1j*phases@hard.packets.conj()/math.sqrt(2)
    labels=hard.labels;index={label:i for i,label in enumerate(labels)}
    hard_plane=np.asarray([weight*np.prod(alpha**np.asarray(label),axis=1)/math.sqrt(math.prod(math.factorial(n) for n in label)) for label in labels])
    hard_creator=np.zeros_like(hard_plane)
    overlap=hard.packets.conj().T@packet
    for row,label in enumerate(labels):
        for mode,n in enumerate(label):
            if n:
                lower=list(label);lower[mode]-=1;hard_creator[row]+=math.sqrt(n)*overlap[mode]*hard_plane[index[tuple(lower)]]
    b,flow=noether['phases'],noether['coexact'];shared=np.exp(-(np.sum(b*b,axis=1)[:,None]+np.sum(phases*phases,axis=1)[None,:])/4)
    z=b@phases.T/2;sinh,cosh=np.sinh(z),np.cosh(z);fphase=flow@phases.T
    gamma_plane=.5j*np.sum(fphase*shared*sinh,axis=0)
    gamma_creator=np.sum(shared*(2*(flow@packet)[:,None]*sinh+fphase*(b@packet)[:,None]*cosh),axis=0)/(2*math.sqrt(2))
    sine_plane=1j*shared*sinh;sine_creator=(b@packet)[:,None]*shared*cosh/math.sqrt(2)
    return np.vstack((hard_plane,gamma_plane,sine_plane)),np.vstack((hard_creator,gamma_creator,sine_creator))


def original_LAB_effect_moments(model: Any,data: dict[str,Any],sample: dict[str,Any],pack: CoexactOperatorPack,frozen_E2_coefficients: np.ndarray) -> dict[str,Any]:
    coefficient=original_LAB_effect_coefficients(model,data,sample,frozen_E2_coefficients)
    scalar=model.noether_bank.scalar_gram();norm=float(np.einsum('fs,st,ft->',coefficient.conj(),scalar,coefficient,optimize=True).real)
    count=len(pack.terms)//2;phases=np.asarray([term.function.phase for term in pack.terms[:count]])
    plane,creator=LAB_scalar_plane_kernels(model,phases,pack.packet)
    left=plane.T@coefficient.conj().T;left_gamma=creator.T@coefficient.conj().T
    row=np.zeros(pack.channel_count,complex)
    for i,term in enumerate(pack.terms[:count]):
        c=term.coefficient();row[term.channels]+=left[i]@c;row[term.channels+pack.base_channel_count]+=left_gamma[i]@c
    scalar_values,_=model.noether_bank.scalar_values(sample['em'][:52]);value=coefficient@scalar_values
    actual=model.basis_at(sample['q'],sample['em'],sample['point'])['charged']@frozen_E2_coefficients
    return dict(coexact_exact_original_LAB_norm_squared=norm,LAB_cross_row=row,
        original_LAB_function_reconstruction_error=float(np.linalg.norm(value-actual)),
        original_saved_E2_NOT_moving_refit_normalized_or_whitened=True,
        norm_scope='Exact all52 conditional q/theta moment; q/theta integration and physical E0 contractivity must be observed, not assumed')
