"""Original frozen Href: REAL all-mode quantum electric/Coulomb source kernels.

Finite-Strang Euclidean kernels, not a beta_norm source filter, RPA, or a
macro/full-Phi imaginary-time certificate. All four raw Refholes are retained.
"""
from __future__ import annotations
import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any
import numpy as np
from induced_phi_em_source_translation import SOURCE_BYTES as TRANSLATION_BYTES
from induced_phi_em_ygraph_clock_transfer import checked
from induced_phi_em_macro_preparation import build_native
from induced_phi_em_gaussian_variational import matter_hamiltonian
from induced_reference_electric_graph import graph_state
from induced_reference_field_following import path_derivative

ROOT=Path(__file__).resolve().parent
SOURCE_BYTES={**TRANSLATION_BYTES,Path(__file__).name:Path(__file__).read_bytes()}
REVISION='1178517cdc4236ce3a4af6a26bed880eb20679df709826463e4db7819ea0d208'


class OverlapCofactors:
    """Established SVD compound-matrix convention, no division at singulars.

    Same mathematics as supplier's frozen endpoint-cofactor class. Excluded
    products use prefix/middle/suffix products, not inverse overlaps or masks.
    """
    def __init__(self, overlap:np.ndarray)->None:
        U,singular,Vh=np.linalg.svd(overlap)
        self.left,self.right=U.conj().T,Vh.conj().T
        self.phase=np.linalg.det(U)*np.linalg.det(Vh)
        n=len(singular)
        prefix=np.ones(n+1);suffix=np.ones(n+1)
        prefix[1:]=np.cumprod(singular)
        suffix[:-1]=np.cumprod(singular[::-1])[::-1]
        self.one=prefix[:-1]*suffix[1:]
        self.two=np.zeros((n,n))
        for i in range(n):
            middle=1.
            for j in range(i+1,n):
                value=prefix[i]*middle*suffix[j+1]
                self.two[i,j]=self.two[j,i]=value
                middle*=singular[j]
        self.determinant=complex(self.phase*prefix[-1])

    def jet(self,L:np.ndarray,R:np.ndarray,S:np.ndarray)->tuple[complex,complex,complex,complex]:
        l,r,s=(self.left@value@self.right for value in (L,R,S))
        first_left=self.phase*np.sum(np.diag(l)*self.one)
        first_right=self.phase*np.sum(np.diag(r)*self.one)
        mixed=self.phase*(np.sum(np.diag(s)*self.one)+np.sum((np.outer(np.diag(l),np.diag(r))-l*r.T)*self.two))
        return self.determinant,complex(first_left),complex(first_right),complex(mixed)


def combine(active:tuple[complex,complex,complex,complex],auxiliary:tuple[complex,complex,complex,complex])->tuple[complex,complex]:
    a,al,ar,alr=active;b,bl,br,blr=auxiliary
    b61,b62,b63=b**61,b**62,b**63
    norm=a*b63
    current=alr*b63+63*(al*br+ar*bl+a*blr)*b62+63*62*a*bl*br*b61
    return complex(norm),complex(current)


def half_heat(native:dict[str,Any],profile:dict[str,Any],theta:np.ndarray,step:float)->np.ndarray:
    h=matter_hamiltonian(native,profile,theta)
    values,vectors=np.linalg.eigh(h)
    return (vectors*np.exp(-step*values/2))@vectors.conj().T


def endpoint(native:dict[str,Any],theta:np.ndarray,source_site:int,directions:np.ndarray)->tuple[list[np.ndarray],list[np.ndarray]]:
    frames=[];vertices=[]
    alpha=np.array([1,1j])/np.sqrt(2)
    for profile in native['Pprofiles']:
        state=graph_state(native,profile,theta)
        frame=state['occupied']
        if profile['load']==1:
            raw=np.eye(4*native['V'])[:,4*source_site:4*source_site+4]
            frame=np.column_stack((frame,raw))
        frames.append(frame)
        vertices.append(sum(alpha[i]*path_derivative(native,profile,theta,f,1) for i,f in enumerate(directions)))
    return frames,vertices


def kernel_jets(frames:list[np.ndarray],left_vertices:list[np.ndarray],propagated:list[np.ndarray],nocc:int)->tuple[complex,complex,complex,complex]:
    all_matrices=[]
    for Fbra,Jbra,Vprop in zip(frames,left_vertices,propagated):
        size=Fbra.shape[1]
        Vbase,Vcurrent=Vprop[:,:size],Vprop[:,size:]
        M=Fbra.conj().T@Vbase
        L=Fbra.conj().T@Jbra.conj().T@Vbase
        R=Fbra.conj().T@Vcurrent
        S=Fbra.conj().T@Jbra.conj().T@Vcurrent
        all_matrices.append((M,L,R,S))
    def coefficients(matrices:tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray],selection:np.ndarray)->tuple[complex,complex,complex,complex]:
        M,L,R,S=(a[np.ix_(selection,selection)] for a in matrices)
        return OverlapCofactors(M).jet(L,R,S)
    auxiliary=coefficients(all_matrices[1],np.arange(nocc))
    neutral=combine(coefficients(all_matrices[0],np.arange(nocc)),auxiliary)
    charged=np.zeros(2,complex)
    for b in range(4):
        indices=np.r_[np.arange(nocc),nocc+b]
        charged+=np.array(combine(coefficients(all_matrices[0],indices),auxiliary))/4
    return neutral[0],charged[0],neutral[1],charged[1]


def summarize(values:np.ndarray)->dict[str,Any]:
    count=len(values)
    mean=np.mean(values,axis=0)
    re=np.std(np.real(values),axis=0,ddof=1)/np.sqrt(count)
    im=np.std(np.imag(values),axis=0,ddof=1)/np.sqrt(count)
    return dict(real=np.real(mean).tolist(),imag=np.imag(mean).tolist(),sampling_SE_real=re.tolist(),sampling_SE_imag=im.tolist(),samples=count,deterministic_integral_certificate=False)


def actual_quantum_kernel(mean:dict[str,Any],beta:F,samples:int,steps:list[int],seed:int)->dict[str,Any]:
    native=build_native(mean,2);V,M=native['V'],native['M'];g=64.;s=2.
    source_site=int(np.flatnonzero(np.all(native['sites']==(1,1,0),axis=1))[0]);ref=0
    f=np.cos(np.pi*native['sites'][:,0])/np.sqrt(V)
    directions=np.zeros((2,3*V));directions[0,1::3]=f;directions[1,2::3]=f
    maximum=max(steps);bfloat=float(beta)
    if samples<4 or beta<=0 or min(steps)<=0 or any(maximum%n for n in steps):
        raise ValueError('Positivebeta/fourormoresamples/dividingStranglevels required')
    lv,lb=np.linalg.eigh(native['Lplus'])
    positive=lv>1e-12
    A0map=lb[:,positive]*np.sqrt(lv[positive])
    if A0map.shape[1]!=V-1:
        raise ArithmeticError('Actual realGaussCoulomb zeromode not removed')
    reference_free=0.
    for profile in native['Pprofiles']:
        values=np.linalg.eigvalsh(matter_hamiltonian(native,profile,np.zeros(3*V)))
        reference_free+=profile['load']*values[:2*V].sum()
    common_phase=np.exp(bfloat*reference_free)
    free_scalar=(1+g*bfloat/(2*s))**(-M/2)
    normalizer=free_scalar*common_phase
    rng=np.random.default_rng(seed)
    ledgers={n:[] for n in steps};photon_ledgers={n:[] for n in steps};current_photon_ledgers={n:[] for n in steps};wrong_ref={n:[] for n in steps}
    source_raw=[];source_current_raw=[];reverse_errors=[]
    electric_only=[];density_kernel=[];density_wrong=[]
    rho=np.zeros(V);rho[source_site]=1;rho[ref]=-1
    actual_basis_Coulomb_energy=float(g*rho@native['Lplus']@rho)
    fractions=np.arange(2*maximum+1)/(2*maximum)
    for sample in range(samples):
        middle=rng.normal(0,np.sqrt(s),M)
        difference=rng.normal(0,np.sqrt(2*g*bfloat/(1+g*bfloat/(2*s))),M)
        increments=rng.normal(0,np.sqrt(g*bfloat/maximum),(2*maximum,M))
        motion=np.vstack((np.zeros(M),np.cumsum(increments,axis=0)))
        bridge=motion-fractions[:,None]*motion[-1]
        coordinates=middle+(fractions[:,None]-.5)*difference+bridge
        fields=coordinates@native['Q'].T
        fine_A0=np.sqrt(2*g*bfloat/maximum)*rng.normal(size=(maximum,V-1))@A0map.T
        Fket,Jket=endpoint(native,fields[0],source_site,directions)
        Fbra,Jbra=endpoint(native,fields[-1],source_site,directions)
        # ActualSOURCEprobability/control is at ketTheta, not normalizedemptyZ.
        initial=[np.column_stack((Fj,Jj@Fj)) for Fj,Jj in zip(Fket,Jket)]
        raw0=kernel_jets(Fket,Jket,initial,2*V)
        source_raw.append(raw0[1]);source_current_raw.append(raw0[3])
        theta_ket=directions@fields[0];theta_bra=directions@fields[-1]
        summed_alpha=fine_A0.sum(axis=0)
        neutral_background=np.exp(1j*128*summed_alpha.sum())
        charged_ref3=neutral_background*np.exp(1j*summed_alpha[ref])
        electric_only.append(free_scalar*np.array(kernel_jets(Fbra,Jbra,initial,2*V)))
        density_kernel.append(np.exp(-1j*summed_alpha@rho))
        density_wrong.append(np.exp(-1j*summed_alpha[source_site]))
        for n in steps:
            width=maximum//n;dt=bfloat/n
            propagated=[a.copy() for a in initial]
            actual_words=[np.eye(4*V,dtype=complex),np.eye(4*V,dtype=complex)] if sample==0 else None
            blocks=[]
            for j in range(n):
                midpoint=width*(2*j+1)
                alpha=fine_A0[j*width:(j+1)*width].sum(axis=0)
                D=np.repeat(np.exp(-1j*alpha),4)
                for si,profile in enumerate(native['Pprofiles']):
                    H=half_heat(native,profile,fields[midpoint],dt)
                    word=(H*D[None,:])@H
                    propagated[si]=word@propagated[si]
                    if actual_words is not None:
                        actual_words[si]=word@actual_words[si]
                        blocks.append((si,H,alpha))
            values=np.array(kernel_jets(Fbra,Jbra,propagated,2*V))
            weights=normalizer*values*np.array([neutral_background,charged_ref3,neutral_background,charged_ref3])
            ledgers[n].append(weights)
            photon_ledgers[n].append(weights[1]*np.outer(theta_bra,theta_ket))
            current_photon_ledgers[n].append(weights[3]*np.outer(theta_bra,theta_ket))
            wrong_ref[n].append(normalizer*values[1]*neutral_background)
            if actual_words is not None:
                reverse=[np.eye(4*V,dtype=complex),np.eye(4*V,dtype=complex)]
                for si,H,alpha in reversed(blocks):
                    reverse[si]=((H*np.repeat(np.exp(1j*alpha),4)[None,:])@H)@reverse[si]
                reverse_errors.append(max(float(np.linalg.norm(reverse[si]-actual_words[si].conj().T,2)) for si in range(2)))
    data={}
    for n in steps:
        values=np.asarray(ledgers[n]);photon=np.asarray(photon_ledgers[n]);current_photon=np.asarray(current_photon_ledgers[n]);wrong=np.asarray(wrong_ref[n])
        m=np.mean(values,axis=0)
        # Pairedpath reversedA0/order/endpoints is an exact proposalinvolution.
        # Primaryunpairedcomplexstatistics retained; this is notphasequenching.
        data[str(n)]=dict(ALLcomplex_quantum_source_kernels=summarize(values),column_order=['neutralSOURCEheat','rawALL4RefholechargedSOURCEheat','neutralcomplexCURRENTsourceheat','rawALL4RefholechargedcomplexCURRENTsourceheat'],charged_photon_SOURCE_heat_kernel=summarize(photon),charged_complexCURRENT_photon_SOURCE_heat_kernel=summarize(current_photon),WRONG_omitted_Ref3_density_phase_mean=summarize(wrong),physical_Ref3_phase_effect=summarize(values[:,1]-wrong),raw_heat_ratio_charged_over_neutral=dict(real=float(np.real(m[1]/m[0])),imag=float(np.imag(m[1]/m[0]))),p_half_normalized_heat_ratio=dict(real=float(np.real(2*m[1]/m[0])),imag=float(np.imag(2*m[1]/m[0]))),not_mass_or_Green_value=True)
    differences={f'{left}_minus_{right}':summarize(np.asarray(ledgers[left])-np.asarray(ledgers[right])) for left,right in zip(steps,steps[1:])}
    if max(reverse_errors)>1e-10 or max(abs(value-.5) for value in source_raw)>1e-10 or not np.all(np.isfinite(np.asarray([ledgers[n] for n in steps]))):
        raise ArithmeticError('Actualquantumkernel domain/source/reversal failed')
    electric_array=np.asarray(electric_only)
    benchmarks=dict(actual_ELECTRIC_ONLY_same_SOURCE_kernels=summarize(electric_array),exact_translation_and_fixedN_ELECTRIC_charged_minus_half_neutral='0 (fullintegral), notperpath',actual_ELECTRIC_charged_minus_half_neutral_sampling=summarize(electric_array[:,1]-.5*electric_array[:,0]),WRONG_ELECTRIC_identity_neutral_and_charged=[1.,.5],actual_integer_charged_basis_density=dict(mobile_active_source_occupation=3,other_mobile_onsite_each=2,auxiliary_flavours=63,Ref=3,rho=rho.tolist(),actual_Coulomb_energy=actual_basis_Coulomb_energy,exact_HS_density_heat_target=float(np.exp(-bfloat*actual_basis_Coulomb_energy)),actual_SHARED_real_A0_HS_density_kernel=summarize(np.asarray(density_kernel)),WRONG_omitting_Ref3_density_kernel=summarize(np.asarray(density_wrong))))
    return dict(N=2,V=V,M=M,source_position=[1,1,0],Ref_position=[0,0,0],real_noncompact_EM_modes=M,ALLcharged_species=64,active_source_N_to_C=[2*V,2*V+1],auxiliary_count=63,auxiliary_N=2*V,Ref_N_to_C=[4,3],fixed_uniform_background=128,fixed_reference_background=4,a=1/64,UEphys=1,g_electric_AND_Coulomb=g,bare_B_squared=None,variance=s,beta=str(beta),samples=samples,seed=seed,Strang_steps=steps,common_NEUTRAL_free_CAR_reference_energy=float(reference_free),exact_free_Gaussian_EM_scalar=float(free_scalar),shared_zeroMean_real_A0_modes=V-1,SOURCE_raw_Refhole_probability_range=[float(np.min(np.real(source_raw))),float(np.max(np.real(source_raw)))],actual_complexSOURCE_current_norm_squared_sampling=summarize(np.asarray(source_current_raw)),physical_reversed_path_operator_error=max(reverse_errors),component_benchmarks=benchmarks,rows=data,paired_Strang_level_differences=differences,scope='ACTUALfiniteStrang quantumHref kernels integratingall17REALelectricBrownian+SHARED7A0realGaussianCoulombfields andall64CAR, originalSOURCE unchanged/raw4Refholes/current(y+iz)/sqrt2. SamplingSE notintegralorTrottercontinuumcertificate; sourceNoe/Phi/heavyquantumevolution NOTsimulated andrealunitaryfullPhi4096prices NOTextendedtoheat.')


def run(translation_path:Path,beta:F,samples:int,steps:list[int],seed:int)->dict[str,Any]:
    started=time.monotonic();raw=translation_path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!='865e8a36c76393e3614ea1b9b77b1fe6bf124e2a4ad69e8d4d51a4bf2344efd0':
        raise ArithmeticError('Actual17f351 source translation epoch required')
    translation=json.loads(raw)
    _,response=checked(translation['input_descriptor']['input_receipts']['response'])
    _,source=checked(response['input_descriptor']['input_receipts']['source'])
    _,transition=checked(source['input_descriptor']['input_receipts']['transition'])
    _,receiver=checked(transition['input_descriptor']['input_receipts']['receiver'])
    _,energy=checked(receiver['input_descriptor']['input_receipts']['source_energy'])
    _,green=checked(energy['input_descriptor']['input_receipts']['green'])
    action_raw,action=checked(green['input_descriptor']['input_receipts']['original_action'])
    _,four=checked(green['input_descriptor']['input_receipts']['response'])
    _,inserted=checked(four['input_descriptor']['input_receipts']['inserted'])
    _,force=checked(inserted['input_descriptor']['input_receipts']['force'])
    _,current=checked(force['input_descriptor']['input_receipts']['current'])
    _,retarded=checked(current['input_descriptor']['input_receipts']['retarded'])
    _,field=checked(retarded['input_descriptor']['input_receipts']['clock'])
    _,bridge=checked(field['input_descriptor']['input_receipts']['bridge'])
    _,mean=checked(bridge['input_descriptor']['input_receipts']['mean'])
    if action['real_EM_Coulomb']['Ref_bare_charge']!=1 or action['real_EM_Coulomb']['Ref_occupied']!=4 or action['real_EM_Coulomb']['Ref_background']!=-4:
        raise ArithmeticError('Actual primitiveRef/background manifest changed')
    descriptor=dict(plan=translation['input_descriptor']['plan'],plan_revision=REVISION,scientific_execution_revision=REVISION,input_receipts=dict(translation=dict(path=str(translation_path),sha256=hashlib.sha256(raw).hexdigest())),source_abi=translation['input_descriptor']['source_abi'],original_action_sha256=hashlib.sha256(action_raw).hexdigest(),native_N=2,beta=str(beta),samples=samples,steps=steps,seed=seed,source='ORIGINALunfilteredrealGaussian2/HermY000AP222 atnativeMIDPOINTframe, localE=.5sumALL4cdag_x rRef, x(1,1,0),Ref0; SAMEa1/64active1m.2+aux63m2,g64PiANDCoulomb',observable='NativeEuclideanfiniteStrangsourceheat/current/photonkernels, NOTsourcebetaNormgenerator/imaginarytimefilterreplacement orrealTimeGreen',clock4096_UNCHANGED=True,physical_conversions='JA=aJTheta, A=Theta/a; currentSOURCEheat*a², photonSOURCEheat/a², current×photonSOURCEheat no neta',macro_fullPhi_analytic_continuation_claim=False)
    return dict(input_descriptor=descriptor,input_sha256=hashlib.sha256(json.dumps(descriptor,sort_keys=True).encode()).hexdigest(),actual_real_EM_Coulomb_quantum_kernels=actual_quantum_kernel(mean,beta,samples,steps,seed),
        exact_finite_Strang_quantum_kernel='S_delta=e^-delta gPi²/2 e^-delta HCAR(Theta)/2 e^-delta g rhoLplusrho e^-delta HCAR(Theta)/2 e^-delta gPi²/2, S_delta^n. ElectricREALheatkernel Brownianbridge(allM), endpointsGaussian source measure diagonalizedEXACT: midpointN(0,s),diffN(0,2g beta/(1+g beta/(2s))), freeScalar(1+g beta/(2s))^-M/2; bridgecov2g beta[min(t,u)-tu]. Phi/heavyfrozenHrefONLY. SharedA0alpha=sqrt(2g delta)Lplus^1/2 xi_real, densityHS e^-i alpha rho exactGaussianidentity. EVERYspecies seesSAMEalpha; no Coulomb meanfield/separateflavourHS. Uniformbackground128 contributesexp(+i128sumalpha), chargedRef3−delta0 exp(+i sumalpha_Ref), all4holesSAMERefphase. Primitiveg64Both, no doubledC, compactEM/subsetmodes/bareB.',
        inverse_free_CURRENT_source_jets='ActiveNEUTRALendpointW16; chargedALL4 rawF_b=[W16,physicalf_x,b], noQsea/emptyZ/whitening: summed4det/4. Propagate F and Jplus F throughliteral32modeCAR heat/densitymatrices. Left/right/mixedoverlap M,L,R,S withphysicalJplus†/Jplus endpoints. SVDphase detUdetVdag andallproducts singularvaluesexcluding1/2, neverdivide; det(M+zL+wR+zwS) coeffzw yieldsSOURCEJdagU J, includingnonSlaterCURRENTpacket exactly. Combine active a andAUXb^63: aLRb63+63(aL bR+aR bL+a bLR)b62+63*62 a bL bR b61. FullTOTALcurrentcrossflavours/PH/sea retainedbeforepowers. establishedsupplierfrozenOverlapCofactorsconvention, no inverse/pseudoinverse/absdet/clipping/phasequenching. Photoninsertionsmultiplybra/ket realTheta_i asactualSOURCEpackets.',
        scope='Physicalfinite-Strang originalHref quantumkernel+exploratoryMonteCarlo, notexactcontinuum/trottererrorcertificate orfullPhiheat/provedspectralgap/pole/commoncone/Compton. Nativeimage notmacrosourceidentity, allfrozenbytes/clocks/sourceprobabilitiesuntouched. Additionalprobe beta isdeclaredinput, notpreparation/filter/survivalgate.',
        source_at_import={name:dict(sha256=hashlib.sha256(data).hexdigest(),bytes_utf8=data.decode()) for name,data in SOURCE_BYTES.items()},source_unchanged_since_import={name:(ROOT/name).read_bytes()==data for name,data in SOURCE_BYTES.items()},runtime=dict(executable=sys.executable,python=sys.version,numpy=np.__version__),elapsed_seconds=time.monotonic()-started)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--translation',type=Path,required=True)
    parser.add_argument('--beta',type=F,required=True)
    parser.add_argument('--samples',type=int,required=True)
    parser.add_argument('--steps',type=int,nargs='+',required=True)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=run(args.translation,args.beta,args.samples,args.steps,args.seed)
    if not all(result['source_unchanged_since_import'].values()):raise ArithmeticError('Source changed during actual calculation')
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('input_sha256','actual_real_EM_Coulomb_quantum_kernels','elapsed_seconds')},indent=2))
