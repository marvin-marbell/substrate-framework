"""Specific Pauli SOURCE metric: spin-staggered identity and full-Phi prices."""
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
from scipy.linalg import block_diag
from induced_phi_ir_well_transfer import rational
from induced_phi_em_ygraph_evolving_field_clock import su,sl
from induced_phi_em_ygraph_clock_transfer import exact,positive_exact
from induced_reference_electric_graph import to_spin_basis

sys.set_int_max_str_digits(0)
ROOT=Path(__file__).resolve().parent
OWN_BYTES=Path(__file__).read_bytes()
import induced_phi_pauli_magnetic_join as pauli
quantum=pauli.quantum
SOURCE_BYTES={**pauli.SOURCE_BYTES,Path(__file__).name:OWN_BYTES}
PAULI_SHA='847ff43b96417a222425c922c98aa474877d3ec23ba9558ff9c61073736bbf15'


def spin_structure(mean:dict[str,Any])->dict[str,Any]:
    native=quantum.build_native(mean,4);V=native['V'];mode=pauli.magnetic_shape(native)
    rng=np.random.default_rng(20261017);theta=native['Q']@(np.sqrt(2)*rng.normal(size=native['M']))
    records=[]
    for profile in native['Pprofiles']:
        state=quantum.graph_state(native,profile,theta);B=state['B'];plus=B[:,:2]
        gamma=native['frame']/np.sqrt([native['g0']]+[native['gs']]*3)[:,None,None]
        gamma5=gamma[0]@gamma[1]@gamma[2]@gamma[3]
        tau=[-1j*plus.conj().T@gamma[j]@gamma5@plus for j in (1,2,3)]
        siteG=[]
        for site in native['sites']:
            G=np.eye(2,dtype=complex)
            for t,n in zip(tau,site):G=G@np.linalg.matrix_power(t,int(n))
            siteG.append(G)
        G=np.asarray(block_diag(*siteG));kg=G.conj().T@state['K']@G;rg=G.conj().T@state['R']@G
        parity=(-1.)**np.sum(native['sites'],axis=1);mask=(parity[:,None]==parity[None,:])[:,:,None,None]
        def blocks(A:np.ndarray)->np.ndarray:return A.reshape(V,2,V,2).transpose(0,2,1,3)
        def scalar_error(A:np.ndarray)->float:
            b=blocks(A);return float(np.linalg.norm(b-np.trace(b,axis1=2,axis2=3)[:,:,None,None]*np.eye(2)/2))
        Fop=pauli.operator(native,profile,theta,mode);fb=to_spin_basis(Fop,B);A=G.conj().T@fb[np.ix_(state['pp'],state['pp'])]@G
        x=int(np.flatnonzero(np.all(native['sites']==(1,1,0),axis=1))[0]);Q=np.eye(4*V)-state['P'];D=Q[:,4*x:4*x+4]@Q[4*x:4*x+4,:]/2;db=to_spin_basis(D,B)
        dg=[G.conj().T@db[np.ix_(sector,sector)]@G for sector in (state['pp'],state['nn'])]
        # Forbidden extension only: a two-link spatial-Pauli term, NOT added to H.
        bad=np.zeros_like(state['K'])
        for path in profile['paths']:
            if path['axis']==2 and path['length']==2:
                i,j=path['source'],path['target'];h=.01*tau[2]*path['seam']*np.exp(1j*np.sum(theta[list(path['edges'])]));bad[2*i:2*i+2,2*j:2*j+2]+=h;bad[2*j:2*j+2,2*i:2*i+2]+=h.conj().T
        errors=dict(K_spin_scalar=scalar_error(kg),R_spin_scalar=scalar_error(rg),R_odd_spatial_parity=float(np.linalg.norm(blocks(rg)*(~mask))),Pauli_force_even_parity_spin_trace=float(np.linalg.norm(np.trace(blocks(A)*mask,axis1=2,axis2=3))),charged_D_diagonal_spin_scalar=max(scalar_error(d) for d in dg),charged_D_diagonal_odd_parity=max(float(np.linalg.norm(blocks(d)*(~mask))) for d in dg))
        falsifier=scalar_error(G.conj().T@(state['K']+bad)@G)
        if max(errors.values())>1e-9 or falsifier<.01:raise ArithmeticError('Actual preparation-only spin-staggered structural identity/falsifier failed')
        records.append(dict(load=profile['load'],actual_structural_errors=errors,FORBIDDEN_two_link_spatial_Pauli_extension_spin_scalar_error=falsifier,forbidden_operator_NOT_in_actual_H=True))
    return dict(N=4,V=V,M=native['M'],seed=20261017,profiles=records,physical_H_not_spin_scalar='The actual one-link Wilson Gamma0 term transforms into spinful odd-parity hops. This is a canonicalY preparation identity, NOT a symmetry of physical H or its heat/real-time evolution.')


def run(path:Path)->dict[str,Any]:
    started=time.monotonic();raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=PAULI_SHA:raise ArithmeticError('Frozenb50d/847f physicalPauli epoch required')
    cap=json.loads(raw);_,record=quantum.checked(cap['input_descriptor']['input_receipts']['prior_joined_heat'])
    for key in ('full_source','contact','transfer','gram','force','join','heat'):_,record=quantum.checked(record['input_descriptor']['input_receipts'][key])
    heat=record;mean=pauli.contact.transfer.packets.force.joined.mean_from_source(heat)
    _,record=quantum.checked(heat['input_descriptor']['input_receipts']['translation']);chain={}
    for key in ('response','source','transition','receiver','source_energy','green','response','inserted','force','current','retarded','clock'):
        spec=record['input_descriptor']['input_receipts'][key];_,record=quantum.checked(spec);chain[key]=(spec,record)
    green=chain['green'][1];field=chain['clock'][1];gd=green['input_descriptor'];N,V=gd['N'],gd['V'];a=F(gd['a']);p=F(1,2);s=F(2);g=F(gd['UEphys'])/a
    if (N,V,gd['Lambda'],gd['variance'],gd['clock'])!=(262144,262144**3,'2^8192','2','4096'):raise ArithmeticError('Actual macro source/law/clock substitution forbidden')
    native=quantum.build_native(mean,4);profiles=native['Pprofiles'];c=[pauli.scalars(profile) for profile in profiles]
    if any(z['path'][2][1]!=0 for z in c):raise ArithmeticError('Spin-staggering requires actual nearest-neighbour-only spatialPhi5')
    enc=mean['centre_enclosure'];g0L,g0U,gsL,gsU=[rational(enc[k]) for k in ('g0_lower','g0_upper','gs_lower','gs_upper')]
    aux=c[1];margin=abs(aux['onsite'])-2*abs(aux['path'][2][0]);kL=abs(aux['path'][1][1])*sl(gsL)
    spatial=pauli.joined.full.coefficient_data(profiles[1])[1];Kcap=2*spatial*su(gsU);dL=2*abs(aux['onsite'])*sl(g0L)+g;dU=2*abs(aux['onsite'])*su(g0U)+g
    PH=2*kL*kL*margin*margin/(dU*dU*(1+(Kcap/dL)**2)**2);B=p*63*PH
    sums=[abs(z['onsite'])+6*sum(abs(z['path'][length][0]) for length in (1,2)) for z in c]
    fmax=su(F(2,V));rows=[fmax*z/sl(F(8)) for z in sums]
    # Every Ref-hole conditional active component is one Slater P+rank1.
    # Neutral mean vanishes pointwise by spin scalar/even-parity mechanism.
    # Variance <= ||F||_F^2; only active rank1 can contribute a mean <= row.
    Uphi=p*((sums[0]**2+63*sums[1]**2)/2+rows[0]**2);GA=p*s/a**2
    lambdaL=min(B,GA);lambdaU=max(Uphi,GA)
    drows=[2*fmax/sl(F(8))*3*sum(abs(z['path'][length][0])*su(F(length)) for length in (1,2)) for z in c]
    count=[2*V+1,126*V];Rsum=sum(n*r for n,r in zip(count,rows));DRsum=sum(n*r for n,r in zip(count,drows))
    Lie=exact(field['alltime_original_reference_field_link_Lie_L2_upper']);eps=exact(green['charged_sector_recomputed_alltime_original_Phi_reference_wave_error'])
    eGram=su(p)*DRsum*Lie;eta=eGram/sl(lambdaL);fullL=lambdaL*(1-eta)**2
    DE=su(p)*su(F(N,4))*Lie;forceWave=eGram+Rsum*DE;photonWave=su(s)*DE/a;eWave=su(forceWave**2+photonWave**2)
    eProp=eWave+su(lambdaU)*eps;kernel=(su(lambdaU)+eGram)*eProp+su(lambdaU)*eWave
    if margin<=0 or not 0<eta<1 or Uphi<B:raise ArithmeticError('Positive actualSOURCE joint metric lost')
    values=dict(aux_PH_norm_squared_lower=PH,RAWcharged_aux63_PH_Schur_lower=B,RAW_reference_Pauli_SOURCE_Gram_upper=Uphi,exact_RAW_reference_photon_SOURCE_Gram_eigenvalue=GA,reference_joint_SOURCE_Gram_eigenvalue_lower=lambdaL,reference_joint_SOURCE_Gram_eigenvalue_upper=lambdaU,fullPhi_sameEU_joint_SOURCE_map_difference_upper=eGram,fullPhi_joint_SOURCE_Gram_relative_error_upper=2*eta+eta**2,fullPhi_joint_SOURCE_Gram_eigenvalue_lower=fullL,fullPhi_referenceE0_joint_SOURCE_wave_difference_upper=eWave,CONDITIONAL_original4096_joint_wave_difference_upper=eProp,CONDITIONAL_original4096_joint_relative_wave_difference_upper=eProp/sl(fullL),CONDITIONAL_original4096_joint_SAMEfullGram_kernel_difference_upper=kernel/fullL)
    desc=dict(plan=gd['plan'],plan_revision=pauli.REVISION,scientific_execution_revision=pauli.REVISION,input_receipts=dict(magnetic=dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest()),field=chain['clock'][0],green=chain['green'][0]),source_abi=gd['source_abi'],N=N,V=V,M=gd['M'],a=gd['a'],UEphys=gd['UEphys'],Lambda=gd['Lambda'],variance=gd['variance'],clock=gd['clock'],charged_counts=green['charged_sector_counts'],source='UNCHANGEDoriginalfullPhi222/Gaussian2/canonicalY000/all64/AP222/Ref4to3EU BEFORE any heat; probabilityp=1/2 ONLYatSOURCE.',probe='SpecificchargedmatterPhi-potentialgradient betaSigma_z/sinqx and physicalAy/Azcosqx; NOTtotalcanonicalPhi force. Heavy/boson/Noe force not removed.',metric='ExactunfilteredGaussian/CAR integral Gamma_REF=diag(Gamma_Pauli,4096,4096); Gamma_Pauli NOTnumericallysampled/whitened. FullGamma uses SAMEEU and strictlypositive lower, SAMEfullGamma both kernel comparands.')
    proof='For even N>=4, in Gamma0 source-spin basis tau_i=-i(Gamma_i Gamma5)++, G_x=prod_i tau_i^x_i is periodic. Gdag K G is scalar-spin nearest-neighbour hopping, hence R=(1+K²/d²)^-1 scalar-spin and even spatial parity. Pauli A_even (onsite/two-link) is traceless-spin; A_odd (one-link) meets zero odd-parity R. Thus TrR A=0 for each flavour at every Theta. Q Rx Q diagonal Gamma0 blocks are also scalar-spin/even spatial parity, so charged all4hole mean TrD F=0. Allforce/photon SOURCE cross entries vanish pointwise. This preparation-only identity DOES NOT extend to physical Wilson H. Source upper: ||F_alpha||_F <= (|onsite|+6sum|c0_l|)/sqrt2 by triangle of actual forward/reverse blocks and sum f²=1; Ref-hole conditional Slater variance bounded by ||F||_F², neutralmeans zero, active rank1 mean <= row. Auxiliary lower: scalar-spin trace selects only K_z; one-z-link coefficient >=2|ci|sqrt(gs)(|onsite|-2|c02|)/sqrt8, conservative N4 folded3-z subtraction; Frobenius projection gives 2|ci|²gs margin². sqrtR singular lower gives stated PH bound. Orthogonal auxiliary63PH sectors prove source Schur lower. Fullcolour derivative row prices actual orderedone/twolink transport differences sqrt(L)*Lie without pricing Phi amplitudes (derivative removed them). All4Ref-hole sum implies exactphoton4096I alsofullPhi. Fullmap singular perturbation yields positive fullGamma. GenericPhi-independentTheta/CAR packet extension of existing76ad/c42 sectoruniform block proof remains parent-review CONDITIONAL; sourcePhi dynamics/nonstationarysurvival retained, no extension of realunitary prices to heat.'
    return dict(input_descriptor=desc,input_sha256=hashlib.sha256(json.dumps(desc,sort_keys=True).encode()).hexdigest(),**{k:positive_exact(v) for k,v in values.items()},native_actual_spin_staggered_structural_control=spin_structure(mean),physical_proof_for_parent_review=proof,source_at_import={name:dict(sha256=hashlib.sha256(data).hexdigest(),bytes_utf8=data.decode()) for name,data in SOURCE_BYTES.items()},source_unchanged_since_import={name:(ROOT/name).read_bytes()==data for name,data in SOURCE_BYTES.items()},runtime=dict(executable=sys.executable,python=sys.version,numpy=np.__version__),elapsed_seconds=time.monotonic()-started)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--magnetic',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();r=run(args.magnetic)
    if not all(r['source_unchanged_since_import'].values()):raise ArithmeticError('Source changed during physical calculation')
    args.output.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps(dict(input_sha256=r['input_sha256'],bounds={k:[v['binary_exponent_lower'],v['binary_exponent_upper']] for k,v in r.items() if isinstance(v,dict) and 'binary_exponent_lower' in v},native=r['native_actual_spin_staggered_structural_control'],elapsed_seconds=r['elapsed_seconds']),indent=2))
