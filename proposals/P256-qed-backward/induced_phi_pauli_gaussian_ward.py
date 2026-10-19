"""Same-source Gaussian Ward estimator with actual canonicalY derivatives."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any
import numpy as np
from induced_reference_electric_graph import to_spin_basis,from_spin_basis

ROOT=Path(__file__).resolve().parent
OWN_BYTES=Path(__file__).read_bytes()
import induced_phi_pauli_initial_moment as initial
pauli=initial.pauli
quantum=initial.quantum
SOURCE_BYTES={**initial.SOURCE_BYTES,Path(__file__).name:OWN_BYTES}
MOMENT_SHA='2be1430a2087bd625b11eb847ea4d46ede3b8453da5af08d94d9e2bbbecd1bc0'


def vertices(native:dict[str,Any],profile:dict[str,Any],theta:np.ndarray,mode:dict[str,Any],direction:np.ndarray)->tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    Fop=pauli.operator(native,profile,theta,mode);H=initial.one_link_Wilson(native,profile,theta);DF=np.zeros_like(Fop);DH=np.zeros_like(H)
    for path in profile['paths']:
        x,y=path['source'],path['target'];phase=path['seam']*np.exp(1j*np.sum(theta[list(path['edges'])]));dphase=1j*np.sum(direction[list(path['edges'])])*phase
        v=path['scalars'][0]*(mode['mode'][x]+mode['mode'][y])*mode['delta']/2*dphase
        DF[4*x:4*x+4,4*y:4*y+4]+=v;DF[4*y:4*y+4,4*x:4*x+4]+=v.conj().T
        if path['length']==1:
            v=path['scalars'][0]*native['frame'][0]*dphase;DH[4*x:4*x+4,4*y:4*y+4]+=v;DH[4*y:4*y+4,4*x:4*x+4]+=v.conj().T
    return Fop,H,DF,DH


def tangent(native:dict[str,Any],profile:dict[str,Any],theta:np.ndarray,direction:np.ndarray)->tuple[dict[str,Any],np.ndarray]:
    state=quantum.graph_state(native,profile,theta,section=False);B=state['B'];pp,nn=state['pp'],state['nn']
    Dh=quantum.path_derivative(native,profile,theta,direction);Dhb=to_spin_basis(Dh,B);DY=(-1j*Dhb[np.ix_(pp,nn)])/state['denominator'];Y,R=state['Y'],state['R'];DR=-R@(DY@Y+Y@DY)@R;DYR=DY@R+Y@DR
    dp=np.empty_like(Dhb);dp[np.ix_(pp,pp)]=-DR;dp[np.ix_(nn,nn)]=DR;dp[np.ix_(pp,nn)]=-1j*DYR;dp[np.ix_(nn,pp)]=1j*DYR
    return state,from_spin_basis(dp,B)


def density_moment(states:list[dict[str,Any]],DP:list[np.ndarray],ops:list[tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]],x:int)->tuple[complex,complex]:
    cov=0j;dcov=0j
    for state,dP,(Fop,H,DF,DH) in zip(states,DP,ops):
        P=state['P'];Q=np.eye(len(P))-P
        cov+=state['load']*np.trace(P@Fop@Q@H)
        dcov+=state['load']*np.trace(dP@Fop@Q@H+P@DF@Q@H-P@Fop@dP@H+P@Fop@Q@DH)
    P=states[0]['P'];Q=np.eye(len(P))-P;dP=DP[0];Fop,H,DF,DH=ops[0];slice_x=slice(4*x,4*x+4)
    D=Q[:,slice_x]@Q[slice_x,:]/2;DD=-(dP[:,slice_x]@Q[slice_x,:]+Q[:,slice_x]@dP[slice_x,:])/2
    L=Fop@Q@H-H@P@Fop;DL=DF@Q@H-Fop@dP@H+Fop@Q@DH-DH@P@Fop-H@dP@Fop-H@P@DF
    return (cov+np.trace(D@L))/2,(dcov+np.trace(DD@L+D@DL))/2


def physical(mean:dict[str,Any],samples:int,seed:int)->dict[str,Any]:
    if samples<4:raise ValueError('Actual sampling uncertainty requires atleast4 samples')
    native=quantum.build_native(mean,4);V,M=native['V'],native['M'];a=float(native['a']);mode=pauli.magnetic_shape(native);x=int(np.flatnonzero(np.all(native['sites']==(1,1,0),axis=1))[0]);f=np.sqrt(2/V)*np.cos(2*np.pi*native['sites'][:,0]/4);direction=np.zeros(3*V);direction[1::3]=f
    if np.linalg.norm(native['Q']@native['Q'].T@direction-direction)>1e-10 or abs(np.linalg.norm(direction)-1)>1e-10:raise ArithmeticError('Actual Gaussian Ward coexact direction failed')
    rng=np.random.default_rng(seed);ward=[];ordinary=[];controls=[];errors=[]
    for sample in range(samples):
        theta=native['Q']@rng.normal(0,np.sqrt(2),M);states=[];DP=[];ops=[]
        for p in native['Pprofiles']:
            state,dP=tangent(native,p,theta,direction);states.append(state);DP.append(dP);ops.append(vertices(native,p,theta,mode,direction));P=state['P']
            errors.append(float(np.linalg.norm(dP@P+P@dP-dP,2))+float(np.linalg.norm(dP-dP.conj().T,2)))
        value,derivative=density_moment(states,DP,ops,x);ward.append(2/a*derivative);ordinary.append(float(direction@theta/a)*value)
        if sample<4:
            W=[quantum.graph_state(native,p,theta)['occupied'] for p in native['Pprofiles']];_,_,independent=initial.all64_moment(native,W,[v[0] for v in ops],[v[1] for v in ops],x)
            controls.append(float(abs(independent-value)))
    if max(controls)>1e-6 or max(errors)>1e-9:raise ArithmeticError('Actual all4hole CAR density moment/source tangent failed')
    w=np.array(ward);o=np.array(ordinary)
    return dict(N=4,V=V,M=M,samples=samples,seed=seed,original_Gaussian_variance=2,ALLcharged_flavours=64,source_position=[1,1,0],actual_SAME_SOURCE_Gaussian_Ward_magnetic_CAR_moment=quantum.summarize(w),actual_same_samples_ordinary_photon_coordinate_moment=quantum.summarize(o),actual_correlated_Ward_minus_ordinary=quantum.summarize(w-o),actual_conditional_initial_heat_ratio_slope_if_native_Pi_Coulomb_selection_accepted=quantum.summarize(-2*w),actual_inversefree_ALL4hole_jet_vs_density_formula_errors=controls,actual_projector_tangent_Hermitian_error=max(errors),exact_same_source_identity='For unfiltered realcoexact Gaussianvariance2, E[(dy.Theta)/a *m(Theta)]=(2/a)E[d_dy m(Theta)]. ActualDy=Qdy_unit; everysourceY/P derivative, Fphi Peierlsderivative and HWilson1 Peierlsderivative retained. No reference/source replacement, omittedsource derivative, fittedspeed, filter or state-selection gate.',exact_CAR_density_formula='With pointwise neutralandall4hole chargedFmean0 proved06db, m_C=p[sum_alpha load Tr(Palpha Falpha Qalpha Halpha)+Tr(Dactive(Factive Qactive Hactive-Hactive Pactive Factive))], Dactive=Q RxQ/2,p=1/2. Derived by weightedrank1conditionalSlaters: meanF*meanH rank1term cancels rank1 covariance term; neutralmeanF0 and all4trace sumF0 remove only actual SOURCE mean terms. NOT applied afterheat. Independent fullcofactorjets/all63coherentproducts checked.',scope='Actual equal-time CAR component moment; primitivePi/Coulomb selection remains separate parent-review symmetry extension, not numericalclaim here. NativefrozenPhi, chargedpartialPhi gradient, no fullcanonicalPhi force. HeavyPHsource6af1 and originalLambda/clock preserved. MonteCarlo integral notdeterministiccertificate, no finitebeta/pole/commoncone/vacuum/QED promotion.')


def run(path:Path,samples:int,seed:int)->dict[str,Any]:
    started=time.monotonic();raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=MOMENT_SHA:raise ArithmeticError('Frozen41d2/2be1 initialmoment epoch required')
    cap=json.loads(raw);_,record=quantum.checked(cap['input_descriptor']['input_receipts']['beta']);_,record=quantum.checked(record['input_descriptor']['input_receipts']['metric']);_,record=quantum.checked(record['input_descriptor']['input_receipts']['magnetic']);_,record=quantum.checked(record['input_descriptor']['input_receipts']['prior_joined_heat'])
    for key in ('full_source','contact','transfer','gram','force','join','heat'):_,record=quantum.checked(record['input_descriptor']['input_receipts'][key])
    mean=pauli.contact.transfer.packets.force.joined.mean_from_source(record)
    desc=dict(plan=cap['input_descriptor']['plan'],plan_revision=pauli.REVISION,scientific_execution_revision=pauli.REVISION,input_receipts=dict(moment=dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest())),source_abi=cap['input_descriptor']['source_abi'],native_N=4,samples=samples,seed=seed,source='UNCHANGEDoriginalcanonicalY/Gaussian2/ALL64/AP222/ALL4Refholes/BEFOREheatE. ExactGaussianintegration-by-parts estimator changes NOsource/preparation/probability.',a='1/64',UEphys='1',g_Pi_AND_Coulomb='64',Lambda='2^8192',chi='2^-4096',original_clock='4096',derivative='AnalyticDy actualP_Y: DY=DK/d,DR=-R(DY Y+Y DY)R; no eigengapdenominators/finite-differencefit. Analytic actualPhi/one-linkWilson orderedPeierlsvertices included.')
    return dict(input_descriptor=desc,input_sha256=hashlib.sha256(json.dumps(desc,sort_keys=True).encode()).hexdigest(),actual_same_source_Pauli_Gaussian_Ward_moment=physical(mean,samples,seed),source_at_import={name:dict(sha256=hashlib.sha256(data).hexdigest(),bytes_utf8=data.decode()) for name,data in SOURCE_BYTES.items()},source_unchanged_since_import={name:(ROOT/name).read_bytes()==data for name,data in SOURCE_BYTES.items()},runtime=dict(executable=sys.executable,python=sys.version,numpy=np.__version__),elapsed_seconds=time.monotonic()-started)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--moment',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--samples',type=int,required=True);p.add_argument('--seed',type=int,required=True);a=p.parse_args();r=run(a.moment,a.samples,a.seed)
    if not all(r['source_unchanged_since_import'].values()):raise ArithmeticError('Source changed during actual Gaussian Ward calculation')
    a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(dict(input_sha256=r['input_sha256'],physical=r['actual_same_source_Pauli_Gaussian_Ward_moment'],elapsed_seconds=r['elapsed_seconds']),indent=2))
