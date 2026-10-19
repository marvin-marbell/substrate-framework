"""Actual original Gaussian/CAR magnetic source initial Hamiltonian moment."""
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

ROOT=Path(__file__).resolve().parent
OWN_BYTES=Path(__file__).read_bytes()
import induced_phi_pauli_beta_continuation as continuation
pauli=continuation.pauli
quantum=pauli.quantum
SOURCE_BYTES={**continuation.SOURCE_BYTES,Path(__file__).name:OWN_BYTES}
BETA_SHA='20d4716a5ebaa9664f84858b2a946648041e628d467979c3e9c5bd1e20a174e5'


def one_link_Wilson(native:dict[str,Any],profile:dict[str,Any],theta:np.ndarray)->np.ndarray:
    V=native['V'];H=np.zeros((4*V,4*V),complex)
    for path in profile['paths']:
        if path['length']==1:
            x,y=path['source'],path['target'];hop=path['scalars'][0]*native['frame'][0]*path['seam']*np.exp(1j*np.sum(theta[list(path['edges'])]))
            H[4*x:4*x+4,4*y:4*y+4]+=hop;H[4*y:4*y+4,4*x:4*x+4]+=hop.conj().T
    return H


def all64_moment(native:dict[str,Any],W:list[np.ndarray],force:list[np.ndarray],H:list[np.ndarray],x:int)->tuple[complex,complex,complex]:
    identity=np.eye(4*native['V'],dtype=complex);aux=pauli.jet(W[1],W[1],force[1],H[1],identity);b,bF,_,_=aux
    creator=np.zeros((4*native['V'],4),complex);creator[4*x:4*x+4]=np.eye(4);norm=0j;moment=0j;firstF=0j
    for hole in range(4):
        activeW=np.column_stack((W[0],creator[:,hole]));active=pauli.jet(activeW,activeW,force[0],H[0],identity);a,aF,_,_=active;n,m=quantum.combine(active,aux)
        norm+=n/4;moment+=m/4;firstF+=(aF*b**63+63*a*bF*b**62)/4
    return norm,firstF,moment


def physical(mean:dict[str,Any],samples:int,seed:int)->dict[str,Any]:
    if samples<4:raise ValueError('Actual sampling uncertainty requires atleast4 samples')
    native=quantum.build_native(mean,4);V,M=native['V'],native['M'];a=float(native['a']);mode=pauli.magnetic_shape(native);x=int(np.flatnonzero(np.all(native['sites']==(1,1,0),axis=1))[0])
    f=np.sqrt(2/V)*np.cos(2*np.pi*native['sites'][:,0]/4);direction=np.zeros(3*V);direction[1::3]=f
    Efree=sum(p['load']*np.linalg.eigvalsh(quantum.matter_hamiltonian(native,p,np.zeros(3*V)))[:2*V].sum() for p in native['Pprofiles'])
    rng=np.random.default_rng(seed);values=[];bare=[];controls=[];norms=[];means=[]
    for sample in range(samples):
        theta=native['Q']@rng.normal(0,np.sqrt(2),M);_,W,force=pauli.endpoints(native,theta,mode);H1=[one_link_Wilson(native,p,theta) for p in native['Pprofiles']]
        norm,firstF,moment=all64_moment(native,W,force,H1,x);Ay=float(direction@theta/a);values.append(moment*Ay);bare.append(moment);norms.append(norm);means.append(firstF)
        if sample<4:
            H=[quantum.matter_hamiltonian(native,p,theta) for p in native['Pprofiles']];_,firstFull,full=all64_moment(native,W,force,H,x)
            controls.append(dict(actual_FULL_CAR_vs_one_link_Wilson_selected_moment_error=float(abs((full-Efree*firstFull)-moment)),raw_source_force_mean_abs=float(abs(firstFull)),native_actual_H_CAR_other_pieces_not_removed=True))
    if max(abs(z-.5) for z in norms)>1e-8 or max(abs(z) for z in means)>1e-8 or max(c['actual_FULL_CAR_vs_one_link_Wilson_selected_moment_error'] for c in controls)>1e-5:raise ArithmeticError('Actual all4hole/sourceprobability/CARselection moment contract failed')
    values=np.array(values)
    return dict(N=4,V=V,M=M,samples=samples,seed=seed,ALLcharged_flavours=64,source_position=[1,1,0],original_Gaussian_variance=2,source_probability_before_heat='1/2',actual_RAW_PhiPauli_Ay_one_link_Wilson_Hamiltonian_moment=quantum.summarize(values),actual_initial_conditional_heat_ratio_slope_if_native_electric_Coulomb_selection_accepted=quantum.summarize(-2*values),actual_RAW_PhiPauli_one_link_Wilson_CAR_moment_without_photon_coordinate=quantum.summarize(np.array(bare)),actual_FULL_CAR_operator_selection_controls=controls,source_probability_error=float(max(abs(z-.5) for z in norms)),raw_source_force_mean_error=float(max(abs(z) for z in means)),reference_SOURCE_Gram_not_sample_whitened=True,physical_interpretation='Initial mixed source Hamiltonian moment, not finitebeta fit or field speed/mass. Exact SOURCE crossPhi/Ay=0, so ratioKPhi,Ay/Zcreator initial derivative=-M/p=-2M if fullnativeH selection holds. No derivative inferred from noisy beta tables.',selection_for_parent_review='The reduced all4Refhole source has canonicalY scalar-spin and even combinedGamma0/sublattice parity after fixed spin-staggering. F_Pauli has only spin-vector pieces or spin-scalar/odd-parity pieces. Actual CAR onsite/Phi5/two-linkWilson pieces are scalar-spin/even combinedparity; one-linkWilsonPhi0 alone can couple. Actual fullCAR-vsH1 controls executed. PrimitivePi and densityCoulomb preserve these fixed colour/parity representations; hence proposed exact mixedmoments vanish for those operators. That latter analytic extension is explicitly parent-review, NOT an unexecuted numerical Pi/Coulomb moment claim.',scope='N4 frozenPhi reference. Phi/SU4/heavy time not propagated. ChargedmatterPhi-gradient only; actualneutralheavyPHnoise6af1 retained separate, not totalforce. No ensemblemagneticbridge/pole/commoncone/vacuum/retarded/QED promotion; MonteCarlo integral not deterministiccertificate.')


def run(path:Path,samples:int,seed:int)->dict[str,Any]:
    started=time.monotonic();raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=BETA_SHA:raise ArithmeticError('Frozen9a52/20d4 actualbeta epoch required')
    cap=json.loads(raw);_,record=quantum.checked(cap['input_descriptor']['input_receipts']['metric']);_,record=quantum.checked(record['input_descriptor']['input_receipts']['magnetic']);_,record=quantum.checked(record['input_descriptor']['input_receipts']['prior_joined_heat'])
    for key in ('full_source','contact','transfer','gram','force','join','heat'):_,record=quantum.checked(record['input_descriptor']['input_receipts'][key])
    mean=pauli.contact.transfer.packets.force.joined.mean_from_source(record)
    desc=dict(plan=cap['input_descriptor']['plan'],plan_revision=pauli.REVISION,scientific_execution_revision=pauli.REVISION,input_receipts=dict(beta=dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest())),source_abi=cap['input_descriptor']['source_abi'],native_N=4,samples=samples,seed=seed,source='UNCHANGEDoriginalcanonicalY/Gaussian2/all64/AP222/ALL4Refholes/E_BEFOREactualheat. SAMEoriginalmacroPhiLambda2^8192meanmidpointnative. No fit/filter/gate or source/caller changes.',a='1/64',UEphys='1',g_Pi_AND_Coulomb='64',Lambda='2^8192',chi='2^-4096',original_clock='4096',operator='Literal one-linkWilsonGamma0 part extracted from ACTUALfullH, NOT a replacementH. FullCAR moment selection compared onactualsource; analyticPi/Coulomb selection parent-review.')
    return dict(input_descriptor=desc,input_sha256=hashlib.sha256(json.dumps(desc,sort_keys=True).encode()).hexdigest(),actual_initial_Pauli_magnetic_source_moment=physical(mean,samples,seed),source_at_import={name:dict(sha256=hashlib.sha256(data).hexdigest(),bytes_utf8=data.decode()) for name,data in SOURCE_BYTES.items()},source_unchanged_since_import={name:(ROOT/name).read_bytes()==data for name,data in SOURCE_BYTES.items()},runtime=dict(executable=sys.executable,python=sys.version,numpy=np.__version__),elapsed_seconds=time.monotonic()-started)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--beta-receipt',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--samples',type=int,required=True);p.add_argument('--seed',type=int,required=True);a=p.parse_args();r=run(a.beta_receipt,a.samples,a.seed)
    if not all(r['source_unchanged_since_import'].values()):raise ArithmeticError('Source changed during actual initialmoment calculation')
    a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(dict(input_sha256=r['input_sha256'],physical=r['actual_initial_Pauli_magnetic_source_moment'],elapsed_seconds=r['elapsed_seconds']),indent=2))
