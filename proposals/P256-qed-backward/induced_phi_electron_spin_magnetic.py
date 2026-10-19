"""Actual microscopic positive-band spin/current/Foldy CONTROL.

Same rootPhi/g0/gs/a and literal1/2linkPeierls matrices as Pauli bandloop;
no phenomenological magnetic term. Distinct bareband control, NOT original
E SOURCE/primitive interacting electron pole or photon-renormalized charge.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any
import numpy as np
ROOT=Path(__file__).resolve().parent
OWN_BYTES=Path(__file__).read_bytes()
import induced_phi_pauli_band_volume as volume
control=volume.control
pauli=control.pauli
quantum=control.quantum
SOURCE_BYTES={**volume.SOURCE_BYTES,Path(__file__).name:OWN_BYTES}
VOLUME_SHA='7ccf67dddd91d89e085ae9b3efd9a98b052a1a4388af280fb18979edba9237e5'


def h(profile:dict[str,Any],terms:list[dict[str,Any]],p:np.ndarray,a:float)->np.ndarray:
    return volume.h_at(profile,terms,(a*p)[None])[0]


def current(terms:list[dict[str,Any]],pout:np.ndarray,pin:np.ndarray,a:float,axis:int)->np.ndarray:
    out=np.zeros((4,4),complex);q=a*(pout-pin)
    for term in terms:
        if term['axis']!=axis:continue
        L=term['length'];pathsum=sum(np.exp(1j*m*q[axis]) for m in range(L))
        out+=1j*a*pathsum*(term['hop']*np.exp(1j*L*a*pin[axis])-term['hop'].conj().T*np.exp(-1j*L*a*pout[axis]))
    return out


def rest_derivative(terms:list[dict[str,Any]],a:float,axis:int,order:int,only_nearest:bool=False)->np.ndarray:
    out=np.zeros((4,4),complex)
    for term in terms:
        if term['axis']!=axis or (only_nearest and term['length']!=1):continue
        z=(1j*a*term['length'])**order*term['hop'];out+=z+z.conj().T
    return out


def positive(profile:dict[str,Any],terms:list[dict[str,Any]],p:np.ndarray,a:float,B:np.ndarray)->np.ndarray:
    H=h(profile,terms,p,a);E=float(np.sqrt(np.trace(H@H).real/4));P=(np.eye(4)+H/E)/2
    e,U=np.linalg.eigh(B.conj().T@P@B)
    if e[0]<=0:raise ArithmeticError('Actual physical positiveband projected-rest section vanished')
    return P@B@(U/np.sqrt(e))@U.conj().T


def physical(mean:dict[str,Any],qvalues:list[float])->dict[str,Any]:
    native=quantum.build_native(mean,4);a=native['a'];gamma=native['frame']/np.sqrt([native['g0']]+[native['gs']]*3)[:,None,None]
    eb,Ub=np.linalg.eigh(gamma[0]);B0=Ub[:,eb>0]
    if B0.shape!=(4,2):raise ArithmeticError('Actual physicalrest positivebeta rank changed')
    spins=[-1j*gamma[j]@gamma[k]/2 for j,k in ((2,3),(3,1),(1,2))]
    algebra=max(float(np.linalg.norm(spins[2]@gamma[1]-gamma[1]@spins[2]-1j*gamma[2])),float(np.linalg.norm(spins[2]@gamma[2]-gamma[2]@spins[2]+1j*gamma[1])))
    if algebra>1e-12:raise ArithmeticError('Actual microscopic spin rotation algebra failed')
    results=[]
    for profile in native['Pprofiles']:
        terms=volume.primitive(profile);H0=h(profile,terms,np.zeros(3),a);M=float(np.trace(gamma[0]@H0).real/4)
        if M<=0 or np.linalg.norm(H0-M*gamma[0])>1e-10:raise ArithmeticError('Actual microscopic restmass/Clifford branch failed')
        Rminus=(np.eye(4)-gamma[0])/2;axisresults=[]
        for axis,(momentum_axis,current_axis) in enumerate(((1,2),(2,0),(0,1))):
            # Physical S=-iGamma_j Gamma_k/2. Existing PhiPauli uses
            # +iGamma_j Gamma_k=-2S, so its sign is NOT silently retagged.
            eS,US=np.linalg.eigh(B0.conj().T@spins[axis]@B0);B=B0@US;Sigma=2*B.conj().T@spins[axis]@B
            Hp=rest_derivative(terms,a,momentum_axis,1);Hpp=rest_derivative(terms,a,momentum_axis,2)
            curvature=B.conj().T@(Hpp+Hp@Rminus@Hp/M)@B;inverse_mass=float(np.trace(curvature).real/2);mass=1/inverse_mass
            Wprime=Rminus@Hp@B/(2*M);J=current(terms,np.zeros(3),np.zeros(3),a,current_axis)
            derivative=(Wprime.conj().T@J@B-B.conj().T@J@Wprime)/2
            mu=float(np.real(np.trace(Sigma@derivative)/(2j)));g=4*mass*mu
            nearest=rest_derivative(terms,a,momentum_axis,2,only_nearest=True)
            wrong_curvature=float(np.trace(B.conj().T@(nearest+Hp@Rminus@Hp/M)@B).real/2);wrong_g=4*mu/wrong_curvature
            rest_phi=np.zeros((4,4),complex)
            delta=gamma[0]@(1j*gamma[momentum_axis+1]@gamma[current_axis+1])/np.sqrt(8)
            rest_phi+=float(pauli.scalars(profile)['onsite'])*delta
            for term in terms:rest_phi+=float(complex(term['scalars'][0]).real)*delta*2
            phi_spin=float(np.trace(Sigma@(B.conj().T@rest_phi@B)).real/2)
            rows=[]
            for q in qvalues:
                p=np.zeros(3);p[momentum_axis]=q/2;Wp=positive(profile,terms,p,a,B);Wm=positive(profile,terms,-p,a,B)
                vertex=Wp.conj().T@current(terms,p,-p,a,current_axis)@Wm
                measured_mu=float(np.real(np.trace(Sigma@vertex)/(2j*q)));measured_g=4*mass*measured_mu
                rows.append(dict(physical_q=q,actual_positiveband_Peierls_spin_vertex=control.matrix(vertex),actual_spin_mu_coefficient=measured_mu,actual_g_from_independent_inertial_curvature=measured_g,actual_Foldy_section_orthonormality_error=max(float(np.linalg.norm(Wp.conj().T@Wp-np.eye(2))),float(np.linalg.norm(Wm.conj().T@Wm-np.eye(2))))))
            if abs(g-2)>1e-10 or np.linalg.norm(curvature-inverse_mass*np.eye(2))>1e-10 or abs(rows[-1]['actual_g_from_independent_inertial_curvature']-g)>1e-6:raise ArithmeticError('Actual microscopic spin/Foldy/g2 consumer failed')
            axisresults.append(dict(spin_axis=axis,physical_spin_eigenvalues=eS.tolist(),actual_rest_energy_M=M,actual_inertial_mass=mass,actual_one_and_two_link_rest_seagull_norm=float(np.linalg.norm(Hpp)),actual_wrong_omitted_two_link_seagull_norm=float(np.linalg.norm(nearest)),actual_tree_spin_mu_coefficient=mu,actual_g_tree= g,WRONG_omitted_two_link_seagull_inertial_g=wrong_g,actual_true_PhiPauli_uniform_positiveband_spin_mass_coefficient=phi_spin,finite_q_rows=rows))
        results.append(dict(role=profile['label'],load=profile['load'],axis_results=axisresults))
    return dict(a=a,g0=native['g0'],gs=native['gs'],charged_flavours=64,actual_spin_rotation_algebra_error=algebra,physical_bare_Peierls_charge=1,actual_microscopic_positiveband_spin_results=results,physical_derivation='Actual1/2linkh gives H0=M beta, restHprime=cm Gamma and improvedWilson restHdoubleprime=0 byFULLseagull cancellation. Physical S_i=-iGamma_jGamma_k/2, [S_i,Gamma_j]=iGamma_k; sourcePhiPauli isbeta(+iGamma_jGamma_k)/sqrt8=-2betaS_i/sqrt8. Canonicalpositiveband W=Pplus Bplus(BplusdagPplusBplus)^-1/2 usespositiveoverlap, no arbitrarymomentum-eigenvectorphase. Independentcurvature Hdoubleprime+HprimePminusHprime/M defines inertialmass M/cm². FullPeierls J_A matrix between W(+q/2),W(-q/2) gives i e cm² q Sigma/(2M)+O(q³). Therefore mu=cm²/(2M) andg=4minertial mu/e=2. Droppingactual2linkseagull spoilsinertialg; no magneticterm inserted.',scope='DISTINCTbarephysicalpositiveband microscopiccontrol. NOT originalcanonicalY/Gaussian2/E/Ref SOURCE or primitivePi+Coul interactingelectronpole. No anomalousloopg/eR, photonmodeleg/commoncone/QEDcompletion. Rho/A0/negativebandloop873/7cc remainseparate; this singleparticlecurrentFOLDY is not their state retag.')


def run(path:Path,qvalues:list[float])->dict[str,Any]:
    started=time.monotonic();raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=VOLUME_SHA:raise ArithmeticError('Frozen00e/7cc primitiveband volume epoch required')
    if not qvalues or any(q<=0 or not np.isfinite(q) for q in qvalues) or qvalues[-1]!=min(qvalues):raise ValueError('Actualspin probes require positive explicitphysicalq endingat smallest')
    cap=json.loads(raw);mean=cap['input_descriptor']['actual_root_mean_input']
    desc=dict(plan=cap['input_descriptor']['plan'],plan_revision=pauli.REVISION,scientific_execution_revision=pauli.REVISION,input_receipts=dict(volume=dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest())),original_source_abi_UNCHANGED=cap['input_descriptor']['original_source_abi_UNCHANGED'],actual_root_mean_input=mean,physical_q_probes=qvalues,source_state='Distinctactualphysicalpositiveband projectedrestsection, NOToriginalE SOURCE.',original_clock_UNCHANGED='4096',original_Lambda_UNCHANGED='2^8192',original_chi_UNCHANGED='2^-4096')
    return dict(input_descriptor=desc,input_sha256=hashlib.sha256(json.dumps(desc,sort_keys=True).encode()).hexdigest(),actual_microscopic_electron_spin_magnetic_Foldy_control=physical(mean,qvalues),source_at_import={name:dict(sha256=hashlib.sha256(data).hexdigest(),bytes_utf8=data.decode()) for name,data in SOURCE_BYTES.items()},source_unchanged_since_import={name:(ROOT/name).read_bytes()==data for name,data in SOURCE_BYTES.items()},runtime=dict(executable=sys.executable,python=sys.version,numpy=np.__version__),elapsed_seconds=time.monotonic()-started)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--volume',type=Path,required=True);p.add_argument('--q-values',type=float,nargs='+',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=run(a.volume,a.q_values)
    if not all(r['source_unchanged_since_import'].values()):raise ArithmeticError('Source changed during actualmicroscopicspin currentconsumer')
    a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(dict(input_sha256=r['input_sha256'],physical=r['actual_microscopic_electron_spin_magnetic_Foldy_control'],elapsed_seconds=r['elapsed_seconds']),indent=2))
