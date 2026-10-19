"""Exact AP finite-box Fourier Pauli/Peierls response, distinct bareCAR control.

Uses actual primitive source0 path matrices, exact shifted AP momenta, BOTH
q sectors and bands. Independent N4 literal-site response capsule is consumer.
No original E SOURCE/vacuum/primitiveinteraction or literal macro retag.
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
import induced_phi_pauli_band_response_control as control
pauli=control.pauli
quantum=control.quantum
SOURCE_BYTES={**control.SOURCE_BYTES,Path(__file__).name:OWN_BYTES}
CONTROL_SHA='87373e6ab0d8d29ff05829fef0c82f267ad69f0bb2fd333c2270a113f1c660de'


def primitive(profile:dict[str,Any])->list[dict[str,Any]]:
    terms=[p for p in profile['paths'] if p['source']==0]
    if len(terms)!=6 or any(p['seam']!=1 for p in terms):raise ArithmeticError('Actual one/twolink primitive representative changed')
    return terms


def h_at(profile:dict[str,Any],terms:list[dict[str,Any]],k:np.ndarray)->np.ndarray:
    h=np.broadcast_to(profile['onsite'],(len(k),4,4)).copy()
    for p in terms:
        z=np.exp(1j*p['length']*k[:,p['axis']])[:,None,None]*p['hop']
        h+=z+z.conj().transpose(0,2,1)
    return h


def supplied(profile:dict[str,Any],N:int,deltaPhi:np.ndarray,a:float)->tuple[dict[str,Any],np.ndarray,dict[str,Any]]:
    V=N**3;q=2*np.pi/N;indices=np.asarray(list(np.ndindex((N,N,N))));k=(2*np.pi*indices+np.pi)/N
    terms=primitive(profile);h=h_at(profile,terms,k);e,U=np.linalg.eigh(h)
    if np.max(e[:,1])>=0 or np.min(e[:,2])<=0:raise ArithmeticError('Physical Fourier halfbands not gapped')
    W=U[:,:,:2];C=np.zeros((5,5),complex);Cgg=np.zeros_like(h);current={}
    for axis in (1,2):
        j=np.zeros_like(h);c=np.zeros_like(h)
        for p in terms:
            if p['axis']==axis:
                z=np.exp(1j*p['length']*k[:,axis])[:,None,None]*p['hop'];j+=1j*p['length']*(z-z.conj().transpose(0,2,1));c-=p['length']**2*(z+z.conj().transpose(0,2,1))
        current[axis]=j
        C[axis,axis]=a*a/V*np.einsum('vsi,vst,vti->',W.conj(),c,W)
    for p in terms:
        if p['axis']==0:
            z=np.exp(1j*p['length']*k[:,0])[:,None,None]*p['hop'];Cgg-=2*(1-np.cos(q*p['length']))/V*(z+z.conj().transpose(0,2,1))
    contact_gauge=complex(np.einsum('vsi,vst,vti->',W.conj(),Cgg,W))
    onsite=pauli.scalars(profile)['onsite'];coefficient=1/np.sqrt(2*V)
    blocks=[];gaps=[];gauge_blocks=[]
    for sign in (1,-1):
        target=indices.copy();target[:,0]=(target[:,0]-sign)%N;ku=(2*np.pi*target+np.pi)/N
        hu=h_at(profile,terms,ku);eu,Zall=np.linalg.eigh(hu);Z=Zall[:,:,2:]
        d=eu[:,None,2:]-e[:,:2,None]
        scalar=np.full(V,onsite,complex)
        for p in terms:scalar+=p['scalars'][0]*(np.cos(p['length']*k[:,p['axis']])+np.cos(p['length']*ku[:,p['axis']]))
        F=(-1j*sign*coefficient*scalar)[:,None,None]*deltaPhi
        operators=[F,a*coefficient*current[1],a*coefficient*current[2],np.broadcast_to(coefficient*np.eye(4),(V,4,4)),np.broadcast_to(-1j*sign*coefficient*np.eye(4),(V,4,4))]
        blocks.append(np.stack([W.conj().transpose(0,2,1)@O@Z for O in operators]));gaps.append(d)
        Jg=1j*coefficient*(hu-h)
        gauge_blocks.append(W.conj().transpose(0,2,1)@Jg@Z)
    amplitudes=np.concatenate(blocks,axis=1).reshape(5,2*V,4);gap=np.concatenate(gaps,axis=0).reshape(2*V,4)
    jg=np.concatenate(gauge_blocks,axis=0).reshape(2*V,4)
    gauge_bubble=complex(np.sum(2*np.abs(jg)**2/gap));error=abs(contact_gauge-gauge_bubble)
    if error>1e-8:raise ArithmeticError('Exact finiteAP puregauge Ward failed')
    return dict(amplitudes=amplitudes,delta=gap),C,dict(load=profile['load'],role=profile['label'],actual_pair_gap_min=float(np.min(gap)),actual_PH_transitions=8*V,actual_puregauge_contact_minus_bothinterband_error=error,actual_missing_contact_gauge_defect=abs(gauge_bubble))


def physical(native:dict[str,Any],N:int,frequencies:list[float])->dict[str,Any]:
    if N<=2 or N%2:raise ValueError('Periodic magnetic sin(qx) requires explicit even N>2; N2 Nyquist sinmode is identicallyzero, not a substitute source')
    a=native['a'];g=native['UE']/a;deltaPhi=pauli.magnetic_shape(native)['delta'];q=2*np.pi/N
    specs=[];contacts=[];checks=[]
    for profile in native['Pprofiles']:
        data,C,check=supplied(profile,N,deltaPhi,a);specs.append((profile['load'],data));contacts.append(profile['load']*C);checks.append(check)
    if sum(load for load,_ in specs)!=64:raise ArithmeticError('Actual Fourier charged64 count lost')
    contact=np.add.reduce(contacts);rows=[];bare=(4*np.sin(q/2)**2)/(2*g)*np.eye(2)
    for omega in frequencies:
        B=np.add.reduce([load*control.bubble(data,omega) for load,data in specs]);raw=contact[:3,:3]-B[:3,:3]
        Schur=B[:3,3:]@np.linalg.solve(bare+B[3:,3:],B[3:,:3]);K=raw+Schur
        rows.append(dict(omega_Euclidean=omega,actual_contact_minus_bothinterband=control.matrix(raw),actual_rho_A0_Schur=control.matrix(Schur),actual_material_kernel=control.matrix(K),actual_magnetic_Phi_Ay_divided_by_qtheta=float(np.real(K[0,1])/q),actual_magnetic_Phi_Ay_divided_by_sinq=float(np.real(K[0,1])/np.sin(q)),actual_all64_positive_density_bubble=control.matrix(B[3:,3:])))
    return dict(N=N,V=N**3,AP_momenta='(2pi*n+pi)/N in EACHaxis; qsectors BOTH plus/minus2pi/N; literalfinite sums, no BZsubtraction/recentering.',actual_physical_qtheta=[float(q),0.,0.],actual_primitive_terms_per_species=6,actual_source_basis=['unitCart35 Phi0 betaSigma_z sinqx','unitAphys_y cosqx','unitAphys_z cosqx','unitrho cosqx','unitrho sinqx'],actual_all64_contact=control.matrix(contact),actual_primitive_controls=checks,rows=rows)


def run(path:Path,sizes:list[int])->dict[str,Any]:
    started=time.monotonic();raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=CONTROL_SHA:raise ArithmeticError('Frozen bfb/873 densephysicalcontrol epoch required')
    cap=json.loads(raw);mean=cap['input_descriptor']['actual_root_mean_input'];frequencies=cap['input_descriptor']['frequencies_Euclidean'];native=quantum.build_native(mean,4)
    outputs=[physical(native,N,frequencies) for N in sizes]
    literal=cap['actual_physical_Pauli_Peierls_band_response_control'];four=physical(native,4,frequencies) if 4 not in sizes else outputs[sizes.index(4)]
    errors=[]
    for f,l in zip(four['rows'],literal['rows'],strict=True):
        for field,consumer in (('actual_contact_minus_bothinterband','actual_all64_contact_minus_bothinterband'),('actual_rho_A0_Schur','actual_rho_A0_Schur'),('actual_material_kernel','actual_chargedmaterial_Phi_photon_kernel_with_rho')):
            for part in ('real','imag'):errors.append(float(np.max(np.abs(np.asarray(f[field][part])-np.asarray(l[consumer][part])))))
    contact_error=max(float(np.max(np.abs(np.asarray(four['actual_all64_contact'][part])-np.asarray(literal['physical_contact'][part])))) for part in ('real','imag'))
    if max(errors+[contact_error])>1e-8:raise ArithmeticError('ActualAPFourier supplier disagrees with independent literalN4 mixedcontact/interband/Schur consumer')
    desc=dict(plan=cap['input_descriptor']['plan'],plan_revision=pauli.REVISION,scientific_execution_revision=pauli.REVISION,input_receipts=dict(control=dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest())),original_source_abi_UNCHANGED=cap['input_descriptor']['original_source_abi_UNCHANGED'],actual_sizes=sizes,actual_Euclidean_frequencies=frequencies,actual_root_mean_input=mean,control_state='Distinct actualbareCAR filledphysicalnegativeband atTheta0. NOToriginalcanonicalY/Gaussian2/E SOURCE, not primitivePi+Coul interactingvacuum, no fittedcone/fieldterm/pole or literalmacroclaim.',original_Lambda_UNCHANGED='2^8192',original_chi_UNCHANGED='2^-4096',original_clock_UNCHANGED='4096')
    return dict(input_descriptor=desc,input_sha256=hashlib.sha256(json.dumps(desc,sort_keys=True).encode()).hexdigest(),actual_exact_AP_Pauli_Peierls_volume_rows=outputs,actual_independent_N4_literal_operator_contact_interband_rhoSchur_consumer_error=max(errors+[contact_error]),formula='F(ko,ku)=f_q deltaPhi[onsite+sum c0_l(cos(l ko_axis)+cos(l ku_axis))]; f_q=-i sign/sqrt(2V), ku=ko-sign*q. J_Ay=a/sqrt(2V) d_h/d_k_y, rho_q=I/sqrt(2V) or-i sign I/sqrt(2V). ALL orderedprimitive1/2links retained; no barePhi5 perturbation. Independent literalconsumer prices signs/normalizations/bothbands/ALL1+63counts.',heavy='Actualneutralheavy -Lambda h PhiPhi response from873 remainsseparate/additiveandEMmixedEXACT0. PrimitivePhi boson/self/Noether not assertedzero; outputs only chargedmaterialcontrol.',source_at_import={name:dict(sha256=hashlib.sha256(data).hexdigest(),bytes_utf8=data.decode()) for name,data in SOURCE_BYTES.items()},source_unchanged_since_import={name:(ROOT/name).read_bytes()==data for name,data in SOURCE_BYTES.items()},runtime=dict(executable=sys.executable,python=sys.version,numpy=np.__version__),elapsed_seconds=time.monotonic()-started)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--control',type=Path,required=True);p.add_argument('--sizes',type=int,nargs='+',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=run(a.control,a.sizes)
    if not all(r['source_unchanged_since_import'].values()):raise ArithmeticError('Source changed during actual APFourier volumephysics')
    a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(dict(input_sha256=r['input_sha256'],literal_N4_consumer_error=r['actual_independent_N4_literal_operator_contact_interband_rhoSchur_consumer_error'],rows=r['actual_exact_AP_Pauli_Peierls_volume_rows'],elapsed_seconds=r['elapsed_seconds']),indent=2))
