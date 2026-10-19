"""Physical negative-band Pauli/Peierls response CONTROL, not original SOURCE.

All actual one/two-link vertices/contact, both interband directions and rho/A0
Schur are retained. This is bareCAR quadratic response at Theta0/frozenPhi;
it does NOT claim the primitive Pi+Coul interacting vacuum or retag E Omega.
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
import induced_phi_pauli_gaussian_ward as ward
from induced_phi_backreaction import PhiConfig,models,scalar_paths
pauli=ward.pauli
quantum=ward.quantum
SOURCE_BYTES={**ward.SOURCE_BYTES,Path(__file__).name:OWN_BYTES}
WARD_SHA='9953ff62d8b83d1adeecb00cffb873b4e6f91508fa578c155343b985346540fd'


def matrix(value:np.ndarray)->dict[str,Any]:
    value=np.asarray(value)
    return dict(real=np.real(value).tolist(),imag=np.imag(value).tolist())


def spectrum(h:np.ndarray,operators:list[np.ndarray],rank:int)->dict[str,Any]:
    e,U=np.linalg.eigh(h)
    if not e[rank-1]<0<e[rank]:raise ArithmeticError('Physical control sea/empty bands not separated')
    W,Z=U[:,:rank],U[:,rank:]
    amplitudes=np.stack([W.conj().T@O@Z for O in operators])
    return dict(W=W,Z=Z,delta=e[None,rank:]-e[:rank,None],amplitudes=amplitudes,gap=float(e[rank]-e[rank-1]))


def bubble(data:dict[str,Any],omega:float)->np.ndarray:
    v=data['amplitudes'];d=data['delta']
    # BOTH interband directions, including frequency-odd antisymmetric parts.
    return np.einsum('aij,bij,ij->ab',v,v.conj(),1/(d-1j*omega))+np.einsum('aij,bij,ij->ab',v.conj(),v,1/(d+1j*omega))


def physical(mean:dict[str,Any],frequencies:list[float])->dict[str,Any]:
    native=quantum.build_native(mean,4);V=native['V'];rank=2*V;a=native['a'];g=native['UE']/a
    mode=pauli.magnetic_shape(native);theta=np.zeros(3*V)
    x=np.asarray([site[0] for site in native['sites']]);q=np.pi/2
    c=np.sqrt(2/V)*np.cos(q*x);s=np.sqrt(2/V)*np.sin(q*x)
    dy=np.zeros(3*V);dz=np.zeros_like(dy);dy[1::3]=c;dz[2::3]=c
    rho=[np.diag(np.repeat(f,4)).astype(complex) for f in (c,s)]
    divergence=max(float(np.linalg.norm(native['D']@dy)),float(np.linalg.norm(native['D']@dz)))
    if divergence>1e-12:raise ArithmeticError('Actual photon control directions not transverse')
    species=[];contacts=[];controls=[]
    for profile in native['Pprofiles']:
        h=quantum.matter_hamiltonian(native,profile,theta)
        Fop,_,DFy,_=ward.vertices(native,profile,theta,mode,dy)
        _,_,DFz,_=ward.vertices(native,profile,theta,mode,dz)
        currents=[a*quantum.path_derivative(native,profile,theta,d) for d in (dy,dz)]
        operators=[Fop,*currents,*rho];data=spectrum(h,operators,rank);W=data['W']
        C=np.zeros((5,5),complex)
        for j,DF in enumerate((DFy,DFz),1):C[0,j]=C[j,0]=a*np.trace(W.conj().T@DF@W)
        for i,di in enumerate((dy,dz),1):
            for j,dj in enumerate((dy,dz),1):C[i,j]=a*a*np.trace(W.conj().T@quantum.path_derivative(native,profile,theta,di,2,other=dj)@W)
        # Literal periodic-gauge Ward: no integral/BZ identity substitution.
        df=native['D'].T@c
        Jg=quantum.path_derivative(native,profile,theta,df)
        Cgg=quantum.path_derivative(native,profile,theta,df,2)
        _,_,DFg,_=ward.vertices(native,profile,theta,mode,df)
        gauge=spectrum(h,[Fop,Jg],rank);B=bubble(gauge,0.)
        gauge_word=float(np.linalg.norm(Jg-1j*(rho[0]@h-h@rho[0])))
        ward_A=complex(np.trace(W.conj().T@Cgg@W)-B[1,1]);ward_F=complex(np.trace(W.conj().T@DFg@W)-B[0,1])
        eps=1e-4
        plus=quantum.matter_hamiltonian(native,profile,theta+eps*a*dy);minus=quantum.matter_hamiltonian(native,profile,theta-eps*a*dy)
        current_error=float(np.linalg.norm((plus-minus)/(2*eps)-currents[0]))
        contact_error=float(np.linalg.norm((plus-2*h+minus)/(eps*eps)-a*a*quantum.path_derivative(native,profile,theta,dy,2)))
        if max(gauge_word,abs(ward_A),abs(ward_F),current_error)>1e-7 or contact_error>1e-3:raise ArithmeticError('Actual physical Ward/current/contact discriminator failed')
        controls.append(dict(role=profile['label'],load=profile['load'],actual_gauge_covariance_error=gauge_word,actual_periodic_gauge_AA_contact_minus_fullbubble_error=abs(ward_A),actual_periodic_gauge_PhiA_contact_minus_fullbubble_error=abs(ward_F),actual_literal_current_difference_error=current_error,actual_literal_contact_difference_error=contact_error,missing_contact_gauge_defect=abs(complex(B[1,1])),actual_mixed_contact_Phi_Ay=matrix(np.asarray(C[0,1]))))
        species.append((profile['load'],data));contacts.append(profile['load']*C)
    if sum(load for load,_ in species)!=64:raise ArithmeticError('Actual massresolved all64 load mismatch')
    contact=np.add.reduce(contacts);rows=[];bare_rho=(4*np.sin(q/2)**2)/(2*g)*np.eye(2)
    for omega in frequencies:
        B=np.add.reduce([load*bubble(data,omega) for load,data in species])
        raw=contact[:3,:3]-B[:3,:3]
        schur=B[:3,3:]@np.linalg.solve(bare_rho+B[3:,3:],B[3:,:3])
        K=raw+schur;primitive_E=np.diag([0.,a*a*omega*omega/(2*g),a*a*omega*omega/(2*g)])
        rows.append(dict(omega_Euclidean=omega,actual_all64_contact_minus_bothinterband=matrix(raw),actual_rho_A0_Schur=matrix(schur),actual_chargedmaterial_Phi_photon_kernel_with_rho=matrix(K),actual_primitive_electric_additive_kernel=matrix(primitive_E),actual_photon_block_with_primitive_E=matrix((K+primitive_E)[1:,1:]),actual_all64_density_positive_bubble=matrix(B[3:,3:]),actual_all64_mixed_Phi_photon_bubble=matrix(B[:3,:3])))
    # Actual EM-neutral -Lambda h: upperpositiveh filled. Never give it charge1.
    cfg=PhiConfig(Lambda=1.,**mean['microscopic_config']);load,model,role=models(cfg)[0]
    if load!=1 or role!='neutral heavy -Lambda' or model.improved:raise ArithmeticError('Actual neutral-heavy law changed')
    onsite,paths=scalar_paths(model);hp=[]
    for displacement,coefficients in paths:
        support=[p for p in native['Pprofiles'][0]['paths'] if np.array_equal(displacement,p['displacement'])]
        if len(support)!=V:raise ArithmeticError('Actual neutral-heavy path support changed')
        hp.extend({**p,'scalars':coefficients,'hop':p['seam']*np.einsum('m,mij->ij',coefficients,native['frame'])} for p in support)
    profile=dict(load=1,model=model,label=role,onsite=onsite*native['frame'][0],paths=hp)
    h=-quantum.matter_hamiltonian(native,profile,theta);Fh=-pauli.operator(native,profile,theta,mode);heavy=spectrum(h,[Fh],rank)
    heavy_static=-float(np.real(bubble(heavy,0.)[0,0]));heavy_remainder=float(np.sum(2*np.abs(heavy['amplitudes'][0])**2/heavy['delta']**3))
    return dict(N=4,V=V,rank_per_flavour=rank,physical_qtheta=[float(q),0.,0.],source_basis=['Cartunit35_Phi0_betaSigma_z_sinqx','Aphys_y_unitcosqx','Aphys_z_unitcosqx','rho_unitcosqx','rho_unitsinqx'],a=a,g_Pi_AND_Coulomb=g,charged_flavours=64,actual_band_gaps=[dict(load=load,gap=data['gap']) for load,data in species],physical_contact=matrix(contact),actual_transverse_divergence_error=divergence,actual_operator_and_periodic_gauge_controls=controls,rows=rows,actual_neutralheavy_PhiPhi_static_divided_by_Lambda=heavy_static,actual_neutralheavy_PhiPhi_Euclidean_remainder_upper_coefficient_omega_squared_divided_by_Lambda=heavy_remainder,actual_neutralheavy_EM_mixed_response='EXACT0, charge0, actual h has NOTheta',heavy_formula='Kheavy_PhiPhi(omega)=Lambda*Kheavy_PhiPhi(0)/Lambda+positive_remainder; 0<=remainder<=omega^2/Lambda*reportedcoefficient. Lambda=2^8192 EXACTsymbolic, never floated/floorzeroed. Upperpositive bareh occupied, unoccupiednegative, gapLambda*gapbare.',primitive_Phi_kinetic_and_boson_self_Noether='NOTincluded/NOTassertedzero; this is only actual material quadratic control, not total Phi-field inverse propagator.',control_state='FULLphysicalnegativeband of chargedh(Theta0) plus positiveband of neutralbareh; NOT originalcanonicalY/Gaussian2/E SOURCE, NOT interactingPi+Coulombvacuum. No source/clock/probability byte changes.')


def run(path:Path,frequencies:list[float])->dict[str,Any]:
    started=time.monotonic();raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=WARD_SHA:raise ArithmeticError('Frozen 1b4/9953 Ward epoch required')
    if not frequencies or any(w<0 or not np.isfinite(w) for w in frequencies):raise ValueError('Actual Euclidean frequencies must be explicit finite nonnegative')
    cap=json.loads(raw);_,record=quantum.checked(cap['input_descriptor']['input_receipts']['moment']);_,record=quantum.checked(record['input_descriptor']['input_receipts']['beta']);_,record=quantum.checked(record['input_descriptor']['input_receipts']['metric']);_,record=quantum.checked(record['input_descriptor']['input_receipts']['magnetic']);_,record=quantum.checked(record['input_descriptor']['input_receipts']['prior_joined_heat'])
    for key in ('full_source','contact','transfer','gram','force','join','heat'):_,record=quantum.checked(record['input_descriptor']['input_receipts'][key])
    mean=pauli.contact.transfer.packets.force.joined.mean_from_source(record)
    desc=dict(plan=cap['input_descriptor']['plan'],plan_revision=pauli.REVISION,scientific_execution_revision=pauli.REVISION,input_receipts=dict(ward=dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest())),original_source_abi_UNCHANGED=cap['input_descriptor']['source_abi'],control_state='Distinct bareCAR physicalfilledband response, NOT original E SOURCE nor primitiveinteractingvacuum.',actual_root_mean_input=mean,native_N=4,frequencies_Euclidean=frequencies,Lambda='2^8192',chi='2^-4096',original_clock_UNCHANGED='4096',source_and_law_cutover='NONE. All physicalprimitivePi+Coul SOURCEreceipts preserved; this bareCAR quadratic discriminator is NOTreplacementlaw.')
    return dict(input_descriptor=desc,input_sha256=hashlib.sha256(json.dumps(desc,sort_keys=True).encode()).hexdigest(),actual_physical_Pauli_Peierls_band_response_control=physical(mean,frequencies),source_at_import={name:dict(sha256=hashlib.sha256(data).hexdigest(),bytes_utf8=data.decode()) for name,data in SOURCE_BYTES.items()},source_unchanged_since_import={name:(ROOT/name).read_bytes()==data for name,data in SOURCE_BYTES.items()},runtime=dict(executable=sys.executable,python=sys.version,numpy=np.__version__),elapsed_seconds=time.monotonic()-started)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--ward',type=Path,required=True);p.add_argument('--frequencies',type=float,nargs='+',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=run(a.ward,a.frequencies)
    if not all(r['source_unchanged_since_import'].values()):raise ArithmeticError('Source changed during actual physical band control')
    a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(dict(input_sha256=r['input_sha256'],physical=r['actual_physical_Pauli_Peierls_band_response_control'],elapsed_seconds=r['elapsed_seconds']),indent=2))
