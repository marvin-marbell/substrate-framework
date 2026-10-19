"""ActualnativePhi35→quantumdensityvacuumfeedback CONTROL, noSOURCEcutover.

SAMEactualnonuniformPhi changesh, EVERYorderedPeierlscurrent/contact and ALL64
PH/densitypoles. PositivecanonicaldensityRPA vacuum retains u64Lplus;
NOToriginalcanonicalY/Ref3 SOURCE, fullPi/Phi vacuum orcommoncone.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Any
import numpy as np
ROOT=Path(__file__).resolve().parent
OWN_BYTES=Path(__file__).read_bytes()
import induced_phi_electron_foldy_density as foldy
spin=foldy.spin;volume=spin.volume;control=spin.control;pauli=spin.pauli;quantum=spin.quantum
RPA_PATH=Path('/tmp/induced-native-longitudinal-rpa-feedback.py')
RPA_BYTES=RPA_PATH.read_bytes()
RPA_SHA='dbe6290017f468841e890e657111da9517d60c3113a0181e00da03862cd01e25'
if hashlib.sha256(RPA_BYTES).hexdigest()!=RPA_SHA:raise RuntimeError('Actual020f positivecanonicaldensity supplier changed')
_spec=importlib.util.spec_from_file_location('actual_phi_density_positivecanonical',RPA_PATH)
if _spec is None or _spec.loader is None:raise RuntimeError('Actualdensity supplier unavailable')
rpa:Any=importlib.util.module_from_spec(_spec);sys.modules[_spec.name]=rpa;_spec.loader.exec_module(rpa)
rpa.driver=rpa.load()
SOURCE_BYTES={**foldy.SOURCE_BYTES,Path(__file__).name:OWN_BYTES,str(RPA_PATH):RPA_BYTES,str(rpa.DRIVER):rpa.DRIVER.read_bytes(),str(rpa.driver.PARENT_PATH):rpa.driver.PARENT_PATH.read_bytes()}
VOLUME_SHA='7ccf67dddd91d89e085ae9b3efd9a98b052a1a4388af280fb18979edba9237e5'
RPA_RECEIPT_SHA='020f0cb348ca4991d09e87c6840e890b3de65533605e3c92c56271caf0dd7fa2'


def phi_profiles(native:dict[str,Any],mode:dict[str,Any],tau:float)->tuple[list[dict[str,Any]],list[np.ndarray]]:
    profiles=[];extra=[];V=native['V']
    for index,p in enumerate(native['Pprofiles']):
        onsite,_=pauli.contact.transfer.packets.force.scalar_paths(p['model']);onsitePhi=np.zeros((4*V,4*V),complex)
        for x in range(V):onsitePhi[4*x:4*x+4,4*x:4*x+4]=tau*onsite*mode['mode'][x]*mode['delta']
        paths=[]
        for path in p['paths']:
            x,y=path['source'],path['target'];d=path['scalars'][0]*(mode['mode'][x]+mode['mode'][y])/2*mode['delta']*path['seam']
            paths.append({**path,'hop':path['hop']+tau*d})
        profiles.append({**p,'paths':paths,'actual_Phi_profile_index':index});extra.append(onsitePhi)
    return profiles,extra


def quantum_density_integral(data:dict[str,Any],order:int)->dict[str,Any]:
    # Literal BOTH poles, no unproved REAL-PH shortcut under nonuniformPhi:
    # Xij=G_iG_j*/(Delta-iomega)+G_i*G_j/(Delta+iomega).
    # Equivalent positivecanonicalH=Delta(x²+p²)/2+rho²/2,
    # rho=sqrt2(ReG x+ImG p); retain actualxp/Berryterms.
    x,w=np.polynomial.legendre.leggauss(order);x=(x+1)/2;w=w/2;omega=32*np.tan(np.pi*x/2);jac=32*np.pi/(2*np.cos(np.pi*x/2)**2);energy=0.;C=0.;first=0.;asym=0.;minimum=0.;V=int(data['V']);identity=np.eye(V-1)
    commutator=np.zeros((V-1,V-1),dtype=np.complex128)
    for item in data['species']:commutator+=item['load']*(item['G']@item['G'].conj().T)
    commutator_error=float(np.max(abs(commutator.imag)))
    if commutator_error>1e-8:raise ArithmeticError('ActualcompletePHdensity canonicalcommutator failed')
    for om,weight in zip(omega,w*jac/(2*np.pi),strict=True):
        X=np.zeros((V-1,V-1))
        for item in data['species']:
            G=item['G'];X+=2*((G*(item['load']/(item['delta']-1j*om))[None,:])@G.conj().T).real
        sign,logdet=np.linalg.slogdet(identity+X);smallest=float(np.linalg.eigvalsh((X+X.T)/2)[0]);minimum=min(minimum,smallest);asym=max(asym,float(np.max(abs(X-X.T))))
        if sign<=0 or smallest<-1e-9:raise ArithmeticError('Actualgeneralcanonicaldensity bath lost positivequadratic law')
        energy+=weight*logdet;C+=weight*np.trace(np.linalg.solve(identity+X,X));first+=weight*np.trace(X)
    return dict(frequency_order=order,actual_density_RPA_zero_point_shift=float(energy),actual_density_RPA_C64_expectation=float(C),actual_density_RPA_PH_excitation_energy=float(energy-C),integrated_FIRST_order_C64=float(first),literal_bare_C64=data['bare_C64'],first_order_C64_integral_relative_error=float(abs(first-data['bare_C64'])/data['bare_C64']),actual_density_BOTHpoles_antisymmetric_frequency_kernel_max=asym,actual_unclipped_density_symmetric_Gram_min_eigenvalue=minimum,actual_completePH_density_commutator_error=commutator_error,actual_HCAR_plus_density_RPA_conditional_energy=data['HCAR']+float(energy))


def physical(mean:dict[str,Any],orders:list[int],probes:list[float],photons:list[float])->dict[str,Any]:
    native=quantum.build_native(mean,4);V=native['V'];a=native['a'];mode=pauli.magnetic_shape(native);direction=np.zeros(3*V);direction[1::3]=np.sqrt(2/V)*np.cos(2*np.pi*native['sites'][:,0]/4);zero=np.zeros(3*V)
    source_profiles=native['Pprofiles'];identities=[];rows=[];tables={};actual_points={(0.,0.)}
    for tau in probes:
        actual_points.update({(tau,0.),(-tau,0.)})
    for amplitude in photons:actual_points.update({(0.,amplitude),(0.,-amplitude)})
    for tau,amplitude in zip(probes,photons,strict=True):actual_points.update({(u*tau,v*amplitude) for u in (-1,1) for v in (-1,1)})
    for tau,amplitude in sorted(actual_points):
        profiles,extra=phi_profiles(native,mode,tau);perturbed={**native,'Pprofiles':profiles}
        def literal_h(actual:dict[str,Any],profile:dict[str,Any],theta:np.ndarray)->np.ndarray:
            index=profile['actual_Phi_profile_index']
            return quantum.matter_hamiltonian(actual,profile,theta)+extra[index]
        supplier=SimpleNamespace(primitives=SimpleNamespace(matter_hamiltonian=literal_h));theta=a*amplitude*direction
        data=rpa.spectral(supplier,perturbed,theta);answers=[quantum_density_integral(data,o) for o in orders]
        row=dict(actual_CartPhi35_normalizedmode_amplitude=tau,actual_Aphysical_y_normalizedmode_amplitude=amplitude,actual_HCAR=data['HCAR'],actual_literal_bare_C64=data['bare_C64'],actual_first_order_bothpole_C64_join_relative_error=data['first_order_C64_join_relative_error'],actual_quantum_density_vacuum_feedback=answers);rows.append(row);tables[(tau,amplitude)]=row
        if amplitude==0. and tau in (0.,probes[-1]):
            for i,(p,p0) in enumerate(zip(profiles,source_profiles,strict=True)):
                H=literal_h(perturbed,p,zero);F=pauli.operator(native,p0,zero,mode);expected=quantum.matter_hamiltonian(native,p0,zero)+tau*F
                error=float(np.linalg.norm(H-expected));J=a*quantum.path_derivative(perturbed,p,zero,direction);C=a*a*quantum.path_derivative(perturbed,p,zero,direction,order=2);eps=2e-3;hp=literal_h(perturbed,p,eps*a*direction);hm=literal_h(perturbed,p,-eps*a*direction);errJ=float(np.linalg.norm((hp-hm)/(2*eps)-J));errC=float(np.linalg.norm((hp-2*H+hm)/(eps*eps)-C))
                alpha=.31*np.sin(2*np.pi*native['sites'][:,0]/4);gradient=native['D'].T@alpha;gaugephase=np.repeat(np.exp(1j*alpha),4);gauged=literal_h(perturbed,p,gradient);gaugeError=float(np.linalg.norm(gauged-gaugephase[:,None]*H*gaugephase.conj()[None,:]));densitydata=rpa.spectral(supplier,perturbed,gradient);gaugeFeedback=quantum_density_integral(densitydata,orders[-1]);gaugeEnergy=abs(gaugeFeedback['actual_density_RPA_zero_point_shift']-answers[-1]['actual_density_RPA_zero_point_shift'])
                if max(error,gaugeError)>1e-10 or errJ>1e-8 or errC>1e-6 or gaugeEnergy>1e-7:raise ArithmeticError('ActualPhi/fullh/Peierlscurrent/contact/gauge quantumdensity consumer failed')
                identities.append(dict(phi_amplitude=tau,role=p['label'],actual_hPhi_equals_hPlusPhiF_error=error,actual_FULLmodified1_2link_current_finite_difference_error=errJ,actual_FULLmodified1_2link_contact_finite_difference_error=errC,actual_nonuniformPhi_U1_gauge_covariance_error=gaugeError,actual_nonuniformPhi_density_quantumvacuum_gauge_energy_error=gaugeEnergy))
    baseline=tables[(0.,0.)];curvatures=[]
    def E(row:dict[str,Any],index:int,part:str)->float:
        if part=='CAR':return row['actual_HCAR']
        answer=row['actual_quantum_density_vacuum_feedback'][index]
        if part=='densityZPE':return answer['actual_density_RPA_zero_point_shift']
        return answer['actual_HCAR_plus_density_RPA_conditional_energy']
    for tau,amplitude in zip(probes,photons,strict=True):
        by_order=[]
        for index,o in enumerate(orders):
            fields={}
            for part in ('CAR','densityZPE','CAR_plus_densityZPE'):
                phi=(E(tables[(tau,0.)],index,part)+E(tables[(-tau,0.)],index,part)-2*E(baseline,index,part))/(tau*tau)
                photon=(E(tables[(0.,amplitude)],index,part)+E(tables[(0.,-amplitude)],index,part)-2*E(baseline,index,part))/(amplitude*amplitude)
                mixed=sum(u*v*E(tables[(u*tau,v*amplitude)],index,part) for u in (-1,1) for v in (-1,1))/(4*tau*amplitude)
                fields[part]=dict(Phi35_Phi35=phi,Phi35_Aphysical_y=mixed,Aphysical_y_Aphysical_y=photon)
            by_order.append(dict(frequency_order=o,actual_paired_finite_amplitude_quantum_feedback_curvatures=fields))
        curvatures.append(dict(phi_amplitude=tau,photon_Aphysical_amplitude=amplitude,actual_NO_fit_finite_amplitude_consumer=by_order))
    if sum(p['load'] for p in source_profiles)!=64:raise ArithmeticError('Actualall64load changed')
    return dict(N=4,V=V,a=a,g_Pi_AND_Coulomb=native['UE']/a,actual_root_g0=native['g0'],actual_root_gs=native['gs'],ALLcharged_flavours=64,actual_nonuniform_CartPhi35_mode=mode['mode'].tolist(),actual_photon_Aphysical_unitmode_norm=float(np.linalg.norm(direction)),actual_poles_and_Canonical_density_vacuum_feedback_rows=rows,actual_complete_Phi_h_current_contact_and_quantumgauge_consumers=identities,actual_Phi_Photon_quantumfeedback_finite_amplitude_curvatures=curvatures,scope='NEWdistinctconditionalcanonicaldensityRPA vacuum builtonactualbareall64negativeCARbands, literalnativePhi35+ALLupdatedorderedpathh/current/contact, sameu64Lplus. BOTHdensitypoles/firstorderjoinnativeC, wholepositive63realdensitybath logdet/expectedC/PHexcitationretained. NOToriginalcanonicalY/Gaussian2/AP222/E/Ref3 SOURCE, fullPhi/SU4/Pi interactingvacuum, eR/pole/commoncone. No fittedquadratic, coefficientrows areactualfinite±Phi/±A differences with independentfrequencyorders. No heavyEMcoupling; heavyPhi force2^8192 belongsseparatebranch; primitivePhi/colour fulltimeNOTpropagated.')


def run(volume_path:Path,rpa_path:Path,orders:list[int],probes:list[float],photons:list[float])->dict[str,Any]:
    start=time.monotonic();raw=volume_path.read_bytes();rparaw=rpa_path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=VOLUME_SHA or hashlib.sha256(rparaw).hexdigest()!=RPA_RECEIPT_SHA:raise ArithmeticError('Frozen00e/7cc andnative020f quantumdensity epochs required')
    if len(probes)!=len(photons) or not probes or any(x<=0 for x in probes+photons) or min(orders)<4:raise ValueError('ExplicitpositivepairedPhi/photonamplitudes andfrequencyorders required')
    cap=json.loads(raw);producer=json.loads(rparaw)
    if producer['provenance']['own_sha256']!=RPA_SHA:raise ArithmeticError('Actualpositivecanonicaldensityproducer interface changed')
    mean=cap['input_descriptor']['actual_root_mean_input'];desc=dict(plan=cap['input_descriptor']['plan'],plan_revision=pauli.REVISION,scientific_execution_revision=pauli.REVISION,input_receipts=dict(volume=dict(path=str(volume_path),sha256=hashlib.sha256(raw).hexdigest()),quantum_density_RPA=dict(path=str(rpa_path),sha256=hashlib.sha256(rparaw).hexdigest(),supplier_execution_epoch=producer['assigned_execution_epoch'])),actual_root_mean_input=mean,original_source_abi_UNCHANGED=cap['input_descriptor']['original_source_abi_UNCHANGED'],actual_Phi_probes=probes,actual_physical_photon_probes=photons,actual_frequency_orders=orders,original_Lambda_UNCHANGED='2^8192',original_chi_UNCHANGED='2^-4096',original_clock_UNCHANGED='4096',branch='DistinctpositivecanonicaldensityRPAquantumvacuumΦfeedback; no originalSOURCEretag.')
    return dict(input_descriptor=desc,input_sha256=hashlib.sha256(json.dumps(desc,sort_keys=True).encode()).hexdigest(),actual_native_Phi_quantumdensity_photon_feedback=physical(mean,orders,probes,photons),source_at_import={name:dict(sha256=hashlib.sha256(data).hexdigest(),bytes_utf8=data.decode()) for name,data in SOURCE_BYTES.items()},source_unchanged_since_import={name:(ROOT/name).read_bytes()==data for name,data in SOURCE_BYTES.items()},runtime=dict(executable=sys.executable,python=sys.version,numpy=np.__version__),elapsed_seconds=time.monotonic()-start)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--volume',type=Path,required=True);p.add_argument('--density',type=Path,required=True);p.add_argument('--orders',type=int,nargs='+',required=True);p.add_argument('--phi',type=float,nargs='+',required=True);p.add_argument('--photon',type=float,nargs='+',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=run(a.volume,a.density,a.orders,a.phi,a.photon)
    if not all(r['source_unchanged_since_import'].values()):raise RuntimeError('ActualPhi density consumer sourcechanged duringexecution')
    a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(dict(input_sha256=r['input_sha256'],physical=r['actual_native_Phi_quantumdensity_photon_feedback'],elapsed_seconds=r['elapsed_seconds']),indent=2))
