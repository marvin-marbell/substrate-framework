#!/usr/bin/env python3
"""Actual charge-dependent photon response: charged sector minus neutral vacuum.

Same coupled H/current/contact, measured interacting photon band, independent
quantum-time consumer in BOTH sectors. Not an asymptotic Compton S-matrix.
"""
from __future__ import annotations
import argparse,hashlib,json,math,os
from pathlib import Path
if __name__=='__main__':
    for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import numpy as np
from noncompact_native_current import Coupled,accept,adverse,source_operators
from noncompact_photon_response import coordinate,complete_source_charged_ground,source_density,response,quantum_time_response,trace_observable,encoded,F,INDEPENDENT
from noncompact_photon_spectrum import full_source_measure
from original_charged_principal import original_frame,portable
from rotation_band import RotationBand
ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--n',type=int,default=5);p.add_argument('--M',type=int,default=3)
    p.add_argument('--UE',type=float,default=.37);p.add_argument('--K',type=float,default=.9)
    p.add_argument('--eta',type=float,default=.5);p.add_argument('--intervals',type=int,default=1024)
    args=p.parse_args()
    if args.eta<=0:raise ValueError('Retarded finite-frequency diagnostic requires positive broadening')
    selected,frame=original_frame(RotationBand(a=1.,m=.2,r=.12,v=.3,improved=False),5,np.zeros(3),True)
    neutral=Coupled(args.n,args.M,0.,4,args.UE,args.K,frame)
    charged=Coupled(args.n,args.M,0.,5,args.UE,args.K,frame)
    en,vn,resn=neutral.solve(4);E0=float(en[0]);psi=vn[:,0]
    if en[1]-en[0]<1e-7:raise AssertionError('Complete neutral degenerate density must replace a selected eigenvector')
    Qpsi,leakage=coordinate(neutral,psi);q=Qpsi-np.vdot(psi,Qpsi)*psi
    indices,values,X,bands,certificate=full_source_measure(neutral,E0,q,3000)
    ranked=sorted((r for r in bands if r['energy_interval'][0]>1e-7),key=lambda r:r['residue'],reverse=True)
    dominant=ranked[0]
    margin=dominant['residue']-dominant['residue_error_upper']-max(r['residue']+r['residue_error_upper'] for r in ranked[1:])
    if margin<=0:raise AssertionError('Actual complete coordinate spectrum does not resolve a dominant band')
    u=np.zeros(neutral.size,dtype=complex);j=dominant['begin'];u[indices]=X[:,j]
    accept(np.linalg.norm(neutral.action(u)-values[j]*u),'actual photon band eigenpair in whole engine',2e-8)
    point=np.asarray([neutral.reshape(psi)@t.T for t in source_operators(neutral,charged)[0]])
    sector,weights=charged.source_triality(point)
    Ec,G,gap,resc=complete_source_charged_ground(charged,sector)
    rho,source=source_density(neutral,charged,psi,G,E0)
    neutralG=psi[:,None];neutralrho=np.ones((1,1),dtype=complex)
    outputs=[]
    for a,b,label in ((F,F,'restored_restored'),(F,INDEPENDENT,'restored_independent')):
        z=complex(dominant['energy'],args.eta)
        positive,cplus,firstplus,secondplus=response(charged,Ec,G,a,b,z)
        vacuum,c0,first0,second0=response(neutral,E0,neutralG,a,b,z)
        measured_plus=trace_observable(positive,rho)
        measured_zero=trace_observable(vacuum,neutralrho)
        connected=measured_plus-measured_zero
        actual_plus,time_plus=quantum_time_response(charged,Ec,G,rho,a,b,z,args.intervals)
        actual_zero,time_zero=quantum_time_response(neutral,E0,neutralG,neutralrho,a,b,z,args.intervals)
        measured_time=actual_plus-actual_zero
        tolerance=time_plus['numerical_acceptance_tolerance']+time_zero['numerical_acceptance_tolerance']
        accept(abs(connected-measured_time),'same-state charge-dependent response versus two actual quantum-time histories',tolerance)
        controls={}
        for name,wrong in (
            ('unsubtracted_charged_response',measured_plus),
            ('zero_charge_response',0j),
            ('missing_charge_contact',connected-trace_observable(cplus,rho)+trace_observable(c0,neutralrho)),
            ('missing_second_ordering',connected+trace_observable(secondplus,rho)-trace_observable(second0,neutralrho))):
            controls[name]=adverse(abs(wrong-measured_time),'same-state charge-dependent response versus two actual quantum-time histories',tolerance)
        band_factor=dominant['residue']*float(F@a)*float(F@b)
        outputs.append(dict(source=label,z=encoded(z),raw_charged=encoded(measured_plus),actual_neutral_background=encoded(measured_zero),charge_dependent_connected=encoded(connected),actual_two_history_response=encoded(measured_time),neutral_quantum_time=time_zero,charged_quantum_time=time_plus,numerical_acceptance_tolerance=tolerance,charge_contact=encoded(trace_observable(cplus,rho)-trace_observable(c0,neutralrho)),measured_interacting_band_factor=band_factor,band_amputated_charge_response=encoded(band_factor*connected),same_assertion_controls=controls))
    names=('noncompact_connected_photon.py','noncompact_connected_photon.md','noncompact_photon_response.py','noncompact_photon_spectrum.py','noncompact_native_current.py')
    print(json.dumps(portable(dict(law='different noncompact root; same complete coupled light/photon H in both charge sectors',native_N5_frame=selected['G'],E0=E0,Ec=Ec,source_sector=sector,source_sector_weights=weights,charged_cluster_gap=gap,source_density=source,neutral_residual=max(resn),charged_residual=max(resc),dominant_actual_photon_band=dominant,dominance_margin=margin,full_neutral_Q_spectral_certificate=certificate,exact_Q_output_leakage=leakage,photon_band_infinite_comparison=neutral.full_residual_bound(u,float(values[j])),responses=outputs,scope='finite graph, finite Galerkin, actual complete-reference source-addressed charge density, finite positive broadening; charge-dependent diagnostic band response, NOT LSZ scattering proof',unpaid=['reference separation and thermodynamic photon phase','stable below-pair-threshold asymptotic incoming/outgoing photon states','nonforward spatial source/angle/recoil and inclusive charged packet','full scalar/heavy/colour normal transfer','mechanical spin and Lorentz identification'],source_hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names})),indent=2,allow_nan=False))

if __name__=='__main__':main()
