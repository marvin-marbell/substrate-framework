#!/usr/bin/env python3
"""Actual full4Dirac canonical SF current/contact and charged-source Ward probe.

The frozen0226 literal-Haar consumer is unchanged.  These are derivatives of
its actual SF factors in the ORIGINAL site-major full-Fock representation,
including scalar detB, mixed matrix-log and nonlinear exponential contact.
They are NOT an assembled -log(K)/a current, a differentiated occupied-sea
compression, an interacting pole, or a photon/Compton certificate.
Local U1 gauge and the Ref0/external-Q source morphism are exercised before
choosing tree gauge; transverse and Phi contacts are checked independently.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import numpy as np
import scipy
import physical_c_compact_haar_charged_source as consumer

SOURCE_BYTES=Path(__file__).read_bytes()
CONSUMER_BYTES=Path(consumer.__file__).read_bytes()
if hashlib.sha256(CONSUMER_BYTES).hexdigest()!='0226e06a911fbdbdc4cf5f40a2dba8ded5873590305fc37bdbcfb2e04b213259':raise ArithmeticError('Current probe requires frozen0226 fullFock Haar consumer')
ALPHA=consumer.ALPHA


def canonical_jet(law: Any,phases: np.ndarray,remainder: np.ndarray,links: np.ndarray,
                  number: int,origin: float,directions: tuple[np.ndarray,np.ndarray],
                  phi_directions: tuple[np.ndarray,np.ndarray] | None=None) -> dict[str,Any]:
    """Complete canonical SF jet; no physical current/pole or vacuum adoption."""
    potential=law.potential(phases,remainder,links,directions,phi_directions)
    generators=[]
    for suffix in ('','x','y','xy'):
        generator,_=ALPHA.second_quantized(potential['h'+suffix],number)
        generator+=(potential['pref'+suffix]-(origin if suffix=='' else 0))*np.eye(len(generator))
        generators.append(generator/2)
    value,first,second,mixed=ALPHA.matrix_exp_jet(*generators)
    return dict(SF=value,SFx=first,SFy=second,SFxy=mixed,generators=generators,potential=potential)


def packet_creation(packet: np.ndarray) -> np.ndarray:
    """Exact physical-site 8CAR creation into four coherent orthogonal Ref3 holes."""
    neutral,_=consumer.basis(4);charged,_=consumer.basis(5);lookup={m:i for i,m in enumerate(charged)}
    result=np.zeros((4,56,70),complex)
    for column,mask in enumerate(neutral):
        for bit in range(8):
            if not mask>>bit&1:result[:,lookup[mask|(1<<bit)],column]+=(-1)**((mask&((1<<bit)-1)).bit_count())*packet[bit]
    return result


def source_jet(law: Any,phases: np.ndarray,links: np.ndarray,
               directions: tuple[np.ndarray,np.ndarray]) -> dict[str,np.ndarray]:
    packets=law.source_packet_jet(phases,links,directions)
    return {suffix:packet_creation(packets['psi'+suffix]) for suffix in ('','_x','_y','_xy')}


def gauge_generator(number: int,q: int,chi: np.ndarray,external: bool=False) -> np.ndarray:
    masks,_=consumer.basis(number)
    values=[q*((mask&15).bit_count()-2)*chi[0]+q*((mask>>4).bit_count()-2)*chi[1]-(q*chi[1] if external else 0) for mask in masks]
    return np.diag(values)


def commutator(left: np.ndarray,right: np.ndarray) -> np.ndarray:
    return left@right-right@left


def gauge_probe(law: Any,phases: np.ndarray,remainder: np.ndarray,links: np.ndarray,
                number: int,origin: float,q: int,chi: np.ndarray) -> dict[str,Any]:
    gradient=np.zeros_like(phases);gradient[:,0]=[chi[0]-chi[1],chi[1]-chi[0]]
    jet=canonical_jet(law,q*phases,remainder,links,number,origin,(q*gradient,q*gradient))
    g=gauge_generator(number,q,chi)
    expected_first=1j*commutator(g,jet['SF']);expected_second=-commutator(g,commutator(g,jet['SF']))
    no_exp_contact=scipy.linalg.expm_frechet(jet['generators'][0],jet['generators'][3],compute_expm=False)
    first_error=float(np.linalg.norm(jet['SFx']-expected_first));second_error=float(np.linalg.norm(jet['SFxy']-expected_second))
    if max(first_error,second_error)>3e-8:raise ArithmeticError('Full canonical SF violated actual U1 gauge Ward/contact')
    return dict(number=number,full_dimension=len(g),first_Ward_defect=first_error,mixed_Ward_contact_defect=second_error,
        OMITnonlinear_exponential_contact_defect=float(np.linalg.norm(no_exp_contact-expected_second)),
        primitive_scalar_gauge_first=jet['potential']['prefx'],primitive_scalar_gauge_mixed=jet['potential']['prefxy'])


def source_gauge_probe(law: Any,phases: np.ndarray,links: np.ndarray,q: int,chi: np.ndarray) -> dict[str,Any]:
    gradient=np.zeros_like(phases);gradient[:,0]=[chi[0]-chi[1],chi[1]-chi[0]]
    jet=source_jet(law,q*phases,links,(q*gradient,q*gradient));gn=gauge_generator(4,q,chi);gc=gauge_generator(5,q,chi,True)
    expected=np.array([1j*(gc@s-s@gn) for s in jet['']]);mixed=np.array([-(gc@(gc@s-s@gn)-(gc@s-s@gn)@gn) for s in jet['']])
    first=float(np.linalg.norm(jet['_x']-expected));second=float(np.linalg.norm(jet['_xy']-mixed))
    obsolete=gauge_generator(5,q,chi)
    wrong=np.array([1j*(obsolete@s-s@gn) for s in jet['']])
    if max(first,second)>3e-11:raise ArithmeticError('Ref0/external-Q source gauge morphism failed')
    return dict(source_first_Ward_defect=first,source_mixed_contact_defect=second,
        OMITexternal_compensator_source_Ward_defect=float(np.linalg.norm(jet['_x']-wrong)),
        source_contract='Refprimitive0, neutralexternal0 -> chargedexternal-Q atsite1; primitive4Dirac cdag with actual Wilson path, four orthogonalRef3 holes. NOT RefQ law.')


def finite_probe(law: Any,phases: np.ndarray,remainder: np.ndarray,links: np.ndarray,
                 number: int,origin: float,directions: tuple[np.ndarray,np.ndarray],
                 phi_directions: tuple[np.ndarray,np.ndarray] | None=None) -> dict[str,Any]:
    jet=canonical_jet(law,phases,remainder,links,number,origin,directions,phi_directions)
    dx,dy=directions;px,py=(np.zeros_like(remainder),np.zeros_like(remainder)) if phi_directions is None else phi_directions
    def value(x: float,y: float) -> np.ndarray:
        potential=law.potential(phases+x*dx+y*dy,remainder+x*px+y*py,links)
        return consumer.canonical_SF(potential,number,origin)
    rows=[]
    for step in (.004,.002,.001):
        first=(value(step,0)-value(-step,0))/(2*step)
        mixed=(value(step,step)-value(step,-step)-value(-step,step)+value(-step,-step))/(4*step*step)
        rows.append(dict(step=step,first_absolute_defect=float(np.linalg.norm(first-jet['SFx'])),mixed_absolute_defect=float(np.linalg.norm(mixed-jet['SFxy']))))
    generator=jet['generators'][0];only_h,_=ALPHA.second_quantized(jet['potential']['hx'],number)
    without_scalar=scipy.linalg.expm_frechet(generator,only_h/2,compute_expm=False)
    only_primitive_contact=scipy.linalg.expm_frechet(generator,jet['generators'][3],compute_expm=False)
    if rows[-1]['first_absolute_defect']>2e-5 or rows[-1]['mixed_absolute_defect']>2e-5:raise ArithmeticError('Full canonical current/contact differs from actual nonlinear field consumer')
    return dict(number=number,full_dimension=len(generator),SF_norm=float(np.linalg.norm(jet['SF'])),
        current_SFx_norm=float(np.linalg.norm(jet['SFx'])),contact_SFxy_norm=float(np.linalg.norm(jet['SFxy'])),finite_differences=rows,
        OMITdetB_scalar_current_defect=float(np.linalg.norm(without_scalar-jet['SFx'])),
        OMITnonlinear_exponential_contact_defect=float(np.linalg.norm(only_primitive_contact-jet['SFxy'])))


def run() -> dict[str,Any]:
    parameters=consumer.reference.NativeParameters();active,reference,normal=consumer.reference.native_law(parameters)
    auxiliary,_,_=consumer.reference.native_law(parameters);auxiliary.kappa=.08
    _,la=consumer.meanbody(active);_,lu=consumer.meanbody(auxiliary)
    phases=np.array([[0.,.12,.17],[.23,-.19,.08]]);remainder=np.zeros((2,4,4,4),complex);links=np.broadcast_to(np.eye(4),(2,3,4,4)).astype(complex)
    q=3;beta=100.;chi=np.array([.31,-.17]);dy=np.zeros_like(phases);dz=np.zeros_like(phases)
    dy[:,1]=[1.,-1.];dz[:,2]=[1.,-1.];dy/=np.sqrt(2*beta);dz/=np.sqrt(2*beta)
    zero=np.zeros_like(remainder);normal_field=np.broadcast_to(normal,(2,4,4,4)).copy()
    gauge=[gauge_probe(active,phases,remainder,links,n,la,q,chi) for n in (4,5)]
    gauge.append(gauge_probe(auxiliary,phases,remainder,links,4,lu,1,chi))
    transverse=[finite_probe(active,q*phases,remainder,links,n,la,(q*dy,q*dz)) for n in (4,5)]
    transverse.append(finite_probe(auxiliary,phases,remainder,links,4,lu,(dy,dz)))
    phi=[finite_probe(active,q*phases,remainder,links,n,la,(q*dy,np.zeros_like(dy)),(zero,normal_field)) for n in (4,5)]
    source=source_gauge_probe(active,phases,links,q,chi)
    inputs=dict(Q=q,Naux=2,kaux=.08,geometry=[2,1,1],bare_mass=parameters.critical_bare_mass,spacing=parameters.spacing,
        phases=phases.tolist(),canonical_beta_CALIBRATION=beta,phase_directions=[dy.tolist(),dz.tolist()],gauge_chi=chi.tolist(),
        Phi_mixed_direction_real=np.real(normal_field).tolist(),Phi_mixed_direction_imag=np.imag(normal_field).tolist(),
        origin_loglambda_active=la,origin_loglambda_auxiliary=lu,
        semantics='Primitive SF first/mixed jets in actual complete site-major canonical Fock; fixed scalar origins, exactdetB pref and nonlinear contact. LocalGauge Ward beforetree quotient, sourceexternalcompensator retained. No physical -log(fullK)/a current, connection/current amputation, Photon/NF/fullPhi/colour kinetic or actual shared Compton is inferred.',
        source_factory='frozen0226 finiteHaar/complete4Dirac/fullGauss only. beta100 is canonical-direction CALIBRATION, NOT complete16Nauxkaux4 induced Maxwell coupling. Native model has shortL1/2 windings.')
    return dict(gauge=gauge,source_gauge=source,transverse_current_contact=transverse,mixed_Phi_current_contact=phi,input=inputs)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True);args=parser.parse_args();started=time.time();report=run();inputs=report['input']
    report['execution']=dict(source_at_IMPORT=hashlib.sha256(SOURCE_BYTES).hexdigest(),source_bytes_at_IMPORT=SOURCE_BYTES.decode(),consumer_at_IMPORT=hashlib.sha256(CONSUMER_BYTES).hexdigest(),consumer_bytes_at_IMPORT=CONSUMER_BYTES.decode(),reference_at_IMPORT=hashlib.sha256(consumer.REFERENCE_BYTES).hexdigest(),ledger_at_IMPORT=hashlib.sha256(consumer.ALPHA_BYTES).hexdigest(),input_sha256=hashlib.sha256(json.dumps(inputs,sort_keys=True).encode()).hexdigest(),input_json=inputs,python=sys.version,numpy=np.__version__,scipy=scipy.__version__,runtime=platform.platform(),startup_unchanged=Path(__file__).read_bytes()==SOURCE_BYTES and Path(consumer.__file__).read_bytes()==CONSUMER_BYTES,elapsed_seconds=time.time()-started)
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ('input','execution')}))


if __name__=='__main__':main()
