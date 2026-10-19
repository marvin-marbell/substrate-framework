#!/usr/bin/env python3
"""Literal large finite-box seam/current/contact consumer and rounding prices.

The displayed matrix image is a diagnostic, never the physical implicit Phi
centre. Bounds price its distance to every actual full-ten sector centre.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
from fractions import Fraction
import hashlib
import itertools
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any
import numpy as np
import mpmath as mp
from mpmath.ctx_iv import MPIntervalContext
from induced_phi_backreaction import PhiConfig,models,scalar_paths,matrix_unit_basis,SELECTED,IMPORT_BYTES as BASE_BYTES

ROOT=Path(__file__).resolve().parent
IMPORT_BYTES={**BASE_BYTES,Path(__file__).name:Path(__file__).read_bytes()}
IMPORT_HASHES={name:hashlib.sha256(data).hexdigest() for name,data in IMPORT_BYTES.items()}
MODE_INDICES=((8,0,0),(0,8,0),(0,0,8),(8,8,0),(8,8,8),(5,7,11),(64,0,0),(45,45,0),(37,37,37))
CENTRES=(1,1j,-1,-1j)


def bound(value: dict[str,str]) -> Fraction:
    return Fraction(int(value['numerator']),int(value['denominator']))


def endpoints(value: Any) -> tuple[Fraction,Fraction]:
    result=[]
    for sign,mantissa,exponent,_ in value._mpi_:
        result.append(Fraction((-1 if sign else 1)*int(mantissa))*Fraction(2)**int(exponent))
    return result[0],result[1]


def encoded(value: Fraction) -> dict[str,str]:
    with mp.workdps(80):display=str(mp.nstr(mp.mpf(value.numerator)/value.denominator,60))
    return dict(numerator=str(value.numerator),denominator=str(value.denominator),decimal_display=display)


def exact(context: Any,value: Fraction|int|float) -> Any:
    rational=Fraction(value)
    return context.mpf(rational.numerator)/rational.denominator


def rounding_prices(cfg: PhiConfig,mean: dict[str,Any],scales: np.ndarray) -> dict[str,Any]:
    c: Any=MPIntervalContext();c.dps=180
    enclosure=mean['centre_enclosure']
    midpoint0=(bound(enclosure['g0_lower'])+bound(enclosure['g0_upper']))/2
    midpointS=(bound(enclosure['gs_lower'])+bound(enclosure['gs_upper']))/2
    half0=(bound(enclosure['g0_upper'])-bound(enclosure['g0_lower']))/2
    halfS=(bound(enclosure['gs_upper'])-bound(enclosure['gs_lower']))/2
    sector=bound(enclosure['all64_Gram_radius_upper'])
    image_squared=[Fraction(float(scale))**2 for scale in scales]
    image_delta=c.sqrt(exact(c,(image_squared[0]-midpoint0)**2+sum((image_squared[i]-midpointS)**2 for i in range(1,4))))
    root_delta=c.sqrt(exact(c,half0**2+3*halfS**2))+exact(c,sector)+image_delta
    minimum=bound(enclosure['minimum_Gram_eigenvalue_lower'])
    if minimum<=0:raise ArithmeticError('No positive actual full-sector Gram root bound')
    denominator=c.sqrt(exact(c,minimum))+exact(c,min(Fraction(float(scale)) for scale in scales))
    frame_delta=root_delta/denominator
    rows=[]
    for load,model,role in models(cfg):
        onsite,paths=scalar_paths(model)
        cap=abs(exact(c,float(onsite)))
        for _,scalars in paths:
            for scalar in scalars:
                real,imag=exact(c,float(scalar.real)),exact(c,float(scalar.imag))
                cap+=2*c.sqrt(real*real+imag*imag)
        heavy=role=='neutral heavy -Lambda'
        multiplier=2**mean['Lambda_exponent'] if heavy else 1
        species=1 if heavy else int(load)
        h=4*cap*frame_delta
        j=4*cap*frame_delta
        jj=8*cap*frame_delta
        rows.append(dict(role=role,species_count=species,physical_band_multiplier=str(-multiplier if heavy else multiplier),
            unscaled_rotation_band_image_error_upper=encoded(endpoints(h)[1]),
            unscaled_rotation_colour_current_image_error_upper=encoded(endpoints(j)[1]),
            unscaled_rotation_colour_contact_image_error_upper=encoded(endpoints(jj)[1]),
            physical_one_particle_band_image_error_upper=encoded(endpoints(multiplier*h)[1]),
            physical_one_particle_colour_current_image_error_upper=encoded(endpoints(multiplier*j)[1]),
            physical_one_particle_colour_contact_image_error_upper=encoded(endpoints(multiplier*jj)[1]),
            physical_one_particle_EM_current_image_error_upper=encoded(endpoints(exact(c,0) if heavy else 4*exact(c,cfg.a)*cap*frame_delta)[1]),
            physical_one_particle_EM_contact_image_error_upper=encoded(endpoints(exact(c,0) if heavy else 8*exact(c,cfg.a)**2*cap*frame_delta)[1]),
            physical_full_species_CAR_band_image_error_upper=encoded(endpoints(4*mean['finite_size']**3*species*multiplier*h)[1])))
    return dict(Gram_distance_from_actual_fullG_upper=encoded(endpoints(root_delta)[1]),
        Cartesian_frame_distance_upper=encoded(endpoints(frame_delta)[1]),profiles=rows,
        derivation='Exact000 enclosure halfwidth plus ALL64 distance from exact000 plus actual emitted binary64 scale-square error. Positive matrix square-root Sylvester bound ||sqrtG-Cimage||F<=||G-Cimage^2||F/(sqrt(lambda_minG)+lambda_minCimage). Literal one/two-link path lengths<=2 and unitary generators price linear band/current/contact images. FullCAR bound retains4V per flavour: it can be large and is NOT suppressed or promoted to a small full-reference error.',
        image_role='DIAGNOSTIC/one-particle image only; physical reference uses the exact implicit FULLG_t, not this midpoint')


def sparse_curve(cfg: PhiConfig,size: int,frame: np.ndarray,twist: tuple[int,...],mode: tuple[int,...]) -> list[dict[str,Any]]:
    positions=sorted({0,1,size-2,size-1,size//7,size//3})
    sites=np.asarray(list(itertools.product(positions,repeat=3)),dtype=int)
    generator=matrix_unit_basis()[7]/np.sqrt(8)
    values,vectors=np.linalg.eigh(generator)
    qtheta=2*np.pi*np.asarray(mode)/size
    integers=np.asarray([0 if t<2 else -1 for t in twist])
    ptheta=2*np.pi*(integers+np.asarray(twist)/4)/size
    wave=np.exp(1j*sites@ptheta)[:,None,None]*np.eye(4)[None]
    step=1e-3
    def link(coordinates,axis,epsilon):
        cosine=np.cos(coordinates@qtheta)
        group=(vectors[None]*np.exp(1j*epsilon*cosine[:,None]*values[None])[:,None,:])@vectors.conj().T
        seam=np.where(coordinates[:,axis]==size-1,CENTRES[twist[axis]],1)
        return group*seam[:,None,None]
    rows=[]
    for _,model,role in models(cfg):
        onsite,paths=scalar_paths(model)
        def actual(epsilon):
            output=float(onsite)*(frame[0]@wave)
            for displacement,scalars in paths:
                axis=int(np.flatnonzero(displacement)[0]);length=int(displacement[axis])
                direction=np.eye(3,dtype=int)[axis]
                F=np.einsum('m,mij->ij',scalars,frame)
                forward=np.broadcast_to(np.eye(4),(len(sites),4,4)).copy().astype(complex)
                reverse=forward.copy()
                for j in range(length):
                    forward=forward@link((sites+j*direction)%size,axis,epsilon)
                    reverse=reverse@link((sites-displacement+j*direction)%size,axis,epsilon)
                reverse=reverse.conj().transpose(0,2,1)
                target=(sites+displacement)%size;source=(sites-displacement)%size
                target_wave=np.exp(1j*target@ptheta)[:,None,None]*np.eye(4)[None]
                source_wave=np.exp(1j*source@ptheta)[:,None,None]*np.eye(4)[None]
                output+=(F@forward+forward@F)@target_wave/2
                output+=(F.conj().T@reverse+reverse@F.conj().T)@source_wave/2
            return output
        expected_j=np.zeros_like(wave);expected_jj=np.zeros_like(wave)
        wrong_j=np.zeros_like(wave);wrong_jj=np.zeros_like(wave)
        for displacement,scalars in paths:
            axis=int(np.flatnonzero(displacement)[0]);length=int(displacement[axis])
            direction=np.eye(3,dtype=int)[axis];F=np.einsum('m,mij->ij',scalars,frame)
            forward_cos=np.asarray([np.cos((sites+j*direction)@qtheta) for j in range(length)])
            reverse_cos=np.asarray([np.cos((sites-displacement+j*direction)@qtheta) for j in range(length)])
            sf,sr=np.sum(forward_cos,axis=0),np.sum(reverse_cos,axis=0)
            phase=np.exp(1j*ptheta@displacement)
            fj=1j*(F@generator+generator@F)/2;rj=-1j*(F.conj().T@generator+generator@F.conj().T)/2
            fjj=-(F@generator@generator+generator@generator@F)/2
            rjj=-(F.conj().T@generator@generator+generator@generator@F.conj().T)/2
            expected_j+=(sf[:,None,None]*phase*fj+sr[:,None,None]*phase.conjugate()*rj)@wave
            expected_jj+=(sf[:,None,None]**2*phase*fjj+sr[:,None,None]**2*phase.conjugate()*rjj)@wave
            wrong_j+=(sf[:,None,None]*phase*1j*(F@generator)-sr[:,None,None]*phase.conjugate()*1j*(generator@F.conj().T))@wave
            wrong_jj+=(np.sum(forward_cos**2,axis=0)[:,None,None]*phase*fjj+np.sum(reverse_cos**2,axis=0)[:,None,None]*phase.conjugate()*rjj)@wave
        zero,plus,minus,hplus,hminus=[actual(epsilon) for epsilon in (0.,step,-step,step/2,-step/2)]
        observed_j=(4*(hplus-hminus)/step-(plus-minus)/(2*step))/3
        observed_jj=(4*(hplus+hminus-2*zero)/(step/2)**2-(plus+minus-2*zero)/step**2)/3
        j_error=float(np.linalg.norm(observed_j-expected_j)/(1+np.linalg.norm(expected_j)))
        jj_error=float(np.linalg.norm(observed_jj-expected_jj)/(1+np.linalg.norm(expected_jj)))
        if j_error>1e-7 or jj_error>1e-5:raise ArithmeticError('Literal finiteIR SU4 seam current/contact failed')
        rows.append(dict(twist=list(twist),mode=list(mode),role=role,current_relative_error=j_error,contact_relative_error=jj_error,
            wrong_ordered_current_relative_error=float(np.linalg.norm(observed_j-wrong_j)/(1+np.linalg.norm(observed_j))),
            wrong_omitted_two_link_cross_contact_relative_error=float(np.linalg.norm(observed_jj-wrong_jj)/(1+np.linalg.norm(observed_jj)))))
    return rows


def run(mean_path: str) -> dict[str,Any]:
    started=time.monotonic();raw=Path(mean_path).read_bytes();mean=json.loads(raw)
    size=mean['finite_size'];cfg=PhiConfig(Lambda=1.,**mean['microscopic_config'])
    enclosure=mean['centre_enclosure']
    g0=(bound(enclosure['g0_lower'])+bound(enclosure['g0_upper']))/2
    gs=(bound(enclosure['gs_lower'])+bound(enclosure['gs_upper']))/2
    scales=np.sqrt(np.asarray([float(g0),float(gs),float(gs),float(gs)]))
    frame=scales[:,None,None]*SELECTED
    modes: list[dict[str,Any]]=[dict(index=list(mode),q_physical=(2*np.pi*np.asarray(mode)/(size*cfg.a)).tolist(),
        q_norm_physical=float(np.linalg.norm(2*np.pi*np.asarray(mode)/(size*cfg.a)))) for mode in MODE_INDICES]
    if any(not .01<=row['q_norm_physical']<=.12 for row in modes):raise ArithmeticError('Named compatible modes do not cover the requested physicalIR range')
    rows=[]
    for twist in itertools.product(range(4),repeat=3):
        for mode in MODE_INDICES:rows.extend(sparse_curve(cfg,size,frame,twist,mode))
    worst_cross=max(row['wrong_omitted_two_link_cross_contact_relative_error'] for row in rows)
    if worst_cross<1e-4:raise ArithmeticError('Lowq two-link contact adversary did not discriminate')
    images=rounding_prices(cfg,mean,scales)
    descriptor=dict(mean_receipt_sha256=hashlib.sha256(raw).hexdigest(),mean_input_sha256=mean['input_sha256'],
        finite_size=size,Lambda_exponent=mean['Lambda_exponent'],microscopic_config=mean['microscopic_config'],modes=modes,
        physical_centre='Each actual implicit fullten G_t from the actual finite-grid certificate; common floating midpoint is ONLY a separately priced operator image')
    return dict(branch='NEW literal finiteIR seam/current/contact diagnostic consumer, NOT photon or scattering admission',
        input_descriptor=descriptor,input_sha256=hashlib.sha256(json.dumps(descriptor,sort_keys=True).encode()).hexdigest(),
        all64_current_contact_rows=rows,modes=modes,operator_image_rounding_prices=images,
        max_current_error=max(row['current_relative_error'] for row in rows),max_contact_error=max(row['contact_relative_error'] for row in rows),
        wrong_ordered_current_max=max(row['wrong_ordered_current_relative_error'] for row in rows),wrong_two_link_cross_contact_max=worst_cross,
        physical_mean_representation=mean['exact_centre_definition'],sector_policy=mean['sector_policy'],
        samples='216 source sites including seams/edges/corners, nearest-neighbor wraps and generic interior positions, 4orbital columns. These are sampled native controls, not allV site enumeration. Unnormalized plane image has common amplitude1; physical normalized mode is image/sqrtV.',
        limitations='Current/contact operator images and compatible finite modes only. No implicit-centre midpoint substitution, quantumphotonvacuum, RPA collective-state transfer, commoncone or Compton promotion. FullCAR image prices may be large.',
        source_at_import=IMPORT_HASHES,own_source_at_import=IMPORT_BYTES[Path(__file__).name].decode(),
        all_sources_unchanged_at_completion=all((ROOT/name).read_bytes()==data for name,data in IMPORT_BYTES.items()),
        runtime=dict(python=sys.version,executable=sys.executable,numpy=np.__version__,mpmath=mp.__version__,platform=platform.platform()),elapsed_seconds=time.monotonic()-started)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--mean',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();report=run(args.mean)
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({key:report[key] for key in ('max_current_error','max_contact_error','wrong_ordered_current_max','wrong_two_link_cross_contact_max','source_at_import','input_sha256','elapsed_seconds')},indent=2))
