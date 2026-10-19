#!/usr/bin/env python3
"""Actual NEW real-cover C current/contact, shell crossing and Dminus probes.

No scalar-photon Gibbs trace, source gate, fake harmonic ground or imported
Root/Lambda receipt. These deliberately smaller COMPLETE-column regulators
exercise the SAME ordered chart/outside primitive used by the packet CLI.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any

if __name__=='__main__':
    for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
        os.environ[name]='1'

import numpy as np
from scipy.linalg import block_diag,eigh,expm
import physical_c_real_cover_kernel as kernel

SOURCE_BYTES=Path(__file__).read_bytes()
KERNEL_BYTES=Path(kernel.__file__).read_bytes()


def covariance_generator(masks: list[int],lam: np.ndarray,number: int) -> np.ndarray:
    # Physical rho includes the neutral background and the actual Ref3 hole.
    rho=np.array([[(mask & 15).bit_count()-2,(mask >> 4).bit_count()-(2 if number==4 else 3)] for mask in masks])
    return np.diag(rho@lam)


def current_and_temporal_Ward() -> dict[str,Any]:
    p=kernel.NativeParameters(photon_grade=0,normal_order=3,colour_order=1)
    field=kernel.field_regulator(p)
    lam_in=np.array([.4,-.3]);lam_out=np.array([-.2,.5])
    d=(kernel.real_graph()['D'].T@lam_in).reshape(2,3)
    rows=[]
    for number in (4,5):
        actual=kernel.native_factor(p,number,0,field,(d,d))
        f,fx,fy,fxy=(actual['F'][key] for key in ('','x','y','xy'))
        k=f@f.conj().T
        j=fx@f.conj().T+f@fx.conj().T
        contact=fxy@f.conj().T+fx@fy.conj().T+fy@fx.conj().T+f@fxy.conj().T
        g_in=np.kron(np.eye(field['dimension']),covariance_generator(actual['masks'],lam_in,number))
        g_out=np.kron(np.eye(field['dimension']),covariance_generator(actual['masks'],lam_out,number))
        expected_j=1j*(g_in@k-k@g_in)
        expected_contact=-(g_in@g_in@k-2*g_in@k@g_in+k@g_in@g_in)
        first=float(np.linalg.norm(j-expected_j,'fro')/np.linalg.norm(k,'fro'))
        second=float(np.linalg.norm(contact-expected_contact,'fro')/np.linalg.norm(k,'fro'))
        if max(first,second)>2e-7:raise ArithmeticError('Actual spatial Wilson/SF/Duhamel current Ward failed')
        if number==4:neutral_lambda=float(eigh(k,eigvals_only=True)[-1])
        h,hx,hxx=kernel.physical_H_jet(k,j,contact,neutral_lambda,p.spacing)
        hfirst=float(np.linalg.norm(hx-1j*(g_in@h-h@g_in),'fro')/np.linalg.norm(h,'fro'))
        hsecond=float(np.linalg.norm(hxx+(g_in@g_in@h-2*g_in@h@g_in+h@g_in@g_in),'fro')/np.linalg.norm(h,'fro'))
        if max(hfirst,hsecond)>2e-7:raise ArithmeticError('ACTUAL reflected K logarithm Hamiltonian current/contact Ward failed')
        temporal=g_out-g_in
        # NEW explicit temporal Wilson parallel transport: K(A0,A)=e^(i rho.A0)K(A).
        full_first,full_second=kernel.temporal_pe_jet(k,j,contact,temporal)
        endpoint_first=1j*(g_out@k-k@g_in)
        endpoint_second=-(g_out@g_out@k-2*g_out@k@g_in+k@g_in@g_in)
        temporal_error=float(np.linalg.norm(full_second-endpoint_second,'fro')/np.linalg.norm(k,'fro'))
        wrong_temporal=2j*temporal@j+contact
        wrong_temporal_error=float(np.linalg.norm(wrong_temporal-endpoint_second,'fro')/np.linalg.norm(k,'fro'))
        wrong_factor_contact=fx@fy.conj().T+fy@fx.conj().T
        wrong_factor_error=float(np.linalg.norm(wrong_factor_contact-expected_contact,'fro')/np.linalg.norm(k,'fro'))
        if temporal_error>2e-7 or min(wrong_temporal_error,wrong_factor_error)<1e-6:
            raise ArithmeticError('Actual contact oracle did not distinguish the adverse deletion')
        rows.append(dict(number=number,carrier_dimension=len(actual['masks']),complete_columns=actual['complete_column_count'],
            spatial_first_Ward=first,spatial_mixed_Ward=second,
            temporal_endpoint_first_Ward=float(np.linalg.norm(full_first-endpoint_first,'fro')/np.linalg.norm(k,'fro')),
            temporal_endpoint_contact_Ward=temporal_error,
            adverse_delete_true_temporal_seagull=wrong_temporal_error,
            adverse_delete_actual_factor_contact=wrong_factor_error,
            actual_outside_factor_norm=actual['outside_factor_norm']))
        rows[-1].update(actual_Klog_H_first_Ward=hfirst,actual_Klog_H_contact_Ward=hsecond,
            actual_Klog_H_current_Frobenius_norm=float(np.linalg.norm(hx,'fro')),
            actual_Klog_H_contact_Frobenius_norm=float(np.linalg.norm(hxx,'fro')),
            fixed_neutral_lambda_N=neutral_lambda,
            actual_Klog_H_min=float(eigh(h,eigvals_only=True)[0]))
    return dict(rows=rows,parameters=asdict(p),
        primitive='NEW temporal kernel K(A0,A)=exp(i sum_x rho_x A0_x)K(A), baseK=FFdag. This EXACT external Pe parallel transport is not a claim of equality with56 temporal fieldword. rho includes active charge and protected Refhole/background. First temporal density and second density² seagull are retained.',
        derivative='Actual chart/outside F jets produce Kx/Kxy. Physical H/current/contact use Frechet log of THIS complete reflected K at fixed neutral lambda_N. Temporal Pe currents are distinct kernel generating insertions. No primitive h56 Hamiltonian current is promoted after fieldword change.',
        scope='Complete210/168-row and420/336-column carriers at this smaller real5-coordinate photon-vacuum regulator; not full-family QED scattering.')


def overlap_shell_source_crossing() -> dict[str,Any]:
    p=kernel.NativeParameters(photon_grade=0,normal_order=5,colour_order=1)
    field=kernel.field_regulator(p)
    neutral=kernel.native_factor(p,4,0,field)
    charged=kernel.native_factor(p,5,0,field)
    yn,yw=kernel.hermite_rule(p.normal_order)
    y=math.sqrt(2)*p.normal_sigma*yn
    ybasis=np.column_stack([yw*np.polynomial.hermite.hermval(yn,[0.]*i+[1.])/math.sqrt(2**i*math.factorial(i)) for i in range(p.normal_order)])
    core=ybasis.T@np.diag(abs(y)<=p.chart_radius/2)@ybasis
    outside=ybasis.T@np.diag(abs(y)>=p.chart_radius)@ybasis
    k=(neutral['vectors']*neutral['values'])@neutral['vectors'].conj().T
    law,_,_=kernel.native_law(p)
    mean=law.potential(np.zeros((2,3)),np.zeros((2,4,4,4)),np.broadcast_to(np.eye(4),(2,3,4,4)).copy())
    generator,_=kernel.alpha.second_quantized(mean['h'],4)
    _,v=eigh(generator)
    phi=core@np.eye(p.normal_order)[:,0]
    trial=np.kron(phi,v[:,-1]);trial/=np.linalg.norm(trial)
    one=k@trial;two=k@one
    out=np.kron(outside,np.eye(70))
    one_price=float(np.linalg.norm(out@one)**2/np.linalg.norm(one)**2)
    two_price=float(np.linalg.norm(out@two)**2/np.linalg.norm(two)**2)
    if two_price<=1e-24 or two_price<=100*one_price:
        raise ArithmeticError('Actual overlap-shell kernel did not communicate core and outside')
    coefficients=kernel.source_coefficients(field,neutral['source_packet_blocks'])
    creation=kernel.creation_matrices(neutral['masks'],charged['masks'])
    source=kernel.apply_source(coefficients,creation,two/np.linalg.norm(two))
    common=neutral['values'][-1]
    spectral=source@charged['vectors'].conj()
    weights=np.sum(abs(spectral)**2,axis=0)*charged['values']/common
    energy=-np.log(charged['values']/common)
    half=float(sum(weights));raw=float(np.linalg.norm(source)**2)
    return dict(parameters=asdict(p),actual_core_one_step_outside_probability=one_price,
        actual_core_two_step_outside_probability=two_price,
        actual_Ref4_to3_source_raw_Gram=raw,actual_whole_charged_half_Gram=half,
        actual_source_first_output_mean_TRANSFER=float(weights@energy/half),
        literal_TRANSFER_Pe4_Gram=float(sum(weights[(energy>=0)&(energy<=4)])),
        actual_outside_H_min=neutral['outside_H_min'],
        actual_nonzero_shell_phi_coordinates=y.tolist(),
        all_positive_neutral_count=len(neutral['values']),all_positive_charged_count=len(charged['values']),
        interpretation='ActualK=AchAchdag+BoutBoutdag has zero one-step core/outside matrix element at this support regulator, but K² communicates through the shared overlap shell. Source remains unfiltered across core/shell/outside; its complete charged halfstep is actually executed.')


def physical_source_Dminus() -> dict[str,Any]:
    alpha=kernel.alpha
    _,m4=alpha.second_quantized(np.eye(8),4);_,m5=alpha.second_quantized(np.eye(8),5)
    creation=kernel.creation_matrices(m4,m5)
    t0,t1=alpha.matrix_unit_basis()[3]/math.sqrt(8),alpha.matrix_unit_basis()[7]/math.sqrt(8)
    v0,v1=expm(.23j*t0),expm(-.31j*t1)
    dv0,dv1=1.3j*t0@v0,-.9j*t1@v1
    w=v0@v1.conj().T;dw=dv0@v1.conj().T+v0@dv1.conj().T
    packet=np.vstack((w,np.eye(4)))/2
    derivative=np.vstack((dw,np.zeros((4,4))))/2
    source=np.einsum('ba,bgf->agf',packet,creation)
    ds=np.einsum('ba,bgf->agf',derivative,creation)
    hermitian=block_diag(1.3*t0,-.9*t1)
    h4,_=alpha.second_quantized(hermitian,4);h5,_=alpha.second_quantized(hermitian,5)
    a4,a5=1j*h4,1j*h5
    aref=.9j*t1.conj()
    covariant=ds-np.einsum('gh,ahf->agf',a5,source)+np.einsum('agf,fh->agh',source,a4)-np.einsum('ab,bgf->agf',aref,source)
    error=float(np.linalg.norm(covariant))
    wrong_without_ref=float(np.linalg.norm(covariant+np.einsum('ab,bgf->agf',aref,source)))
    if error>1e-10 or wrong_without_ref<1e-3:raise ArithmeticError('Actual physical Gamma(V)/Ref Dminus cancellation failed')
    return dict(actual_physical_source_Dminus_norm=error,
        actual_wrong_bare_frame_derivative_norm=float(np.linalg.norm(ds)),
        adverse_omit_Refhole_connection_norm=wrong_without_ref,
        primitive='PhysicalW=V0 Uref V1dag, bodyGamma(V) and actual Ref3 anti-fundamental holeV1* share the SAME connection. DminusSource=ds-A5s+sA4-ARefhole.s=0 for this independently assembled Wilson source. No Gamma(V) derivative price is reintroduced.',
        global_graph_core='Globally the source is bounded Wilsonline CAR creation, independent of CartesianPhi; smooth SU4 path and literalinteger realEM character map compactPhi/SU4 x Schwartz realEM packet into the actual Friedrichs graph core. The finite probe verifies the physical-frame/ref cancellation, not all huge-family derivative constants.')


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    args=parser.parse_args();started=time.time()
    result=dict(actual_current_and_temporal_contact=current_and_temporal_Ward(),
        actual_overlap_shell_source_crossing=overlap_shell_source_crossing(),
        actual_physical_source_Dminus=physical_source_Dminus())
    inputs=json.dumps(dict(kernel_sha256=hashlib.sha256(KERNEL_BYTES).hexdigest(),
        source='unchanged frozen56 primitive four-spin Wilson path, Ref4->3',no_source_gate=True),sort_keys=True,separators=(',',':'))
    result['execution']=dict(source_at_IMPORT=hashlib.sha256(SOURCE_BYTES).hexdigest(),source_bytes_at_IMPORT=SOURCE_BYTES.decode(),
        kernel_source_at_IMPORT=hashlib.sha256(KERNEL_BYTES).hexdigest(),kernel_source_bytes_at_IMPORT=KERNEL_BYTES.decode(),
        captured_ledger_sha256=kernel.LEDGER_HASH,captured_ledger_bytes=kernel.LEDGER_BYTES.decode(),
        frozen_Alpha56_source_at_IMPORT=kernel.alpha.SOURCE_HASHES,
        frozen_Alpha56_source_bytes_at_IMPORT={key:value.decode() for key,value in kernel.alpha.SOURCE_BYTES.items()},
        input_json=inputs,input_sha256=hashlib.sha256(inputs.encode()).hexdigest(),runtime_seconds=time.time()-started,
        runtime=dict(executable=sys.executable,python=sys.version,numpy=np.__version__,platform=platform.platform()),
        own_startup_unchanged=Path(__file__).read_bytes()==SOURCE_BYTES,
        kernel_startup_unchanged=Path(kernel.__file__).read_bytes()==KERNEL_BYTES)
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(output=args.output,source=result['execution']['source_at_IMPORT'],kernel=result['execution']['kernel_source_at_IMPORT'],
        Ward=result['actual_current_and_temporal_contact']['rows'],cross=result['actual_overlap_shell_source_crossing']['actual_core_two_step_outside_probability'],
        Dminus=result['actual_physical_source_Dminus']['actual_physical_source_Dminus_norm'],runtime_seconds=result['execution']['runtime_seconds'])))


if __name__=='__main__':
    main()
