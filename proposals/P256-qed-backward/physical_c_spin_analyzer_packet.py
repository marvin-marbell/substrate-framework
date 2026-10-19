#!/usr/bin/env python3
"""NEW spin-inclusive normalized FRAME EFFECT on the actual C packet kernel.

The source/incoming packet and coherent-singlet receiver remain unchanged.
Sixteen matrix-unit analyzers use the actual Ref3 anti-fundamental physical
frame. The complete charged propagation is unitarily transported and really
mixes Ref channels in that frame; no static numeric-label assumption is made.
This finite detector probability is NOT a Compton cross section or mass proof.
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
from typing import Any,cast
if __name__=='__main__':
    for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
        os.environ[name]='1'
import numpy as np
from scipy.linalg import eigh,expm,qr
import physical_c_real_cover_kernel as kernel
SOURCE_BYTES=Path(__file__).read_bytes();KERNEL_BYTES=Path(kernel.__file__).read_bytes()


def algebraic_spin_frame(phi: np.ndarray) -> np.ndarray:
    """Gauge-equivariant measurable frame, modulo the connected diagonalZ4.

    H=Phi0+.137i[Phi1,Phi2], B=Phi3+.219Phi1. These have simple H spectrum
    and connected B eigenvector graph at the actual reference and generically.
    Their exceptional algebraic sets have Lebesgue measure zero. The detector
    is defined a.e.; we DO NOT claim a globally smooth normal-Phi frame/core.
    """
    h=phi[0]+.137j*(phi[1]@phi[2]-phi[2]@phi[1])
    eigenvalues,frame=eigh(h)
    if np.min(np.diff(eigenvalues))<1e-11:
        raise ArithmeticError('Native detector reaches exceptional degenerate physical spin chart')
    b=frame.conj().T@(phi[3]+.219*phi[1])@frame
    phases=np.ones(4,complex);visited={0}
    while len(visited)<4:
        candidates=[(abs(b[i,j]),i,j) for i in visited for j in range(4) if j not in visited]
        magnitude,i,j=max(candidates)
        if magnitude<1e-11:raise ArithmeticError('Native physical spin frame graph is disconnected')
        phases[j]=phases[i]*np.exp(-1j*np.angle(b[i,j]));visited.add(j)
    frame=frame*phases
    frame*=np.exp(-1j*np.angle(np.linalg.det(frame))/4)
    return frame


def physical_Ref_transport(field: dict[str,Any],reference: np.ndarray,direction: np.ndarray,p: kernel.NativeParameters) -> tuple[np.ndarray,dict[str,Any]]:
    """Exact finite DVR unitary for ALL4Ref3 channels; photon variables untouched."""
    nodes,weights=kernel.hermite_rule(p.normal_order)
    yn=math.sqrt(2)*p.normal_sigma*nodes
    ynbasis=np.column_stack([weights*np.polynomial.hermite.hermval(nodes,[0.]*i+[1.])/math.sqrt(2**i*math.factorial(i)) for i in range(p.normal_order)])
    origin=algebraic_spin_frame(reference)
    frames=np.array([algebraic_spin_frame(reference+y*direction)@origin.conj().T for y in yn])
    # Ref3 hole transforms by V*, not V. Normal Hermite/DVR is square, so
    # this is an EXACT unitary, unlike projecting a variable frame in phgrade1.
    transport=np.einsum('ni,nab,nj->aibj',ynbasis,frames.conj(),ynbasis,optimize=True)
    flat=transport.reshape(4*p.normal_order,4*p.normal_order)
    defect=float(np.linalg.norm(flat.conj().T@flat-np.eye(len(flat))))
    if defect>1e-10:raise ArithmeticError('Actual fullRef3 physical-frame transport lost unitarity')
    t=kernel.alpha.matrix_unit_basis()[7]/math.sqrt(8)
    epsilon=2e-5
    up,um=expm(1j*epsilon*t),expm(-1j*epsilon*t)
    base=algebraic_spin_frame(reference)@origin.conj().T
    transformed=algebraic_spin_frame(up@reference@up.conj().T)@origin.conj().T
    # An overall centre/root phase cancels in every matrix-unit analyzer.
    e=np.zeros((4,4),complex);e[0,2]=1
    a=base.conj()@e@base.T
    actual=transformed.conj()@e@transformed.T
    expected=up.conj()@a@up.T
    equivariance=float(np.linalg.norm(actual-expected))
    am=(algebraic_spin_frame(um@reference@um.conj().T)@origin.conj().T).conj()
    ap=transformed.conj()
    derivative=(ap@e@ap.conj().T-am@e@am.conj().T)/(2*epsilon)
    connection=-1j*t.conj()
    covariant=derivative-(connection@a-a@connection)
    dminus=float(np.linalg.norm(covariant))
    if equivariance>1e-9 or dminus>1e-7:raise ArithmeticError('Actual Ref3 analyzer connection transport failed')
    return transport,dict(actual_fullRef3_transport_unitarity_defect=defect,
        actual_analyzer_SU4_equivariance_defect=equivariance,
        actual_Ref3_analyzer_Dminus_norm=dminus,
        adverse_static_numeric_spin_label_derivative=float(np.linalg.norm(connection@a-a@connection)),
        physical_hole_representation='Vstar; connection=-i tstar. Operator analyzer transforms Vstar M Vtranspose.',
        frame_definition='Gauge-equivariant a.e. Cart frame from actualPhi; calibrated identity at reference. Physical field dependence is a NEW detector input, not a retag of old coherent receiver. Exceptional algebraic nullsets do not change the L2 operator; no smooth normal-Phi graph claim.',
        retained_normal_DVR_coordinates=yn.tolist())


def transport_Ref(state: np.ndarray,transport: np.ndarray,p: kernel.NativeParameters,adjoint: bool=False) -> np.ndarray:
    tensor=state.reshape(4,6,p.normal_order,p.colour_order,56)
    operator=transport
    if adjoint:operator=operator.conj().transpose(2,3,0,1)
    return np.einsum('aibj,bpjkg->apikg',operator,tensor,optimize=True).reshape(4,-1)


def receiver_seeds(neutral: list[dict[str,Any]],targets: list[np.ndarray],field: dict[str,Any],transport: np.ndarray,p: kernel.NativeParameters) -> tuple[list[np.ndarray],np.ndarray,float,np.ndarray]:
    outgoing=np.zeros((6,6));outgoing[2,0]=1 # actual projected a_outdag
    outgoing=np.kron(np.kron(outgoing,np.eye(p.normal_order)),np.eye(p.colour_order))
    blocks=[]
    for row,target in zip(neutral,targets,strict=True):
        _,m5=kernel.alpha.second_quantized(np.eye(8),5)
        creation=kernel.creation_matrices(row['masks'],m5)
        source=kernel.apply_source(kernel.source_coefficients(field,row['source_packet_blocks']),creation,target)
        physical=np.einsum('ij,ajf->aif',outgoing,source.reshape(4,field['dimension'],56),optimize=True).reshape(4,-1)
        framed=transport_Ref(physical,transport,p,adjoint=True)
        allspin=np.zeros((16,4,physical.shape[1]),complex)
        for a in range(4):
            for b in range(4):allspin[4*a+b,a]=framed[b]
        blocks.append(allspin.reshape(16,-1))
    r=np.concatenate(blocks,axis=1)
    gram=r.conj()@r.T
    weights=eigh((gram+gram.conj().T)/2,eigvals_only=True)
    mu=float(weights[-1])
    if mu<=0:raise ArithmeticError('Actual inclusive frame effect has zero normalization')
    return blocks,r,mu,weights


def thin_effect_difference(r1: np.ndarray,r0: np.ndarray) -> float:
    """Exact nonzero spectrum of R1R1dag-R0R0dag, no ambient square allocation."""
    _,small=cast(tuple[tuple[np.ndarray,np.ndarray],np.ndarray],
                 qr(np.column_stack((r1.T,r0.T)),mode='raw',pivoting=False))
    signed=np.r_[np.ones(16),-np.ones(16)]
    matrix=(small*signed)@small.conj().T
    return float(np.max(abs(eigh((matrix+matrix.conj().T)/2,eigvals_only=True))))


def packet_detector(p: kernel.NativeParameters) -> dict[str,Any]:
    field=kernel.field_regulator(p);law,reference_phi,direction=kernel.native_law(p)
    transport,transport_receipt=physical_Ref_transport(field,reference_phi,direction,p)
    mean=law.potential(np.zeros((2,3)),np.zeros((2,4,4,4)),np.broadcast_to(np.eye(4),(2,3,4,4)).copy())
    generator,_=kernel.alpha.second_quantized(mean['h'],4);_,v=eigh(generator);sea=v[:,-1]
    neutral=[];charged=[];reference=[];adverse=[]
    # The deliberate adverse TRANSFER deletes transverse Wilson coupling ONLY;
    # keep actual real-field heat/Coulomb, incoming packet and ORIGINAL source.
    no_transverse=dict(field,points=[(np.zeros_like(q),y,xi) for q,y,xi in field['points']])
    for z in range(4):
        neutral.append(kernel.native_factor(p,4,z,field))
        charged.append(kernel.native_factor(p,5,z,field))
        reference.append(kernel.native_factor(p,4,z,field,reference_fields=True))
        adverse.append((kernel.native_factor(p,4,z,no_transverse),kernel.native_factor(p,5,z,no_transverse)))
    lambda_N=max(r['values'][-1] for r in neutral);lambda0=max(r['values'][-1] for r in reference)
    lambda_adverse=max(pair[0]['values'][-1] for pair in adverse)
    targets=[];targets0=[];targets_adverse=[]
    for row,row0,pair in zip(neutral,reference,adverse,strict=True):
        xi=np.zeros(row['dimension'],complex);xi[:70]=sea/2
        for result,r,scale in [(targets,row,lambda_N),(targets0,row0,lambda0),(targets_adverse,pair[0],lambda_adverse)]:
            result.append(r['vectors']@((r['values']/scale)**p.prefix_steps*(r['vectors'].conj().T@xi)))
    for states in (targets,targets0,targets_adverse):
        norm=math.sqrt(sum(float(np.linalg.norm(s)**2) for s in states))
        for z in range(4):states[z]/=norm
    receiver_blocks,r,mu,weights=receiver_seeds(neutral,targets,field,transport,p)
    _,r0,mu0,weights0=receiver_seeds(neutral,targets0,field,transport,p)
    deltaA=thin_effect_difference(r,r0)
    incoming=np.zeros((6,6));incoming[1,0]=1
    incoming=np.kron(np.kron(incoming,np.eye(p.normal_order)),np.eye(p.colour_order))
    outcomes=[]
    for label,states,sectors,scale in [('actual',targets,charged,lambda_N),('adverse_delete_transfer_transverse_Wilson_only',targets_adverse,[pair[1] for pair in adverse],lambda_adverse)]:
        photon=[(incoming@s.reshape(field['dimension'],70)).reshape(-1) for s in states]
        gph=sum(float(np.linalg.norm(s)**2) for s in photon)
        spectral=[];half=raw=pe4=0.
        for n,c,s in zip(neutral,sectors,photon,strict=True):
            creation=kernel.creation_matrices(n['masks'],c['masks'])
            source=kernel.apply_source(kernel.source_coefficients(field,n['source_packet_blocks']),creation,s/math.sqrt(gph))
            energy=-np.log(c['values']/scale)
            weighted=(source@c['vectors'].conj())*np.sqrt(c['values']/scale)
            w=np.sum(abs(weighted)**2,axis=0)
            raw+=float(np.linalg.norm(source)**2);half+=float(sum(w));pe4+=float(sum(w[(energy>=0)&(energy<=4)]))
            spectral.append((weighted,energy))
        # Each law uses its SAME target for the detector; source packets remain
        # the original actual ones even for the adverse transfer deletion.
        blocks,_,detector_mu,detector_weights=receiver_seeds(neutral,states,field,transport,p)
        events=[]
        for t in (0.,.25,.5,1.):
            amplitudes=np.zeros((4,4),complex);naive_amplitudes=np.zeros((4,4),complex)
            mixing_price=0.
            for c,block,(weighted,energy) in zip(sectors,blocks,spectral,strict=True):
                physical=(weighted*np.exp(-1j*t*energy/p.spacing))@c['vectors'].T
                framed=transport_Ref(physical,transport,p,adjoint=True)
                amplitudes+=(block.conj()@framed.reshape(-1)).reshape(4,4)
                # Deliberately wrong static Ref labels: evolve the framed source
                # with IRef tensor K instead of Tdag(IRef tensorK)T.
                framed_initial=transport_Ref(weighted@c['vectors'].T,transport,p,adjoint=True)
                naive_spectral=framed_initial@c['vectors'].conj()
                naive=(naive_spectral*np.exp(-1j*t*energy/p.spacing))@c['vectors'].T
                naive_amplitudes+=(block.conj()@naive.reshape(-1)).reshape(4,4)
                mixing_price+=float(np.linalg.norm(framed-naive)**2)
            inclusive=float(np.sum(abs(amplitudes)**2)/(detector_mu*half))
            # Coherent norm must include ALL centres, not just the lastblock.
            coherent_gram=sum(float(np.linalg.norm(sum(b[4*a+a] for a in range(4)))**2) for b in blocks)
            coherent=float(abs(np.trace(amplitudes))**2/(coherent_gram*half))
            offdiag=amplitudes-np.diag(np.diag(amplitudes))
            if inclusive>1+1e-9:raise ArithmeticError('Actual normalized inclusive frame effect exceeds1')
            events.append(dict(time_PHYSICAL=t,actual_inclusive_frame_effect_probability=inclusive,
                original_coherent_singlet_effect_probability=coherent,
                actual_spin_amplitude_Frobenius_squared=float(np.sum(abs(amplitudes)**2)),
                actual_spin_trace_absolute_squared=float(abs(np.trace(amplitudes))**2),
                actual_nonzero_offdiagonal_spin_amplitude_Frobenius_norm=float(np.linalg.norm(offdiag)),
                actual_amplitude_real=np.real(amplitudes).tolist(),actual_amplitude_imag=np.imag(amplitudes).tolist(),
                adverse_static_Ref_label_amplitude_error=float(np.linalg.norm(naive_amplitudes-amplitudes)),
                actual_Ref_channel_mixing_norm_squared=mixing_price))
        outcomes.append(dict(law=label,ONE_neutral_lambda_N=float(scale),actual_incoming_projected_photon_Gram=gph,
            actual_source_raw_Gram=raw,actual_whole_charged_half_Gram=half,literal_Pe4_TRANSFER_Gram=pe4,
            actual_receiver_mu_max=detector_mu,actual_receiver_Gram_eigenweights=detector_weights.tolist(),
            actual_receiver_rank_at_relative1e_12=int(np.sum(detector_weights>detector_mu*1e-12)),events=events))
    actual=outcomes[0]['events'][-1];deleted=outcomes[1]['events'][-1]
    if actual['actual_inclusive_frame_effect_probability']<=1e-14 or actual['actual_nonzero_offdiagonal_spin_amplitude_Frobenius_norm']<=1e-9:
        raise ArithmeticError('Actual spin-analyzer construction has no discriminating cross-polarized amplitude')
    if deleted['actual_inclusive_frame_effect_probability']>1e-14:
        raise ArithmeticError('Deleting actual transverse transfer coupling did not remove cross-polarized transition')
    return dict(parameters=asdict(p),physical_Ref3_connection=transport_receipt,outcomes=outcomes,
        actual_raw_frame_effect_difference_norm=deltaA,actual_reference_receiver_mu_max=mu0,
        actual_raw_frame_difference_over_mu0=deltaA/mu0,
        normalized_frame_effect_difference_upper=2*deltaA/mu0,
        reference_Gram_eigenweights=weights0.tolist(),actual_Gram_eigenweights=weights.tolist(),
        effect='E=sum_ab|r_ab><r_ab|/mu_max <=I. Not an orthogonal projector; no Gram minimum/rank-full claim. Rawframe stability ||E1-E0||<=2||R1R1dag-R0R0dag||/mu0 for actual positive normalizers.',
        physical_frame='Ref3 matrix-unit analyzers transported by actualPhi Vstar and source electron columns by the SAME reference frame. Complete charged evolution in that coordinate frame is Tdag(IRef tensor KC)T, not static labels. No source/energy gate or quarter normalization added.',
        scope='Actual finite real5-mode interacting C packet detector, bothchart/outside complete columns, all4native centrecycles, ALL neutral70/charged56 carriers and fourRef3 channels. Naux0; one normal/colour direction; no full macro probability, crosssection, physicalmass or induced-Maxwell promotion.')


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    args=parser.parse_args();start=time.time();p=kernel.NativeParameters(colour_order=1)
    result=packet_detector(p)
    inputs=json.dumps(dict(parameters=asdict(p),kernel_epoch=hashlib.sha256(KERNEL_BYTES).hexdigest(),NEW_inclusive_frame_effect=True,source_unchanged=True,all_Ref3_channels=True),sort_keys=True,separators=(',',':'))
    result['execution']=dict(source_at_IMPORT=hashlib.sha256(SOURCE_BYTES).hexdigest(),source_bytes_at_IMPORT=SOURCE_BYTES.decode(),
        kernel_source_at_IMPORT=hashlib.sha256(KERNEL_BYTES).hexdigest(),kernel_bytes_at_IMPORT=KERNEL_BYTES.decode(),
        input_json=inputs,input_sha256=hashlib.sha256(inputs.encode()).hexdigest(),runtime_seconds=time.time()-start,
        runtime=dict(executable=sys.executable,python=sys.version,numpy=np.__version__,platform=platform.platform()),
        own_startup_unchanged=Path(__file__).read_bytes()==SOURCE_BYTES,kernel_startup_unchanged=Path(kernel.__file__).read_bytes()==KERNEL_BYTES)
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(output=args.output,source=result['execution']['source_at_IMPORT'],input_sha256=result['execution']['input_sha256'],
        inclusive_probability=result['outcomes'][0]['events'][-1]['actual_inclusive_frame_effect_probability'],
        coherent_probability=result['outcomes'][0]['events'][-1]['original_coherent_singlet_effect_probability'],runtime_seconds=result['execution']['runtime_seconds'])))
if __name__=='__main__':main()
