#!/usr/bin/env python3
"""NEW bare-electric physical creation graph and inclusive packet experiment.

Captured kernel6a66 is untouched. Whole a† on the retained neutral state has
Gram 1+Nf; compress S a† JOINTLY, not S P a†. Receiver normalization uses
its UNCOMPRESSED ket Gram, with the same physical Ref3 transport. Gaussian
and SAME-Phi0 smooth-boundary controls share the original source. All results
are finite Naux0 evidence; no macro source/window or Maxwell claim follows.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict,replace
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
from scipy.linalg import eigh
import physical_c_bare_electric_kernel as bare
import physical_c_spin_analyzer_packet as spin
import physical_c_bare_electric_external_graph as graph
ROOT=Path(__file__).resolve().parent
SOURCE_BYTES=Path(__file__).read_bytes()
MODULES=[('bare',bare),('spin',spin),('external_graph',graph)]
DEPENDENCIES={name:Path(cast(str,module.__file__)).read_bytes() for name,module in MODULES}
if hashlib.sha256(DEPENDENCIES['bare']).hexdigest()!='6a66ca3a45b05e8df46cc1cec90aa775d97cd60341186639fdaf1dfd5fce9e64':
    raise ArithmeticError('This packet experiment binds exact bare-electric6a66')


def creation_basis(field: dict[str,Any],mode: int,p: bare.BareElectricParameters) -> np.ndarray:
    """Exact a_mode† phi0 P(x)=phi0(sqrt2*x P-dP/sqrt2)."""
    basis=field['basis'];width=math.sqrt(math.sinh(2*math.asinh(1.))) if mode<2 else 1.
    x=np.array([q[mode]*width for q,_,_ in field['points']])
    result=math.sqrt(2)*x[:,None]*basis
    stride=p.normal_order*p.colour_order
    result[:,(mode+1)*stride:(mode+2)*stride]-=basis[:,:stride]
    return result


def packets(field: dict[str,Any],centre: int,p: bare.BareElectricParameters) -> np.ndarray:
    law,_,_=bare.reference.native_law(bare.reference_parameters(p));e=bare.alpha.charge_contract(bare.alpha.Parameters())['bare_e']
    graph=bare.reference.real_graph();generator=bare.alpha.matrix_unit_basis()[7]/math.sqrt(8)
    from scipy.linalg import expm
    result=[]
    for q,_,xi in field['points']:
        phase=(e*graph['Q']@q).reshape(2,3)
        colour=np.broadcast_to(np.eye(4),(2,3,4,4)).astype(complex).copy()
        colour[0,0]=np.exp(1j*np.pi*centre/2)*expm(1j*xi*generator)
        result.append(law.source_packet_jet(phase,colour,(np.zeros((2,3)),np.zeros((2,3))))['psi'])
    return np.array(result)


def nodal_source(packet: np.ndarray,creation: np.ndarray,nodal: np.ndarray) -> np.ndarray:
    """CAR applied sparsely; all canonical masks retained, no ambient operator."""
    result=np.zeros((4,len(nodal),creation.shape[1]),complex)
    for bit in range(8):
        output,input_=np.nonzero(creation[bit])
        result[:,:,output]+=np.einsum('na,ng,g->ang',packet[:,bit,:],nodal[:,input_],creation[bit,output,input_],optimize=True)
    return result


def joint_source(field: dict[str,Any],created_basis: np.ndarray,packet: np.ndarray,creation: np.ndarray,state: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    nodal=created_basis@state.reshape(field['dimension'],70)
    source=nodal_source(packet,creation,nodal)
    projected=np.einsum('ni,ang->aig',field['basis'],source,optimize=True).reshape(4,-1)
    return projected,source


def frame_blocks(source: np.ndarray,transport: np.ndarray,p: bare.BareElectricParameters) -> np.ndarray:
    framed=spin.transport_Ref(source,transport,bare.reference_parameters(p),adjoint=True)
    result=np.zeros((16,4,source.shape[1]),complex)
    for a in range(4):
        for b in range(4):result[4*a+b,a]=framed[b]
    return result.reshape(16,-1)


def normalized_prefix(rows: list[dict[str,Any]],scale: float,boundary: list[np.ndarray],m: int) -> tuple[list[np.ndarray],float]:
    states=[r['vectors']@((r['values']/scale)**m*(r['vectors'].conj().T@xi)) for r,xi in zip(rows,boundary,strict=True)]
    norm=sum(float(np.linalg.norm(s)**2) for s in states)
    if norm<=0:raise ArithmeticError('Actual neutral preparation has zero norm')
    return [s/math.sqrt(norm) for s in states],.5*math.log(norm)


def actual_packet(p: bare.BareElectricParameters) -> dict[str,Any]:
    field=bare.field_regulator(p);pr=bare.reference_parameters(p)
    law,phi,direction=bare.reference.native_law(pr)
    transport,transport_receipt=spin.physical_Ref_transport(field,phi,direction,pr)
    mean=law.potential(np.zeros((2,3)),np.zeros((2,4,4,4)),np.broadcast_to(np.eye(4),(2,3,4,4)).copy())
    h,_=bare.alpha.second_quantized(mean['h'],4);_,v=eigh(h);sea=v[:,-1]
    neutral=[];charged=[];neutral0=[];charged0=[]
    for z in range(4):
        neutral.append(bare.native_factor(p,4,z,field));charged.append(bare.native_factor(p,5,z,field))
        neutral0.append(bare.native_factor(p,4,z,field,normal_reference=True));charged0.append(bare.native_factor(p,5,z,field,normal_reference=True))
    scale=max(r['values'][-1] for r in neutral);scale0=max(r['values'][-1] for r in neutral0)
    boundary=[]
    for row in neutral:
        xi=np.zeros(row['dimension'],complex);xi[:70]=sea/2;boundary.append(xi)
    smooth=[];delta=1/2**20
    for r,xi in zip(neutral0,boundary,strict=True):
        energy=-np.log(r['values']/scale0);f=np.zeros_like(energy);inside=(energy>=0)&(energy<delta)
        f[inside]=np.exp(-energy[inside]**2/(delta**2-energy[inside]**2))
        smooth.append(r['vectors']@(f*(r['vectors'].conj().T@xi)))
    prepgram=sum(float(np.linalg.norm(s)**2) for s in smooth)
    if prepgram<=0:raise ArithmeticError('Actual common smooth preparation Gram vanishes')
    smooth=[s/math.sqrt(prepgram) for s in smooth]
    high=bare.field_regulator(replace(p,quadrature_order=4));check=bare.field_regulator(replace(p,quadrature_order=5))
    b_in=creation_basis(high,0,p)
    packet_high=[packets(high,z,p) for z in range(4)];packet_check=[packets(check,z,p) for z in range(4)]
    _,m5=bare.alpha.second_quantized(np.eye(8),5);creation=bare.reference.creation_matrices(neutral[0]['masks'],m5)
    outcomes=[]
    for label,initial in [('original_Gaussian',boundary),('NEW_same_Phi0_bare_smooth_boundary',smooth)]:
        target,logtau=normalized_prefix(neutral,scale,initial,p.prefix_steps)
        target0,logtau0=normalized_prefix(neutral0,scale0,initial,p.prefix_steps)
        stride=p.normal_order*p.colour_order
        gfull=1+sum(float(np.linalg.norm(s.reshape(6,stride,70)[1])**2) for s in target)
        gfull0=1+sum(float(np.linalg.norm(s.reshape(6,stride,70)[1])**2) for s in target0)
        gprojected=sum(float(np.linalg.norm(s.reshape(6,stride,70)[0])**2) for s in target)
        d=[s-s0 for s,s0 in zip(target,target0,strict=True)]
        weighted_error=math.sqrt(sum(float(np.linalg.norm(s)**2+np.linalg.norm(s.reshape(6,stride,70)[1])**2) for s in d)/gfull0)
        sources=[];sources0=[];receiver=[];receiver0=[];gramspin=np.zeros((4,4),complex)
        gramspin0=np.zeros((4,4),complex);gramcross=np.zeros((4,4),complex)
        raw=raw0=projectedraw=pointcreationgram=old_feedback_error=quad_error=quad_exact_error=0.
        for z,(s,s0) in enumerate(zip(target,target0,strict=True)):
            quadsrc,_=joint_source(high,b_in,packet_high[z],creation,s)
            src=graph.projected_joint_source(s,0,z,creation,p)
            src0=graph.projected_joint_source(s0,0,z,creation,p)
            sources.append(src/math.sqrt(gfull));sources0.append(src0/math.sqrt(gfull0))
            part_in=graph.graph_parts(s,0,z,creation,p);part_in0=graph.graph_parts(s0,0,z,creation,p)
            raw+=float(np.real(np.trace(graph.full_cross_gram(part_in,part_in,p))))/gfull
            raw0+=float(np.real(np.trace(graph.full_cross_gram(part_in0,part_in0,p))))/gfull0
            projectedraw+=float(np.linalg.norm(src)**2)/gfull
            pointcreationgram+=float(np.linalg.norm(b_in@s.reshape(field['dimension'],70))**2)
            checksrc,_=joint_source(check,creation_basis(check,0,p),packet_check[z],creation,s)
            quad_error+=float(np.linalg.norm(checksrc-quadsrc)**2)/gfull
            quad_exact_error+=float(np.linalg.norm(checksrc-src)**2)/gfull
            # Historical bounded P a† P control, normalized by SAME fullGram
            # here solely to isolate the omitted intermediate-state feedback.
            oldph=np.zeros((6,stride,70),complex);oldph[1]=s.reshape(6,stride,70)[0]
            old=bare.reference.apply_source(bare.reference.source_coefficients(field,neutral[z]['source_packet_blocks']),creation,oldph.reshape(-1))
            old_feedback_error+=float(np.linalg.norm(src-old)**2)/gfull
            rout=graph.projected_joint_source(s,1,z,creation,p)
            rout0=graph.projected_joint_source(s0,1,z,creation,p)
            receiver.append(frame_blocks(rout,transport,p));receiver0.append(frame_blocks(rout0,transport,p))
            part_out=graph.graph_parts(s,1,z,creation,p,transport)
            part_out0=graph.graph_parts(s0,1,z,creation,p,transport)
            gramspin+=graph.full_cross_gram(part_out,part_out,p)
            gramspin0+=graph.full_cross_gram(part_out0,part_out0,p)
            gramcross+=graph.full_cross_gram(part_out,part_out0,p)
        if abs(pointcreationgram-gfull)>1e-10:raise ArithmeticError('Actual uncompressed creation quadrature disagrees with CCR graph norm')
        eigen=eigh((gramspin+gramspin.conj().T)/2,eigvals_only=True);mu=float(eigen[-1])
        if mu<=0:raise ArithmeticError('Actual FULL receiver Gram normalization vanishes')
        full_effect_price=graph.full_frame_effect_price(gramspin,gramspin0,gramcross)
        r=np.concatenate(receiver,axis=1);r0=np.concatenate(receiver0,axis=1)
        gram_projected=r.conj()@r.T;mu_projected=float(eigh((gram_projected+gram_projected.conj().T)/2,eigvals_only=True)[-1])
        difference=spin.thin_effect_difference(r,r0)
        spectral=[];spectral0=[];half=half0=physical4=transfer4=negative=moment=source_gap=0.
        for c,c0,src,src0 in zip(charged,charged0,sources,sources0,strict=True):
            energy=-np.log(c['values']/scale);energy0=-np.log(c0['values']/scale0)
            w=(src@c['vectors'].conj())*np.sqrt(c['values']/scale)
            w0=(src0@c0['vectors'].conj())*np.sqrt(c0['values']/scale0)
            weights=np.sum(abs(w)**2,axis=0)
            half+=float(sum(weights));half0+=float(np.linalg.norm(w0)**2);moment+=float(weights@energy)
            physical4+=float(sum(weights[(energy>=0)&(energy<=4*p.spacing)]));transfer4+=float(sum(weights[(energy>=0)&(energy<=4)]))
            negative+=float(sum(weights[energy<0]));source_gap+=float(sum(weights[energy<=min(energy)+1e-8]))
            spectral.append((w,energy));spectral0.append((w0,energy0))
        events=[]
        for t in (0.,.25,.5,1.):
            amp=np.zeros((4,4),complex);whole_error=0.
            for c,c0,block,(w,en),(w0,en0) in zip(charged,charged0,receiver,spectral,spectral0,strict=True):
                evolved=(w*np.exp(-1j*t*en/p.spacing))@c['vectors'].T
                evolved0=(w0*np.exp(-1j*t*en0/p.spacing))@c0['vectors'].T
                whole_error+=float(np.linalg.norm(evolved-evolved0)**2)
                framed=spin.transport_Ref(evolved,transport,pr,adjoint=True)
                amp+=(block.conj()@framed.reshape(-1)).reshape(4,4)
            probability=float(np.sum(abs(amp)**2)/(mu*half))
            if probability>1+1e-9:raise ArithmeticError('FULL calibrated normalized frame effect exceeds1')
            events.append(dict(time_PHYSICAL=t,inclusive_FULLcalibrated_probability=probability,
                offdiagonal_amplitude_norm=float(np.linalg.norm(amp-np.diag(np.diag(amp)))),
                trace_absolute_squared=float(abs(np.trace(amp))**2),whole_charged_half_time_reference_error=math.sqrt(whole_error),
                whole_charged_relative_half_time_reference_error=math.sqrt(whole_error/half0),
                amplitude_real=np.real(amp).tolist(),amplitude_imag=np.imag(amp).tolist()))
        outcomes.append(dict(preparation=label,common_preparation_Gram=1. if label=='original_Gaussian' else prepgram,
            common_preparation_cost=0. if label=='original_Gaussian' else -math.log(prepgram),
            actual_log_tau=logtau,reference_log_tau=logtau0,actual_neutral_vector_error=math.sqrt(sum(float(np.linalg.norm(s)**2) for s in d)),
            TRUE_creation_Gram=gfull,reference_TRUE_creation_Gram=gfull0,projected_creation_Gram=gprojected,
            TRUE_creation_omitted_norm_squared=gfull-gprojected,actual_TRUE_creation_weighted_reference_error=weighted_error,
            actual_uncompressed_source_Gram=raw,reference_uncompressed_source_Gram=raw0,
            actual_JOINT_P_S_aDag_P_source_Gram=projectedraw,source_uncompressed_projection_loss=raw-projectedraw,
            adverse_intermediate_P_S_P_aDag_P_feedback_error=math.sqrt(old_feedback_error),
            actual_quad4_vs5_jointsource_error=math.sqrt(quad_error),
            actual_quad5_vs_exact_Gaussian_jointsource_error=math.sqrt(quad_exact_error),
            actual_whole_half_Gram=half,reference_whole_half_Gram=half0,actual_mean_PHYSICAL=moment/half/p.spacing,
            literal_PHYSICAL_Pe0_4_Gram=physical4,literal_TRANSFER_Pe0_4_Gram=transfer4,negative_energy_weight=negative,
            actual_lowest_gap_branch_source_weight=source_gap,full_receiver_mu=mu,projected_receiver_mu=mu_projected,
            FULL_receiver_Gram_eigenweights=np.repeat(eigen,4).tolist(),FULL_receiver_rank_relative1e_12=int(4*np.sum(eigen>mu*1e-12)),
            actual_projected_raw_frame_effect_difference=difference,actual_FULL_frame_effect_price=full_effect_price,events=events))
    return dict(parameters=asdict(p),ONE_neutral_lambda_N=float(scale),reference_ONE_neutral_lambda_N=float(scale0),
        physical_Ref3_transport=transport_receipt,outcomes=outcomes,
        consumer='Actual P_C S a_fdag P_N. TRUE a† on retained state, followed by S, then chargedGalerkin projection. NO intermediate photon projection. Olddeclared P a†P is only a control, not physicalCCR.',
        receiver='EXACT fulluncompressed Gaussian Wilson source graphGram uses WdagW=I before projection; actual projected overlap divided by FULL mu. Same measurable physicalRef3 transport; E<=I, not orthoprojector, no Gmin. Source colourvacuum and normalDVR are explicit regulator restrictions.',
        scope='NEWbare6a66 exactK Naux0/completeN4,N5/all4centres/twochannels. Allph=0,1 dynamics with TRUE creation/source external graph and explicit chargedfield projection loss. Full field/interacting continuum and macro source/current/Pe4 probability remain unproved. No inducedMaxwell, physicalelectronpole or crosssection claim.')


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    args=parser.parse_args();start=time.time()
    p=bare.BareElectricParameters();result=actual_packet(p)
    inputs=json.dumps(dict(parameters=asdict(p),quadrature_controls=[4,5],source_unchanged=True,
        external_consumer='WHOLEaDag_JOINTSourceCompression_FULLreceiverGram',dependencies={k:hashlib.sha256(v).hexdigest() for k,v in DEPENDENCIES.items()}),sort_keys=True,separators=(',',':'))
    unchanged=Path(__file__).read_bytes()==SOURCE_BYTES and all(Path(cast(str,module.__file__)).read_bytes()==DEPENDENCIES[name] for name,module in MODULES)
    result['execution']=dict(source_at_IMPORT=hashlib.sha256(SOURCE_BYTES).hexdigest(),source_bytes_at_IMPORT=SOURCE_BYTES.decode(),
        dependency_bytes_at_IMPORT={k:v.decode() for k,v in DEPENDENCIES.items()},dependency_sha256={k:hashlib.sha256(v).hexdigest() for k,v in DEPENDENCIES.items()},
        frozen_reference_sha256=bare.REFERENCE_HASH,input_json=inputs,input_sha256=hashlib.sha256(inputs.encode()).hexdigest(),
        runtime_seconds=time.time()-start,startup_unchanged=unchanged,runtime=dict(executable=sys.executable,python=sys.version,numpy=np.__version__,platform=platform.platform()))
    content=json.dumps(result,indent=2)+'\n';Path(args.output).write_text(content)
    print(json.dumps(dict(output=args.output,source=result['execution']['source_at_IMPORT'],input_sha256=result['execution']['input_sha256'],
        summary=[{k:o[k] for k in ('preparation','TRUE_creation_Gram','projected_creation_Gram','actual_whole_half_Gram','literal_PHYSICAL_Pe0_4_Gram')} for o in result['outcomes']],runtime_seconds=time.time()-start)))
if __name__=='__main__':main()
