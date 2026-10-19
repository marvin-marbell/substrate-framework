"""Conditional full-Phi transfer of ACTUAL12 bounded physical Wilson receivers.

Bind an actual parent uniform source-action receipt, never a free epsilon.
Original finalized body/controls bytes are not modified by this consumer.
"""
from __future__ import annotations
import argparse
from fractions import Fraction as F
import hashlib
from itertools import combinations
import json
from pathlib import Path
import sys
import time
from typing import Any
import zipfile

import numpy as np
from scipy.linalg import expm
from threadpoolctl import threadpool_limits
from joined_phi_u1_nonlocal_body_transport import sqrt_upper
from joined_phi_u1_nonlocal_body_transport_controls import creation

ROOT=Path(__file__).resolve().parent
BODY='joined_phi_u1_nonlocal_body_transport_receipt.json'
BODY_SHA='354738d91318fbb048182f809a10c9a44fdbf8536710f46634dcaa5bc38eb537'
JET='induced_phi_global_normal_h_jet_control_receipt.json'
JET_SHA='e48a53776184f9eda572d5f06ef061f880cfb13935b41f3a707b25d1994e58b8'
PROOF={
 'endpoint_uniformity':'T uses a fixed normalized scalar packet, fullHaar localGamma(U), neutral filled fibre and fundamental active KatoW(r), with W(0)=I. For ANY bodyvector v (arbitrary realTheta dependence and arbitrary fixed chargedFock vector), CAR intertwining cdag(z)Gamma(W)=Gamma(W)cdag(Wdag z) cancels every spectator/auxiliary W exactly. Ref is static and has no KatoWref. Thus receiver difference depends on ONE fundamentalactive W-I, not full-Fockgradient, particle count or wholeGammaW-I.',
 'fundamental_W':'Along radial segment inside supported3R tube, ||W(r)-I||<=b1|r|, b1=active directional_Kato_first.upper from actual repaired support-wide hjet receipt. Its producer derives b1=8D1/gap_distance. This is not the wholeN FockC1 or an unsupported componentwise norm.',
 'physical_Wilson':'LocalGamma(U) and Ref endpoint transport exactly cancel localgauges in the physical ordered Wilson receiver. The full anchored scalar phase qTheta+delta lambdaC is shared on both sides; delta=e_x-e0 removes arbitrary anchor. Residual ordered SU4 path Vell(C) remains, NOT falsely identified with TExTdag. For generators Tr(tatb)=deltaab/2, ||C_e^a ta||op<=|C_e|/sqrt2 and unitary product telescoping gives ||Vell-I||<=ell|C|/sqrt2<=ell|r|/sqrt2 (valid also repeated ordered edges).',
 'creation_norm':'On preserved Ref4 background, Ex=.5sum4 cdag(z_alpha)Ralpha, orthogonal Refholes. For the four-column receiver difference Z, operatornorm<=.5||Z||HS<=||Z||op. Consequently ||(Eell_full T-T Eell_body)v||<=etaell||v||, etaell=(b1+ell/sqrt2)*prepared_radius_L2. For colocal ell0, eta0=b1*radius. This Fubini L2 packet bound holds for EVERY bodyvector, not merely the initial Slater. Full Wilson receivers have norm<=1 on Ref4.',
 'Duhamel':'Actual source-action receipt supplies uniform ||Hfull T-T(Hbody+Ephi)||<=epsilon in both fixedtotalN8324 background128/Ref4 and source129/Ref3 sectors, with common reference Ephi. Then first-leg difference <=|t|epsilon+etaell. Second-leg difference <=|t|epsilon*sqrt(.5)+eta0; initial physical E0fullTpsi is compared with T E0psi, rather than replacing it for free. Cauchy-Schwarz gives kernel error<=2|t|epsilon*sqrt(.5)+etaell*sqrt(.5)+eta0. A conservative exact rational square-root upper supplies the implemented price.',
 'signal':'Add this uniform endpoint+Duhamel error to the ACTUAL12 body rational remainder at its unchanged2^-36 clock. The surviving imaginary signal is conditional on the body first-moment floating1percent allowance; NOT an exact interval signal lower. Root existence/selection and static/positive slow pencil remain conditional. No ground, freechargedpole, commoncone or originalSOURCE admission.'}


def endpoint_control() -> dict[str,Any]:
    rng=np.random.default_rng(222101066)
    raw=rng.normal(size=(4,4))+1j*rng.normal(size=(4,4));g=(raw+raw.conj().T)/2
    c1=np.array([[.2,.3j],[-.3j,-.2]]);c2=np.array([[.1,.4],[.4,-.1]])
    r=.03;w=expm(1j*r*g);v=expm(1j*r*c1)@expm(1j*r*c2)
    def exterior(mat: np.ndarray,count: int) -> np.ndarray:
        sets=list(combinations(range(4),count))
        return np.asarray([[np.linalg.det(mat[np.ix_(a,b)]) for b in sets] for a in sets])
    gamma_b=exterior(w,2);gamma_s=np.kron(np.eye(2),exterior(w,3))
    cs=[creation(4,2,i) for i in range(4)]
    ebody=np.concatenate([.5*cs[2+a] for a in range(2)],axis=0)
    physical=np.concatenate([.5*sum((v[i,a]*cs[2+i] for i in range(2)),np.zeros_like(cs[0])) for a in range(2)],axis=0)
    transformed=w.conj().T[:,2:]@v
    car=np.concatenate([.5*sum((transformed[i,a]*cs[i] for i in range(4)),np.zeros_like(cs[0])) for a in range(2)],axis=0)
    defect=float(np.linalg.norm(physical@gamma_b-gamma_s@car,2))
    diff=float(np.linalg.norm(physical@gamma_b-gamma_s@ebody,2))
    upper=r*(np.linalg.norm(g,2)+np.linalg.norm(c1,2)+np.linalg.norm(c2,2))
    wrong_W=float(np.linalg.norm(ebody@gamma_b-gamma_s@ebody,2))
    wrong_Wilson=float(np.linalg.norm(physical@gamma_b-ebody@gamma_b,2))
    if defect>1e-12 or diff>upper or wrong_W<1e-3 or wrong_Wilson<1e-3:
        raise ArithmeticError('Uniform CAR/Wilson endpoint control failed or blind')
    return dict(law='Finite4activeorbital count2/3, two orthogonal staticRefholes; independent noncommuting ordered SU4-subblock matrices',
                radius=r,CAR_intertwining_defect=defect,operatornorm_endpoint_difference=diff,
                elementary_uniform_upper=float(upper),wrong_exact_TExTdag_endpoint_defect=diff,
                omitted_fundamental_W_defect=wrong_W,omitted_ordered_Wilson_defect=wrong_Wilson,
                no_particle_count_multiplier=True,scope='Dense finite-CAR algebra control, not N4 source or interval signal measurement')


def run(source_path: Path, source_sha: str, output: Path) -> dict[str,Any]:
    start=time.monotonic();sys.set_int_max_str_digits(0)
    own=Path(__file__).read_bytes()
    raws={BODY:(ROOT/BODY).read_bytes(),JET:(ROOT/JET).read_bytes(),source_path.name:source_path.read_bytes()}
    for name,sha in ((BODY,BODY_SHA),(JET,JET_SHA),(source_path.name,source_sha)):
        if hashlib.sha256(raws[name]).hexdigest()!=sha:raise ArithmeticError('Actual input receipt mismatch: '+name)
    body=json.loads(raws[BODY]);jet=json.loads(raws[JET]);source=json.loads(raws[source_path.name])
    if source['kind']!='CONDITIONAL_ACTUAL_COMPACT_PHI_BODY_UNIFORM_SOURCE_INTERTWINER_ACTION':raise ArithmeticError('Wrong source law receipt')
    if not source['action_input']['uniform_body_Hilbert_map'] or not source['all_inputs_and_sources_unchanged']:
        raise ArithmeticError('No actual uniform source-action evidence')
    weights=source['source_weights']
    if (weights['body_Theta_dimensions'],weights['normal_dimensions'],weights['fixed_total_particle_number'])!=(129,5760,8324):
        raise ArithmeticError('Wrong fullN4 fixed-particle body/source contract')
    if source['input_sha256'].get(JET)!=JET_SHA:raise ArithmeticError('Source and endpoint fundamental Kato epochs differ')
    epsilon=F(source['action_input']['epsilon']['upper']);radius=F(weights['prepared_radius_L2']['upper'])
    if epsilon<0 or radius<0:raise ArithmeticError('Negative norm budget')
    active=next(z for z in jet['profiles'] if z['role']=='active filled charged')
    b1=F(active['directional_Kato_first']['upper']);root_half=sqrt_upper(F(1,2))
    eta0=b1*radius
    rows=[]
    for z in body['receiver_rows']:
        ell=int(z['length']);eta=(b1+ell*root_half)*radius;t=F(z['clock']['t_exact'])
        transfer=2*t*epsilon*root_half+eta*root_half+eta0
        rem=F(z['clock']['rational_remainder_upper']);total=rem+transfer
        # The floating observed signal is never converted into an interval lower.
        signal=z['clock']['observed_imaginary_signal_with_one_percent_allowance']
        lower=signal-float(total)
        if lower<=0:raise ArithmeticError('Actual bound does not preserve body floating signal allowance')
        rows.append(dict(path=z['path'],row=z['row'],col=0,length=ell,time_exact=str(t),
                         eta_receiver_uniform_exact=str(eta),eta_source_uniform_exact=str(eta0),
                         endpoint_Duhamel_kernel_error_exact=str(transfer),body_remainder_exact=str(rem),
                         total_error_exact=str(total),total_error_float=float(total),
                         observed_signal_with_declared_float_allowance=signal,
                         observed_signal_minus_total_error=lower,
                         conditional_full_Phi_nonzero_relative_to_FLOAT_allowance=True))
    control=endpoint_control()
    for name,raw in raws.items():
        path=source_path if name==source_path.name else ROOT/name
        if path.read_bytes()!=raw:raise ArithmeticError('Receipt changed during composition: '+name)
    if Path(__file__).read_bytes()!=own:raise ArithmeticError('Own source changed')
    captures={**raws,Path(__file__).name:own,
              'joined_phi_u1_nonlocal_body_transport.py':(ROOT/'joined_phi_u1_nonlocal_body_transport.py').read_bytes(),
              'joined_phi_u1_nonlocal_body_transport_controls.py':(ROOT/'joined_phi_u1_nonlocal_body_transport_controls.py').read_bytes()}
    archive=output.with_name(output.stem+'_input_bytes.zip')
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for name,raw in captures.items():z.writestr(name,raw)
    result=dict(kind='CONDITIONAL_ACTUAL12_PHYSICAL_WILSON_BOUNDED_KERNEL_SOURCE_TRANSFER',proof=PROOF,
                source_action_bound_exact=str(epsilon),source_radius_L2_upper_exact=str(radius),fundamental_active_W1_upper_exact=str(b1),
                receiver_rows=rows,independent_uniform_CAR_endpoint_control=control,
                actual_norm_price_input_bound=True,unbounded_current_graph_theorem_used=False,
                arithmetic='Exact rational upper error coefficients from actual supplier interval upper endpoints; body moment signal FLOATING with declared1percent allowance, NOT INTERVAL lower',
                assumptions=['Exact selected currentN4 root and static gaps/positive slow pencil','Declared source isometry and fullHaar/Gauss quotient construction','Actual supplier uniform action applies both background and source sectors with common Ephi'],
                excluded=['Original SOURCE/CURRENT427 adoption','Ground','Free charged pole','Common cone','FullN4 measured evolution','Future Wick/Slater propagation'],
                input_sha256={k:hashlib.sha256(v).hexdigest() for k,v in captures.items()},
                source_action_receipt_path=str(source_path),source_action_receipt_sha256=source_sha,
                all_inputs_and_source_unchanged=True,own_source_utf8=own.decode(),
                immutable_inputs=dict(path=archive.name,sha256=hashlib.sha256(archive.read_bytes()).hexdigest()),
                governing_plan=body['governing_plan'],body_captured_revision=body['captured_parent_revision'],source_captured_plan=source['plan'],
                runtime=dict(executable=sys.executable,python=sys.version),elapsed_seconds=time.monotonic()-start)
    output.write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-receipt',type=Path,required=True)
    parser.add_argument('--source-sha',required=True)
    parser.add_argument('--output',type=Path,default=ROOT/'joined_phi_u1_nonlocal_body_transport_source_transfer_receipt.json')
    args=parser.parse_args()
    with threadpool_limits(limits=1):result=run(args.source_receipt,args.source_sha,args.output)
    print(json.dumps(dict(kind=result['kind'],rows=len(result['receiver_rows']),
                         minimum_observed_signal_minus_total_error=min(z['observed_signal_minus_total_error'] for z in result['receiver_rows']),
                         endpoint_control=result['independent_uniform_CAR_endpoint_control'],elapsed_seconds=result['elapsed_seconds'])),flush=True)
