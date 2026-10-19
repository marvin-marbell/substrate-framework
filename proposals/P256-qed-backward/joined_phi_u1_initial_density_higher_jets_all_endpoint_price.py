"""Price-only repaired continuation: immutable n4 guard, all64x64 CAR support.

No initial coefficient rerun; no root solve; the original native-only Dcar
premise is retained as withdrawn evidence, not silently reused.
"""
from __future__ import annotations
from fractions import Fraction as F
import json
import math
from pathlib import Path
import sys
import time
import zipfile
from typing import Any
import numpy as np
from threadpoolctl import threadpool_limits
import joined_phi_u1_initial_density_higher_jets as jets

ROOT=Path(__file__).resolve().parent
PREFIX='joined_phi_u1_initial_density_higher_jets_all_endpoint_price'


def run()->dict[str,Any]:
    start=time.monotonic()
    guard_path=ROOT/'joined_phi_u1_initial_density_higher_jets_interval_actual_receipt_20261010.json'
    guard_raw=guard_path.read_bytes();guard=json.loads(guard_raw)
    conditional_path=ROOT/'joined_phi_u1_initial_density_higher_jets_actual_receipt_20261010.json'
    conditional_raw=conditional_path.read_bytes()
    if guard['conditional_receipt_sha256']!=jets.digest(conditional_raw):raise ArithmeticError('n4 source epoch mismatch')
    data,sp,inputs,_=jets.body.full_inputs();geo=jets.current.geometry(data,sp[0])
    supplier=jets.moments.MomentBounds(data,order=7)
    requests=[supplier.request_word([j]) for j in range(len(geo['k']))]
    force=[max(F(r['force_Lp_upper'][p]) for r in requests) for p in range(8)]
    horizon=[max(F(r['states']['raw_source']['horizon_a_upper'][p])/F(r['states']['raw_source']['norm_upper']) for r in requests) for p in range(8)]
    # A one-body block may be long or diagonal after repeated CAR merging.
    # Enumerate every64x64 row/column pair; native pairs alone were inadequate.
    role_prices=[]
    for role,bb in enumerate(supplier.b[:2]):
        op=[b/4 for b in bb]
        maximum=F(0);witness=None
        for row in range(64):
            for col in range(64):
                incident=np.flatnonzero((geo['y']==row)|(geo['x']==col))
                value=2*sum((op[int(j)] for j in incident),F(0))
                if value>maximum:maximum=value;witness=dict(row=row,col=col,signed_paths=[int(j) for j in incident])
        role_prices.append(dict(role=sp[role].profile['role'],Dcar_all_endpoint_op_upper_exact=str(maximum),witness=witness,
                                native_op_upper_exact=[str(v) for v in op],native_nuclear_upper_exact=[str(v) for v in bb]))
    dcar=max(F(r['Dcar_all_endpoint_op_upper_exact']) for r in role_prices)
    saved=json.loads(inputs[jets.body.ARCHIVE])['input_descriptor']['exact_NEW_allq_BO_mean']
    g0=F(saved['centre_exact_rational'][0]);gs=F(saved['centre_exact_rational'][1])
    rootg0=F(saved['parameters']['g0']['upper']);rootgs=F(saved['parameters']['gs']['upper'])
    if not(0<g0<=rootg0 and 0<gs<=rootgs):raise ArithmeticError('Original physical mean not enclosed by retained root-frame upper')
    # Root-priced onsite, not a measured matrix norm or an unproved unit cap.
    onsite=jets.moments.upper(jets.moments.ar.iv.sqrt(jets.moments.ar.exact(rootg0)))*max(abs(F(s.profile['onsite'])) for s in sp[:2])
    source_name='joined_phi_u1_actual_compact_source_action_certificate_verified_receipt_20261010.json'
    transfer_name='joined_phi_u1_nonlocal_body_transport_verified_source_transfer_receipt_20261010.json'
    source_raw=(ROOT/source_name).read_bytes();transfer_raw=(ROOT/transfer_name).read_bytes()
    source=json.loads(source_raw);transfer=json.loads(transfer_raw)
    epsilon=F(source['action_input']['epsilon']['upper']);radius=F(source['source_weights']['prepared_radius_L2']['upper']);b1=F(transfer['fundamental_active_W1_upper_exact'])
    eta0=b1*radius;etan=8*eta0;eta_mass=2*eta0+eta0*eta0
    if eta0>F(1,4):raise ArithmeticError('Physical raw norm<=1 not proved by sqrt(.5)<=.75 plus eta0')
    def endpoint(t):
        delta_source=eta0+t*epsilon;delta_bg=t*epsilon
        return dict(Bs=4*delta_source*(2+delta_source)+etan/2,Bg=4*delta_bg*(2+delta_bg)+etan)
    now=endpoint(jets.CLOCK);initial=endpoint(F(0))
    physical=now['Bs']+initial['Bs']+now['Bg']+initial['Bg']+8*eta_mass
    rows=[]
    for row in guard['all12']:
        x=row['receiver'];incident=np.flatnonzero((geo['x']==x)|(geo['y']==x))
        B=sum((supplier.b[0][int(j)] for j in incident),F(0))
        tails={n:jets.higher_budget(B,dcar,onsite,force,horizon,n) for n in (6,8)}
        lower=sum(jets.CLOCK**int(n)*F(c['enclosure']['lower'])/math.factorial(int(n)) for n,c in row['initial_coefficients'].items())
        upper=sum(jets.CLOCK**int(n)*F(c['enclosure']['upper'])/math.factorial(int(n)) for n,c in row['initial_coefficients'].items())
        error=F(tails[6]['Taylor_remainder_upper_exact'])
        rows.append(dict(path=row['path'],receiver=x,length=row['length'],initial_coefficients=row['initial_coefficients'],
                         initial_n4_guard_sha256=jets.digest(guard_raw),Bx_native_nuclear_upper_exact=str(B),Dcar_all_endpoint_op_upper_exact=str(dcar),
                         sixth_full_H_remainder_upper_exact=str(error),eighth_full_H_remainder_if_actual_n6_exact=tails[8]['Taylor_remainder_upper_exact'],
                         exact_positive_tail_provenance=tails,
                         coordinate_density_change_exact=dict(lower=str(lower-error),upper=str(upper+error)),
                         both_time_physical_density_change_exact=dict(lower=str(lower-error-physical),upper=str(upper+error+physical)),
                         physical_signed_margin_lower_exact=str(lower-error-physical),physical_sign_controlled=lower>error+physical))
    captures={**jets.body.CAPTURED,**inputs,guard_path.name:guard_raw,conditional_path.name:conditional_raw,source_name:source_raw,transfer_name:transfer_raw}
    for module in list(sys.modules.values()):
        file=getattr(module,'__file__',None)
        if file and Path(file).parent==ROOT and Path(file).suffix=='.py':captures.setdefault(Path(file).name,Path(file).read_bytes())
    for name,raw in captures.items():
        if (ROOT/name).read_bytes()!=raw:raise ArithmeticError('Price-only source/input changed '+name)
    archive=ROOT/(PREFIX+'_input_bytes_20261010.zip')
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for name,raw in sorted(captures.items()):z.writestr(name,raw)
    result: dict[str,Any]=dict(status='INITIAL_DENSITY_N4_ALL_ENDPOINT_PAID_SIXTH_PHYSICAL_SIGNS_CONTROLLED' if all(r['physical_sign_controlled'] for r in rows) else 'INITIAL_DENSITY_N4_ALL_ENDPOINT_PAID_SIXTH_FAILED_REGIME',
                clock_exact=str(jets.CLOCK),clock_adopted=False,all12=rows,
                Dcar_proof='For arbitrary CAR-generated row/col block A, nuclear([B_j,A])<=2 op(B_j) nuclear(A); nonzero products require col_j=row_A or row_j=col_A. All64x64 pairs enumerated, including long/diagonal. The physical Clifford B=aGamma0+i bGammaaxis has op(B)=|a|+|b|, bounded outward by nuclearTriangle/4 from the exact physical-frame root supplier. Identical auxiliary commutators collapse before expectation, no63 factor here.',
                Bx_proof='Initial [H,n_x] is the active signed incident hop sum; every literal native path and AP seam is retained. Nuclear root bounds price dGamma of each hop.',
                root_frame_provenance=dict(original_fixed_mean_g0_exact=str(g0),original_fixed_mean_gs_exact=str(gs),retained_frame_g0_upper_exact=str(rootg0),retained_frame_gs_upper_exact=str(rootgs),onsite_block_upper_exact=str(onsite),role_prices=role_prices),
                moment_provenance=dict(order=7,force_Lp_native_max_exact=[str(x) for x in force],per_normalized_raw_channel_horizon_a_max_exact=[str(x) for x in horizon],
                    common_norm_factor_removal='algebraic cancellation in the linear positive hierarchy; raw+.5background total mass1 is retained, not source normalization',native_only=True,all63_auxiliary=True,all129=True),
                physical_provenance=dict(epsilon_exact=str(epsilon),prepared_radius_L2_exact=str(radius),b1_exact=str(b1),eta0_exact=str(eta0),eta_n_exact=str(etan),eta_mass_upper_exact=str(eta_mass),
                    source_Bs_t_exact=str(now['Bs']),source_Bs_0_exact=str(initial['Bs']),background_Bg_t_exact=str(now['Bg']),background_Bg_0_exact=str(initial['Bg']),additional_mass_price_exact=str(8*eta_mass),total_both_time_price_exact=str(physical),
                    derivation='n norm4; rank4 density trace-nuclear spectator bound eta_n=8b1R. Physical actual E0 eta0=b1R; source vector error eta0+t epsilon, background t epsilon. Source BODY norm²=.5; sqrt(.5)<=.75 and eta0<=.25 establish physical raw norm<=1 without renormalization. Both0/t costs and mass deviation <=2eta0+eta0² retained.'),
                physical_target='xi=E0_FULL Tsrc psi, NOT Tsrc E0_BODY psi; own physical initial density is subtracted at t0 before the finite-time excess comparison',
                physical_raw_mass_interval_exact=dict(lower=str(max(F(0),F(1,2)-eta_mass)),upper=str(min(F(1),F(1,2)+eta_mass))),
                inequality_stage='Initial outward n2/n4 BODY jet plus paid SAME-full-H sixth tail; then both source/background physical t0/t endpoints and 8etaMass are paid. This encloses original source-minus-half-background density CHANGE, and also source-minus-actual-physical-raw-mass background CHANGE without normalization. It is not a photon/current kernel, ground-state, pole, cone or source adoption theorem.',
                original_native_only_Dcar_tail_premise='WITHDRAWN: max over native endpoint pairs can miss long/diagonal CAR-generated endpoints; old receipts remain evidence only',
                original_n6_symbolic_failure='OPEN: order5 exceeds1000000 grouped symbols; no partial n6 used, paid sixth after actual n4 makes it unnecessary for this signed density claim',
                initial_arithmetic='derived outward binary64 Balls with exact-rational Gauss and directed elementary fixedmean Bloch supplier',
                initial_guard_receipt=dict(path=guard_path.name,sha256=jets.digest(guard_raw),source_sha256=guard['source_sha256']),
                source_sha256=jets.digest(Path(__file__).read_bytes()),input_archive=dict(path=archive.name,sha256=jets.digest(archive.read_bytes()),files={n:jets.digest(v) for n,v in captures.items()}),
                runtime=dict(executable=sys.executable,python=sys.version,numpy=np.__version__),elapsed_seconds=time.monotonic()-start,
                original_source_parameters_unchanged=True,raw_mass_body_exact='1/2',source_renormalized=False,whole_kernel_pole_ground_cone_or_source_admission=False)
    output=ROOT/(PREFIX+'_actual_receipt_20261010.json');output.write_text(json.dumps(result,indent=2)+'\n')
    (ROOT/(PREFIX+'_natural_exit_capsule_20261010.json')).write_text(json.dumps(dict(numeric_exit=0,natural_exit=True,source_stable=True,
        source_sha256=result['source_sha256'],receipt=dict(path=output.name,sha256=jets.digest(output.read_bytes())),input_archive=result['input_archive'],elapsed_seconds=result['elapsed_seconds']),indent=2)+'\n')
    return result


if __name__=='__main__':
    with threadpool_limits(limits=1):result=run()
    print(json.dumps(dict(status=result['status'],actual12=len(result['all12']),elapsed_seconds=result['elapsed_seconds'])),flush=True)
