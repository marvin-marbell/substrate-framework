#!/usr/bin/env python3
"""NEW coupled quantumEM/CAR source lifted through the actual Phi well.

This proves reference graph domination and constructs a normalized macro
preparation; it does NOT promote finite-box EM energy/dynamics to macro poles.
"""
from __future__ import annotations
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any
from induced_phi_em_gaussian_variational import build_native
from induced_phi_em_momentum_dressing import build_dressing,evaluate_dressing,SOURCE_BYTES as DRESS_BYTES
from induced_phi_ir_well_transfer import rational

ROOT=Path(__file__).resolve().parent
SOURCE_BYTES={**DRESS_BYTES,Path(__file__).name:Path(__file__).read_bytes()}


def encoded(value: Fraction | int) -> dict[str,str]:
    f=Fraction(value);return dict(numerator=str(f.numerator),denominator=str(f.denominator))


def run(mean_path: Path,well_path: Path,pure_phi_path: Path,source_path: Path) -> dict[str,Any]:
    started=time.monotonic();inputs={}
    for role,path in (('mean',mean_path),('well',well_path),('pure_phi',pure_phi_path),('EM_source',source_path)):
        raw=path.read_bytes();inputs[role]=dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),capsule=json.loads(raw))
    mean=inputs['mean']['capsule'];well=inputs['well']['capsule'];pure=inputs['pure_phi']['capsule'];em=inputs['EM_source']['capsule']
    if pure['input_descriptor']['input_receipts']['mean']['sha256']!=inputs['mean']['sha256']:
        raise ArithmeticError('ProjectedPhi and newEM source use different actual mean epochs')
    state=em['descriptor'];variance=Fraction(float(state['s_density_variance']));kappa=Fraction(float(state['kappa']))
    N=mean['finite_size'];V=N**3;M=2*V+1
    old2=Fraction(192);old4=Fraction(129024);new2=1/(4*variance);new4=3*new2**2
    if new2>old2 or new4>old4:raise ArithmeticError('Actual Gaussian reference moments do not dominate the old graph proof')
    oldsum4=M*old4+M*(M-1)*old2**2;newsum4=M*new4+M*(M-1)*new2**2
    if newsum4>oldsum4:raise ArithmeticError('Fullmacro momentum graph dominance failed')
    # Actual native cubic density is a separate source, not the old axis0 pure source.
    native=build_native(mean,2);rows=[]
    for axis in range(3):
        model=build_dressing(native,axis);result=evaluate_dressing(model,float(kappa),float(variance))
        rows.append(dict(axis=axis,energy=result['energy'],current_norm=float(sum(x*x for x in result['current'])**.5)))
    defect=max(row['energy'] for row in rows)-min(row['energy'] for row in rows)
    if defect>1e-6:raise ArithmeticError('Actual native cube-average construction has an unexpected orientation energy difference')
    descriptor=dict(inputs={role:{key:entry[key] for key in ('path','sha256')} for role,entry in inputs.items()},
        macroN=N,macroV=V,physical_EM_coordinates=M,charged_flavours=64,heavy_neutral_flavours=1,
        state='NEW3orientation quantumEM/CAR cubeaverage liftedbyprojectedheavyPhi; oldaxis0/pure222/rho128 unchanged',
        preparation_variance=str(variance),preparation_kappa=str(kappa),clock=4096,
        plan='issue-222-induced-photon-delta13-continuation-20261008',plan_revision='85975947857ac7816cf0d8674f69229861014434b125a07a6996af0ecf72c4eb')
    coordinate2=M*variance+kappa*kappa*V*128**2
    return dict(input_descriptor=descriptor,input_sha256=hashlib.sha256(json.dumps(descriptor,sort_keys=True).encode()).hexdigest(),
        exact_macro_state='For exactimplicit cubic222 mean C, all64 originalchargedcentre negativeSlaters Omega0 and eachorientationj, define normalized momentumwave g_s(p) tensor64 Gamma(exp[-i kappa (Qp)_(x,j) Gamma_j]) Omega0. Q is ANY exactrealorthonormal coexact+harmonic frame ofkerD, allM=2V+1 modes. Ref4 bareQref1,nref4,background−4 NET0, neutralheavy factor unchanged. PhysicalrealGauss lift is an isometry. NEWrho_cube=(rho_0+rho_1+rho_2)/3. Lift SAMErho_cube through actual slow45/fast90V−45 Gaussian-cut Phi well and constantphase projectedheavy section, jointnormalizedHaar/FP w^-1/2 and onlyglobalZ4. Normalizationexact1; no compactification or clippedharmonic/Gibbs trace.',
        exact_current_source_functional='Fixednormalizedrho_cube: Z_cube[A]=(Z_0[A]+Z_1[A]+Z_2[A])/3, W=logZ_cube; never averageW/vertices. Eachorientation isC invariant whenfullaction reviewed, sooddjetsvanish; chi4=meanchi4_j+3Var(chi2_j), complexbilinear. This is a NEWcubeaverage, not modification of anypreviousdensity.',
        exact_graph_dominance=dict(old_scalar_Pi2=encoded(old2),old_scalar_Pi4=encoded(old4),new_scalar_Pi2=encoded(new2),new_scalar_Pi4=encoded(new4),
            macro_old_sum_Pi2_squared=encoded(oldsum4),macro_new_sum_Pi2_squared=encoded(newsum4),dominance=True,
            proof='U_j commutes EVERYphysicalPi_a andEVERYrho_x. NewGaussian momentum moments are smaller thanoldH2 bump moments separately atorders2/4; expansion of(sumPi²)² givesexactallM graph dominance. EntireCoulombgraph norm invariant underU. OriginalchargedCAR graph bound is an operatornorm independentof the initialSlater/EM state. Therefore the oldreferencegraph cap dominates thisNEWquantumreference; graph is conserved by SAMEexactHref atalltimes. No stationarity assumption orWick aftertimeevolution.'),
        full_Physical_Phi_transfer={name:value for name,value in pure['new_projected_section_transfer']['prices'].items() if name in (
            'physical_state_norm_distance_upper','density_trace_norm_distance_upper','difference_graph_norm_upper','weighted_Phi_action_error',
            'all_flavour_current_action_error','all_flavour_contact_action_error','all_flavour_current_expectation_error','all_flavour_contact_expectation_error')},
        transfer_proof='The actual heavyprojectedsection D1/D2 price, scalarwell/minmax/Agmon, Haar/FP metric and Gausschargeoperator caps are unchanged. ChargedPhi/colour freezing is an operatornorm estimate validforANY64CAR/EM input. Newreference has dominatedinitialgraph and sameconservedHrefgraph. Thus its oldconservative64x norm/graph/Heinz/current/contact prices remain valid; convexity preservesrho_cube tracebounds. This is a new same-Href state theorem, NOT oldpacket-as-vacuum inheritance.',
        coordinate_second_moment_upper=encoded(coordinate2),noncompact_tail_probability_formula='P(||Theta||>R)<=coordinate_second_moment_upper/R²; P(||Pi||>R)<=M/(4variance R²). Nonzero tails, no toron clipping.',
        native_cube_energy_consumer=dict(N=2,V=8,orientations=rows,max_orientation_energy_defect=defect,
            scope='Actual threepreparedorientation sources atSAMEfiniteN/mean/variance/kappa. Thiscontrol isnot macroenergy orpole verification.'),
        mathematical_review_required='OldactualPhi/root/jointGauss/Heinz proof andprojectedheavy64x bounds, plusreferencegraph-domination argument. Exactmacro source isconstructed; its lowenergy/stationarity/EM causalconsumer remainsunproved.',
        limitations='NO N2/N6-to-macro EMenergy/current reduction, photonvacuum, commoncone, stationarity, finiteclockEMpropagator orresidue/dressedCompton/QEDpromotion. Anactualnormalizedmacrostate andcontrolledPhi transfer do not resolve its EM dynamics.',
        source_at_import={name:dict(sha256=hashlib.sha256(raw).hexdigest(),bytes_utf8=raw.decode()) for name,raw in SOURCE_BYTES.items()},
        source_unchanged_since_import={name:(ROOT/name).read_bytes()==raw for name,raw in SOURCE_BYTES.items()},
        runtime=dict(python=sys.version,executable=sys.executable,platform=platform.platform()),elapsed_seconds=time.monotonic()-started)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for role in ('mean','well','pure_phi','source','output'):parser.add_argument('--'+role,type=Path,required=True)
    args=parser.parse_args();report=run(args.mean,args.well,args.pure_phi,args.source)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({name:report[name] for name in ('input_sha256','coordinate_second_moment_upper','native_cube_energy_consumer','elapsed_seconds')},indent=2))
