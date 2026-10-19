#!/usr/bin/env python3
"""Actual weighted bare-field creator graph on a normalized auxiliary sea line.

This exposes massive canonical particle-hole components created when bare
field momentum differentiates the field-dependent auxiliary sea. It does NOT
identify the bare Hermite ladder with an interacting asymptotic photon. No
microscopic sector is deleted; a proposed sea-band photon would be a distinct
observable/preparation, requiring its own actual spectral/current admission.
"""
from __future__ import annotations
import argparse
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any
if __name__=='__main__':
    for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import numpy as np
from scipy.linalg import eigh
import physical_c_bare_electric_loops as loops
SOURCE_BYTES=Path(__file__).read_bytes();LOOP_BYTES=Path(loops.__file__).read_bytes()
if hashlib.sha256(LOOP_BYTES).hexdigest()!='01245b29b505d724ed8b301d07cb00c022e1f5a1a300cec3db0661c03c943e00':
    raise ArithmeticError('Auxiliary photon graph binds precise-loop01245')


def actual_metric(ktext: str,beta_text: str,omega: float,order: int) -> dict[str,Any]:
    count=Fraction(beta_text)/(16*Fraction(ktext)**4)
    if count.denominator!=1 or count.numerator%2:raise ArithmeticError('Exact even Naux required')
    naux=count.numerator;k=float(ktext);beta=float(beta_text)
    b=loops.background((4,3,1));norm=math.sqrt(b['normE']);profile=b['a']/norm
    law=loops.make_law(k,b);nodes,weights=np.polynomial.hermite.hermgauss(order)
    metric_integral=defect_after_SF_integral=0.;rows=[]
    for node,weight in zip(nodes,weights/math.sqrt(math.pi),strict=True):
        a=node/math.sqrt(omega);phases=a*profile/math.sqrt(beta);direction=profile/math.sqrt(beta)
        state=law.potential(phases,b['phi'],b['links'],(direction,direction))
        energies,frame=eigh(state['h']);n=len(energies)//2
        current=frame[:,:n].conj().T@state['hx']@frame[:,n:]
        gaps=energies[None,n:]-energies[:n,None]
        if min(gaps.ravel())<=0:raise ArithmeticError('Actual canonical band gap closes')
        amplitudes2=abs(current)**2/gaps**2
        metric=float(naux*np.sum(amplitudes2))
        # Field a_ref† derivative coefficient1/sqrt(2omega). Single canonical
        # particle-hole SF amplitude ratioexp(-gap/2), squaredexp(-gap).
        after=float(naux*np.sum(amplitudes2*np.exp(-gaps))/(2*omega))
        metric_integral+=float(weight)*metric;defect_after_SF_integral+=float(weight)*after
        rows.append(dict(field_A=float(a),actual_tensorsea_quantum_metric=metric,
            actual_min_particle_hole_gap_TRANSFER=float(min(gaps.ravel())),
            actual_point_relative_SF_defect_graph_Gram=after))
    defect=metric_integral/(2*omega)
    return dict(kappa=ktext,beta_theta=beta_text,Naux=str(naux),reference_packet_omega=omega,quadrature_order=order,
        normalized_mode_link_norm_squared=float(np.sum(profile**2)),all_straight_torons_zero=b['max_line_toron'],
        omega_meaning='Declared Gaussian envelope width parameter, not a measured interacting photon frequency.',
        actual_Gaussian_weighted_auxiliary_metric=metric_integral,
        actual_whole_bare_field_creator_Gram=1+defect,
        actual_sea_band_creator_component_Gram=1.,actual_UNCOMPRESSED_massive_auxiliary_creator_Gram=defect,
        actual_Gaussian_average_point_relative_SF_defect_Gram=defect_after_SF_integral,rows=rows,
        graph_identity='Choose parallel sea phase on REAL A: <I,Iprime>=0. For normalized Gaussian phi_omega, a_refdag(I phi)=I a_refdag(phi)-Iprime phi/sqrt(2omega). Components orthogonal; wholeGram=1+<G>/(2omega), G=N sum|hprime_-+|²/(Eplus-Eminus)².',
        output_limit='The SF defect price is average of pointwise ratios to lambdaVac(A), NOT the full field-K output norm or globally normalized sourcehalf. Original fixedbody boundary and Gauss have not been replaced by this sea-line packet.')


def uniform_Riccati_metric_bound() -> dict[str,Any]:
    """Dimension-free actual instantaneous-sea metric, pureAbelian meanPhi."""
    mp=loops.mp
    with mp.workdps(180):
        formula={}
        # After factoring k² out of dx, every positive numerator factor
        # increases and every positive separation decreases on[0,1e-6].
        # A conservative integer13 at the endpoint avoids rounding an
        # actual12+O(k) bound DOWN to12 at macro tiny k.
        for label,k in [('macro',mp.mpf('1e-250')),('ceiling',mp.mpf('1e-6'))]:
            b=1/(2*k)-3;f=mp.mpf(3)
            upper_component=f/(b*(b-1/b))
            x=upper_component/mp.sqrt(1-upper_component**2)
            separation=b-1/b-2*x*f/b
            db=df=2*mp.sqrt(3)
            da=db/b**2;dc=db*f/b**2+df/b
            dd=db+2*f*df/b+f*f*db/b**2
            dx=(da*x+dc+x*dd+x*x*dc)/separation
            formula[label]=dx*dx/(16*k**4)
        if formula['ceiling']>=13:raise ArithmeticError('Conservative Riccati interval ceiling failed')
        metric=mp.mpf(13);b=1/(2*mp.mpf('1e-250'))-3
        length=mp.mpf(2)**24
        omega=2*mp.asinh(mp.sin(mp.pi/length))
        defect=metric/(2*omega)
        return dict(kappa='1e-250',unit_L2_mode_profile=True,
            instantaneous_tensorsea_metric_uniform_upper=mp.nstr(metric,80),
            Riccati_formula_at_macro_approximation=mp.nstr(formula['macro'],80),
            Riccati_formula_at_interval_ceiling=mp.nstr(formula['ceiling'],80),
            certified_conservative_interval='0<kappa<=1e-6; metric<=13. Monotone factored-Riccati expression; avoids rounded12+O(k) mislabel.',
            macro_L=str(int(length)),conditional_leading_fundamental_omega_TRANSFER=mp.nstr(omega,80),
            bare_reference_creator_defect_Gram_upper=mp.nstr(defect,80),
            whole_reference_creator_Gram_upper=mp.nstr(1+defect,80),
            canonical_point_SF_defect_ratio_upper=mp.nstr(defect/b**2,80),
            profile_volume_dependence='For a normalized Fourier mode each link amplitude is O(V^-1/2), while Frobenius derivative uses sumlink|profile|²=1. Metric upper is independent ofV; creator defect scales1/omega, not1/V. Wavepacket bound replaces1/omega by sum|fmode|²/omega_mode.',
            derivation='For T=[[A,C],[Cdag,D]], A=B^-1,C=-B^-1F,D=B+FdagB^-1F; eigeninterlacing gives positiveband>=b and negativeband<=1/b. Positiveband is graph(X); ||X||<=c/sqrt(1-c²),c=f/[b(b-1/b)]. Riccati derivative solves (A-XCdag)dX-dX(D+CdagX)=-dA X-dC+X dD+X dCdag X. Hermitianpart separation>=b-1/b-2xf/b. Frobenius RHS bounds yield displayeddx. Slatermetric<=||dX||F² and exactN/beta=1/(16k^4). Canonical particle-hole SF normratio²<=b^-2. All phases and any periodicvolume; no short-winding assumption needed for this spectralmetric bound.',
            scope='MeanPhi exactly0/SU4I, actualC=T4 identity; instantaneous auxiliarysea-line packet only. The leadingomega is conditional Maxwell dispersion, NOT an observed fullK pole. Uniformbound is not actual interacting preparedB graph or sourceM0/window; originalfixedbody boundary/fullGauss/intertwining remain unproved.')


def leading_Riccati_metric_remainder() -> dict[str,Any]:
    """Actual instantaneous metric tends to ONE, not only bounded by thirteen."""
    mp=loops.mp
    with mp.workdps(360):
        records={}
        for label,k in [('macro',mp.mpf('1e-250')),('ceiling',mp.mpf('1e-6'))]:
            b=1/(2*k)-3;f=mp.mpf(3);c=f/(b*(b-1/b));x=c/mp.sqrt(1-c*c)
            separation=b-1/b-2*x*f/b
            # All extents>2: forward/backward entries are distinct, giving
            # EXACT sqrt(beta)||dB||F=sqrt(beta)||dF||F=1 for unitL2profile.
            da=1/b**2;dc=f/b**2+1/b;dd=1+2*f/b+f*f/b**2
            # dX0=-4k²dF. Bound residual of the differentiated Riccati
            # equation at dX0, then divide by its Hermitian separation.
            residual=da*x+f/b**2+6*k/b+x*dd+x*x*dc
            residual+=4*k*k*(1/b+3+f*f/b+2*x*f/b)
            relative_derivative_error=residual/(separation*4*k*k)
            metric_error=2*relative_derivative_error+relative_derivative_error**2+2*x*x+x**4
            records[label]=dict(relative_derivative_error=mp.nstr(relative_derivative_error,80),
                metric_absolute_error=mp.nstr(metric_error,80),
                metric_error_divided_by_kappa=mp.nstr(metric_error/k,80))
        if mp.mpf(records['ceiling']['metric_error_divided_by_kappa'])>=60:
            raise ArithmeticError('Actual leading-metric conservative ceiling failed')
        omega=2*mp.asinh(mp.sin(mp.pi/mp.mpf(2)**24))
        # Gaussian envelope width for the exact leading Mehler ground,
        # distinct from its TRANSFER eigenvalue gap.
        packet_width=mp.sinh(omega)
        return dict(exact_mode_geometry='Every spatial extent>2, unitL2 real linkprofile; meanPhi0/SU4I, auxiliarycharge1.',
            conservative_actual_tensorsea_metric_interval='1-60kappa <= G(A) <= 1+60kappa, uniformlyA andvolume for0<kappa<=1e-6.',
            evaluations=records,
            leading_mechanism='PositiveTF occupiedgraph X=-4k²F+remainder. Actual derivative remainder hasrelativebound from displayedSylvesterresidual. Metric=Tr[(I+XdagX)^-1 dXdag(I+XXdag)^-1dX]; exactN/beta=1/(16k^4) and ||dF||F²=1/beta give leadingONE. Factoredpositive ceiling bounds allk<=1e-6; NO numeric k1e-250 eigensystem.',
            conditional_leading_fundamental_Mehler_packet_width=mp.nstr(packet_width,80),
            conditional_original_FIXEDmeanbody_Gaussian_projection_Gram_leading=mp.nstr(mp.sqrt(packet_width/(packet_width+1)),80),
            conditional_original_FIXEDmeanbody_projection_cost_leading=mp.nstr(-mp.log(mp.sqrt(packet_width/(packet_width+1))),80),
            conditional_bare_creator_massive_component_Gram_leading=mp.nstr(1/(2*packet_width),80),
            scope='Metric theorem concerns actual instantaneous auxiliaryline, noGauss/Coul/Phi/activecarrier. Gaussian projection numbers are explicit LEADING reference-boundary calculations, NOT actualprepared macroGram, globalREALcover vacuum, fullK photonpole or bareB admission. Mehlerwidth sinhOmega is not Omega.')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True);args=parser.parse_args();start=time.time()
    # Strong canonical finite-k evidence; no numerical k1e-250 claims.
    cases=[actual_metric(k,b,omega,order) for k,b,omega in [('0.02','100',1.),('0.01','100',1.),('0.01','1000',1.),('0.01','1000',.1)] for order in (5,7)]
    result: dict[str,Any]=dict(cases=cases,interpretation='Barefield ladder and dressed lowphoton envelopecreator are different operators on an interacting sea line. Metric contribution is not suppressed by Naux; SF can suppress its massive canonical output. No physicalincoming packet normalization from Gaussian envelopealone.',
        source_unchanged=True,all_primitive_occupations_retained=True,macro_photon_or_source_pole_admitted=False,
        macro_uniform_auxiliary_metric=uniform_Riccati_metric_bound(),
        macro_leading_auxiliary_metric=leading_Riccati_metric_remainder(),
        derivation='Occupiedprojector first derivative comes from actual intrinsicC scalar/valence generator hC and its actual analytic Peierls jet; phase-invariant canonical Slater metric=sum|-+|²/gap². Tensor product multiplies metric by EXACT Naux, not a huge Fock allocation.',
        execution=dict(source_at_IMPORT=hashlib.sha256(SOURCE_BYTES).hexdigest(),source_bytes_at_IMPORT=SOURCE_BYTES.decode(),loop_at_IMPORT=hashlib.sha256(LOOP_BYTES).hexdigest(),loop_bytes_at_IMPORT=LOOP_BYTES.decode(),
            captured_primitive=loops.IMPORT_EPOCH,runtime_seconds=time.time()-start,executable=sys.executable,python=sys.version,numpy=np.__version__,
            startup_unchanged=Path(__file__).read_bytes()==SOURCE_BYTES and Path(loops.__file__).read_bytes()==LOOP_BYTES))
    inputs=json.dumps(dict(cases=[[r['kappa'],r['beta_theta'],r['reference_packet_omega'],r['quadrature_order']] for r in cases],mode_normalized=True,packet='NEW auxiliarysea-line Gaussian',macro_kappa='1e-250',interval_ceiling_kappa='1e-6',macro_L=2**24,geometry_for_leading_metric='all_extents>2',metric_bound_constants=[13,60],captured_loop=result['execution']['loop_at_IMPORT']),sort_keys=True,separators=(',',':'))
    result['execution'].update(input_json=inputs,input_sha256=hashlib.sha256(inputs.encode()).hexdigest())
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(output=args.output,source=result['execution']['source_at_IMPORT'],summary=[{k:c[k] for k in ('kappa','beta_theta','reference_packet_omega','quadrature_order','actual_whole_bare_field_creator_Gram','actual_UNCOMPRESSED_massive_auxiliary_creator_Gram')} for c in cases])))
if __name__=='__main__':main()
