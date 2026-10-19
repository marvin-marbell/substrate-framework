#!/usr/bin/env python3
"""Conditional bare6a full-chart SF bridge to pure-Abelian centre coordinates.

Set only normalY and colourLiexi to zero in chart MULTIPLICATION; retain actual
geometry, measure, physical Fock frame, heat, jointGauss and outsidecolumn.
This does not replace the microscopic law by a Wilson/Gaussian action. It
prices how the pureAbelian determinant submanifold can be used conditionally.
No macro preparation Gram, photon graph, physical sourcewindow or pole follows.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
if __name__=='__main__':
    for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import mpmath as mp
import numpy as np
from scipy.linalg import expm
import physical_c_bare_electric_kernel as bare
SOURCE_BYTES=Path(__file__).read_bytes();CORE_BYTES=Path(bare.__file__).read_bytes()
if hashlib.sha256(CORE_BYTES).hexdigest()!='6a66ca3a45b05e8df46cc1cec90aa775d97cd60341186639fdaf1dfd5fce9e64':
    raise ArithmeticError('Fullchart bridge requires exact frozen bare6a')


def left_factor_price(volume,count,k,frob,colour):
    db=k*(frob+frob*frob)+3*colour
    df=4*k*frob*(1+colour)+3*colour
    b0=1/(2*k)-3;bmax=1/(2*k)+3
    if db>=b0:raise ArithmeticError('Relative B conditioning unavailable')
    eta_b=-mp.log1p(-db/b0);eta=eta_b+2*mp.log1p(df/b0)
    condition=max(1/(b0-db),bmax+db)*(4+df)**2
    spread=2*mp.log(condition)+2*eta
    flow=eta*mp.expm1(spread/2)/spread
    logarithm=count*volume*(eta_b+eta+4*flow)
    return logarithm,dict(db=mp.nstr(db,70),df=mp.nstr(df,70),positive_B0=mp.nstr(b0,70),
        eta_h=mp.nstr(eta,70),onebody_LEFT_halfSF_flow=mp.nstr(flow,70),
        ALL_occupation_log_bound=mp.nstr(logarithm,70),occupation_range='0..4V')


def macro_bridge():
    with mp.workdps(180):
        volume=mp.mpf(2)**72;radius=mp.mpf('1e-1000');sigma=mp.mpf('1e-1500')
        # CartTr(TaTb)=4delta, t=Cart/sqrt8, Lie radius8sigma:
        # ||X||op<=||X||F=||xi||/sqrt2; ||exp(iX)-I||<=||X||op.
        colour=4*mp.sqrt(2)*sigma;cases=[]
        for label,frob,u in [('colour_only',mp.mpf(0),colour),('normal_and_colour',2*radius,colour)]:
            rows=[];total=mp.mpf(0)
            for name,count,k in [('active',mp.mpf(1),mp.mpf('.1256305951044483')),('aux',mp.mpf(10)**1150/16,mp.mpf('1e-250'))]:
                value,row=left_factor_price(volume,count,k,frob,u);total+=value;row.update(species=name);rows.append(row)
            eps=mp.expm1(total);delta=2*eps+eps*eps;nt=mp.mpf(2)**20;m=nt/2
            neutral=2*delta/(1-delta);tau=mp.exp(-m/nt)
            cases.append(dict(comparison=label,rows=rows,relative_chart_LEFT_SF_error_upper=mp.nstr(eps,80),
                COMPLETE_K_relative_norm_error_upper=mp.nstr(delta,80),
                SAME_nonzero_smooth_boundary_normalized_neutral_vector_error_upper=mp.nstr(2*m*neutral/tau,80)))
        return dict(cases=cases,colour_Lie_operator_radius=mp.nstr(colour,80),original_Wilson_source_operator_error_upper=mp.nstr(colour,80),
            actual_normalizer_requirement='Same nonzero smooth boundary for BOTH reference and actual; Delta1/Nt. No actual interacting preparationGram is inferred.',
            original_source='PhysicalPsi=[e^(iQtheta00)U00;I]/2 unchanged. Source comparison sets ONLYxi=0; jointfourRef creation error<=||U-U0|| since eachcolumn<=||U-U0||/2. No quarter norm or sourcegate.',
            geometry='Same actualCart/SU4/REAL coordinates, FPcoarea/Haar, physicalGamma(V) connection, allZ4centre/cycle characters, exactREALGauss. Referencepotential is evaluated atPhi=C andU=zI in SAME bodyframe; z canbe absorbed into Abelianphases. Geometry/heat/source are NOT silently replaced by flat ones.',
            factor='F1=[M A0,B], F0=[A0,B], M=SF(Y,xi)SF(0,0)^-1 onlyonchart. Outsidecolumn B is unchanged. DeltaK=D Kchart0+Kchart0 Ddag+D Kchart0 Ddag. Field/Gauss factors remaincommon contractions; NO fullF1=MfullF0 assertion.',
            derivation='DeltaB<=k(2r+4r²)+3u; DeltaF<=8kr(1+u)+3u. B/L conditioning precedes log. IntrinsicC averaging iscontractive; all0..4V occupationgrades included with scalarpref. LEFT halfSF Duhamel flow uses actual spectralspread, not norm(dh) alone.',
            limit='SFbridge does not prove literal Gauss pathdet mapping, aux boundary overlap, interacting fullphoton graphprice, chargednorm/sourceM0, physicalPe4 or physicalpole. Current/filter/packet acceptance remains separate.',macro_source_window_admitted=False)


def actual_alloccupation_smoke():
    p=bare.BareElectricParameters();law,_,_=bare.reference.native_law(bare.reference_parameters(p))
    phases=np.array([[.17,-.23,.31],[-.11,.19,-.29]])
    colour=np.broadcast_to(np.eye(4),(2,3,4,4)).astype(complex).copy()
    t=bare.alpha.matrix_unit_basis()[7]/math.sqrt(8);xi=1e-5
    colour[0,0]=expm(1j*xi*t)
    remainder=np.zeros((2,4,4,4));remainder[:,0]=1e-5*bare.alpha.matrix_unit_basis()[2]
    reference=law.potential(phases,np.zeros_like(remainder),np.broadcast_to(np.eye(4),(2,3,4,4)).copy())
    actual=law.potential(phases,remainder,colour)
    actual_frob=max(float(np.linalg.norm(y)) for y in remainder)
    actual_colour=float(np.linalg.norm(colour[0,0]-np.eye(4),2))
    with mp.workdps(70):
        logarithm,_=left_factor_price(mp.mpf(2),mp.mpf(1),mp.mpf(str(law.kappa)),mp.mpf(str(actual_frob)),mp.mpf(str(actual_colour)))
        upper=float(mp.expm1(logarithm))
    rows=[]
    for n in range(9):
        h0,_=bare.alpha.second_quantized(reference['h'],n);h1,_=bare.alpha.second_quantized(actual['h'],n)
        sf0=expm((h0+reference['pref']*np.eye(len(h0)))/2);sf1=expm((h1+actual['pref']*np.eye(len(h1)))/2)
        from scipy.linalg import solve
        relative=solve(sf0.T,sf1.T).T-np.eye(len(h0))
        observed=float(np.linalg.norm(relative,2))
        if observed>upper:raise ArithmeticError('Actual alloccupation LEFT factor violates derived bridge')
        rows.append(dict(occupation=n,canonical_carriers=len(h0),actual_LEFT_M_minus_I_norm=observed,proved_upper=upper))
    return dict(actual_parameters=dict(frob=actual_frob,colour_operator_delta=actual_colour,kappa=law.kappa),rows=rows,
        scope='Actual intrinsicC scalar+valence all256 totalcarriers across0..8 onV2 at diagnosticfiniteperturbation; not numericalmacroPhi/sigma substitution.')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True);args=parser.parse_args();start=time.time()
    result=dict(macro_bridge=macro_bridge(),actual_alloccupation_smoke=actual_alloccupation_smoke())
    inputs=json.dumps(dict(core=hashlib.sha256(CORE_BYTES).hexdigest(),normal_radius='1e-1000',colour_sigma='1e-1500',V=2**72,Nt=2**20),sort_keys=True,separators=(',',':'))
    result['execution']=dict(source_at_IMPORT=hashlib.sha256(SOURCE_BYTES).hexdigest(),source_bytes_at_IMPORT=SOURCE_BYTES.decode(),
        core_at_IMPORT=hashlib.sha256(CORE_BYTES).hexdigest(),core_bytes_at_IMPORT=CORE_BYTES.decode(),input_json=inputs,input_sha256=hashlib.sha256(inputs.encode()).hexdigest(),
        runtime_seconds=time.time()-start,python=sys.version,executable=sys.executable,startup_unchanged=Path(__file__).read_bytes()==SOURCE_BYTES and Path(bare.__file__).read_bytes()==CORE_BYTES)
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(output=args.output,source=result['execution']['source_at_IMPORT'],cases=result['macro_bridge']['cases'])))
if __name__=='__main__':main()
