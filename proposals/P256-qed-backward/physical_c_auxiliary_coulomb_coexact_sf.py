#!/usr/bin/env python3
"""Actual coexact mixed sea/Coul kernel plus full SF C SF output control.

The NF tensor character retains every occupation grade; its exponent is
computed without allocating an NF Fock space.  Full Tminus is retained in
output norms.  Finite scaled integerQ controls are not retagged as macroQ.
These are conditional pure-Abelian fiber controls, not a RAWB/Khalf source,
complete realGauss/centre projection, or a physical pole claim.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from decimal import Decimal
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
import numpy as np
from numpy.typing import NDArray
from scipy.linalg import eigh
from scipy.special import ndtri
from scipy.stats import qmc
import physical_c_auxiliary_coulomb_sf_output as sf
coul=sf.coul
SOURCE_BYTES=Path(__file__).read_bytes();SF_BYTES=sf.SOURCE_BYTES


def proposal_data(b: dict,s: dict,nf: int,u: float) -> dict:
    prior=np.diag(2*u/b['gauge_eigenvalues']);cov=nf*s['cov']
    proposal=np.linalg.inv(np.linalg.inv(prior)+b['gauge_frame'].T@cov@b['gauge_frame'])
    e,r=eigh(proposal)
    return dict(map=(r*np.sqrt(e)).T@b['gauge_frame'].T,cov=cov,
        jacobian=float(np.sqrt(np.linalg.det(proposal)/np.linalg.det(prior))))


def single_character(left: dict,right: dict,nf: int,eta: np.ndarray,cov: np.ndarray,background: np.ndarray) -> NDArray[np.complex128]:
    out: NDArray[np.complex128]=np.empty(len(eta),np.complex128)
    for first in range(0,len(eta),256):
        x=eta[first:first+256]
        log=nf*coul.graph_character_log(left,right,x)+.5*np.einsum('ni,ij,nj->n',x,cov,x)
        out[first:first+256]=np.exp(log+1j*(x@background))
    return out


def coexact_control(bits: int) -> dict:
    b=coul.graph_data((4,3,1));nf=1000000;k=.04;beta_decimal=16*Decimal(nf)*Decimal(str(k))**4
    beta=float(beta_decimal);contract=sf.scaled_integer_charge(str(beta_decimal));q=int(contract['Q'])
    profile=b['a']/np.linalg.norm(b['a']);step=.03
    states=[coul.sea(k,b,x*profile/math.sqrt(beta)) for x in (-step,0.,step)]
    middle=states[1];rank=len(b['gauge_eigenvalues']);source_pair=np.zeros(len(b['targets']));source_pair[0]=1;source_pair[1]=-1
    background=q*source_pair
    uniforms=qmc.Sobol(2*rank,scramble=True,rng=np.random.default_rng(222942)).random_base2(bits)
    if np.any((uniforms<=0)|(uniforms>=1)):raise ArithmeticError('Sobol inverse-normal endpoint; no clipped probability floor')
    normal=ndtri(uniforms);proposal=proposal_data(b,middle,nf,.02)
    eta=normal[:,:rank]@proposal['map'];zeta=normal[:,rank:]@proposal['map']
    cov=proposal['cov'];jacobian=proposal['jacobian'];data=sf.complement_data(middle)
    zero=np.zeros_like(background)
    left=single_character(middle,middle,nf,eta,cov,zero);right=single_character(middle,middle,nf,zeta,cov,zero)
    unsigned_den=0j;signed_den=0j;unsigned_num=0j;signed_num=0j
    absolute_weight=float(np.mean(abs(left*right)))
    for signa,signz in itertools.product((1,-1),repeat=2):
        a,z=signa*eta,signz*zeta
        base=(left if signa==1 else left.conjugate())*(right if signz==1 else right.conjugate())
        phase=np.exp(1j*((a+z)@background));correction=np.empty(len(a),complex)
        for first in range(0,len(a),256):correction[first:first+256]=nf*sf.complement_log(data,a[first:first+256],z[first:first+256])
        unsigned_den+=np.mean(base)/4;signed_den+=np.mean(base*phase)/4
        unsigned_num+=np.mean(base*np.expm1(correction))/4;signed_num+=np.mean(base*phase*np.expm1(correction))/4
    unsigned_increment=unsigned_num/unsigned_den;signed_increment=signed_num/signed_den
    gram: NDArray[np.complex128]=np.zeros((3,3),np.complex128)
    bare: NDArray[np.complex128]=np.zeros((3,3),np.complex128)
    for i,j in itertools.combinations_with_replacement(range(3),2):
        forward=single_character(states[i],states[j],nf,eta,cov,background)
        reversed_eta=single_character(states[i],states[j],nf,-eta,cov,background)
        gram[i,j]=jacobian*np.mean((forward+reversed_eta)/2);gram[j,i]=gram[i,j].conjugate()
        bare[i,j]=np.exp(nf*coul.graph_character_log(states[i],states[j],zero[None])[0]);bare[j,i]=bare[i,j].conjugate()
    normalized=gram/np.sqrt(np.outer(gram.diagonal().real,gram.diagonal().real))
    g0=jacobian*np.mean(left.real);gb=gram[1,1];intermediate={}
    for u in (.01,.04):
        p=proposal_data(b,middle,nf,u);x=normal[:,:rank]@p['map']
        value=p['jacobian']*np.mean(single_character(middle,middle,nf,x,p['cov'],background).real)
        intermediate[str(u)]=value
    half=intermediate['0.01'];square=intermediate['0.04']
    encode=lambda x:[float(x.real),float(x.imag)]
    return dict(bits=bits,samples=len(normal),shape=[4,3,1],NF=nf,k=k,beta=beta,UA=.02*beta,
        charge_contract=contract,source_pair=source_pair.tolist(),background=background.tolist(),
        profile_unitL2=float(np.linalg.norm(profile)),profile_divergence=float(np.linalg.norm(b['incidence']@profile.ravel())),
        g0=encode(g0),signed_g=encode(gb),signed_normalized_Gram_ratio=encode(gb/g0),
        signed_single_HS_cancellation_price=float(jacobian*np.mean(abs(left))/abs(gb)),
        signed_double_HS_cancellation_price=absolute_weight/abs(signed_den),
        unsigned_fullSF_relative_norm2_increment=encode(unsigned_increment),
        signed_fullSF_relative_norm2_increment=encode(signed_increment),
        unsigned_fullSF_output_outside_fraction=float((unsigned_increment/(1+unsigned_increment)).real),
        signed_fullSF_output_outside_fraction=float((signed_increment/(1+signed_increment)).real),
        signed_doubleHS_projected_channel_norm2=encode(jacobian**2*signed_den),
        intermediate_Rhalf_output_outside_fraction=1-abs(half)**2/gb.real,
        intermediate_C_output_outside_fraction=1-abs(gb)**2/square.real,
        charged_Coul_Gram_real=gram.real.tolist(),charged_Coul_Gram_imag=gram.imag.tolist(),
        normalized_charged_Coul_Gram_real=normalized.real.tolist(),normalized_charged_Coul_Gram_imag=normalized.imag.tolist(),
        unfiltered_Gram_real=bare.real.tolist(),
        canonicalA_nodes=[-step,0.,step],normalized_charged_transverse_metric=-math.log(abs(normalized[0,2]))/(2*step)**2,
        unfiltered_transverse_metric=-math.log(abs(bare[0,2]))/(2*step)**2,
        full_SF_endpoint_loglambda=[nf*x['loglambda'] for x in states],
        measurement_contract='Common actual-Cov importance Sobol normals, exact measure-preserving independent eta/zeta sign orbits, and antithetic mixed-Gram eta. These remove odd quadrature noise, NOT occupationgrades or source/pole channels. QMC convergence evidence, NOT a certified quadrature bound. Whole SF scalar+valence sqrt(lambda_i lambda_j) retained before normalization. Full output retains Tminus through expm1; no subtraction of unit-sized norms.',
        geometry_scope='Actual nonzero coexact curl on periodic4x3x1; short/self edges and oddy are explicitly finitecontrols, NOT the even macrogeometry proof.')


def macro_preparation_cost() -> dict:
    screening=sf.macro_signed_screening()
    with sf.mp.workdps(210):
        mp=sf.mp;beta=mp.mpf('1e150');u=mp.mpf('.02');dimension=mp.mpf(2)**72-1
        gaussian_log=-dimension*mp.log1p(2*u*beta)/2
        ratio_error=mp.mpf(screening['normalized_signed_ratio_absolute_error_bound'])
        relative_error=ratio_error/(2+ratio_error)
        ratio=mp.mpf(screening['Gaussian_signed_Gram_ratio'])
        lower=gaussian_log+mp.log1p(-relative_error);upper=gaussian_log+mp.log1p(relative_error)
        signed_lower=lower+mp.log(ratio-ratio_error);signed_upper=upper+mp.log(ratio+ratio_error)
        return dict(unsigned_Gaussian_log_Gram=mp.nstr(gaussian_log,90),
            actual_unsigned_log_Gram_interval=[mp.nstr(lower,90),mp.nstr(upper,90)],
            actual_signed_log_Gram_interval=[mp.nstr(signed_lower,90),mp.nstr(signed_upper,90)],
            unsigned_relative_Gram_error_bound=mp.nstr(relative_error,70),
            additional_static_source_log_cost=-float(mp.log(ratio)),
            identity='g0_Gaussian=(1+2u beta)^[-(V-1)/2], from actual collective rhoCov/beta=L and exact HS priorCov=2uLplus. Same bounded wrapped-sheet/saddle/actual-character error as the signed screening proof.',
            cost_contract='Absolute auxiliary Coul preparation is extremely small. Only the charged/neutral ratio is O(1). This is NOT a free Gram, actual neutral lambda_N bound, RAWB/Khalf preparation, or source/pole admission.')


def retained_matrix_control() -> dict:
    b=coul.graph_data((4,3,1));s=coul.sea(.04,b,.007*b['a']);data=sf.complement_data(s)
    rng=np.random.default_rng(222943);eta=rng.normal(0,.2,(7,len(b['targets'])));zeta=rng.normal(0,.2,eta.shape)
    parity=np.max(abs(np.exp(coul.graph_character_log(s,s,-eta))-np.exp(coul.graph_character_log(s,s,eta)).conjugate()))
    expected=np.exp(coul.graph_character_log(s,s,eta)+coul.graph_character_log(s,s,zeta)+sf.complement_log(data,eta,zeta))
    direct=[];wrong=[]
    normalization=np.linalg.det(data['Tplus'])
    for a,z in zip(eta,zeta,strict=True):
        Qe=np.diag(np.repeat(np.exp(1j*a),4));Qz=np.diag(np.repeat(np.exp(1j*z),4))
        uncentered=np.linalg.det(s['W'].conj().T@Qe@s['T']@Qz@s['W'])/normalization
        direct.append(np.exp(-2j*np.sum(a+z))*uncentered);wrong.append(uncentered)
    return dict(seed=222943,real_cover_eta=eta.tolist(),real_cover_zeta=zeta.tolist(),
        actual_centered_fulltransfer_factorization_error=float(np.max(abs(np.array(direct)-expected))),
        exact_single_character_sign_symmetry_error=float(parity),
        omitted_center_phase_falsifier_error=float(np.max(abs(np.array(wrong)-expected))),
        whole_scalar_pref=s['pref'],whole_loglambda=s['loglambda'],
        scope='Independent direct48x48 onebody T and centered24x24 occupied determinant versus retained-positive-plus-negative factorization; arbitrary nonmeanfree eta/zeta, NOT a source-text/mock test.')


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument('--capsule',required=True);args=parser.parse_args();start=time.time()
    rows=[coexact_control(bits) for bits in (10,12,14,16)]
    execution=dict(source_sha256=hashlib.sha256(SOURCE_BYTES).hexdigest(),source_bytes_at_IMPORT=SOURCE_BYTES.decode(),
        SF_source_sha256=hashlib.sha256(SF_BYTES).hexdigest(),SF_source_bytes_at_IMPORT=SF_BYTES.decode(),
        Coul_source_sha256=hashlib.sha256(coul.SOURCE_BYTES).hexdigest(),Coul_source_bytes_at_IMPORT=coul.SOURCE_BYTES.decode(),
        loops_source_bytes_at_IMPORT=coul.loops.SOURCE_BYTES.decode(),core_sha256=hashlib.sha256(coul.loops.CORE_BYTES).hexdigest(),core_source_bytes_at_IMPORT=coul.loops.CORE_BYTES.decode(),
        frozen_dependencies=coul.loops.IMPORT_EPOCH,root_charge_source_sha256=hashlib.sha256(coul.ROOT_CHARGE_BYTES).hexdigest(),root_charge_source_bytes_at_IMPORT=coul.ROOT_CHARGE_BYTES.decode(),
        argv=sys.argv,python=sys.executable,python_version=platform.python_version(),numpy=np.__version__,
        input_parameters=dict(shape=[4,3,1],NF=1000000,k=.04,u=.02,bits=[10,12,14,16],seed=222942,canonicalA_step=.03),
        thread_environment={name:os.environ.get(name) for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')},elapsed_seconds=time.time()-start)
    report=dict(rows=rows,macro_preparation_cost=macro_preparation_cost(),retained_matrix_control=retained_matrix_control(),
        scope='Actual conditional coexact pureAbelian auxiliary-sea/Coul/SF output. Complete Gauss/centre, outside/Phi/noncentralSU4, ORIGINALfixedbody RAWB/Khalf source/tangent and physicalpole remain independent debts.')
    Path(args.capsule).write_text(json.dumps(dict(report=report,execution=execution),indent=2)+'\n')
    print(json.dumps(dict(capsule=args.capsule,report=report,elapsed_seconds=execution['elapsed_seconds']),indent=2),flush=True)


if __name__=='__main__':main()
