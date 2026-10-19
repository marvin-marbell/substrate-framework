#!/usr/bin/env python3
"""Relative completed canonical/Gauss/source traces on longer native rings.

This is a NEW native epoch. Reviewed Source741, frozen Source2a/Root6911 and
all prior capsules remain unchanged. No macro Pe/source/receiver admission.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import marshal
import math
import os
from pathlib import Path
import sys
import time
from typing import Any
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
ROOT=Path(__file__).resolve().parent
OWN_BYTES=Path(__file__).read_bytes()
LOADED_MAIN_CODE=sys._getframe().f_code
MAIN_CODE_MATCH=LOADED_MAIN_CODE==compile(OWN_BYTES,LOADED_MAIN_CODE.co_filename,'exec',dont_inherit=True,optimize=sys.flags.optimize)
if not MAIN_CODE_MATCH:raise ArithmeticError('Loaded relative-ring source differs from captured bytes')
import numpy as np
from scipy.linalg import eigh
from scipy.stats import qmc
import covariant_canonical_gauss_boundary_kernel as supplier
PLAN='issue-222-induced-photon-delta13-continuation-20261008'
REVISION='7650a3636f3d39a74243a986afb1683be8fae895c3f5afb77720b05330363a51'
SUPPLIER_HASH='741fbc1f196061cb7f0e2c479770e992304ac2de90492f002ef4cbd931315d50'


def require(ok: bool,reason: str) -> None:
    if not ok:raise ArithmeticError(reason)


def canonical_polynomial(matrices: np.ndarray,degree: int) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    """ALL fixed-N coefficients, using a scaled characteristic polynomial.

    Scaling is matrix scalar arithmetic only; no occupation is removed and
    no Fermi/Slater projector or inverse(I+M) is substituted.
    """
    scale=np.maximum(np.max(np.sum(abs(matrices),axis=2),axis=1),1.)
    normalized=matrices/scale[:,None,None]
    roots=np.linalg.eigvals(normalized)
    polynomial=np.zeros((len(matrices),degree+1),complex);polynomial[:,0]=1
    for root in roots.T:
        for n in range(degree,0,-1):polynomial[:,n]+=root*polynomial[:,n-1]
    return polynomial,scale,normalized


def source_diagonal(matrices: np.ndarray,middle: np.ndarray,number: int,one: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    polynomial,scale,normalized=canonical_polynomial(matrices,number)
    right=middle@one
    applied=right.copy()
    # Apply the polynomial adjugate to the actual Ref columns. There is no
    # reason to allocate/multiply its full4V by4V matrix for each degree.
    for degree in range(1,number+1):
        applied=polynomial[:,degree,None,None]*right-normalized@applied
    diagonal=np.einsum('ir,bir->br',one.conj(),applied,optimize=True)*scale[:,None]**number
    return diagonal,polynomial[:,number]*scale**number


def source_minor(matrices: np.ndarray,middle: np.ndarray,number: int,one: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    diagonal,coefficient=source_diagonal(matrices,middle,number,one)
    return np.sum(diagonal,axis=1),coefficient


def ring_half(gamma: np.ndarray,sites: int,kappa: float,theta: float,y: float) -> tuple[np.ndarray,float,np.ndarray]:
    """Physical Root4V SF at a uniform absolute holonomy theta.

    Every link has A=theta/L, so the source must retain its actual integer
    path phase. A symmetric gamma-index normal shear mixes the full physical
    Clifford components; there is no invariant-spin-plane truncation here.
    Returns (sqrt(T4), sqrt(detBplus), T4): the scalar belongs to ONE SF
    halfstep. The immutable supplier returns detBplus for a WHOLE step.
    """
    shift=np.zeros((sites,sites),complex)
    for x in range(sites):shift[x,(x+1)%sites]=np.exp(1j*theta/sites)
    links=(np.kron(shift,np.eye(4)),np.eye(4*sites),np.eye(4*sites))
    shear=np.array([[.7,0.,0.,.4],[0.,.13,0.,0.],[0.,0.,-.11,0.],[.4,0.,0.,-.3]])
    normal=[y*np.kron(np.eye(sites),sum((shear[mu,nu]*gamma[nu] for nu in range(4)),np.zeros((4,4),complex))) for mu in range(4)]
    half,whole_prefactor,transfer=supplier.physical_spatial_onebody(gamma,kappa,links,normal,np.eye(4*sites))
    return half,math.sqrt(whole_prefactor),transfer


def integer_source_paths(sites: int) -> tuple[np.ndarray,np.ndarray]:
    paths=np.zeros((sites,sites),int)
    for x in range(1,sites):
        # Positive tie preserves the reviewed two-cell path convention.
        if 2*x<sites:paths[x,:x]=-1
        else:paths[x,x:]=1
        divergence=paths[x]-np.roll(paths[x],1)
        expected=np.zeros(sites,int);expected[x]=1;expected[0]=-1
        require(bool(np.array_equal(divergence,expected)),'Every coherent body/Ref component has a literal INTEGER Gauss path')
    return paths,paths.sum(axis=1)/sites


def source_packet(gamma: np.ndarray,sites: int,kappa: float,width: float) -> tuple[np.ndarray,dict[str,Any]]:
    _,_,T=ring_half(gamma,sites,kappa,0.,0.)
    values,vectors=eigh(T);stable=vectors[:,values<1]
    signed=np.array([x if 2*x<sites else x-sites for x in range(sites)])
    packet=np.exp(-signed*signed/(4*width*width));packet/=np.linalg.norm(packet)
    seed=np.kron(packet[:,None],np.eye(4))
    one=stable@stable.conj().T@seed;one/=np.linalg.norm(one)
    paths,harmonic=integer_source_paths(sites)
    return one,dict(Gaussian_amplitudes=packet,positive_T4_band_count=stable.shape[1],
        canonical_source_coefficients=one,integer_paths=paths,source_harmonic_generators=harmonic,
        source_path_max_L1=int(np.max(np.sum(abs(paths),axis=1))),packet_width_lattice_units=width,
        scope='One declared coherent canonical positive-band packet with all4 Ref holes. Gaussian input is projected by the REAL full-ring T4 stable band before literal paths, never replaced by bare gamma0.')


def gauss_samples(sites: int,copies: int,method: str,points_power: int,seed: int) -> tuple[np.ndarray,np.ndarray]:
    dimensions=2*(sites-1)
    if method=='exact':
        # One insertion has local powers0..4*(1+copies), target2*(1+copies).
        quadrature=4*(1+copies)+1
        points=np.asarray(list(itertools.product(range(quadrature),repeat=dimensions)),float)/quadrature
    else:points=qmc.Sobol(dimensions,scramble=True,rng=seed).random_base2(points_power)
    angles=2*math.pi*points
    first=np.zeros((len(points),sites));second=np.zeros_like(first)
    first[:,1:]=angles[:,:sites-1];second[:,1:]=angles[:,sites-1:]
    gauges=np.stack((np.repeat(np.exp(1j*first),4,axis=1),np.repeat(np.exp(1j*second),4,axis=1)),axis=1)
    # Strip one auxiliary flavour's EXACT background BEFORE its integer power.
    background=np.exp(-2j*(first.sum(axis=1)+second.sum(axis=1)))
    return gauges,background


def common_vacuum_scale(gamma: np.ndarray,sites: int,kappa: float) -> float:
    """Largest neutral SF-HALF eigenvalue at the ONE common flat reference."""
    _,prefactor,T=ring_half(gamma,sites,kappa,0.,0.)
    values=eigh(T,eigvals_only=True)
    return prefactor*math.exp(.5*float(np.sum(np.log(values[-2*sites:]))))


def centre_ref_control(sites: int,copies: int) -> dict[str,Any]:
    roots=np.exp(.5j*math.pi*np.arange(4))
    number=2*sites*(1+copies)
    neutral=complex(np.mean(roots**(number%4)*roots**4))
    charged=complex(np.mean(roots**((number+1)%4)*roots**3))
    require(abs(neutral-1)<1e-10 and abs(charged-neutral)<1e-10,
        'This declared canonical native epoch must have a nonzero globalZ4 Ref4/Ref3-compensated sector')
    return dict(actual_global_Z4_neutral_weight=neutral,actual_global_Z4_charged_weight=charged,
        total_neutral_canonical_particles=number,neutral_Ref_particles=4,charged_Ref_particles=3,
        exact_neutral_centre_exponent_mod4=(number+4)%4,exact_charged_centre_exponent_mod4=(number+1+3)%4,
        source_global_centre_charge_compensation_error=float(abs(neutral-charged)))


def complete_pair(gamma: np.ndarray,sites: int,body: float,aux: float,copies: int,theta: float,
                  coordinates: np.ndarray,normal_heat: np.ndarray,one: np.ndarray,harmonic: np.ndarray,
                  gauges: np.ndarray,background: np.ndarray,batch: int,
                  weights: np.ndarray | None=None,auxiliary_reference: float=1.) -> dict[str,Any]:
    number=2*sites
    body_scale=common_vacuum_scale(gamma,sites,body);aux_scale=common_vacuum_scale(gamma,sites,aux)
    body_halves=[];aux_halves=[];body_pref=[];aux_pref=[]
    for y in coordinates:
        h,p,_=ring_half(gamma,sites,body,theta,float(y));body_halves.append(h);body_pref.append(p/body_scale)
        h,p,_=ring_half(gamma,sites,aux,theta,float(y));aux_halves.append(h);aux_pref.append(p/aux_scale)
    dressed=np.exp(1j*theta*np.repeat(harmonic,4))[:,None]*one
    all_columns=np.column_stack((dressed,one));ref_columns=one.shape[1]
    neutral=0j;source=0j;undressed=0j;aux_min_real=math.inf;aux_max_imag=0.;negative_terms=0;term_count=0
    norm_phase=np.linalg.norm(dressed-one,2)
    for z in range(len(coordinates)):
        for w in range(len(coordinates)):
            hz=body_halves[z];hw=body_halves[w];az=aux_halves[z];aw=aux_halves[w]
            heat=float(normal_heat[z,w]**2)
            # L and M each have TWO SF halfsteps; retain all FOUR half scalars.
            body_factor=(body_pref[z]*body_pref[w])**2;aux_factor=(aux_pref[z]*aux_pref[w])**2
            for start in range(0,len(gauges),batch):
                pair=gauges[start:start+batch];bg=background[start:start+batch]
                weight=1. if weights is None else weights[start:start+batch]
                g=pair[:,0];r=pair[:,1]
                # The same polynomial closes the neutral cyclic coefficient
                # and the true charged minor: e_N(L M)=e_N(M L).
                L=hw[None,:,:]@(g[:,:,None]*hw[None,:,:])
                M=hz[None,:,:]@(r[:,:,None]*hz[None,:,:])
                diagonal,coefficient=source_diagonal(M@L,M,number,all_columns)
                minor=np.sum(diagonal[:,:ref_columns],axis=1)
                deleted=np.sum(diagonal[:,ref_columns:],axis=1)
                neutral_body=coefficient*body_factor
                auxC=aw[None,:,:]@(g[:,:,None]*(aw@az)[None,:,:]*r[:,None,:])@az[None,:,:]
                ac,ascale,_=canonical_polynomial(auxC,number)
                stripped=aux_factor*ac[:,number]*ascale**number*bg
                aux_min_real=min(aux_min_real,float(np.min(stripped.real)))
                aux_max_imag=max(aux_max_imag,float(np.max(abs(stripped.imag))))
                auxiliary=(stripped/auxiliary_reference)**copies
                # Neutral/source active background is2 per site; spectator
                # backgrounds have already been stripped separately per flavour.
                neutral_terms=neutral_body*bg*auxiliary
                neutral+=heat*np.sum(weight*neutral_terms)/len(gauges)
                source+=heat*np.sum(weight*body_factor*minor*bg*auxiliary)/len(gauges)
                undressed+=heat*np.sum(weight*body_factor*deleted*bg*auxiliary)/len(gauges)
                negative_terms+=int(np.count_nonzero(neutral_terms.real<0));term_count+=len(neutral_terms)
    return dict(neutral_positive_completed_trace=neutral,source_positive_completed_trace=source,
        wrong_deleted_integer_source_phase_trace=undressed,actual_source_open_phase_operator_norm=float(norm_phase),
        stripped_single_auxiliary_character_real_min=aux_min_real,stripped_single_auxiliary_character_imag_max=aux_max_imag,
        individual_neutral_character_negative_real_fraction=negative_terms/term_count,
        common_neutral_SF_half_vacuum_scale_body=body_scale,common_neutral_SF_half_vacuum_scale_auxiliary=aux_scale,
        ONE_common_auxiliary_canonical_reference_scalar=auxiliary_reference,
        importance_reweighting_used=weights is not None,
        normalization='ONE common neutral SF-HALF vacuum constant per species is divided from EACH of the FOUR SF halves in BOTH neutral/charged two-clock words. Scalar product is (p_half_z*p_half_w)^2/W_half^4, identical to the historical correctly grouped whole-step scalars. No charged own-max normalization.')


def characterize(rows: list[dict[str,Any]],method: str) -> dict[str,Any]:
    neutral:np.ndarray=np.array([complex(row['neutral_positive_completed_trace']) for row in rows],dtype=np.complex128)
    source:np.ndarray=np.array([complex(row['source_positive_completed_trace']) for row in rows],dtype=np.complex128)
    realN=np.real(neutral);realS=np.real(source);imagN=np.imag(neutral);imagS=np.imag(source)
    meanN=float(np.mean(realN));meanS=float(np.mean(realS))
    require(bool(realN.min()>0 and realS.min()>0),'A completed positive sector must not silently become zero/negative')
    if method=='exact':
        require(bool(np.max(abs(imagN))<1e-8*realN.min() and np.max(abs(imagS))<1e-8*realS.min()),
            'Exact tensor quadrature must complete the signed Gauss integral to a real positive trace')
    n=realN/meanN;s=realS/meanS
    return dict(neutral_relative_density=n,source_relative_density=s,
        neutral_relative_Fourier=np.fft.fft(n)/len(n),source_relative_Fourier=np.fft.fft(s)/len(s),
        neutral_relative_common_shift_variation=float(np.ptp(n)),source_relative_common_shift_variation=float(np.ptp(s)),
        actual_positive_source_to_neutral_two_clock_ratio=realS/realN,
        neutral_positive_mean=meanN,source_positive_mean=meanS,
        maximum_neutral_imaginary_fraction=float(np.max(abs(imagN))/meanN),
        maximum_source_imaginary_fraction=float(np.max(abs(imagS))/meanS),
        Fourier_scope='DFT on the stated finite cyclic grid; these are alias-folded sampled coefficients, not continuum Fourier coefficients or a priced theta-Haar quadrature.',
        meaning='Positive FULL canonical/Gauss/Phi two-clock diagonal field kernel, normalized by its OWN completed common-shift mean. Not a Gaussian action ratio, grandcanonical Z, Pe window or the macro125-clipped prepared source.')


def full_component_crosscheck(gamma: np.ndarray) -> dict[str,Any]:
    sites=2;body=.1248905293249106;aux=.04
    h,p,_=ring_half(gamma,sites,body,.31,.00013);a,pa,_=ring_half(gamma,sites,aux,.31,.00013)
    states=supplier.occupations(8,4)
    rng=np.random.default_rng(222)
    angles=rng.normal(size=(2,sites));g=[np.diag(np.repeat(np.exp(1j*x),4)) for x in angles]
    C=h@g[0]@h@h@g[1]@h
    poly,scale,_=canonical_polynomial(C[None,:,:],4)
    direct=np.trace(supplier.exterior(C,states));coefficient=poly[0,4]*scale[0]**4
    error=float(abs(direct-coefficient)/max(1.,abs(direct)))
    require(error<1e-10,'Efficient canonical polynomial keeps ALL70 occupations')
    one,_=source_packet(gamma,sites,body,.8)
    L=h@g[0]@h;M=h@g[1]@h
    numerical,_=source_minor((M@L)[None,:,:],M[None,:,:],4,one)
    exact=np.trace(supplier.inserted_minor(L,M,np.eye(8),4,one,one))
    minor_error=float(abs(numerical[0]-exact)/max(1.,abs(exact)))
    require(minor_error<1e-10,'Source recurrence remains the true canonical charged minor, not an inverse or forwarding echo')
    charged=supplier.occupations(8,5)
    creators=supplier.creation(one,states,charged)
    active=supplier.exterior(L,states);propagator=supplier.exterior(M,charged)
    literal=sum((np.trace(active@creator.conj().T@propagator@creator) for creator in creators),0j)
    CAR_error=float(abs(numerical[0]-literal)/max(1.,abs(literal)))
    require(CAR_error<1e-10,'The actual70/56 full4-component CAR word must equal the polynomial source minor')
    # Whole sum of principal minors is independently executed above; these
    # controls precede larger rings where no dense Fock array is allocated.
    return dict(full_physical_modes=8,canonical_number=4,canonical_components=70,
        efficient_vs_actual_full_exterior_trace_relative_error=error,
        source_polynomial_vs_actual_principal_minor_recurrence_relative_error=minor_error,
        source_polynomial_vs_literal_full70_56_CAR_relative_error=CAR_error,
        full4_component_normal_shear=True)


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--rings',type=int,nargs='+',default=[2,3])
    parser.add_argument('--shifts',type=int,default=16)
    parser.add_argument('--aux-kappa',type=float,default=.04)
    parser.add_argument('--aux-copies',type=int,default=1)
    parser.add_argument('--method',choices=('exact','sobol'),default='exact')
    parser.add_argument('--points-power',type=int,default=12)
    parser.add_argument('--seeds',type=int,nargs='+',default=[222])
    parser.add_argument('--normal-amplitude',type=float,default=.0002)
    parser.add_argument('--packet-width',type=float,default=.8)
    parser.add_argument('--batch',type=int,default=256)
    args=parser.parse_args();start=time.monotonic()
    require(hashlib.sha256(supplier.OWN_BYTES).hexdigest()==SUPPLIER_HASH,'Reviewed Source741 is an immutable supplier')
    parameters=supplier.Parameters();gamma,_=supplier.phi_frame(parameters)
    body=max(1/(2*(math.expm1(a*m)+4)) for a in parameters.spacings for m in parameters.masses)
    _,ground,H,half=supplier.normal_heat();heat=half@half
    coordinates=np.linspace(-args.normal_amplitude,args.normal_amplitude,len(ground))
    check=full_component_crosscheck(gamma)
    shifts=2*math.pi*np.arange(args.shifts)/args.shifts;cases=[]
    for sites in args.rings:
        centre=centre_ref_control(sites,args.aux_copies)
        one,packet=source_packet(gamma,sites,body,args.packet_width)
        paths,harmonic=integer_source_paths(sites)
        seed_rows=[]
        for seed in args.seeds:
            gauges,background=gauss_samples(sites,args.aux_copies,args.method,args.points_power,seed)
            rows=[complete_pair(gamma,sites,body,args.aux_kappa,args.aux_copies,float(theta),coordinates,heat,
                one,harmonic,gauges,background,args.batch) for theta in shifts]
            seed_rows.append(dict(seed=seed,actual_character_samples=len(gauges),rows=rows,relative=characterize(rows,args.method)))
        cases.append(dict(sites=sites,physical_components_per_site=4,canonical_number_per_flavour=2*sites,
            auxiliary_copies=args.aux_copies,actual_body_kappa=body,actual_auxiliary_kappa=args.aux_kappa,
            actual_global_centre_control=centre,
            shifted_holonomies=shifts,source_packet=packet,normal_coordinates=coordinates,normal_H=H,
            normal_ground=ground,actual_normal_heat=heat,method=args.method,replicates=seed_rows,
            integration_scope='Exact local-U1 Gauss characters plus true global Z4 Ref compensation; no numeric full-SU4 macro Haar claim. All canonical occupations of every flavour remain inside their determinant coefficient.'))
    compiled=dict(supplier.frozen.COMPILED);compiled[Path(__file__).stem]=hashlib.sha256(OWN_BYTES).hexdigest()
    compiled['covariant_canonical_gauss_boundary_kernel']=SUPPLIER_HASH
    captured={name:supplier.frozen.SNAPSHOT[name].decode() for name in supplier.frozen.COMPILED}
    captured[Path(__file__).stem]=OWN_BYTES.decode();captured['covariant_canonical_gauss_boundary_kernel']=supplier.OWN_BYTES.decode()
    unchanged=all((ROOT/(name+'.py')).read_text()==captured[name] for name in compiled)
    require(unchanged,'Scientific source changed during relative-ring execution')
    result=dict(plan=PLAN,governing_revision=REVISION,native_epoch='NEW full4-component relative completed two-clock ring kernels',
        parameters=vars(args)|{'output':str(args.output)},actual_coefficient_control=check,cases=cases,
        actual_frozen_body_kappa=body,actual_frozen_body_8k=8*body,
        excluded_C_completed_body='Criticalκ=.1256305951044483 belongs to AlphaC new law; it is NOT substituted into Root6911 and no completed-C winding claim is made.',
        compiled_source_hashes=compiled,captured_source_bytes=captured,compiled_source_unchanged=unchanged,
        main_code_matches_captured_source=MAIN_CODE_MATCH,loaded_main_code_sha256=hashlib.sha256(marshal.dumps(LOADED_MAIN_CODE)).hexdigest(),
        runtime=dict(executable=sys.executable,python=sys.version,numpy=np.__version__,scipy=sys.modules['scipy'].__version__),
        elapsed_seconds=time.monotonic()-start,macro_conditioned_admission=False,
        limitation='These are genuine relative POSITIVE completed canonical/Gauss/source traces at finite diagonal field endpoints and clock2. Sobol estimates are labelled stochastic and never promoted to an interval bound or exact macro state; actual Nt/125-clip/Pe/global receiver joins remain separate.')
    args.output.write_text(json.dumps(supplier.portable(result),indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(args.output),source_sha256=compiled[Path(__file__).stem],
        variations=[dict(sites=c['sites'],neutral=c['replicates'][0]['relative']['neutral_relative_common_shift_variation'],
            source=c['replicates'][0]['relative']['source_relative_common_shift_variation']) for c in cases],
        elapsed_seconds=result['elapsed_seconds'],macro_conditioned_admission=False)))
    print('Natural exit: relative completed full-component Pauli/Gauss/Phi/source ring traces; no macro admission')


if __name__=='__main__':main()
