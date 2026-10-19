#!/usr/bin/env python3
"""Actual physical Root occupied-minor Haar compression, CAR/source consumer.

The finite four-site ring is an explicit scientific probe, not the macro cubic
law. Every neutral/charged active occupation and all Ref4 holes are retained.
An auxiliary occupied-frame compression is an exact reference identity, never
an interacting Slater assertion. The full canonical complement keeps its scope.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import time
from typing import Any
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import numpy as np
from scipy.linalg import eigh
from scipy.special import roots_hermite
from scipy.sparse import coo_matrix,diags
from scipy.sparse.linalg import eigsh
import covariant_canonical_gauss_boundary_kernel as supplier
import covariant_auxiliary_charge_clock_bridge as clock

ROOT=Path(__file__).resolve().parent


def physical(kappa: float,amplitude: float) -> tuple[np.ndarray,float,np.ndarray]:
    gamma,_=supplier.phi_frame(supplier.Parameters())
    sites=4;shift=np.zeros((sites,sites),complex)
    for x in range(sites):shift[x,(x+1)%sites]=1
    fields=amplitude*np.cos(2*math.pi*np.arange(sites)/sites)
    links=[np.kron(shift,np.eye(4)),np.kron(np.diag(np.exp(1j*fields)),np.eye(4)),np.eye(4*sites)]
    h,p,t=supplier.physical_spatial_onebody(gamma,kappa,links,[np.zeros_like(links[0]) for _ in range(4)],np.eye(4*sites))
    return h,math.sqrt(p),t


def graph(kappa: float,amplitude: float) -> np.ndarray:
    gamma,_=supplier.phi_frame(supplier.Parameters());O,_=clock.chiral_coordinates(gamma,4)
    _,_,t=physical(kappa,amplitude)
    _,vectors=eigh(O.conj().T@t@O)
    upper=np.array([4*x+s for x in range(4) for s in (0,1)])
    lower=np.array([4*x+s for x in range(4) for s in (2,3)])
    occupied=vectors[:,-8:]
    X=occupied[upper]@np.linalg.inv(occupied[lower])
    if np.linalg.norm(X-X.conj().T)>1e-12:
        raise ValueError('Actual physical graph must have true Hermitian phase')
    return (X+X.conj().T)/2


def charge_patterns(body: int) -> tuple[list[tuple[int,...]],dict[tuple[int,...],int]]:
    patterns=Counter()
    for counts in itertools.product(range(5),repeat=4):
        if sum(counts)!=8+body:continue
        rho=list(n-2 for n in counts);rho[0]-=body
        if sum(rho)!=0:raise ValueError('Literal active+Ref Gauss charge must be zero')
        multiplicity=math.prod(math.comb(4,n) for n in counts)
        patterns[tuple(rho[1:])]+=multiplicity
    return sorted(patterns),dict(patterns)


def compact_tail(X: np.ndarray,Xp: np.ndarray,N: int,radius: float=.5) -> dict[str,Any]:
    """Uniform principal-Haar tail from actual graph pair coefficients."""
    W=(Xp*X.T).reshape(4,2,4,2).sum(axis=(1,3))
    edges={(0,1),(1,2),(2,3),(0,3)}
    nearest=min(float(W[x,y].real) for x,y in edges)
    negative=sum(max(0.,-float(W[x,y].real))*2
        for x in range(4) for y in range(x+1,4) if (x,y) not in edges)
    activity=2*N*(nearest-negative)
    bias=2*N*sum(abs(float(W[x,y].imag)) for x in range(4) for y in range(x+1,4))
    graphnorm=max(float(np.linalg.norm(X,2)),float(np.linalg.norm(Xp,2)))
    remainder=8*N*graphnorm**4/(1-graphnorm**2)
    C=4*N*max(float(sum(abs(W[x,y]) for y in range(4) if x!=y)) for x in range(4))
    if activity<=0:raise ValueError('Actual graph does not supply the declared compact-tail activity')
    lower_log=-bias-remainder-1.5*math.log(2*math.pi*C)+3*math.log(math.erf(math.pi*math.sqrt(C/2)))
    upper_log=bias+remainder-activity*radius**2/math.pi**2-lower_log
    return dict(principal_rooted_angle_radius=radius,actual_graph_activity=activity,
        all_angle_odd_bias_upper=bias,uniform_logdet_remainder_upper=remainder,
        actual_uncharged_Haar_integral_lower_log10=lower_log/math.log(10),
        relative_compact_outer_tail_log10_upper=upper_log/math.log(10),
        proof='Pair spin-traces of Xprime_ij X_ji retain their actual imaginary/odd bias. Negative non-nearest real pairs cost the length2 ring path. On the rooted compact torus, any |chi_x|>.5 forces sum_edges(1-cos Delta chi)>=.5^2/pi^2. Uniform higher trace-log remainder and an explicit diagonal-Gaussian cube integral supply the uncharged denominator lower bound; no parity assumption.')


def gauss(kappa: float,nodes: int,beta: float=4096.) -> dict[str,Any]:
    N=int(round(beta/(16*kappa**4)))
    X=graph(kappa,.07);Xp=graph(kappa,.0704)
    W=Xp*X.T
    M0=Xp@X
    lap=np.array([[2.,-1.,0.],[-1.,2.,-1.],[0.,-1.,2.]])
    covariance=np.linalg.inv(beta*lap)
    chol=np.linalg.cholesky(covariance)
    z,w=roots_hermite(nodes);w/=math.sqrt(math.pi)
    mesh=np.array(list(itertools.product(range(nodes),repeat=3)))
    chi=np.sqrt(2)*z[mesh]@chol.T
    weight=np.prod(w[mesh],axis=1)
    rho=sorted(set(itertools.chain(*(charge_patterns(body)[0] for body in (0,1,-1)))))
    rho_array=np.array(rho)
    values=np.zeros(len(rho),complex);normal=0.
    maximum_imaginary=0.;logrange=[math.inf,-math.inf]
    maxgraph=max(float(np.linalg.norm(X,2)),float(np.linalg.norm(Xp,2)))
    omitted=2*8*N*maxgraph**6/(3*(1-maxgraph**2))
    tail=compact_tail(X,Xp,N)
    for c,prob in zip(chi,weight):
        g=np.repeat(np.exp(1j*np.r_[0.,c]),2)
        delta1=np.sum(W*np.expm1(1j*(np.angle(g)[None,:]-np.angle(g)[:,None])))
        M=Xp@(g[:,None]*X*g.conj()[None,:])
        delta2=np.trace(M@M-M0@M0)
        logratio=N*(delta1-delta2/2)
        maximum_imaginary=max(maximum_imaginary,abs(float(logratio.imag)))
        correction=float(logratio.real)+.5*float(c@np.linalg.solve(covariance,c))
        logrange[0]=min(logrange[0],correction);logrange[1]=max(logrange[1],correction)
        density=math.exp(correction)
        normal+=prob*density
        values+=prob*density*np.exp(1j*(rho_array@c))
    values/=normal
    gaussian=np.exp(-np.einsum('ij,jk,ik->i',rho_array,covariance,rho_array)/2)
    errors=np.abs(values-gaussian)
    return dict(kappa=kappa,Naux_exact=str(N),beta=beta,quadrature_nodes_per_dimension=nodes,
        charge_patterns=[list(r) for r in rho],actual_normalized_Haar_charge_kernel=[[float(z.real),float(z.imag)] for z in values],
        Gaussian_charge_diagonal=gaussian.tolist(),actual_kernel_CAR_norm_difference=float(errors.max()),
        actual_charge_minimum_real=float(np.min(np.real(values))),maximum_charge_imaginary=float(np.max(abs(np.imag(values)))),
        actual_auxiliary_graph_norm=maxgraph,logdet_series_remainder_upper=omitted,
        actual_background_phase_log_imaginary_max=maximum_imaginary,
        actual_relative_uncharged_Haar_normalization=normal,log_relative_density_core_range=logrange,
        actual_compact_Haar_outer_tail=tail,
        all_neutral_occupation_components=math.comb(16,8),all_charged_components_per_Ref=math.comb(16,9),Ref_holes=4,
        canonical_reference_scope='Exact true-graph occupied-minor compression of the actual physical16-mode Root. The auxiliary canonical complement is NOT declared zero and is not an interacting Slater.',
        logdet_scope='Two-term stable trace-log of graph products, with displayed deterministic all-N bound on omitted terms; normalized exact occupied kernel up to that bound.',
        compact_integral_scope='Converged Gaussian quadrature of the compact-rooted-Haar core with an explicit displayed all-angle outer-tail bound; quadrature convergence is numerical, not a fabricated certified integration error.',
        macro_cubic_kappa6_claim=False)


def source_matrix() -> tuple[Any,list[tuple[int,...]],list[tuple[int,...]],float]:
    k=1/(2*(math.expm1(.01)+4))
    gamma,_=supplier.phi_frame(supplier.Parameters())
    modes=16;neutral=supplier.occupations(modes,8);charged=supplier.occupations(modes,9)
    index={state:i for i,state in enumerate(charged)}
    offsets=np.array([-1,0,1]);amplitudes=np.ones(3)/math.sqrt(3)
    coeff=np.zeros((modes,4),complex)
    for offset,a in zip(offsets,amplitudes):
        momentum=offset*2*math.pi/4
        links=[np.exp(1j*momentum)*np.eye(4),np.eye(4),np.eye(4)]
        _,_,T=supplier.physical_spatial_onebody(gamma,k,links,[np.zeros((4,4),complex) for _ in range(4)],np.eye(4))
        v,u=eigh(T);band=u[:,v<1]@u[:,v<1].conj().T;energy=-math.log(float(max(v[v<1])))
        for x in range(4):coeff[4*x:4*x+4]+=a*math.exp(energy/2)*np.exp(1j*momentum*x)*band/math.sqrt(8)
    rows=[];cols=[];data=[]
    for col,state in enumerate(neutral):
        for mode in range(modes):
            if mode in state:continue
            target=tuple(sorted((*state,mode)));s=(-1)**sum(i<mode for i in state)
            for ref in range(4):
                value=s*(-1)**ref*coeff[mode,ref]
                if abs(value)>1e-15:
                    rows.append(ref*len(charged)+index[target]);cols.append(col);data.append(value)
    S=coo_matrix((data,(rows,cols)),shape=(4*len(charged),len(neutral))).tocsr()
    return S,neutral,charged,float(np.linalg.norm(coeff,'fro'))


def sparse_norm(matrix: Any) -> float:
    rng=np.random.default_rng(222)
    value=eigsh(matrix.conj().T@matrix,k=1,which='LM',return_eigenvectors=False,
        v0=rng.standard_normal(matrix.shape[1]),tol=0)[0]
    return math.sqrt(max(0,float(value)))


def source_consumer(row: dict[str,Any]) -> dict[str,Any]:
    S,neutral,charged,cap=source_matrix()
    lookup={tuple(r):complex(*v) for r,v in zip(row['charge_patterns'],row['actual_normalized_Haar_charge_kernel'])}
    gaussian={tuple(r):v for r,v in zip(row['charge_patterns'],row['Gaussian_charge_diagonal'])}
    def rho(state,body):
        return tuple(sum(4*x<=i<4*x+4 for i in state)-2 for x in range(1,4))
    dn=np.array([lookup[rho(state,0)] for state in neutral]);dc=np.tile([lookup[rho(state,1)] for state in charged],4)
    gn=np.array([gaussian[rho(state,0)] for state in neutral]);gc=np.tile([gaussian[rho(state,1)] for state in charged],4)
    comm=diags(dc)@S-S@diags(dn)
    reference_comm=diags(gc)@S-S@diags(gn)
    difference=diags(dc-gc)@S-S@diags(dn-gn)
    measured=sparse_norm(difference)
    bound=2*row['actual_kernel_CAR_norm_difference']*cap
    if measured>bound*(1+1e-7)+1e-9:
        raise ValueError('Whole-source CAR kernel bound failed')
    return dict(neutral_CAR_dimension=len(neutral),charged_Ref_CAR_dimension=4*len(charged),
        actual_sparse_source_norm=sparse_norm(S),whole_source_Frobenius_CAR_cap=cap,
        actual_Gauss_source_noncommutation=sparse_norm(comm),reference_Gauss_source_noncommutation=sparse_norm(reference_comm),
        actual_whole_source_kernel_difference=measured,proved_whole_source_kernel_difference_upper=bound,
        whole_coherent_source_and_all4_Ref_holes=True,
        source_scope='Explicit NEW finite4-site normalized3-beam native control; original macro source coefficients/state unchanged. Free canonical T<1 projector before literal x-path (Ax=0), Ref4->3. No local norm1 or charge/kernel commutation asserted.')


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();started=time.perf_counter()
    rows=[gauss(k,nodes) for k in (.004,.002,.001) for nodes in (10,14)]
    native_source=source_consumer(rows[-1])
    report=dict(governing_plan_revision='1124ce49865f8f7cc5f912ef9569edc52ad3227fb532fd78d5637658c923b990',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        frozen_Source2a_sha256=hashlib.sha256((ROOT/'covariant_prepared_gauss_state.py').read_bytes()).hexdigest(),
        frozen_Root6911_sha256=hashlib.sha256((ROOT/'covariant_root_transfer.py').read_bytes()).hexdigest(),
        actual_real_Gauss_CAR_kernel_rows=rows,actual_whole_source_CAR_consumer=native_source,
        exact_operator_upgrade='The joint Haar compression is Fprime diag(Crho) F. Uniform EVERY-occupation Fourier error gives CAR operator norm <=||Fprime||||F|| maxrho|Crho-Pq(rho)|. This is a matrix-kernel norm estimate, NEVER a scalar-to-Loewner inference. Pq does not commute through physical matter/source.',
        macro_actual_Grams=None,macro_source_admitted=False,elapsed_seconds=time.perf_counter()-started)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(args.output),elapsed_seconds=report['elapsed_seconds'],
        CAR_kernel_errors=[r['actual_kernel_CAR_norm_difference'] for r in rows],whole_source=native_source,macro_source_admitted=False)))


if __name__=='__main__':
    main()
