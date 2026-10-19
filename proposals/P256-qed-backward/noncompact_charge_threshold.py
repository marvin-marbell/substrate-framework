#!/usr/bin/env python3
"""Noncompact reduced-Coulomb determinant and all-Bloch threshold construction.

Whole quantum light/photon diagnostics are finite Galerkin observations;
analytic untruncated and microscopic-transfer scopes remain separate.
"""
from __future__ import annotations
import argparse,hashlib,itertools,json,math,os
from pathlib import Path
from typing import Any,Callable,cast
if __name__=='__main__':
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.sparse.linalg import LinearOperator,expm_multiply
from bulk_quantum_selection import BulkQuantumLaw
from noncompact_native_current import Coupled,accept,adverse,source_operators
from original_charged_principal import original_frame,portable
from original_uniform_charge_gap import determinant_controls,canonical_controls
from rotation_band import RotationBand

ROOT=Path(__file__).resolve().parent

def constants(G,frame):
    b=np.linalg.solve(G[1:,1:],G[1:,0]);s=float(G[0,0]-G[0,1:]@b)
    if s<=0:raise ValueError('Positive full native Schur metric is required')
    beta=(frame[0]-np.einsum('j,jab->ab',b,frame[1:]))/math.sqrt(s)
    return .2*math.sqrt(s),beta,s

def real_backgrounds(size,centre):
    band=RotationBand(a=1.,m=.2,r=.12,v=.3,improved=False)
    selected,frame=original_frame(band,size,np.asarray(centre)*math.pi/2,True)
    G=selected['G'];mu,beta,s=constants(G,frame)
    law=BulkQuantumLaw(size,band=band)
    fields=np.tile(frame,(law.volume,1,1,1));links=law.toron_links(np.asarray(centre)*math.pi/2)
    rng=np.random.default_rng(2221014)
    A=rng.normal(size=(4,law.volume,3))*8
    durations=np.full(4,.5)
    temporal=rng.normal(size=(4,law.volume))
    temporal[0,0]+=(math.pi/2-2*np.sum(durations[:,None]*temporal))/(2*durations[0])
    h=[law.onebody(fields,links,a) for a in A]
    bound=math.sqrt(G[0,0])*(.2+6*.12)+.3*sum(math.sqrt(G[j,j]) for j in range(1,4))
    determinant=determinant_controls(h,temporal,durations,beta,mu,bound)
    all64=[]
    for theta in itertools.product(range(4),repeat=3):
        native,p=original_frame(band,size,np.asarray(theta)*math.pi/2,True)
        m,_,schur=constants(native['G'],p)
        all64.append(dict(centre=list(theta),mu=m,Schur=schur))
    return frame,dict(native_size=size,centre=list(centre),actual_Gram=G.tolist(),mu=mu,Schur=s,real_unwrapped_A_range=[float(A.min()),float(A.max())],temporal_determinant=determinant,global_canonical=canonical_controls(frame,band,rng),all64=all64,min_sampled_native_mu=min(r['mu'] for r in all64),N0_admission_claimed=False)

def coulomb_gaussian(UE,dt):
    rows=list(itertools.combinations(range(8),4))
    nx=np.asarray([sum(i<4 for i in r) for r in rows],float)
    z,w=hermgauss(40);z*=math.sqrt(2);w/=math.sqrt(math.pi)
    angles=math.sqrt(2*dt*UE/3)*z
    measured=np.exp(1j*angles[:,None]*(nx[None,:]-2)).T@w
    exact=np.exp(-dt*UE*(nx-2)**2/3)
    error=accept(np.linalg.norm(measured-exact),'actual whole70 neutral Coulomb Gaussian temporal factor')
    wrong=np.exp(1j*angles[:,None]*nx[None,:]).T@w
    rejection=adverse(np.linalg.norm(wrong-exact),'actual whole70 neutral Coulomb Gaussian temporal factor')
    return dict(UE=UE,dt=dt,complete_neutral_dimension=70,Gaussian_nodes=40,actual_error=error,omitted_background=rejection,identity='exp(-dt UE rho G+ rho)=E exp(i sqrt(2dtUE) Z.G+^(1/2)rho); same static reference included in charged rho')

def bloch_kinetic_controls(model,periodic,zeta,UE):
    rng=np.random.default_rng(2221019+model.occupation)
    v=rng.normal(size=model.size)+1j*rng.normal(size=model.size);v/=np.linalg.norm(v)
    ell=np.tile(periodic.ell,periodic.n)
    target=(UE*(2*zeta*ell+zeta*zeta)[:,None]*model.reshape(v)/6).ravel()
    actual=model.action(v)-periodic.action(v)
    error=accept(np.linalg.norm(actual-target),'requested Bloch fibre full-H harmonic kinetic shift',2e-10)
    wrong=None
    if zeta!=0:
        wrong=adverse(np.linalg.norm(target),'requested Bloch fibre full-H harmonic kinetic shift',2e-10)
    return dict(requested_zeta=zeta,actual_shift_norm=float(np.linalg.norm(actual)),
                independent_kinetic_shift_error=error,hardcoded_zero_twist=wrong,
                zero_twist_is_legitimate_positive_control=zeta==0)

def euclidean(model,E0,source,t):
    factory=cast(Callable[...,Any],LinearOperator)
    op=factory((model.size,model.size),matvec=lambda v:model.action(v)-E0*np.asarray(v).ravel(),rmatvec=lambda v:model.action(v)-E0*np.asarray(v).ravel(),dtype=complex)
    trace=float(model.d*np.sum(model.phenergy)+model.phdim*np.sum(model.coulomb)-model.size*E0)
    solver=cast(Callable[...,Any],expm_multiply)
    columns=np.asarray(source).T
    value=solver(-t*op,columns,traceA=-t*trace)
    return float(np.vdot(columns,value).real)

def quantum_scan(n,M,UE,K,frame,mu,zetas):
    neutral0=Coupled(n,M,0.,4,UE,K,frame)
    en0,v0,r0=neutral0.solve(4);E0=float(en0[0]);psi=v0[:,0]
    if en0[1]-en0[0]<1e-7:raise AssertionError('Neutral ground degeneracy needs its complete projector')
    rows=[]
    for zeta in zetas:
        neutral=Coupled(n,M,zeta,4,UE,K,frame);ev,_,res=neutral.solve(4)
        neutral_bloch=bloch_kinetic_controls(neutral,neutral0,zeta,UE)
        accept(max(0.,E0-float(ev[0])),'observed finite-Galerkin neutral Bloch minimum at zero',2e-7)
        qrows=[]
        for occupation in (3,5):
            model=Coupled(n,M,zeta,occupation,UE,K,frame)
            periodic=Coupled(n,M,0.,occupation,UE,K,frame)
            bloch=bloch_kinetic_controls(model,periodic,zeta,UE)
            ec,vc,rc=model.solve(6);gap=float(ec[0]-E0)
            # A diagnostic observation only: not a finite-cutoff determinant theorem.
            accept(max(0.,mu-gap),'observed finite-Galerkin charge gap versus untruncated target floor',2e-7)
            qrows.append(dict(light_number=occupation,static_reference_charge=4-occupation,complete_reference_dimension=4*model.size,global_gap_from_periodic_neutral=gap,ground_residual=max(rc),requested_Bloch_kinetic=bloch,infinite_comparison=model.full_residual_bound(vc[:,0],float(ec[0]))))
        charged=Coupled(n,M,zeta,5,UE,K,frame)
        prepared=np.asarray([neutral0.reshape(psi)@t.T for t in source_operators(neutral0,charged)[0]]).reshape(4,charged.size)
        norm=float(np.vdot(prepared,prepared).real)
        accept(abs(norm-2),'actual all-reference source crossing Bloch fibre norm2')
        sector,weights=charged.source_triality(prepared)
        ec,vc,rc=charged.source_spectrum(6,sector)
        mask=np.flatnonzero(ec-ec[0]<1e-7);cluster=np.linalg.qr(vc[:,mask])[0]
        residue=float(np.linalg.norm(cluster.conj().T@prepared.T)**2/norm)
        if residue<=1e-10:raise AssertionError('Source-addressed cluster has no actual source residue')
        correlations=[]
        for t in (1.,4.,12.):
            value=euclidean(charged,E0,prepared,t)/norm
            accept(max(0.,value-math.exp(-mu*t)),'actual quantum source heat decay versus target massive exponent',2e-7)
            correlations.append(dict(time=t,actual_normalized_source_heat=value,mass_floor_envelope=math.exp(-mu*t)))
        rows.append(dict(zeta=zeta,neutral_energy=float(ev[0]),neutral_shift=float(ev[0]-E0),neutral_residual=max(res),requested_neutral_Bloch_kinetic=neutral_bloch,charged=qrows,source_sector=sector,source_sector_weights=weights,source_addressed_gap=float(ec[0]-E0),source_ground_residue=residue,actual_source_heat=correlations))
    return dict(oscillator_levels=n,harmonic_radius=M,periodic_neutral_E0=E0,periodic_neutral_residual=max(r0),periodic_neutral_infinite_comparison=neutral0.full_residual_bound(psi,E0),rows=rows,scope='actual finite Galerkin spectra/heat; NOT proof of infinite-basis gap from two Ritz energies; Bloch-changing source includes unit-modulus harmonic section')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--size',type=int,default=5)
    p.add_argument('--n',type=int,default=5);p.add_argument('--M',type=int,default=3)
    p.add_argument('--UE',type=float,default=.37);p.add_argument('--K',type=float,default=.9)
    p.add_argument('--zetas',type=float,nargs='+',default=[-.4,0.,.4])
    args=p.parse_args();frame,native=real_backgrounds(args.size,(1,2,3))
    result=dict(law='explicit noncompact quadratic curl plus exact reduced Coulomb/full CAR',analytic_target='untruncated fixed-Clifford effective charged infimum over ALL harmonic Bloch fibres >= periodic neutral infimum +mu',native=native,Coulomb_Hubbard_Gaussian=coulomb_gaussian(args.UE,.2),quantum=quantum_scan(args.n,args.M,args.UE,args.K,frame,native['mu'],args.zetas),microscopic_BO_lower_transfer_claimed=False,isolated_pole_mobile_spin_photon_phase_or_Compton_claimed=False,source_hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ('noncompact_charge_threshold.py','noncompact_charge_threshold.md','noncompact_native_current.py','original_uniform_charge_gap.py','original_uniform_charge_gap.md')} )
    print(json.dumps(portable(result),indent=2,allow_nan=False))

if __name__=='__main__':main()
