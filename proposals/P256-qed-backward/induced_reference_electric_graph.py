#!/usr/bin/env python3
"""NEW electric-priced graph-Slater EM/CAR source, original full Href unchanged.

The preparation graph uses the actual primitive offdiagonal K(theta), not an
endpoint Gamma or an altered physical band. All real coexact/harmonic photon
coordinates, all1+63 CAR loads and true Gauss Coulomb remain. No vacuum/pole.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any
if __name__=='__main__':
    for _name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
        os.environ[_name]='1'
import numpy as np
from induced_reference_field_following import build_native, matter_hamiltonian, path_derivative, occupied_section, SOURCE_BYTES as FOLLOW_BYTES
from induced_reference_electric_graph_enclosure import SOURCE_BYTES as ENCLOSURE_BYTES

ROOT=Path(__file__).resolve().parent
SOURCE_BYTES={**FOLLOW_BYTES,**ENCLOSURE_BYTES,Path(__file__).name:Path(__file__).read_bytes()}
SOURCE_HASHES={name:hashlib.sha256(raw).hexdigest() for name,raw in SOURCE_BYTES.items()}


def spin_basis(native: dict[str, Any]) -> tuple[np.ndarray,dict[str,float]]:
    """Resolve the actual Phi0/Phi5 planes; never substitute endpoint Gamma."""
    gamma=native['frame']/np.sqrt(np.array([native['g0']]+[native['gs']]*3))[:,None,None]
    gamma5=gamma[0]@gamma[1]@gamma[2]@gamma[3]
    _,vectors=np.linalg.eigh(gamma[0]); plus=vectors[:,2:]; minus=gamma5@plus
    B=np.hstack((plus,minus))
    errors=dict(gamma5_Hermitian_error=float(np.max(abs(gamma5-gamma5.conj().T))),
        gamma5_square_error=float(np.max(abs(gamma5@gamma5-np.eye(4)))),
        basis_unitarity_error=float(np.max(abs(B.conj().T@B-np.eye(4)))))
    if max(errors.values())>1e-12:
        raise ArithmeticError('Frozen orthogonal Phi graph premise failed')
    return B,errors


def to_spin_basis(matrix: np.ndarray,B: np.ndarray) -> np.ndarray:
    V=len(matrix)//4
    return np.einsum('aA,xayb,bB->xAyB',B.conj(),matrix.reshape(V,4,V,4),B,optimize=True).reshape(4*V,4*V)


def from_spin_basis(matrix: np.ndarray,B: np.ndarray) -> np.ndarray:
    V=len(matrix)//4
    return np.einsum('aA,xAyB,bB->xayb',B,matrix.reshape(V,4,V,4),B.conj(),optimize=True).reshape(4*V,4*V)


def graph_state(native: dict[str, Any],profile: dict[str, Any],theta: np.ndarray,
                *,section: bool=True) -> dict[str, Any]:
    """Exactly normalized graph source; d=2Monsite+u prices a single link pair.

    This is a NEW preparation, not an eigen-Slater of Heta. Its physical Href
    is the unchanged original band. No preparation spectrum is manufactured.
    """
    V=native['V']; rank=2*V; B,proof=spin_basis(native)
    pp=np.array([4*x+s for x in range(V) for s in range(2)]); nn=pp+2
    physical=matter_hamiltonian(native,profile,theta); hb=to_spin_basis(physical,B)
    M=hb[np.ix_(pp,pp)]; K=-1j*hb[np.ix_(pp,nn)]
    errors=dict(physical_anti_chiral_error=float(np.max(abs(hb[np.ix_(nn,nn)]+M))),
        physical_offdiagonal_Hermitian_error=float(np.max(abs(K-K.conj().T))))
    if max(errors.values())>1e-11:
        raise ArithmeticError('Actual primitive physical matrix violates graph premise')
    mass0=float(np.trace(B[:,:2].conj().T@profile['onsite']@B[:,:2]).real/2)
    if mass0<=0:
        raise ValueError('Projected negative-Phi0 onsite base requires positive onsite mass')
    denominator=2*mass0+native['UE']/native['a']
    Y=K/denominator; R=np.linalg.inv(np.eye(rank)+Y@Y)
    pb=np.empty_like(hb); pb[np.ix_(pp,pp)]=np.eye(rank)-R; pb[np.ix_(nn,nn)]=R
    pb[np.ix_(pp,nn)]=-1j*Y@R; pb[np.ix_(nn,pp)]=1j*Y@R
    P=from_spin_basis(pb,B)
    result=dict(physical_h=physical,P=P,Y=Y,R=R,M=M,K=K,B=B,pp=pp,nn=nn,
        mass0=mass0,denominator=denominator,load=profile['load'],profile=profile,
        spin_frame_proof=proof,physical_graph_premise=errors)
    if section:
        values,vectors=np.linalg.eigh(R); root=(vectors*np.sqrt(values))@vectors.conj().T
        occ=np.empty((4*V,rank),complex); emp=np.empty_like(occ)
        occ[pp]=-1j*Y@root; occ[nn]=root; emp[pp]=root; emp[nn]=-1j*Y@root
        occ=np.einsum('aA,xAk->xak',B,occ.reshape(V,4,rank),optimize=True).reshape(4*V,rank)
        emp=np.einsum('aA,xAk->xak',B,emp.reshape(V,4,rank),optimize=True).reshape(4*V,rank)
        occupied,phase=occupied_section(profile,occ)
        result.update(occupied=occupied,empty=emp,phase=phase)
    return result


def graph_direction(native: dict[str, Any],state: dict[str, Any],theta: np.ndarray,
                    direction: np.ndarray,order: int=1,*,other: np.ndarray|None=None) -> np.ndarray:
    """Differentiate the ACTUAL primitive one/two-link current/contact first."""
    physical=path_derivative(native,state['profile'],theta,direction,order,other=other)
    hb=to_spin_basis(physical,state['B'])
    return -1j*hb[np.ix_(state['pp'],state['nn'])]/state['denominator']


def graph_tangents(state: dict[str, Any],first: np.ndarray,second: np.ndarray) -> dict[str, Any]:
    """True determinant ray; second derivative includes dGamma(T)^2 PH pairs."""
    Y,R=state['Y'],state['R']; A=Y@first+first@Y
    dR=-R@A@R
    ddR=2*R@A@R@A@R-R@(Y@second+second@Y+2*first@first)@R
    first_block=np.empty_like(state['P']); second_block=np.empty_like(first_block)
    pp,nn=state['pp'],state['nn']
    first_block[np.ix_(pp,pp)]=-dR;first_block[np.ix_(nn,nn)]=dR
    first_block[np.ix_(pp,nn)]=-1j*(first@R+Y@dR)
    first_block[np.ix_(nn,pp)]=first_block[np.ix_(pp,nn)].conj().T
    second_block[np.ix_(pp,pp)]=-ddR;second_block[np.ix_(nn,nn)]=ddR
    second_block[np.ix_(pp,nn)]=-1j*(second@R+2*first@dR+Y@ddR)
    second_block[np.ix_(nn,pp)]=second_block[np.ix_(pp,nn)].conj().T
    dP=from_spin_basis(first_block,state['B']); ddP=from_spin_basis(second_block,state['B'])
    P=state['P']; T=dP@P; Tprime=ddP@P+dP@dP
    result=dict(dP=dP,ddP=ddP,T=T,Tprime=Tprime,metric=float(np.trace(R@first@R@first).real),
        determinant_Berry_connection=0.,scope='Flat Hermitian graph on frozen orthogonalPhi; operator dGamma(T)^2 retains all PH pairs/all64 copies.')
    if 'occupied' in state:
        result['PH']=state['empty'].conj().T@dP@state['occupied']
    return result


def energy_integrand(native: dict[str, Any],theta: np.ndarray,*,direct_Ward: bool=False) -> dict[str, Any]:
    """Physical fullHref energy: exact CAR plus full-link Born+Gauss identity."""
    V=native['V']; rank=2*V; u=native['UE']/native['a']; car=metric=coul=0.; rows=[]
    for profile in native['Pprofiles']:
        state=graph_state(native,profile,theta,section=False); B,R,P=state['B'],state['R'],state['P']
        blocks=P.reshape(V,4,V,4); occupation=P.diagonal().reshape(V,4).sum(axis=1).real
        density_cov=np.diag(occupation)-np.einsum('xayb,ybxa->xy',blocks,blocks,optimize=True).real
        gauss=float(np.trace(native['Lplus']@density_cov)); energy=float(np.einsum('ij,ji->',P,state['physical_h']).real)
        local_metric=0.; derivatives=[]
        for path in profile['paths']:
            if path['length']!=1: continue
            x,y=path['source'],path['target']; edge=path['edges'][0]
            indices=np.array([2*x,2*x+1,2*y,2*y+1])
            derivative=(B[:,:2].conj().T@path['hop']@B[:,2:])*np.exp(1j*theta[edge])/state['denominator']
            local=np.zeros((4,4),complex);local[:2,2:]=derivative;local[2:,:2]=derivative.conj().T
            support=R[np.ix_(indices,indices)];local_metric+=float(np.trace(support@local@support@local).real)
            if direct_Ward:
                full=np.zeros((rank,rank),complex);full[np.ix_(indices,indices)]+=local;derivatives.append((edge,full))
        ward=None
        if direct_Ward:
            transverse=0.
            for coordinate in native['Q'].T:
                derivative=sum((coordinate[edge]*full for edge,full in derivatives),np.zeros((rank,rank),complex))
                transverse+=float(np.trace(R@derivative@R@derivative).real)
            ward=dict(coexact_harmonic_metric=transverse,independent_density_Gauss_metric=gauss,
                full_link_recombination_error=abs(transverse+gauss-local_metric))
        density_error=float(np.max(abs(occupation-2)))
        if density_error>1e-10 or min(gauss,local_metric-gauss)<-1e-8:
            raise ArithmeticError('Graph half-density/Gauss metric premise failed')
        model=profile['model']; norm=3*abs(model.v)/model.a*np.sqrt(native['gs'])/state['denominator']
        rows.append(dict(label=profile['label'],load=profile['load'],actual_CAR_energy_per_V=energy/V,
            full_link_graph_metric_per_V=local_metric/V,realGauss_density_metric_per_V=gauss/V,
            coexact_harmonic_metric_per_V=(local_metric-gauss)/V,graph_denominator=state['denominator'],
            uniform_graph_norm_upper=norm,rational_series_contraction_upper=norm**2,
            mean_rho_error=density_error,projector_error=float(np.max(abs(P@P-P))),direct_Ward=ward,
            physical_graph_premise=state['physical_graph_premise']))
        load=profile['load'];car+=load*energy/V;metric+=load*local_metric/V;coul+=load*gauss/V
    return dict(actual_CAR_energy_per_V=car,full_link_Born_plus_Coulomb_per_V=u*metric,
        physical_Gauss_Coulomb_per_V=u*coul,coexact_harmonic_Born_per_V=u*(metric-coul),
        actual_Href_integrand_per_V=car+u*metric,profiles=rows)


def current_integrand(native: dict[str, Any],theta: np.ndarray,direction: np.ndarray) -> dict[str, Any]:
    """FullHref initial retarded current moment on the true NEW joint source."""
    if native['N']!=2:
        raise ValueError('Full causal backend explicitly N2; no silent macro truncation')
    V=native['V'];u=native['UE']/native['a'];states=[]
    matter=mean=covariance=contact=0.;density_term=0j
    comm_density_mean=np.zeros(V,complex);comm_density_cov=0j
    for profile in native['Pprofiles']:
        state=graph_state(native,profile,theta);h,P,lower,upper=state['physical_h'],state['P'],state['occupied'],state['empty']
        J=path_derivative(native,profile,theta,direction);load=profile['load'];rank=lower.shape[1]
        current_PH=upper.conj().T@J@lower;C=path_derivative(native,profile,theta,direction,2)
        mean+=load*float(np.trace(P@J).real);covariance+=load*float(np.sum(abs(current_PH)**2));contact+=load*float(np.trace(P@C).real)
        hJ=h@J-J@h;matter+=load*float(np.trace(P@(hJ@J-J@hJ)).real)
        transitions=[];complement=np.eye(4*V)-P
        for x in range(V):
            mask=np.repeat((np.arange(V)==x).astype(float),4);Ax=(mask[:,None]-mask[None,:])*J
            transitions.append(upper.conj().T@Ax@lower);comm_density_mean[x]+=load*np.trace(P@Ax)
            Bx=Ax@J-J@Ax;weights=np.repeat(native['Lplus'][x],4)
            density_term+=load*np.sum(np.diag(P@Bx@complement)*weights)
        packed=np.array(transitions).reshape(V,rank*rank)
        comm_density_cov-=load*np.einsum('xa,xy,ya->',packed.conj(),native['Lplus'],packed)
        state.update(J=J);states.append(state)
    density_pair=comm_density_cov+comm_density_mean@native['Lplus']@comm_density_mean
    coulomb_complex=u*(2*density_term.real+2*density_pair)
    electric=fixed_fibre=correction_total=0.
    for coordinate in native['Q'].T:
        derivative_mean=derivative_covariance=correction=0.
        for state in states:
            profile,P=state['profile'],state['P'];lower,upper,J=state['occupied'],state['empty'],state['J']
            first=graph_direction(native,state,theta,coordinate);second=graph_direction(native,state,theta,coordinate,2)
            ray_PH=graph_tangents(state,first,second)['PH']
            derivative_current=path_derivative(native,profile,theta,direction,2,other=coordinate)
            contact_PH=upper.conj().T@derivative_current@lower;comm_PH=upper.conj().T@(J@derivative_current-derivative_current@J)@lower
            load=state['load'];derivative_mean+=load*float(np.trace(P@derivative_current).real)
            derivative_covariance+=load*float(np.sum(abs(contact_PH)**2));correction+=load*float(np.vdot(ray_PH,comm_PH).real)
        scalar=derivative_mean**2+derivative_covariance;fixed_fibre-=2*u*scalar;correction_total-=2*u*correction;electric-=2*u*(scalar+correction)
    return dict(current_mean=mean,conditional_current_covariance=covariance,raw_current_second_moment=covariance+mean**2,literal_contact=contact,
        retarded_initial_tangent=electric+matter+float(coulomb_complex.real),
        parts=dict(primitive_electric=electric,Peierls_CAR=matter,full_real_Gauss_Coulomb=float(coulomb_complex.real)),
        missing_actual_ray_tangent_shift=correction_total,wrong_fixed_fibre_electric=fixed_fibre,
        Coulomb_hermitian_residual=float(abs(coulomb_complex.imag)))


def compute_source(mean: str|Path,*,size: int=6,variance: float=2.,samples: int=4,
                   seed: int=20261008,current: bool=False) -> dict[str, Any]:
    if samples<2 or variance<=0:
        raise ValueError('Need at least2 independent samples and positive Gaussian density variance')
    start=time.monotonic();mean=Path(mean);native=build_native(mean,size);rng=np.random.default_rng(seed);rows=[]
    if tuple(profile['load'] for profile in native['Pprofiles'])!=(1,63):
        raise ValueError('Assigned source requires actual active1 plus auxiliary63 canonical CAR loads')
    direction=np.zeros(3*native['V']);direction[1::3]=np.cos(np.pi*native['sites'][:,0]);direction/=np.linalg.norm(direction)
    for _ in range(samples):
        theta=native['Q']@(np.sqrt(variance)*rng.normal(size=native['M']))
        row=dict(theta=theta.tolist(),energy=energy_integrand(native,theta,direct_Ward=size==2))
        if current: row['current']=current_integrand(native,theta,direction)
        rows.append(row)
    kinetic=native['UE']/native['a']*native['M']/(4*variance*native['V'])
    energies=np.array([row['energy']['actual_Href_integrand_per_V']+kinetic for row in rows])
    result: dict[str,Any]=dict(N=size,V=native['V'],real_EM_coordinates=native['M'],all_charged_flavours=64,
        density_variance=variance,samples=samples,seed=seed,rows=rows,photon_Gaussian_kinetic_per_V=kinetic,
        actual_Href_energy_estimate_per_V=float(np.mean(energies)),sampling_energy_standard_error=float(np.std(energies,ddof=1)/np.sqrt(samples)),
        source='NEW normalized pure realGaussian EM × all64 Hermitian graph Slaters; Y=Kphysical/(2Monsite+UE/a). Preparation ONLY; original physical band/current/Href unchanged.',
        canonical_CAR_scope='All8 flat encodedP sectors only; Q priced separately. Each1+63 flavour has global2V particles, not two per momentum.',
        validity='Frozen orthogonalPhi midpoint diagnostic; independent sample estimates are not interval/ground/stationary/pole/macro-current admission. Source is distinct from a000, pure222/rho128 and momentum preparations.',
        source_hashes=SOURCE_HASHES,source_snapshots={name:raw.decode() for name,raw in SOURCE_BYTES.items()},
        input=dict(path=str(mean),sha256=hashlib.sha256(mean.read_bytes()).hexdigest()),elapsed_seconds=time.monotonic()-start,
        assigned_scientific_epoch='03b34911221ba1898315373edaa7c4c3bcac718462e7fe51a842560eb6933f87')
    if current:
        means=np.array([row['current']['current_mean'] for row in rows]);raw=np.array([row['current']['raw_current_second_moment'] for row in rows])
        tangents=np.array([row['current']['retarded_initial_tangent'] for row in rows]);mean_variance=float(np.var(means,ddof=1));plugin=float(np.mean(raw)-np.mean(means)**2)
        result.update(current_direction=direction.tolist(),retarded_initial_tangent_estimate=float(np.mean(tangents)),
            sampling_tangent_standard_error=float(np.std(tangents,ddof=1)/np.sqrt(samples)),current_mean_estimate=float(np.mean(means)),
            raw_current_second_moment_estimate=float(np.mean(raw)),between_Theta_conditional_mean_variance_unbiased=mean_variance,
            within_Theta_current_covariance_estimate=float(np.mean([row['current']['conditional_current_covariance'] for row in rows])),
            connected_equal_time_plugin_estimate=plugin,connected_equal_time_unbiased_estimate=plugin+mean_variance/samples,
            source_mean_status='No exact C-zero mean assumption; raw moments preserve between-field/all64 correlations. Unbiased finite-sample connected estimate adds conditional-mean samplevariance/n.')
    for name,raw in SOURCE_BYTES.items():
        if (ROOT/name).read_bytes()!=raw:
            raise RuntimeError('Source drift: '+name)
    return result


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--mean',type=Path,required=True)
    parser.add_argument('--size',type=int,default=6);parser.add_argument('--variance',type=float,default=2.)
    parser.add_argument('--samples',type=int,default=4);parser.add_argument('--seed',type=int,default=20261008)
    parser.add_argument('--current',action='store_true');parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=compute_source(args.mean,size=args.size,variance=args.variance,samples=args.samples,seed=args.seed,current=args.current)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    keys=['actual_Href_energy_estimate_per_V','sampling_energy_standard_error','elapsed_seconds']
    if args.current: keys+=['retarded_initial_tangent_estimate','sampling_tangent_standard_error']
    print(json.dumps({key:result[key] for key in keys}));print('Natural exit:',args.output)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
