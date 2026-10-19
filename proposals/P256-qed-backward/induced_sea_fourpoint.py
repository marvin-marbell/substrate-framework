#!/usr/bin/env python3
"""NEW native filled-sea electromagnetic derivatives through four photons.

Literal one/two-link Peierls vertices and both CAR bands, no imported QED box.
Finite antiperiodic grids price Ward identities exactly at their allowed q;
bulk quadrature is explicitly distinct and needs its own alias/convergence price.
This is a sea supplier, not a complete radiative Compton or photon-vacuum admission.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict,replace
import hashlib,itertools,json,math,os
from pathlib import Path
import time
from typing import Any,Iterator,cast
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
ROOT=Path(__file__).resolve().parent
SOURCE_NAMES=('induced_sea_fourpoint.py','induced_electron_selfenergy.py',
              'induced_photon_response.py','induced_photon_mode_frame.py',
              'induced_phi_backreaction.py','improved_phi_band.py','rotation_band.py',
              'sea_selection.py','soft_budget.py','local_orbital_stability.py','compton.py')
SOURCE_SNAPSHOTS={name:(ROOT/name).read_text() for name in SOURCE_NAMES}
SOURCE_HASHES={name:hashlib.sha256(text.encode()).hexdigest() for name,text in SOURCE_SNAPSHOTS.items()}
import numpy as np
from induced_electron_selfenergy import selected_config
from induced_photon_response import SpectralResponse
for name,digest in SOURCE_HASHES.items():
    if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:
        raise RuntimeError(f'Source changed while importing {name}')


def ordered_blocks(n:int) -> Iterator[tuple[tuple[int,...],...]]:
    """Labeled functional derivatives of -Tr log(D0-dH), including contacts."""
    for count in range(1,n+1):
        for assignment in itertools.product(range(count),repeat=n):
            if len(set(assignment))==count:
                yield tuple(tuple(i for i,block in enumerate(assignment) if block==j)
                            for j in range(count))


def occupied_divided_difference(poles,occupied):
    """Confluent zero-T contour integral, no division within equal occupations.

    Sort the step-function labels, not nearly equal pole values. Every nonzero
    recurrence spans an occupied and empty pole, so its denominator has real
    magnitude at least their positive gap sum. Equal/near poles on the SAME
    side have EXACTLY zero divided differences; there is no frequency floor.
    """
    permutation=sorted(range(len(poles)),key=lambda i:not occupied[i])
    values=[poles[i] for i in permutation]
    labels=[occupied[i] for i in permutation]
    if len(values)==1:return np.full_like(values[0],float(labels[0])-.5)
    row=[np.full_like(values[0],float(label)) for label in labels]
    for width in range(1,len(values)):
        next_row=[]
        for i in range(len(values)-width):
            if labels[i]==labels[i+width]:next_row.append(np.zeros_like(values[0]))
            else:
                denominator=values[i+width]-values[i]
                if np.any(denominator.real<=0):raise ArithmeticError('Contour gap/sign ordering violated')
                next_row.append((row[i+1]-row[i])/denominator)
        row=next_row
    return row[0]


def primitive_vertex(model,k,legs,block):
    """Hamiltonian functional derivative, physical A, all actual path matrices."""
    n=len(block);cfg=model.config
    result=np.zeros((len(k),4,4),complex)
    if n==1:result+=legs[block[0]]['polarization'][0]*np.eye(4)
    for axis,length,matrix in model.paths:
        product=1.+0j
        for i in block:
            leg=legs[i]
            path_sum=sum(np.exp(1j*cfg.a*leg['q'][axis]*(ell-(length-1)/2))
                         for ell in range(length)).real
            product*=path_sum*leg['polarization'][axis+1]
        phase=np.exp(1j*length*k[:,axis])[:,None,None]
        forward=phase*matrix
        result+=product*((-1j*cfg.a)**n*forward+(1j*cfg.a)**n*forward.conj().transpose(0,2,1))
    return result


def sea_vertex(config,legs: list[dict[str,Any]],*,finite_N=None,order=4) -> dict[str,Any]:
    """Complete Euclidean and certified subthreshold-continued native sea vertex.

    legs: {q: physical 3-vector, xi: complex Euclidean frequency,
           polarization: complex (A0,Ax,Ay,Az)}; all legs INCOMING, sums zero.
    Return actual complex kernel per physical volume and native contact groups.
    Finite_N uses exact discrete antiperiodic momenta, not bulk substitution.
    """
    n=len(legs)
    if not 1<=n<=4:raise ValueError('Native supplier supports orders1 through4')
    legs=[dict(q=np.asarray(leg['q'],float),xi=complex(leg['xi']),
               polarization=np.asarray(leg['polarization'],complex)) for leg in legs]
    if any(leg['q'].shape!=(3,) or leg['polarization'].shape!=(4,) for leg in legs):
        raise ValueError('Physical q and four-potential polarization required')
    if np.linalg.norm(sum(leg['q'] for leg in legs))>1e-11 or abs(sum(leg['xi'] for leg in legs))>1e-11:
        raise ValueError('Closed sea loop requires exact momentum/frequency conservation')
    # Only physical REAL energy displacements Re(i*xi) enter the analytic
    # strip. Virtual Euclidean frequencies may be unbounded. Occupations
    # remain the ORIGINAL CAR band labels, never Theta(shifted pole.real).
    energy_range=max(abs(sum(-legs[i]['xi'].imag for i in range(n) if mask&(1<<i)))
                     for mask in range(1,1<<n))
    species=config.species or ((config.flavors,config.m),)
    global_gap=min(mass for _,mass in species)*math.sqrt(config.g0)
    if energy_range>=2*global_gap:
        raise ValueError('Real external-energy subset reaches the certified pair threshold; explicit cut/width treatment required')
    analytic=dict(original_primitive_gap_lower=global_gap,
        largest_absolute_real_energy_subset=energy_range,pair_strip_margin=2*global_gap-energy_range,
        proof='Native d0=Phi0*(m+r/a sum(1-cos theta)^2), m,r>0 and orthogonal selected Phi: E>=m*sqrt(G00). Every route interval is a proper external subset, so opposite-CAR pole real separation >=2gap-max|Re(i sum xi)|. Original occupations are unchanged; virtual Euclidean imaginary energies are not bounded.',
        continuation='Unique analytic continuation in this certified no-pair-cut domain. No above-threshold retarded/Feynman discontinuity is supplied.')
    if finite_N is not None:
        if finite_N<3:raise ValueError('Finite AP grid N>=3 required')
        for leg in legs:
            modes=config.a*finite_N*leg['q']/(2*math.pi)
            if np.max(abs(modes-np.rint(modes)))>1e-10:
                raise ValueError('Finite sea Ward identity needs actual integer photon modes')
    groups=np.zeros(n,complex);minimum_gap=math.inf;points=0
    partitions=list(ordered_blocks(n))
    for count,mass in species:
        model=SpectralResponse(replace(config,m=mass,flavors=1,species=()),order)
        if finite_N is None:chunks=model.chunks()
        else:
            p=(2*np.arange(finite_N)+1)*math.pi/finite_N
            grid=np.stack(np.meshgrid(p,p,p,indexing='ij'),axis=-1).reshape(-1,3)
            chunks=((grid,np.full(len(grid),1/finite_N**3)),)
        for k,weight in chunks:
            points+=len(k)
            states={};vertices={};shifts={}
            for mask in range(1<<n):
                subset=[i for i in range(n) if mask&(1<<i)]
                qsum=sum((legs[i]['q'] for i in subset),np.zeros(3))
                xsum=sum(legs[i]['xi'] for i in subset)
                shifted=k-config.a*qsum
                energy,lower,upper=model.projectors(shifted)
                minimum_gap=min(minimum_gap,float(np.min(energy)))
                states[mask]=(energy,(lower,upper),xsum)
                shifts[mask]=qsum
            for blocks in partitions:
                green_count=len(blocks)
                mask=0;route=[]
                for block in blocks:
                    route.append(mask);key=(mask,block)
                    if key not in vertices:
                        qblock=sum((legs[i]['q'] for i in block),np.zeros(3))
                        middle=k-config.a*(shifts[mask]+qblock/2)
                        vertices[key]=primitive_vertex(model,middle,legs,block)
                    mask|=sum(1<<i for i in block)
                for sign_tuple in itertools.product((0,1),repeat=green_count):
                    signs=cast(tuple[int,...],sign_tuple)
                    if green_count>1 and len(set(signs))==1:
                        continue  # Exact zero contour term; no dense matrix work.
                    poles=[];occupied=[]
                    for start,sign in zip(route,signs):
                        energy,_,xi=states[start]
                        poles.append((2*sign-1)*energy+1j*xi);occupied.append(sign==0)
                    matrix=states[route[0]][1][signs[0]]@vertices[route[0],blocks[0]]
                    for start,block,sign in zip(route[1:],blocks[1:],signs[1:]):
                        matrix=matrix@states[start][1][sign]@vertices[start,block]
                    coefficient=occupied_divided_difference(poles,occupied)
                    trace=np.trace(matrix,axis1=1,axis2=2)
                    groups[green_count-1]+=count/config.a**3*np.sum(weight*coefficient*trace)/green_count
    if minimum_gap<=0:raise ArithmeticError('Fixed half-filled primitive crosses zero')
    total=sum(groups)
    return dict(value=total,groups=groups,minimum_single_particle_gap=minimum_gap,
        quadrature_points=points,ordered_partitions=len(partitions),finite_N=finite_N,order=order,
        contour='Exact original occupied/empty confluent contour differences; complex Euclidean frequencies only within the certified real-energy strip. Both CAR bands and all contacts; no pole/frequency floor.',
        group_meaning='group m: m Green factors; n4 m1=h4, m2=h1h3/h2h2, m3=h1h1h2, m4=literal current/density box',
        analytic_domain=analytic,
        scope='Native sea kernel ONLY at fixed explicitly supplied uniform mean and half CAR reference. No photon-vacuum, renormalized matter/source or radiative Compton promotion.')


def encode(value) -> Any:
    if isinstance(value,complex):return dict(real=value.real,imag=value.imag)
    if isinstance(value,np.ndarray):return [encode(x) for x in value]
    if isinstance(value,dict):return {key:encode(x) for key,x in value.items()}
    if isinstance(value,(list,tuple)):return [encode(x) for x in value]
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--N',type=int,default=6)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();started=time.monotonic()
    cfg,mean=selected_config('base',10);unit=2*math.pi/(cfg.a*args.N)
    y=np.array([0.,0.,1.,0.]);x=np.array([0.,1.,0.,0.])
    static_legs=[dict(q=np.array([sign*unit,0.,0.]),xi=0.,polarization=y)
                 for sign in (1,-1,1,-1)]
    static=sea_vertex(cfg,static_legs,finite_N=args.N)
    dynamic_legs=[dict(q=unit*np.array(q,float),xi=xi,polarization=np.asarray(pol,complex))
                  for q,xi,pol in (((1,0,0),.37,(.3,.2,.7,-.4)),
                                   ((0,1,0),-.21,(-.2,.5,-.3,.6)),
                                   ((0,0,1),.19,(.1,-.4,.6,.2)),
                                   ((-1,-1,-1),-.35,(.4,.3,-.2,.5)))]
    dynamic=sea_vertex(cfg,dynamic_legs,finite_N=args.N)
    wards=[]
    for index in range(4):
        replaced=[dict(leg) for leg in dynamic_legs]
        q=replaced[index]['q'];xi=replaced[index]['xi']
        replaced[index]['polarization']=np.r_[1j*xi,2*np.sin(cfg.a*q/2)/cfg.a]
        result=sea_vertex(cfg,replaced,finite_N=args.N)
        box_only=result['groups'][-1]
        scale=sum(abs(x) for x in result['groups'])
        relative=abs(result['value'])/max(1e-30,scale)
        wards.append(dict(leg=index,total=result['value'],groups=result['groups'],relative_residual=relative,
                          missing_contact_box_defect=abs(box_only)))
        if relative>2e-10:raise ArithmeticError('Full native sea four-photon Ward contraction fails')
    cthree_legs=[dict(q=unit*np.array(q,float),xi=xi,polarization=pol) for q,xi,pol in (
        ((1,0,0),.37,(.2,.3,.7,-.1)),((0,1,0),-.21,(.1,-.4,.2,.8)),((-1,-1,0),-.16,(.3,.6,-.2,.5)))]
    cthree=sea_vertex(cfg,cthree_legs,finite_N=args.N)
    causal=[]
    for regulator,virtual in ((1e-5,False),(1e-6,False),(1e-6,True)):
        continued=[dict(leg,xi=(regulator-.1j)*leg['xi']) for leg in dynamic_legs]
        if virtual:
            continued[0]['xi']+=128
            continued[1]['xi']-=128
        measured=sea_vertex(cfg,continued,finite_N=args.N)
        contractions=[]
        for index in range(4):
            replaced=[dict(leg) for leg in continued]
            replaced[index]['polarization']=np.r_[1j*replaced[index]['xi'],
                2*np.sin(cfg.a*replaced[index]['q']/2)/cfg.a]
            proof=sea_vertex(cfg,replaced,finite_N=args.N)
            relative=abs(proof['value'])/max(1e-30,sum(abs(x) for x in proof['groups']))
            if relative>2e-10:raise ArithmeticError('Subthreshold-continued four-photon Ward fails')
            contractions.append(dict(leg=index,relative=relative,value=proof['value']))
        causal.append(dict(regulator=regulator,unbounded_virtual_Euclidean_pair=128 if virtual else 0,
            legs=continued,vertex=measured,Ward=contractions,
            net_frequency=sum(leg['xi'] for leg in continued),
            prescription='Conserving xi=(eta-0.1i)*original xi, giving real energies 0.037,-0.021,0.019,-0.035. Optional virtual pair +/-128 adds no physical real energy. Unique subthreshold strip, not an above-cut causal prescription.'))
    output=encode(dict(config=asdict(cfg),mean_provenance=mean,static=static,dynamic=dynamic,wards=wards,
        actual_dynamic_three_photon=cthree,
        causal_subthreshold=causal,
        static_E_b4_comparison=dict(kernel=static['value'],energy_b4_per_cell_prediction=static['value']/(64*cfg.a),
            formula='Real y-link phase b*cos(2pi*x/N): energy per cell b4 coefficient = Gamma4(q,-q,q,-q)/(64*a). All4potentials incoming. Independent earlier finite full-H spectrum witness retained separately.'),
        source_hashes=SOURCE_HASHES,source_snapshots=SOURCE_SNAPSHOTS,elapsed_seconds=time.monotonic()-started,
        source_unchanged_since_import={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest for name,digest in SOURCE_HASHES.items()},
        remaining_scope='This supplies actual full-frequency native SEA Gamma4, necessary in D second background derivative. It does not yet supply complete gauged electron Gamma2 or integrated D Gamma4 D self-energy derivative; no placeholder/fallback is returned.'))
    if any(not output['source_unchanged_since_import'][name] for name in ('induced_sea_fourpoint.py','induced_electron_selfenergy.py','induced_photon_response.py','induced_photon_mode_frame.py')):
        raise RuntimeError('Owned/frozen primitive or sea source changed during execution')
    args.output.write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print(json.dumps({key:output[key] for key in ('static','dynamic','wards','actual_dynamic_three_photon','causal_subthreshold','static_E_b4_comparison','elapsed_seconds')},indent=2))
    print(f'Natural exit: {args.output}; actual finite-frequency filled-sea four-photon supplier, no full radiative admission')


if __name__=='__main__':main()
