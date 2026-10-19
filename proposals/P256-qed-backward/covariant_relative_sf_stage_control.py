#!/usr/bin/env python3
"""Actual Root SF-half/whole-step curvature discriminator, no wiring test.

The completed ring pair already groups four halfsteps correctly. Its consumer
API is repaired to carry HALF scalars explicitly; the physical law is unchanged.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
ROOT=Path(__file__).resolve().parent
OWN_BYTES=Path(__file__).read_bytes();LOADED=sys._getframe().f_code
MATCH=LOADED==compile(OWN_BYTES,LOADED.co_filename,'exec',dont_inherit=True,optimize=sys.flags.optimize)
if not MATCH:raise ArithmeticError('Loaded SF stage control differs from captured source')
import numpy as np
from scipy.linalg import eigh
import covariant_relative_ring_boundary as ring


def local_curl_links(shape: tuple[int,int,int],angle: float) -> tuple[list[np.ndarray],dict[str,Any]]:
    sites=list(itertools.product(range(shape[0]),range(shape[1]),range(shape[2])))
    lookup={site:index for index,site in enumerate(sites)};phase=np.zeros((3,len(sites)))
    phase[0,lookup[(0,0,0)]]=angle;phase[0,lookup[(1,0,0)]]=-angle
    def neighbour(site: tuple[int,int,int],axis: int) -> tuple[int,int,int]:
        result=list(site);result[axis]=(result[axis]+1)%shape[axis];return result[0],result[1],result[2]
    links=[]
    for axis in range(3):
        shift=np.zeros((len(sites),len(sites)),complex)
        for site in sites:
            index=lookup[site];shift[index,lookup[neighbour(site,axis)]]=np.exp(1j*phase[axis,index])
        links.append(np.kron(shift,np.eye(4)))
    curls=[]
    for site in sites:
        for axis in range(3):
            for other in range(axis):
                curls.append(float(phase[axis,lookup[site]]+phase[other,lookup[neighbour(site,axis)]]
                    -phase[axis,lookup[neighbour(site,other)]]-phase[other,lookup[site]]))
    torons=[]
    for axis in range(3):
        for site in sites:
            if site[axis]!=0:continue
            current=site;value=0.
            for _ in range(shape[axis]):value+=phase[axis,lookup[current]];current=neighbour(current,axis)
            torons.append(value)
    action=sum(1-math.cos(value) for value in curls)
    ring.require(max(abs(value) for value in torons)==0.,'Local-curl probe preserves EVERY toron')
    ring.require(abs(action-4*(1-math.cos(angle)))<1e-13,'Actual plaquette action contains four local curls, not holonomy')
    return links,dict(shape=shape,sites=len(sites),angle=angle,toron_sums=torons,
        nonzero_plaquette_curls=[value for value in curls if value!=0.],plaquette_action=action)


def physical_stage(gamma: np.ndarray,kappa: float,links: list[np.ndarray]) -> dict[str,Any]:
    modes=len(links[0]);number=modes//2
    zeros=[np.zeros((modes,modes),complex) for _ in range(4)]
    half,whole_prefactor,whole=ring.supplier.physical_spatial_onebody(gamma,kappa,links,zeros,np.eye(modes))
    half_values=eigh(half,eigvals_only=True);whole_values=eigh(whole,eigvals_only=True)
    log_prefactor=math.log(whole_prefactor)
    # Exact canonical minmax identity: every Nc eigenoccupation is retained;
    # largest Gamma eigenvalue is the product of the largest Nc onebody values.
    sf_log=.5*log_prefactor+float(np.sum(np.log(half_values[-number:])))
    whole_log=log_prefactor+float(np.sum(np.log(whole_values[-number:])))
    actual_composed_values=eigh(half@half,eigvals_only=True)
    composed_sf_log=log_prefactor+float(np.sum(np.log(actual_composed_values[-number:])))
    composition_error=float(abs(math.expm1(composed_sf_log-whole_log)))
    half_largest_error=float(abs(math.expm1(2*sf_log-whole_log)))
    ring.require(composition_error<2e-11 and half_largest_error<2e-11,
        'Literal physical sqrt(detB)*Gamma(sqrtT) must square to detB*Gamma(T) at the actual field')
    return dict(physical_modes=modes,canonical_number=number,whole_detBplus=whole_prefactor,
        HALF_scalar_sqrt_detBplus=math.sqrt(whole_prefactor),actual_SF_half_largest_log=sf_log,
        actual_whole_step_largest_log=whole_log,literal_composed_SF_largest_relative_error=composition_error,
        HALF_squared_largest_relative_error=half_largest_error,
        WRONG_direct_raw_detB_per_HALF_whole_largest_log=2*log_prefactor+float(np.sum(np.log(whole_values[-number:]))),
        minimum_physical_onebody_eigenvalue=float(whole_values[0]),
        scope='Exact full physical canonical spectral minmax comparison, not an interacting Slater state or isolated-pole gate.')


def curvature(gamma: np.ndarray,angle: float,kappas: list[float]) -> dict[str,Any]:
    shape=(4,3,1);flat,flat_geometry=local_curl_links(shape,0.);curled,geometry=local_curl_links(shape,angle)
    rows=[]
    for kappa in kappas:
        zero=physical_stage(gamma,kappa,flat);nonzero=physical_stage(gamma,kappa,curled)
        denominator=kappa**4*geometry['plaquette_action']
        correct=(zero['actual_whole_step_largest_log']-nonzero['actual_whole_step_largest_log'])/denominator
        wrong=(zero['WRONG_direct_raw_detB_per_HALF_whole_largest_log']-nonzero['WRONG_direct_raw_detB_per_HALF_whole_largest_log'])/denominator
        rows.append(dict(kappa=kappa,theta0=zero,actual_nonzero_LOCALCURL=nonzero,
            physical_whole_step_spatial_plaquette_coefficient=correct,
            WRONG_direct_raw_detB_per_HALF_spatial_plaquette_coefficient=wrong))
    ring.require(abs(rows[-1]['physical_whole_step_spatial_plaquette_coefficient']-16)<1,
        'Actual frozen Root whole-step curvature approaches the original coefficient16')
    ring.require(abs(rows[-1]['physical_whole_step_spatial_plaquette_coefficient']
        -rows[-1]['WRONG_direct_raw_detB_per_HALF_spatial_plaquette_coefficient'])>2.5,
        'Nonconstant local curvature must discriminate wrong direct half-stage prefactors')
    return dict(geometry=geometry,flat_geometry=flat_geometry,rows=rows,
        interpretation='Wrong direct raw-det HALF consumer is a real nonconstant law change. Historical completed ring pairs had FOUR h factors with whole_detB_z*whole_detB_w, hence already correct grouping; no changed-holonomy or rejected-native-law claim follows.')


def interface_and_pair(gamma: np.ndarray) -> dict[str,Any]:
    sites=2;kappa=.04;theta=.47;y=.00017
    def literal(field_theta: float,normal_y: float) -> tuple[np.ndarray,float,np.ndarray]:
        shift=np.zeros((sites,sites),complex)
        for x in range(sites):shift[x,(x+1)%sites]=np.exp(1j*field_theta/sites)
        links=[np.kron(shift,np.eye(4)),np.eye(4*sites),np.eye(4*sites)]
        shear=np.array([[.7,0.,0.,.4],[0.,.13,0.,0.],[0.,0.,-.11,0.],[.4,0.,0.,-.3]])
        normal=[normal_y*np.kron(np.eye(sites),sum((shear[mu,nu]*gamma[nu] for nu in range(4)),np.zeros((4,4),complex))) for mu in range(4)]
        return ring.supplier.physical_spatial_onebody(gamma,kappa,links,normal,np.eye(4*sites))
    half,half_prefactor,_=ring.ring_half(gamma,sites,kappa,theta,y)
    number=2*sites;states=ring.supplier.occupations(4*sites,number)
    sf=half_prefactor*ring.supplier.exterior(half,states)
    _,literal_whole_prefactor,literal_whole=literal(theta,y)
    physical_whole=literal_whole_prefactor*ring.supplier.exterior(literal_whole,states)
    actual_error=float(np.linalg.norm(sf@sf-physical_whole,2)/np.linalg.norm(physical_whole,2))
    ring.require(actual_error<1e-10,'Literal full70 canonical physical SF-half composition')
    sf_scale=ring.common_vacuum_scale(gamma,sites,kappa)
    literal_flat_half,literal_flat_prefactor,literal_flat_T=literal(0.,0.)
    physical_sf_largest=math.sqrt(literal_flat_prefactor)*float(np.prod(eigh(literal_flat_half,eigvals_only=True)[-number:]))
    ring.require(abs(sf_scale/physical_sf_largest-1)<1e-10,'Declared SF vacuum constant has actual HALF stage')
    _,p2,_=ring.ring_half(gamma,sites,kappa,.23,-.00011)
    _,literal_second_prefactor,_=literal(.23,-.00011)
    literal_whole_vacuum_scale=literal_flat_prefactor*float(np.prod(eigh(literal_flat_T,eigvals_only=True)[-number:]))
    old_grouped_factor=literal_whole_prefactor*literal_second_prefactor/literal_whole_vacuum_scale**2
    new_explicit_four_half_factor=(half_prefactor/sf_scale)**2*(p2/sf_scale)**2
    return dict(literal_full70_SF_square_relative_error=actual_error,
        actual_SF_half_vacuum_scale=sf_scale,actual_SF_half_largest=physical_sf_largest,
        new_explicit_four_HALF_scalar_factor=new_explicit_four_half_factor,
        historical_correctly_grouped_WHOLE_scalar_factor=old_grouped_factor,
        grouping_relative_difference=abs(new_explicit_four_half_factor/old_grouped_factor-1),
        meaning='Full-component normal shear and actual holonomies. Grouping orientation is retained; no physical native-law rejection from the interface correction.')


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--angle',type=float,default=.37);parser.add_argument('--kappas',type=float,nargs='+',default=[.025,.0125,.00625])
    args=parser.parse_args();start=time.monotonic();gamma,_=ring.supplier.phi_frame(ring.supplier.Parameters())
    curve=curvature(gamma,args.angle,args.kappas);interface=interface_and_pair(gamma)
    captured={name:ring.supplier.frozen.SNAPSHOT[name] for name in ring.supplier.frozen.COMPILED}
    captured['covariant_canonical_gauss_boundary_kernel']=ring.supplier.OWN_BYTES
    captured['covariant_relative_ring_boundary']=ring.OWN_BYTES;captured[Path(__file__).stem]=OWN_BYTES
    unchanged=all((ROOT/(name+'.py')).read_bytes()==data for name,data in captured.items())
    ring.require(unchanged,'Scientific SF stage sources changed during execution')
    result=dict(plan=ring.PLAN,governing_revision=ring.REVISION,parameters=vars(args)|{'output':str(args.output)},
        actual_local_curvature=curve,actual_SF_interface_and_grouping=interface,
        compiled_source_hashes={name:hashlib.sha256(data).hexdigest() for name,data in captured.items()},
        captured_source_bytes={name:data.decode() for name,data in captured.items()},
        compiled_source_unchanged=unchanged,main_code_matches_captured_source=MATCH,
        elapsed_seconds=time.monotonic()-start,macro_source_admitted=False,
        scope='Corrected explicit half-stage interface, unchanged physically correct completed-pair grouping, actual local-curvature discriminator. No frozen supplier/Source/Root changes, no macro admission.')
    args.output.write_text(json.dumps(ring.supplier.portable(result),indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(args.output),elapsed_seconds=result['elapsed_seconds'],rows=[dict(kappa=r['kappa'],correct=r['physical_whole_step_spatial_plaquette_coefficient'],WRONG_half=r['WRONG_direct_raw_detB_per_HALF_spatial_plaquette_coefficient']) for r in curve['rows']],composition=interface)))
    print('Natural exit: literal full-component SF stages and unchanged grouped pair; local curvature distinguishes wrongful direct raw-det half consumers')

if __name__=='__main__':main()
