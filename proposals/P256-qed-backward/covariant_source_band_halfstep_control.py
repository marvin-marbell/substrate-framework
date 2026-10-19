#!/usr/bin/env python3
"""Source-band linear square-root estimate on actual positive native Haar factors."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
from scipy.linalg import eigh

ROOT=Path(__file__).resolve().parent
SOURCE_BYTES_AT_IMPORT={name:(ROOT/name).read_bytes() for name in (
    'covariant_source_band_halfstep_control.py','covariant_native_haar_range_jets.py',
    'covariant_photon_energy_control.py','covariant_induced_photon.py','sea_selection.py','rotation_band.py')}
from covariant_induced_photon import Parameters,phi_frame
from covariant_native_haar_range_jets import space,auxiliary_half
from covariant_photon_energy_control import gaussian_energy_form_control


def exact_square_half(gamma: np.ndarray,theta: float,coupling: float) -> tuple[np.ndarray,float]:
    B=np.eye(4,dtype=complex)/(.18);F=np.zeros((4,4),complex)
    for edge,(x,y,mu,orientation) in enumerate(((0,1,1,1),(1,2,2,1),(2,3,1,-1),(3,0,2,-1))):
        phase=1+coupling*(np.exp(1j*theta)-1) if edge==3 else 1.
        B[x,y]-=.5*phase;B[y,x]-=.5*np.conjugate(phase)
        z=orientation*gamma[mu,0,1]/2
        F[x,y]+=z*phase;F[y,x]-=z*np.conjugate(phase)
    inverse=np.linalg.inv(B)
    T=np.block([[inverse,-inverse@F],[-F.conj().T@inverse,B+F.conj().T@inverse@F]])
    permutation=[0,4,1,5,2,6,3,7];T=T[np.ix_(permutation,permutation)]
    values,U=eigh(T)
    if values[0]<=0:raise ArithmeticError('Native exact square transfer is not positive')
    return (U*np.sqrt(values))@U.conj().T,float(np.sqrt(np.linalg.det(B).real))


def actual_factor(gamma: np.ndarray,y: np.ndarray,normal_half: np.ndarray,
                  physical: dict[str,Any],coupling: float,grid: int=32) -> np.ndarray:
    middle=[physical['states'][i] for i in physical['middle']]
    light_columns=sorted(set(l for l,_ in middle));aux_columns=sorted(set(a for _,a in middle))
    light_index={l:i for i,l in enumerate(light_columns)};aux_index={a:i for i,a in enumerate(aux_columns)}
    cl=np.array([light_index[l] for l,_ in middle]);ca=np.array([aux_index[a] for _,a in middle])
    lr=np.array(physical['light']);lc=np.array(light_columns)
    ar=np.array(physical['auxiliary']);ac=np.array(aux_columns)
    rows=len(physical['physical']);columns=len(middle);nodes=len(y)
    row_state=np.array([i for i,_,_ in physical['physical']]);row_flux=np.array([n%grid for _,n,_ in physical['physical']])
    result=np.zeros((nodes*rows,nodes*columns),complex)
    for node,value in enumerate(y):
        actual=gamma.copy();actual[1]*=1+float(value);samples=[]
        for theta in 2*math.pi*np.arange(grid)/grid:
            half,pref=exact_square_half(actual,float(theta),coupling)
            light=pref*np.linalg.det(half[lr[:,None,:,None],lc[None,:,None,:]])
            ah,ap=auxiliary_half(actual,.04,float(theta))
            aux=ap*np.linalg.det(ah[ar[:,None,:,None],ac[None,:,None,:]])
            product=light[:,cl][:,None,:]*aux[None,:,ca]
            samples.append(product.reshape(len(physical['states']),columns))
        coefficients=np.fft.fft(np.array(samples),axis=0)/grid
        block=coefficients[row_flux,row_state,:]
        for output in range(nodes):result[output*rows:(output+1)*rows,node*columns:(node+1)*columns]=normal_half[output,node]*block
    return result


def run() -> dict[str,Any]:
    p=Parameters(kappa_power=400);gamma4,_=phi_frame(p)
    values,U=eigh(1j*gamma4[0]@gamma4[1]@gamma4[2]);embedding=U[:,values>0]
    _,frame=eigh(embedding.conj().T@gamma4[0]@embedding);embedding=embedding@frame[:,::-1]
    gamma=np.array([embedding.conj().T@g@embedding for g in gamma4[:3]])
    y=np.array([-.09,0.,.09]);sigma=.03;radius=.12;dy=.09
    k=np.exp(-y*y/(4*sigma*sigma)-sigma*sigma*y*y/(radius*radius*(radius*radius-y*y)));k/=np.linalg.norm(k)
    D=np.zeros((2,3))
    for j in range(2):D[j,j]=math.sqrt(k[j+1]/k[j])/dy;D[j,j+1]=-math.sqrt(k[j]/k[j+1])/dy
    H=2*sigma*sigma*D.T@D;levels,phi=eigh(H)
    normal_half=(phi*np.exp(-levels/2))@phi.T
    physical=space(0,2);F0=actual_factor(gamma,y,normal_half,physical,0.)
    G0=F0.conj().T@F0;values,U=eigh(G0);scale=float(values[-1]);F0/=math.sqrt(scale);values/=scale
    b=math.exp(-.25);selected=values>=b
    band=F0@(U[:,selected]/np.sqrt(values[selected]));band_values=values[selected]
    results=[]
    for coupling in (.05,.01,.001):
        F=actual_factor(gamma,y,normal_half,physical,coupling)/math.sqrt(scale)
        G=F.conj().T@F;av,au=eigh(G)
        if av[0]<=0:raise ArithmeticError('Actual nonzero Haar factor rank was lost')
        inverse=(au/np.sqrt(av))@au.conj().T
        discrepancy=F@(F.conj().T@band)-band*band_values
        sqrt_discrepancy=F@(inverse@(F.conj().T@band))-band*np.sqrt(band_values)
        delta=float(np.linalg.norm(discrepancy,2));actual=float(np.linalg.norm(sqrt_discrepancy,2))
        upper=delta/math.sqrt(b)
        if actual>upper*(1+1e-7)+1e-9:raise ArithmeticError('Actual positive moving-null transfer violates source-band Sylvester estimate')
        union,_=np.linalg.qr(np.column_stack((F0,F)),mode='reduced')
        f=union.conj().T@F;f0=union.conj().T@F0
        global_delta=float(np.linalg.norm(f@f.conj().T-f0@f0.conj().T,2))
        outside=band-F@(np.linalg.solve(G,F.conj().T@band))
        results.append(dict(native_homotopy_parameter=coupling,actual_A_rank=F.shape[1],
            actual_A_has_physical_nullspace=F.shape[0]>F.shape[1],actual_B_low_energy_band_rank=int(selected.sum()),
            actual_operator_discrepancy_on_reference_band=delta,actual_halfstep_discrepancy_on_reference_band=actual,
            proved_linear_Sylvester_halfstep_upper=upper,actual_global_operator_discrepancy=global_delta,
            adverse_global_Holder_sqrt_bound=math.sqrt(global_delta),
            actual_band_leakage_into_moved_A_nullspace_norm=float(np.linalg.norm(outside,2))))
    energy=gaussian_energy_form_control(p,9,.25);cost=energy['maximum_intermediate_energy']/(2*math.log(10))
    inputs=dict(native_active_kappa=.09,native_auxiliary_kappa=.04,active_canonical_number=4,
        auxiliary_canonical_number=2,native_normal_nodes=y.tolist(),normal_sigma=sigma,normal_radius=radius,
        actual_integer_Gauss=True,native_electric_Galerkin_cutoff=2,native_Haar_quadrature=32,
        native_homotopy_parameters=[.05,.01,.001],reference_band_energy=.25,
        macro_conditional_operator_band_error_log10=-314,maximum_word_grade=9)
    return dict(input_descriptor=inputs,input_descriptor_sha256=hashlib.sha256(json.dumps(inputs,sort_keys=True).encode()).hexdigest(),
        source_at_import={name:hashlib.sha256(data).hexdigest() for name,data in SOURCE_BYTES_AT_IMPORT.items()},
        source_bytes_at_import={name:data.decode() for name,data in SOURCE_BYTES_AT_IMPORT.items()},
        actual_native_positive_Haar_normal_Gauss_band_rows=results,
        macro_reference_word_energy=energy['maximum_intermediate_energy'],
        macro_band_inverse_root_log10_price=cost,
        conditional_macro_halfstep_error_log10_upper=-314+cost,
        proof=[
            'For any A,B>=0 on the SAME physical Hilbert, X=sqrtA-sqrtB obeys sqrtA X+X sqrtB=A-B. With Pb=1_[B>=b], XPb=integral0inf exp(-t sqrtA)(A-B)Pb exp(-t sqrtB)dt.',
            'Consequently ||XPb||<=||(A-B)Pb||/sqrtb. No minimum eigenvalue of A, no global B inverse, and no fixed common range are assumed. A may have a nullspace and its nonzero support may move.',
            'The native A=F(t)F(t)dag and B=F(0)F(0)dag are actual canonical detB/Wilson/Haar/integerGauss/fullauxiliary/three-normal-level positive Galerkin transfers, not synthetic diagonal PSD matrices. All auxiliary canonical occupations remain.',
            'A band-selected reference vector can leak into the actual moved A-nullspace; this is measured, not assumed absent. The Sylvester integral prices its contribution linearly in the actual operator-band discrepancy.',
            'The macro price is CONDITIONAL on a supplied normalized operator discrepancy on that actual band. A state-vector or Schatten-density error alone does not imply the required discrepancy. This is not macro source or quartercut admission.'],
        scope='Exact source-band square-root theorem and actual finite positive native discriminator; conditional macro price only.',
        microscopic_admission_proved=False)


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    started=time.monotonic();result=run();result['runtime_seconds']=time.monotonic()-started
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:result[key] for key in ('source_at_import','input_descriptor_sha256',
        'actual_native_positive_Haar_normal_Gauss_band_rows','macro_band_inverse_root_log10_price',
        'conditional_macro_halfstep_error_log10_upper','runtime_seconds')}))


if __name__=='__main__':
    main()
