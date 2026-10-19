#!/usr/bin/env python3
"""Nonlinear two-cell Wilson SF coupled to a true Gaussian coexact oscillator.

All56 canonical N=5 CAR states are retained. This is an active comparison
transfer, not an auxiliary/Root/full microscopic Gauss source admission.
"""
from __future__ import annotations

import argparse
import cmath
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
from scipy.linalg import logm, sqrtm

ROOT=Path(__file__).resolve().parent
SOURCE_BYTES_AT_IMPORT={name:(ROOT/name).read_bytes() for name in (
    'covariant_nonlinear_sf_logjet_native.py','covariant_reference_linear_jet_native.py',
    'covariant_reference_log_jet_words.py','covariant_photon_energy_control.py',
    'covariant_induced_photon.py','sea_selection.py','rotation_band.py')}
from covariant_induced_photon import Parameters, phi_frame
from covariant_reference_linear_jet_native import exterior
from covariant_reference_log_jet_words import sf_majorants
import itertools


def run(p: Parameters) -> dict[str,Any]:
    gamma,_=phi_frame(p);_,basis=np.linalg.eigh(gamma[0])
    plus,minus=basis[:,2:],basis[:,:2]
    spin=np.array([plus.conj().T@g@minus for g in gamma[1:]])
    ix=np.eye(2);sx=np.array([[0.,1.],[1.,0.]])
    identity=np.eye(4);m0=math.expm1(.01);momentum=np.array([.2,.1,.3])
    def onebody(angle: complex) -> tuple[np.ndarray,complex]:
        px,py,pz=momentum
        cy=np.diag(np.cos(py+np.array([angle,-angle])))
        dy=np.diag(np.sin(py+np.array([angle,-angle])))
        b=(4+m0-math.cos(pz))*identity-math.cos(px)*np.kron(sx,ix)-np.kron(cy,ix)
        f=1j*(math.sin(px)*np.kron(sx,spin[0])+np.kron(dy,spin[1])+math.sin(pz)*np.kron(ix,spin[2]))
        bar=-1j*(math.sin(px)*np.kron(sx,spin[0].conj().T)+np.kron(dy,spin[1].conj().T)+math.sin(pz)*np.kron(ix,spin[2].conj().T))
        inverse=np.linalg.inv(b)
        t=np.block([[inverse,-inverse@f],[-bar@inverse,b+bar@inverse@f]])
        return t,complex(np.linalg.det(b))
    t0,det0=onebody(0.)
    eigenvalues=np.linalg.eigvalsh(t0)
    if np.min(eigenvalues)<=0 or np.max(abs(np.log(eigenvalues)))>=math.log(8):
        raise ArithmeticError('Actual nonlinear Wilson reference misses positive log8 spectrum')
    sea=float(np.prod(eigenvalues[-4:]))
    occupations=list(itertools.combinations(range(8),5))
    photon_dimension=4;omega=2*math.asinh(1.)
    annihilation=np.diag(np.sqrt(np.arange(1,photon_dimension)),1)
    field=(annihilation+annihilation.T)/math.sqrt(2*math.sinh(omega))
    coordinates,vectors=np.linalg.eigh(field)
    heat=np.kron(np.eye(len(occupations)),np.diag(np.exp(-omega*np.arange(photon_dimension))))
    photon_projectors=[np.outer(vectors[:,node],vectors[:,node].conj()) for node in range(photon_dimension)]
    # The real q=pi mode has normalized spatial pattern (+1,-1)/sqrt(2).
    def transfer(coupling: complex) -> np.ndarray:
        sf=np.zeros((len(occupations)*photon_dimension,)*2,dtype=complex)
        for coordinate,projector in zip(coordinates,photon_projectors):
            t,detb=onebody(coupling*coordinate/math.sqrt(2))
            square=np.asarray(sqrtm(t),dtype=np.complex128)
            fock=cmath.sqrt(detb/det0)*exterior(square,occupations)/math.sqrt(sea)
            sf+=np.kron(fock,projector)
        return sf@heat@sf
    def hamiltonian(coupling: complex) -> np.ndarray:
        k=transfer(coupling)
        if abs(coupling.imag)<1e-15:
            values,frames=np.linalg.eigh(k)
            if np.min(values)<=0:
                raise ArithmeticError('Actual nonlinear coupled transfer is not positive')
            return -(frames*np.log(values))@frames.conj().T
        return -np.asarray(logm(k),dtype=np.complex128)
    h0=hamiltonian(0j)
    nodes=24
    def circle(radius: float) -> list[np.ndarray]:
        samples=[hamiltonian(radius*cmath.exp(2j*math.pi*(j+.5)/nodes)) for j in range(nodes)]
        return [sum((samples[j]*cmath.exp(-2j*math.pi*n*(j+.5)/nodes) for j in range(nodes)),
                    np.zeros_like(h0))/(nodes*radius**n) for n in range(6)]
    first=circle(.04);second=circle(.06)
    rows=[]
    for n in range(1,6):
        rows.append(dict(grade=n,actual_coefficient_operator_norm=float(np.linalg.norm(second[n],2)),
                         two_circle_difference_estimate=float(np.linalg.norm(second[n]-first[n],2)),
                         hermiticity_defect_estimate=float(np.linalg.norm(second[n]-second[n].conj().T,2))))
    polynomial=[]
    for coupling in (.05,.1,.2):
        actual=hamiltonian(complex(coupling))
        fourth=h0+sum((coupling**n*second[n] for n in range(1,5)),np.zeros_like(h0))
        fifth=fourth+coupling**5*second[5]
        polynomial.append(dict(coupling=coupling,actual_fifth_order_remainder_norm=float(np.linalg.norm(actual-fourth,2)),
                               remainder_divided_by_coupling_power5=float(np.linalg.norm(actual-fourth,2))/coupling**5,
                               actual_after_fifth_subtraction_norm=float(np.linalg.norm(actual-fifth,2))))
    sf=sf_majorants(2,5)
    inputs=dict(native_geometry=[2,1,1],constant_native_link_torons=momentum.tolist(),spacing=.01,mass=1.,
                photon_mode=[math.pi,0.,0.],photon_polarization=[0.,1.,0.],normalized_spatial_pattern=[1/math.sqrt(2),-1/math.sqrt(2)],
                photon_Hermite_dimension=photon_dimension,charged_CAR_number=5,neutral_sea_CAR_number=4,
                cauchy_nodes=nodes,cauchy_radii=[.04,.06],real_comparison_couplings=[.05,.1,.2],
                source_recipe_status='Native operator consumer only; no macro source or receiver is replaced')
    return dict(input_descriptor=inputs,input_descriptor_sha256=hashlib.sha256(json.dumps(inputs,sort_keys=True).encode()).hexdigest(),
        source_at_import={k:hashlib.sha256(v).hexdigest() for k,v in SOURCE_BYTES_AT_IMPORT.items()},
        source_bytes_at_import={k:v.decode() for k,v in SOURCE_BYTES_AT_IMPORT.items()},
        actual_free_onebody_eigenvalues=eigenvalues.tolist(),canonical_CAR_dimension=len(occupations),
        joint_dimension=len(h0),actual_nonlinear_log_jet_rows=rows,actual_real_coupling_remainders=polynomial,
        native_SF_rational_majorants=[str(x) for x in sf['SF_relative_coefficients']],
        physical_operator='Exact det(B)^1/2 Gamma(sqrt(T1)) for nonlinear exp(i g A_y(x)) links, combined with the positive Gaussian mode heat. AllN5 CAR wedges includingPH retained. Complex circles are numerical derivatives of this actual transfer; real samples use its observed positive eigenvalues.',
        numerical_estimate_status='Two-circle and Hermiticity differences are observed numerical estimates, NOT outward interval bounds.',
        scope='Native active-SF/Gaussian comparison quantum transfer at two cells. Short winding makes this unsuitable for identifying the microscopic auxiliary Wilson-loop coefficient on this geometry. The Gaussian mode is the comparison kernel, not a newly introduced bare gauge term. No full microscopic Gauss projection, macro prepared source admission, Coulomb or compact rotor/magnetic remainder is claimed.')


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    started=time.monotonic();result=run(Parameters(kappa_power=400));result['runtime_seconds']=time.monotonic()-started
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:result[key] for key in ('source_at_import','input_descriptor_sha256','joint_dimension',
        'actual_nonlinear_log_jet_rows','actual_real_coupling_remainders','runtime_seconds')}))


if __name__=='__main__':
    main()
