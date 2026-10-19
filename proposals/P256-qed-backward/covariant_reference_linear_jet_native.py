#!/usr/bin/env python3
"""Full CAR exterior-sector native first jet of the stated comparison transfer."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCE_BYTES_AT_IMPORT = {name:(ROOT/name).read_bytes() for name in (
    'covariant_reference_linear_jet_native.py','covariant_photon_energy_control.py',
    'covariant_induced_photon.py','sea_selection.py','rotation_band.py')}
from covariant_induced_photon import Parameters, WilsonConsumer, phi_frame, soft_photon_transfer_vertex, log_first
from covariant_photon_energy_control import reference_linear_log_jet_control


def exterior(matrix: np.ndarray, occupations: list[tuple[int, ...]]) -> np.ndarray:
    return np.array([[np.linalg.det(matrix[np.ix_(row,col)]) for col in occupations]
                     for row in occupations],dtype=complex)


def lift_bilinear(matrix: np.ndarray, occupations: list[tuple[int, ...]]) -> np.ndarray:
    lookup={state:i for i,state in enumerate(occupations)}
    result=np.zeros((len(occupations),len(occupations)),dtype=complex)
    for column,state in enumerate(occupations):
        for removed,j in enumerate(state):
            remaining=list(state);remaining.pop(removed)
            for i in range(len(matrix)):
                if i in remaining:
                    continue
                insertion=sum(k<i for k in remaining)
                final=tuple(sorted(remaining+[i]))
                result[lookup[final],column]+=(-1)**(removed+insertion)*matrix[i,j]
    return result


def matrix_log(matrix: np.ndarray) -> np.ndarray:
    values,vectors=np.linalg.eigh(matrix)
    if np.min(values)<=0:
        raise ArithmeticError('Actual native transfer is not strictly positive')
    return (vectors*np.log(values))@vectors.conj().T


def run(p: Parameters) -> dict[str,Any]:
    gamma,_=phi_frame(p);consumer=WilsonConsumer(gamma,.01,1.)
    step=2*math.pi/p.spatial_size
    pin=np.rint(np.array([.2,.1,.3])/step)*step
    q=np.array([0.,0.,round(.03/step)*step]);polarization=np.array([1.,0.,0.])
    native=soft_photon_transfer_vertex(gamma,.01,1.,pin,q,polarization,emission=True)
    _,basis=np.linalg.eigh(gamma[0]);physical=np.column_stack((basis[:,2:],basis[:,:2]))
    incoming=physical.conj().T@native['initial_canonical_frame']
    outgoing=physical.conj().T@native['final_canonical_frame']
    raw=outgoing.conj().T@native['native_field_derivative']@incoming
    ein=consumer.energy(pin);eout=consumer.energy(pin-q)
    eigenvalues=np.exp(np.repeat([-ein,ein,-eout,eout],2))
    derivative=np.zeros((8,8),dtype=complex)
    derivative[4:,:4]=raw;derivative[:4,4:]=raw.conj().T
    root=np.sqrt(eigenvalues)
    sqrt_derivative=derivative/(root[:,None]+root[None,:])
    r1=sqrt_derivative/root[None,:]
    occupations=list(itertools.combinations(range(8),5))
    sea_indices=(2,3,6,7)
    sea=float(np.prod(eigenvalues[list(sea_indices)]))
    sf0_values=np.array([np.prod(root[list(state)])/math.sqrt(sea) for state in occupations])
    ph_dimension=4;omega=native['photon_energy']
    annihilation=np.diag(np.sqrt(np.arange(1,ph_dimension)),1)
    field=(annihilation+annihilation.T)*native['oscillator_normalization']
    ph_transfer=np.exp(-omega*np.arange(ph_dimension))
    k0=np.kron(sf0_values**2,ph_transfer)
    r1_joint=np.kron(lift_bilinear(r1,occupations),field)
    k1=r1_joint*k0[None,:]+k0[:,None]*r1_joint.conj().T
    whitened=k1/np.sqrt(k0[:,None]*k0[None,:])
    cp_kernel=np.array([[math.sqrt(x*y)*log_first(float(x),float(y)) for y in k0] for x in k0])
    h1=-cp_kernel*whitened
    if np.linalg.norm(h1-h1.conj().T)>1e-10:
        raise ArithmeticError('Actual first Hamiltonian jet is not Hermitian')
    heat=np.kron(np.eye(len(occupations)),np.diag(ph_transfer))
    field_values,field_vectors=np.linalg.eigh(field)
    transfer0=np.diag(eigenvalues)
    def actual_transfer(epsilon: float) -> np.ndarray:
        sf=np.zeros_like(h1)
        for node,value in enumerate(field_values):
            t=transfer0+epsilon*value*derivative
            ev,vec=np.linalg.eigh(t)
            if np.min(ev)<=0:
                raise ArithmeticError('Native positive transfer direction leaves its domain')
            square=(vec*np.sqrt(ev))@vec.conj().T
            fock=exterior(square,occupations)/math.sqrt(sea)
            photon=np.outer(field_vectors[:,node],field_vectors[:,node].conj())
            sf+=np.kron(fock,photon)
        return sf@heat@sf.conj().T
    finite=[]
    for epsilon in (1e-3,3e-4,1e-4):
        direct=-(matrix_log(actual_transfer(epsilon))-matrix_log(actual_transfer(-epsilon)))/(2*epsilon)
        finite.append(dict(epsilon=epsilon,operator_difference=float(np.linalg.norm(direct-h1,2))))
    sea_set=set(sea_indices)
    lookup={state:i for i,state in enumerate(occupations)}
    charged=np.array([[h1[lookup[tuple(sorted(sea_set|{4+a}))]*ph_dimension+1,
                          lookup[tuple(sorted(sea_set|{b}))]*ph_dimension]
                       for b in range(2)] for a in range(2)])
    expected=native['charged_vertex_without_oscillator']*native['oscillator_normalization']
    # The occupied wedge fixes a common positive-electron CAR phase here.
    charged_defect=float(np.linalg.norm(charged-expected))
    if charged_defect>1e-10 or finite[-1]['operator_difference']>2e-5:
        raise ArithmeticError('Native full-CAR derivative fails its physical charged matrix element')
    norm_h=float(np.linalg.norm(h1,2));norm_x=float(np.linalg.norm(whitened,2))
    if norm_h>norm_x+1e-10:
        raise ArithmeticError('Actual CP contraction violates its operator bound')
    inputs=dict(spatial_size=p.spatial_size,spacing=.01,mass=1.,incoming_momentum=pin.tolist(),
                emitted_momentum=q.tolist(),polarization=polarization.tolist(),
                one_body_modes=8,canonical_charged_number=5,neutral_sea_number=4,
                photon_Hermite_dimension=ph_dimension,finite_derivative_steps=[1e-3,3e-4,1e-4],
                native_path='Positive T1+epsilon*A*actual Peierls derivative; tests only first derivative, not higher Peierls jets')
    return dict(input_descriptor=inputs,input_descriptor_sha256=hashlib.sha256(json.dumps(inputs,sort_keys=True).encode()).hexdigest(),
        source_at_import={k:hashlib.sha256(v).hexdigest() for k,v in SOURCE_BYTES_AT_IMPORT.items()},
        source_bytes_at_import={k:v.decode() for k,v in SOURCE_BYTES_AT_IMPORT.items()},
        canonical_full_exterior_dimension=len(occupations),joint_dimension=len(k0),
        native_positive_reference_transfer_minimum=float(np.min(k0)),
        exact_first_log_jet_operator_norm=norm_h,whitened_first_transfer_jet_operator_norm=norm_x,
        cp_contraction_margin=norm_x-norm_h,finite_actual_log_derivatives=finite,
        charged_soft_emission_matrix_defect=charged_defect,
        adverse_drop_occupied_CAR_sector='Native calculation includes every N=5 wedge state, including real PH states; it is not a single charged particle trace.',
        analytic_control=reference_linear_log_jet_control(p),
        scope='Actual positive comparison-transfer derivative on two physical momentum vertices and all56 CAR wedges, four Hermite levels. First source n=0 to n=1 matrix elements are unaffected by the upper photon truncation. This is NOT the macro microscopic Gauss source admission or a proof of all mixed fifth-order words.')


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    started=time.monotonic();result=run(Parameters(kappa_power=400));result['runtime_seconds']=time.monotonic()-started
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:result[key] for key in ('source_at_import','input_descriptor_sha256','joint_dimension',
        'exact_first_log_jet_operator_norm','whitened_first_transfer_jet_operator_norm','finite_actual_log_derivatives',
        'charged_soft_emission_matrix_defect','runtime_seconds')}))


if __name__=='__main__':
    main()
