#!/usr/bin/env python3
"""Material correlated-normal preparation: physical units and graph discriminator.

PNFF-v3 is a candidate, not a relabeling of PNFF-v2 or its photon evidence.
The executable evaluates a genuinely changed native preparation on the same H.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from fractions import Fraction as F
from pathlib import Path
from typing import Any

if __name__=='__main__':
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
        os.environ[key]='1'

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import expm_multiply

from prepared_neutral_full_field import (
    PreparedQuantumSlice, neutral_descriptor, bump_graph_bounds, require,
)
from noncompact_quantitative_normal_budget import native_normal_budget, ceil_log2
from original_charged_principal import portable
from improved_phi_band import NSTAR

ROOT=Path(__file__).resolve().parent
IDENTITY='PNFF-v3-candidate-correlated-actual-well-Hessian'


def hessian_descriptor() -> dict[str,Any]:
    state=neutral_descriptor()
    state.update(identity=IDENTITY,candidate_not_current_join_state=True,
        predecessor_identity=neutral_descriptor()['identity'],
        normal_wave='normalized correlated Gaussian det(A)^(1/4)/pi^(D/4)*exp(-y^T A y/2), with smooth original chart cutoff; not the v2 radial bump',
        precision='A=G^-1/2*(G^1/2 K G^1/2)^1/2*G^-1/2; A G A=K',
        covariance='E yy^T=(2A)^-1; E pp^T=A/2',
        kinetic='Hkin=p^T G p/2; G=chi*physical normal cometric, chi=Lambda^(-1/2), Phi coefficient chi/2, colour coefficient .3chi (cometric .6chi)',
        Hessian='K=Lambda*K_leading+Lambda^(5/4)*K_alignment +actual mixed derivatives; actual original-heavy whole occupied cluster and original scalar Hessian',
        epsilon='cutoff radius Lambda^(-1/5); Gaussian widths derived from A, not declared epsilon widths',
        quadratic_energy='tr[(G^1/2 K G^1/2)^1/2]/2, above the actual classical local-well constant',
        all64_offsets='all64 amplitude1/8 retained in this candidate; Lambda*v_theta ordinary values not subtracted from H',
        branch_floor='E_floor_theta=Lambda*v_theta-light/reference form allowance-beta, not unknown E0; branchwise excess is not automatically conserved',
        local_well_projection='NOT implemented as final preparation: proposed Dirichlet local-well spectral window plus Agmon boundary control requires its own actualH proof',
        full_field_transfer_admitted=False,full2pi_source_graph_admitted=False)
    return state


def termwise_exponents() -> dict[str,Any]:
    """Exact dimensional algebra; distinguishes scale lemma from full-H proof."""
    budget=native_normal_budget(NSTAR)
    # p^2 coefficient Lambda^-1/2/2 and stiffness Lambda^beta/2.
    # Precision sqrt(k/g), covariance sqrt(g/k)/2, energy sqrt(gk)/2.
    rows=[]
    for label,beta in [('leading',F(1)),('alignment',F(5,4))]:
        precision=(beta+F(1,2))/2
        covariance=-precision
        energy=(beta-F(1,2))/2
        rows.append(dict(sector=label,stiffness_Lambda_power=str(beta),
            kinetic_coefficient_Lambda_power='-1/2',precision_Lambda_power=str(precision),
            covariance_Lambda_power=str(covariance),field_RMS_Lambda_power=str(covariance/2),
            energy_Lambda_power=str(energy),momentum_RMS_Lambda_power=str(precision/2),
            chi_times_momentum_Lambda_power=str(precision/2-F(1,2))))
    # Conserved FIRST energy moment at Lambda^3/8 only supplies distance and
    # chiPi; fullQ-weighted distance using crude Cauchy costs another Lambda^3/8.
    field=F(-5,16)
    momentum=F(-1,16)
    energy=F(3,8)
    return dict(rows=rows,physical_units='same H energy; lattice a=1; hbar=1 is microscopic quantum-law debt',
        covariance_identity='A G A=K; no copying scalar chi into covariance denominator',
        positive_Hessian_lower=dict(leading=str(budget.alpha),alignment=str(budget.stiff)),
        actual_cometric_required=True,alignment_kernel_requires_leading_Hessian=True,
        energy_excess_if_actual_floor_and_boundary_proved_Lambda_power=str(energy),
        distance_RMS_from_local_coercivity_Lambda_power=str(field),
        chiPi_from_excess_Lambda_power=str(momentum),
        fullQ_times_distance_crude_Lambda_power=str(energy+field),
        fullQ_times_chiPi_crude_Lambda_power=str(energy+momentum),
        exact_failed_fullQ_implication='first conserved energy moment alone does not bound fullQ-weighted SOURCE jets; crude multiplication grows Lambda^(1/16) and Lambda^(5/16)',
        ordinary_all64_floor_spread='sum theta |wtheta|^2 Lambda*(vtheta-minv); not branchwise zero',
        localized_floor_commutator='[H,Ptheta]=-(chi/2)*(Delta_metric Ptheta+2 grad Ptheta .Pi)+colour metric terms; nonzero at chart boundary',
        no_superselection_assumed=True,
        branch_boundary_native_not_huge_family=True,
        fullQ_moment_proved=False,
        supplied_lower_floor='global Root -B_N is lawful but too coarse for distance; actual local-floor/Agmon/source graphs not supplied',
        quantum_law_debt=True,parameter_debt=True)


def native(time: float) -> dict[str,Any]:
    model=PreparedQuantumSlice()
    state,row=model.correlated_hessian_control()
    H=model.H
    evolve=lambda mat,vec: np.asarray(expm_multiply(-1j*time*mat,vec,traceA=-1j*time*complex(mat.diagonal().sum())))
    propagated=evolve(H,state)
    photon=model.creator@state
    photon_flight=evolve(H,photon)
    relative=np.exp(-1j*time*model.phomega)*(model.creator@propagated)
    source=model.B@state
    norm=float(np.linalg.norm(source));require(norm>0,'new covariance fullRef4 source nonzero')
    source/=norm
    J,Q=model.J,model.contact
    zero=sparse.csr_matrix(H.shape)
    drive=sparse.bmat([[H,zero,zero],[J,H,zero],[Q,2*J,H]],format='csr')
    initial=np.concatenate([source,np.zeros_like(source),np.zeros_like(source)])
    response=evolve(drive,initial).reshape(3,-1)
    step=2e-4
    Hp=model.hamiltonian(step)-model.energy*sparse.eye(len(state),format='csr')
    Hm=model.hamiltonian(-step)-model.energy*sparse.eye(len(state),format='csr')
    plus=evolve(Hp,source);minus=evolve(Hm,source)
    first=float(np.linalg.norm((plus-minus)/(2*step)-response[1]))
    second=float(np.linalg.norm((plus+minus-2*response[0])/(step*step)-response[2]))
    require(first<2e-6 and second<2e-4,'v3 changed covariance actual shared current/contact response')
    momentum=model.axis_operator(0,model.p2[0])/model.normal_scale**2
    kinetic_moment=float(np.vdot(propagated,momentum@propagated).real)
    row.update(state_identity=IDENTITY,dimension=len(state),time=time,source_norm=norm,
        norm_error=abs(float(np.linalg.norm(propagated))-1),
        different_from_v2_native_state_norm=float(np.linalg.norm(state-model.seed)),
        actual_same_H_neutral_rightaction_photon_error=float(np.linalg.norm(photon_flight-relative)),
        wrong_frozen_rightaction_norm=float(np.linalg.norm(photon_flight-np.exp(-1j*time*model.phomega)*photon)),
        first_current_response_norm=float(np.linalg.norm(response[1])),second_current_contact_response_norm=float(np.linalg.norm(response[2])),
        first_response_difference_error=first,second_response_difference_error=second,
        normal_momentum_squared_after=kinetic_moment,
        chi_momentum_norm_after=model.chi*math.sqrt(kinetic_moment),
        actual_source_first_norm=float(np.linalg.norm(H@(model.B@state)-model.B@(H@state))),
        rho_norm_after=float(np.linalg.norm(model.rho@propagated)),
        auxiliary_complement_norm=float(np.linalg.norm(model.complement@propagated)),
        current_native_UE=model.ue,selected_UE_not_executed=True,
        actual_native_positive_Hessian_only=True,ground_claimed=False,
        photon_v2_receipt_not_reused=True,fullN_bridge_proved=False)
    require(row['different_from_v2_native_state_norm']>1e-5,'material changed covariance discriminates state')
    return row


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--time',type=float,default=1.)
    args=parser.parse_args()
    hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
        ('prepared_neutral_hessian.py','prepared_neutral_hessian.md','prepared_neutral_full_field.py','prepared_neutral_full_field.md')}
    result=dict(state=hessian_descriptor(),scale_derivation=termwise_exponents(),native=native(args.time),
        source_sha256=hashes,source_unchanged=all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==s for n,s in hashes.items()),
        conclusion='Materially changed actual-Hessian covariance native response; exact fullQ/local-floor boundary/coherence implications still unpaid, not a joined-state promotion')
    print(json.dumps(portable(result),indent=2,sort_keys=True,allow_nan=False))


if __name__=='__main__':
    main()
