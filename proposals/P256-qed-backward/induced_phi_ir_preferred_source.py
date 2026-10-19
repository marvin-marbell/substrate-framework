#!/usr/bin/env python3
"""Actual joined preferred222 C-paired source and near-rest CAR consumer.

Selection and full-state transfer remain conditional on the displayed reviewed
lemmas, not on this small native CAR control. Frozen source/state bytes unchanged.
"""
from __future__ import annotations
import argparse
from fractions import Fraction
import hashlib
import itertools
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any
import numpy as np
from induced_phi_backreaction import PhiConfig,models,scalar_paths,SELECTED,IMPORT_BYTES as BASE_BYTES
from induced_phi_ir_charge_symmetry import local_charge_unitary

ROOT=Path(__file__).resolve().parent
SOURCE_BYTES={**BASE_BYTES,**{name:(ROOT/name).read_bytes() for name in (
    'induced_phi_ir_mean_certificate.py','induced_phi_ir_well_transfer.py',
    'induced_phi_ir_current_contact.py','induced_phi_ir_charge_symmetry.py',
    'induced_phi_ir_holonomy_refinement.py',Path(__file__).name)}}
SOURCE_HASHES={name:hashlib.sha256(data).hexdigest() for name,data in SOURCE_BYTES.items()}


def rational(value: dict[str,str]) -> Fraction:
    return Fraction(int(value['numerator']),int(value['denominator']))


def annihilation_images(vector: np.ndarray) -> np.ndarray:
    result=np.zeros((8,256),complex)
    for i in range(8):
        for state in range(256):
            if state&(1<<i):result[i,state^(1<<i)]=(-1)**((state&((1<<i)-1)).bit_count())*vector[state]
    return result


def CAR_Hamiltonian(h: np.ndarray) -> np.ndarray:
    result=np.zeros((256,256),complex)
    for state in range(256):
        for j in range(8):
            if not state&(1<<j):continue
            middle=state^(1<<j);annihilation=(-1)**((state&((1<<j)-1)).bit_count())
            for i in range(8):
                if middle&(1<<i):continue
                creation=(-1)**((middle&((1<<i)-1)).bit_count())
                result[middle|(1<<i),state]+=h[i,j]*annihilation*creation
    return result


def paired_charge() -> tuple[np.ndarray,dict[str,Any]]:
    C4,_,control=local_charge_unitary();swap=np.zeros((256,256),complex)
    permutation=tuple(range(4,8))+tuple(range(4))
    for state in range(256):
        occupied=[permutation[i] for i in range(8) if state&(1<<i)]
        inversions=sum(occupied[i]>occupied[j] for i in range(len(occupied)) for j in range(i+1,len(occupied)))
        target=sum(1<<i for i in occupied);swap[target,state]=(-1)**inversions
    # C4 is EVEN, so this tensor product implements both graded CAR blocks.
    C=swap@np.kron(C4,C4)
    if np.linalg.norm(C@C-np.eye(256))!=0:raise ArithmeticError('Paired finite-momentum C is not an exact involution')
    return C,dict(local=control,pair_dimension=256,implementation='Fermionic swap of the two4mode momentum blocks times the tensor product of the actual EVEN16x16 C. Hence a_p ->B a_-p^dag; C^2=I exactly, with swap occupation-inversion signs retained.')


def one_particle(model,frame,p):
    onsite,paths=scalar_paths(model);result=float(onsite)*frame[0]
    for displacement,scalars in paths:
        F=np.einsum('m,mij->ij',scalars,frame);phase=np.exp(1j*model.a*np.asarray(p)@displacement)
        result+=F*phase+F.conj().T*phase.conjugate()
    return result


def slater(columns: np.ndarray) -> np.ndarray:
    result=np.zeros(256,complex)
    for occupied in itertools.combinations(range(8),4):
        index=sum(1<<i for i in occupied)
        result[index]=np.linalg.det(columns[np.asarray(occupied),:])
    norm=float(np.linalg.norm(result))
    if abs(norm-1)>1e-12:raise ArithmeticError('Actual paired CAR Slater norm is not1')
    return result/norm


def native_reference(cfg: PhiConfig,size: int,mean: dict[str,Any]) -> dict[str,Any]:
    enclosure=mean['centre_enclosure']
    g0=(rational(enclosure['g0_lower'])+rational(enclosure['g0_upper']))/2
    gs=(rational(enclosure['gs_lower'])+rational(enclosure['gs_upper']))/2
    frame=np.sqrt(np.asarray([float(g0),float(gs),float(gs),float(gs)]))[:,None,None]*SELECTED
    p=-np.pi*np.ones(3)/(size*cfg.a)
    C,implementation=paired_charge()
    parity=np.diag([(-1)**state.bit_count() for state in range(256)])
    if np.linalg.norm(C@parity-parity@C)!=0:raise ArithmeticError('Actual two-momentum C is not even')
    rows=[]
    for load,model,role in models(cfg):
        hplus=one_particle(model,frame,p);hminus=one_particle(model,frame,-p)
        heavy=role=='neutral heavy -Lambda';sign=-1 if heavy else 1
        h=np.zeros((8,8),complex);h[:4,:4]=sign*hplus;h[4:,4:]=sign*hminus
        values,vectors=np.linalg.eigh(h);occupied=vectors[:,:4]
        state=slater(occupied);transformed=C@state
        # For two normalized rays, trace distance of projectors is exactly
        # 2sqrt(1-|overlap|^2); direct ray minimization avoids sqrt cancellation.
        overlap=np.vdot(state,transformed)
        if abs(overlap)<1e-8:raise ArithmeticError('Actual C-transformed paired source is orthogonal')
        phase=overlap/abs(overlap)
        ray_error=float(np.linalg.norm(transformed-phase*state))
        density=(np.outer(state,state.conj())+np.outer(transformed,transformed.conj()))/2
        invariance=float(np.linalg.norm(C@density@C.conj().T-density))
        number_distribution=[float(sum(abs(state[j])**2 for j in range(256) if j.bit_count()==n)) for n in range(9)]
        images=annihilation_images(state)
        expectation=images.conj()@images.T
        projector=occupied@occupied.conj().T
        projector_error=float(np.linalg.norm(expectation-projector.T))
        energy=float(np.sum(values[:4]));electron_rest=float(np.linalg.eigvalsh(hplus)[2])
        # Independent literal creation/annihilation actions, not dense products
        # of eight redundant256x256 annihilator matrices.
        H=CAR_Hamiltonian(h)
        Hamiltonian_C_error=float(np.linalg.norm(C@H@C.conj().T-H))
        eigenstate_error=float(np.linalg.norm(H@state-energy*state))
        if max(ray_error,invariance,projector_error,Hamiltonian_C_error,eigenstate_error)>1e-10 or abs(number_distribution[4]-1)>1e-12:
            raise ArithmeticError('Actual preferred-sector paired CAR reference consumer failed')
        rows.append(dict(role=role,species_count=1 if heavy else int(load),physical_multiplier='2^8192' if heavy else '1',native_H='-h' if heavy else '+h',
            actual_nearest_rest_p=p.tolist(),opposite_allowed_p=(-p).tolist(),electron_positive_band_energy=electron_rest,
            CAR_slater_energy=energy,C_ray_error=ray_error,C_density_error=invariance,one_particle_projector_error=projector_error,
            manybody_H_C_error=Hamiltonian_C_error,manybody_eigenstate_error=eigenstate_error,number_distribution=number_distribution))
    return dict(implementation=implementation,profiles=rows,
        scope='Actual N262144/a1/64 nearest-rest anti-periodic fermion momenta and primitive one/two-link Hamiltonians on one selected p,-p pair. Each physical profile has4occupied modes of8, not a mock oscillator or extra species. Nativecentre is diagnostic midpoint only, with previously computed physical operator-image prices retained; complete macrosea andPhi shapes remain exactimplicit.',
        no_inference='A small paired CAR eigenstate control is not a numerical simulation or ground proof of the entire interacting noncompactEM macrobox.')


def run(paths: dict[str,str]) -> dict[str,Any]:
    started=time.monotonic();inputs={}
    for role,path in paths.items():
        raw=Path(path).read_bytes();inputs[role]=dict(path=path,sha256=hashlib.sha256(raw).hexdigest(),capsule=json.loads(raw))
    mean=inputs['mean']['capsule'];well=inputs['well']['capsule'];current=inputs['current']['capsule'];charge=inputs['charge']['capsule'];refinement=inputs['refinement']['capsule']
    for candidate in (well,current):
        if candidate['input_descriptor']['mean_input_sha256']!=mean['input_sha256']:raise ArithmeticError('Physical consumers have different finite mean inputs')
    if refinement['input_descriptor']['mean_receipt_sha256']!=inputs['mean']['sha256'] or refinement['input_descriptor']['well_receipt_sha256']!=inputs['well']['sha256']:
        raise ArithmeticError('Preferred-sector discriminator did not use actual same mean/well bytes')
    if charge['input_descriptor']['input_receipts']['mean']['sha256']!=inputs['mean']['sha256'] or charge['input_descriptor']['input_receipts']['current']['sha256']!=inputs['current']['sha256']:
        raise ArithmeticError('C construction did not use actual same mean/current bytes')
    mean_live={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==entry['sha256']==hashlib.sha256(entry['bytes_utf8'].encode()).hexdigest()
        for name,entry in mean['source_at_import'].items()}
    if not all(mean_live.values()) or not all(well['source_unchanged_since_import'].values()) or not all(candidate['all_sources_unchanged_at_completion'] for candidate in (current,charge,refinement)):
        raise ArithmeticError('Joined scientific receipt has a source epoch drift')
    if rational(refinement['actual_all64_margin_lower'])<=0:raise ArithmeticError('No actual positive macro sector discriminator')
    if any(row['EM_charge']!=(0 if row['role']=='neutral heavy -Lambda' else 1) for row in charge['literal_full_field_consumer']['scalar_paths']):
        raise ArithmeticError('Wrong charged-neutral primitive C epoch')
    size=mean['finite_size'];cfg=PhiConfig(Lambda=1.,**mean['microscopic_config'])
    native=native_reference(cfg,size,mean)
    descriptor=dict(input_receipts={role:{key:value[key] for key in ('path','sha256')} for role,value in inputs.items()},
        finite_size=size,Lambda_exponent=mean['Lambda_exponent'],microscopic_config=mean['microscopic_config'],selected_sector=[2,2,2],
        preparation='NEWpreferred222 C-paired density with unchanged frozen64/128 source epochs',clock=4096,
        governing_plan='issue-222-induced-photon-delta13-continuation-20261008',plan_revision='d13d7d28ebfd7cb8386e7442897c596481aeab95cd55def1a8b3a60c1e89f4dd')
    return dict(branch=descriptor['preparation'],input_descriptor=descriptor,input_sha256=hashlib.sha256(json.dumps(descriptor,sort_keys=True).encode()).hexdigest(),
        exact_physical_state_definition='rho_pref,C=(|Psi222><Psi222|+|C Psi222><C Psi222|)/2, with Psi222 the actual normalized macro slow/fast Gaussian-cut jointGauss lift using its EXACT finite implicit222 fullten root, exact positive/heavy Slater fibre, all64 exact negative-band chargedSlaters, original onsiteRef4 and realnoncompactEM H2 packet. C is the actual mobile local16x16 CAR product plus explicitPhi/colour/EM pullback. No midpoint/mean overwrite or energydegeneracy assumption.',
        exact_reference_definition='Apply the SAME C-pairing to sector222 exact charged/noncompactEM reference and samephase convention; unitary and convexity keep the original component budgets. This preparation is new, not a change of finiteSCREEN source or receiver.',
        actual_native_quantum_consumer=native,
        actual_finite_centre=dict(exact_definition=mean['exact_centre_definition'],sector=[2,2,2],symmetry='Exactantiperiodic parity andcube permute allowed momenta; uniquecoupledroot is diagonal(g0,gs,gs,gs). Fullquarterroots elsewhere remain unaltered. RefineddifferencefromEXACT000 is displayed, not a continuum replacement.',
            original000_enclosure=mean['centre_enclosure'],refined_all64_root_distance_upper=refinement['refined_all64_Gram_distance_to_exact000_upper']),
        actual_sector_energy_discriminator=refinement['actual_all64_margin_lower'],
        actual_clock_transfer={name:well['prices'][name] for name in ('physical_state_norm_distance_upper','density_trace_norm_distance_upper','weighted_Phi_action_error','all_flavour_current_action_error','all_flavour_contact_action_error','all_flavour_current_expectation_error','all_flavour_contact_expectation_error')},
        compatible_actual_IR_modes=current['modes'],literal_current_contact_native_errors={name:current[name] for name in ('max_current_error','max_contact_error','wrong_ordered_current_max','wrong_two_link_cross_contact_max')},
        separately_priced_diagnostic_current_images=current['operator_image_rounding_prices'],
        source_symmetry='C^2=I, C H0 Cdag=H0 in the same fixed neutralRef4 sector, and C H[A] Cdag=H[-A]. This explicitly paired density isC-invariant even ifindividualsections have different phase. OddexternalEM functional derivatives, when defined, vanish; nofixedquarter χ3 theorem inherited. Individualelectrons/Refholes breakthis source-symmetry and areoutsideclaim.',
        parent_review_obligations='Two-cut/geometrickernel identity and actualalias theorem; exactlocalwell/minmax energy sandwich; fulloriginal-Hamiltonian/fixedRef4 C intertwining. Positivearithmetic andnativeCAR controls are evidence, not automatic mathematical admission.',
        mean_imported_source_bytes_match_at_actual_join=mean_live,
        limitations='Constructed finite preparedlocalwell source, not globalground/bulkstationarity, noncompactEM/photonvacuum, commoncone, radiativeCompton or completeelectron/photon/QED ontology. NoUV masscountertermfork merged. Exactcharged/EM reference still requires its own photon-state/consumer construction.',
        source_at_import=SOURCE_HASHES,own_source_at_import=SOURCE_BYTES[Path(__file__).name].decode(),all_sources_unchanged_at_completion=all((ROOT/name).read_bytes()==data for name,data in SOURCE_BYTES.items()),
        runtime=dict(python=sys.version,executable=sys.executable,numpy=np.__version__,platform=platform.platform()),elapsed_seconds=time.monotonic()-started)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for role in ('mean','well','current','charge','refinement'):parser.add_argument('--'+role,required=True)
    parser.add_argument('--output',required=True);args=parser.parse_args()
    report=run({role:getattr(args,role) for role in ('mean','well','current','charge','refinement')})
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(input_sha256=report['input_sha256'],source_at_import=report['source_at_import'],native=report['actual_native_quantum_consumer'],elapsed_seconds=report['elapsed_seconds']),indent=2))
