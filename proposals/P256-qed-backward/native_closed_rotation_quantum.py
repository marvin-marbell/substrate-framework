#!/usr/bin/env python3
"""Physical RAW Noether on the SAME cached once-prepared coupled native H.

This is a Galerkin quantum experiment, not a native legal-SOFT/full-Fock proof.
No scalar spin expectation is exponentiated as a physical rotation generator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from fractions import Fraction
from pathlib import Path
from typing import Any, cast

if __name__ == '__main__':
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
        os.environ[key] = '1'

import numpy as np
from scipy.linalg import eigh

from original_total_angular_transfer import primitive_flow, scalar_orbital
from prepared_neutral_shared_photon_state import (conditional_derivative,
    conditional_values, coupled_form, frozen_nodes, generated_weyl_kernel,
    load_shared_native_cache, scalar_native_kernels)
from unrestricted_field_minimum import coordinates

PLAN = 'issue-222-fullfield-closedrotation-continuation-20261006'
REVISION = '57838ca23dbf1424677c1c2f4e183784463faa3f759a85ec2a2b9722ff1fd9ff'


def digest_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()


def physical_chart_pullback(model: Any, sample: dict[str, Any],
                            alpha: np.ndarray, velocity: np.ndarray) -> dict[str, Any]:
    """RAW all60Phi+45colour; inverse FULL Jacobian, not old raw90 slice."""
    point, law = sample['point'], model.law
    df,du,_ = primitive_flow(law,point['phi'],point['links'],np.zeros((law.volume,3)),velocity,alpha)
    left = -1j*np.einsum('xjab,xjcb->xjac',du,point['links'].conj())
    raw = np.c_[coordinates(df,law.basis).reshape(law.volume,60),
        (2*np.einsum('xjkl,alk->xja',left,law.generators)).real.reshape(law.volume,45)].ravel()
    # The already evaluated inverse metric avoids a second full cubic solve.
    pulled = point['cometric']@(point['jacobian'].T@(model.chart.mass*raw))
    error = float(np.linalg.norm(point['jacobian']@pulled-raw))
    if error>1e-8*(1+np.linalg.norm(raw)):
        raise ValueError('RAW full-J Noether pullback failed')
    return dict(normal=pulled[:model.chart.normal_dim],gauge=pulled[model.chart.normal_dim:],
                reconstruction_error=error)


def gaussian_noether_kernels(model: Any, nodes: int) -> dict[str, Any]:
    """ALL52 literal sine weak kernels, with explicit generated-sector MC debt.

    The exact scalar Gram is NOT replaced. Native generated-H quadrature is
    already part of the shared cache; this independent G quadrature is named.
    The vacuum column of G_coexact is exact: G|vac>=the appended gamma function.
    """
    if nodes<128 or nodes%2:
        raise ValueError('At least128 antithetic Gaussian Noether nodes required')
    bank = model.noether_bank
    rng = np.random.default_rng(222712)
    y = rng.standard_normal((nodes//2,52))/math.sqrt(2)
    y = np.concatenate((-y,y),axis=0)
    values,derivatives = zip(*(bank.scalar_values(row) for row in y))
    values,derivatives = np.asarray(values),np.asarray(derivatives)
    sine = np.sin(y@bank.coefficients['phases'].T)
    flow = sine@bank.coefficients['coexact']
    dy = np.einsum('ni,nij->nj',flow,derivatives,optimize=True)
    cross = values.conj().T@dy/nodes
    em = -.5j*(cross-cross.conj().T)
    gram = bank.scalar_gram()
    # Exact known operator column, not a fitted normalization or clipped norm.
    em[:,0] = gram[:,15]
    em[0,:] = gram[15,:]
    harmonics = []
    coulomb = []
    for axis in range(3):
        factors = sine@bank.coefficients['harmonic'][:,axis]
        harmonics.append(values.conj().T@(factors[:,None]*values)/nodes)
    for site in range(model.law.volume):
        factors = sine@bank.coefficients['lambda_C'][:,site]
        coulomb.append(values.conj().T@(factors[:,None]*values)/nodes)
    return dict(coexact=em,harmonic=harmonics,coulomb=coulomb,
        quadrature_nodes=nodes,quadrature_seed=222712,
        generated_G_quadrature_debt=True,exact_vacuum_gamma_column=True,
        literal_sine=True,all_coexact_coordinates=52,
        native_c=float(model.c),native_ue=float(model.ue))


def compile_physical_noether(model: Any, data: dict[str, Any],
                            scalar_nodes: int = 128) -> tuple[dict[str,np.ndarray],dict[str,Any]]:
    """Export WEAK Hermitian forms, never Gram^-1 operators under the same name."""
    manifest = data['manifest']
    guard_path = Path(manifest['argv'][manifest['argv'].index('--noether-guard')+1])
    guard_bytes = guard_path.read_bytes()
    if hashlib.sha256(guard_bytes).hexdigest()!=manifest['input_sha256']['noether_guard']:
        raise ValueError('Shared physical Noether guard differs from the frozen A input')
    guard = json.loads(guard_bytes)
    alpha,velocity = np.asarray(guard['alpha']),np.asarray(guard['velocity'])
    if float(model.c)!=.125 or float(model.ue)!=2.**-20:
        raise ValueError('The SAME immutable A17 native c/UE is required')
    kernels = gaussian_noether_kernels(model,scalar_nodes)
    gramkernel = model.noether_bank.scalar_gram()
    scalar = scalar_native_kernels(model,int(manifest['science']['generated_H_EM_nodes']))
    records = frozen_nodes(model)
    sizes = {s:data[s+'_raw_Gram'].shape[0] for s in ('neutral','charged')}
    forms = {s:np.zeros((d,d),complex) for s,d in sizes.items()}
    errors = []
    for record in records:
        sample = record['sample']
        base,extra = conditional_values(model,sample)
        pull = physical_chart_pullback(model,sample,alpha,velocity)
        errors.append(pull['reconstruction_error'])
        db,de = conditional_derivative(model,sample,base,extra,pull['normal'],np.zeros(3))
        identity = {s:model.fibre.overlap(s) for s in sizes}
        node = {}
        heavy_orbit = scalar_orbital(model.heavy_law,sample['point']['links'],np.zeros((27,3)),velocity)
        heavy_scalar = float(np.trace(model.heavy_projector@heavy_orbit).real)
        for sector in sizes:
            gauge = model.gauge_operators(pull['gauge'],sector)['first']
            dbase = db[sector]+1j*gauge@base[sector]
            dextra = de[sector]+1j*np.einsum('ab,ben->aen',gauge,extra[sector],optimize=True)
            cross = coupled_form(base[sector],extra[sector],dbase,dextra,identity[sector],gramkernel)
            value = -.5j*(cross-cross.conj().T)
            value += heavy_scalar*coupled_form(base[sector],extra[sector],base[sector],extra[sector],identity[sector],gramkernel)
            value += coupled_form(base[sector],extra[sector],base[sector],extra[sector],identity[sector],kernels['coexact'])
            node[sector] = value
        # Harmonic velocity is a genuine operator, not meanY_EM(0)=0.
        for axis in range(3):
            db,de = conditional_derivative(model,sample,base,extra,np.zeros(2430),np.eye(3)[axis])
            for sector in sizes:
                cross = coupled_form(base[sector],extra[sector],db[sector],de[sector],identity[sector],kernels['harmonic'][axis])
                node[sector] += -.5j*(cross-cross.conj().T)
        # rho=lambda_C light density plus SAME Ref/background; lambda_R=0.
        for site in range(model.law.volume):
            local = np.zeros((108,108),complex)
            local[4*site:4*site+4,4*site:4*site+4] = np.eye(4)
            for sector in sizes:
                rho = model.fibre.onebody(local,sector)-2*identity[sector]
                node[sector] += coupled_form(base[sector],extra[sector],base[sector],extra[sector],rho,kernels['coulomb'][site])
        # Scalar orbital only, all original Peierls phases BEFORE projection.
        links,theta = sample['point']['links'],sample['theta']
        for x in range(model.law.volume):
            for axis in range(3):
                target = int(model.law.target[x,axis])
                scalar_velocity = (velocity[x,axis]+velocity[target,axis])/(4j*model.law.band.a)
                if not scalar_velocity:
                    continue
                body = np.zeros((108,108),complex)
                body[4*x:4*x+4,4*target:4*target+4] = scalar_velocity*links[x,axis]*np.exp(1j*theta[axis]/model.law.size)
                displacement = model.em['legs'][3*x+axis]
                kernel = generated_weyl_kernel(scalar,displacement,model.photon_basis.weyl_matrix(displacement))
                for sector in sizes:
                    term = coupled_form(base[sector],extra[sector],base[sector],extra[sector],model.fibre.onebody(body,sector),kernel)
                    node[sector] += term+term.conj().T
        for sector in sizes:
            forms[sector] += sample['importance_weight']/len(records)*node[sector]
    out = {}
    hermiticity = {}
    for sector in sizes:
        coord = data[sector+'_coordinates']
        value = coord.conj().T@forms[sector]@coord
        hermiticity[sector] = float(np.linalg.norm(value-value.conj().T))
        if hermiticity[sector]>1e-7*(1+np.linalg.norm(value)):
            raise ValueError('Physical weak Noether form lost Hermiticity')
        out[sector+'_G'] = (value+value.conj().T)/2
    profiles = {}
    for name,profile in model.source_bank.profiles.items():
        weights = np.abs(np.asarray(profile))**2
        norm = float(np.sum(weights))
        profiles[name] = dict(raw_lattice_norm_squared=norm,
            raw_profile_weight_outside_alpha1=float(np.sum(weights[alpha!=1])/norm),
            scope='raw lattice profile geometry only; NOT coupled SOURCE or full physical L2 mass')
    return out,dict(generator_units='weak_form',Phi_nodes=len(records),
        max_RAW_full_J_reconstruction_error=max(errors),Hermiticity=hermiticity,
        gaussian_kernels={k:v for k,v in kernels.items() if k not in ('coexact','harmonic','coulomb')},
        native_raw_profiles=profiles,exact_neutral_RIGHT=True,derived_Gauss_body_Ref_once=True,Coulomb_count=1,
        no_assigned_intrinsic_body_spin=True,no_reference_position_motion=True,
        scope='actual physical RAW weak generator in SAME native coupled basis; finite Phi/EM quadrature and fullSlaterPH/fullAux/fieldfunction projection debt remain')


def mean_matter_complement(model: Any, data: dict[str,Any]) -> dict[str,Any]:
    """Actual mean Noether matter PH, BEFORE compression, not assigned Sigma."""
    manifest = data['manifest']
    guard_path = Path(manifest['argv'][manifest['argv'].index('--noether-guard')+1])
    guard = json.loads(guard_path.read_bytes())
    alpha,velocity = np.asarray(guard['alpha']),np.asarray(guard['velocity'])
    sample = model.basis_at(np.zeros(2430),np.zeros(55))
    pull = physical_chart_pullback(model,sample,alpha,velocity)
    matrices = np.einsum('xa,aij->xij',pull['gauge'].reshape(27,15),model.law.generators)
    gauge_body = np.zeros((108,108),complex)
    for site in range(27):
        gauge_body[4*site:4*site+4,4*site:4*site+4] = matrices[site]
    light = scalar_orbital(model.law,sample['point']['links'],sample['a'],velocity)+gauge_body
    heavy = scalar_orbital(model.heavy_law,sample['point']['links'],np.zeros((27,3)),velocity)+gauge_body
    reference = model.reference_frame.conj().T@matrices[model.reference_site]@model.reference_frame
    P = model.heavy_projector
    heavy_mean = float(np.trace(P@heavy).real)
    heavy_variance = float(np.trace(P@heavy@(np.eye(108)-P)@heavy).real)
    values = {'neutral':sample['neutral']@data['neutral_state']}
    for name in ('source19','SOFT'):
        values[name] = model.source_at(sample['point']['phi'],sample['point']['links'],
            sample['theta'],sample['a'],selected_profile=name)['source_matrix']@values['neutral']
    rows = {}
    for name,value in values.items():
        sector = 'neutral' if name=='neutral' else 'charged'
        first = model.fibre.onebody(light,sector)+model.fibre.reference(reference,sector)
        second = model.fibre.bilinear_product(light,light,sector,
            reference_A=reference,reference_B=reference)
        outside = second-first@first+heavy_variance*model.fibre.overlap(sector)
        norm = float(np.vdot(value,value).real)
        rows[name] = dict(point_fibre_norm_squared=norm,
            mean_G_matter_full_Wick_squared=float(np.vdot(value,
                (second+2*heavy_mean*first+(heavy_mean**2+heavy_variance)*model.fibre.overlap(sector))@value).real),
            mean_G_matter_unrepresented_PH_squared=float(np.vdot(value,outside@value).real),
            charged_operator_is_not_scalar_spin=True)
    return dict(rows=rows,heavy_PH_variance=heavy_variance,
        actual_guard_full_Gauss_Ref_scalar_orbital=True,
        scope='pointwise full-fibre mean matter contribution only, common Gaussian/Jhalf factored; NOT whole-field L2, full G squared or time propagator',
        no_heavy_spin_stationarity_or_return=True)


def orthogonal_coordinates(gram: np.ndarray) -> np.ndarray:
    values,modes = eigh(gram)
    if values[0]<=0:
        raise ValueError('Shared analytic quotient must remove null functions; no G-specific jitter')
    return modes/np.sqrt(values)[None,:]


def pulse_average(clock: int,step: int,count: int) -> float:
    def theta(u: Fraction) -> Fraction:
        return 10*u**3-15*u**4+6*u**5
    return 2*math.pi*float(theta(Fraction(step+1,count))-theta(Fraction(step,count)))*count/clock


def propagate(H: np.ndarray,G: np.ndarray,initial: np.ndarray,
              clock: int,count: int) -> np.ndarray:
    """Exact constant-interval SA evolution; refinements compare native paths."""
    state = initial.copy()
    dt = clock/count
    for step in range(count):
        energies,modes = eigh(H-pulse_average(clock,step,count)*G,check_finite=False)
        coefficients = modes.conj().T@state
        phase = np.exp(-1j*energies*dt)
        state = modes@((phase[:,None] if coefficients.ndim==2 else phase)*coefficients)
    return state


def load_noether_export(npz: Path,receipt: Path) -> tuple[dict[str,np.ndarray],dict[str,Any]]:
    meta = json.loads(receipt.read_bytes())
    if hashlib.sha256(npz.read_bytes()).hexdigest()!=meta['export_npz_sha256']:
        raise ValueError('Noether action export differs from source-bound receipt')
    with np.load(npz,allow_pickle=False) as stored:
        arrays = {name:stored[name].copy() for name in stored.files}
    return arrays,meta


def run_shared_loop(model: Any,data: dict[str,Any],generators: dict[str,np.ndarray],
                    steps: int = 16) -> tuple[dict[str,np.ndarray],dict[str,Any]]:
    if steps < 2:
        raise ValueError('At least two actual pulse intervals are required')
    gram = {s:data[s+'_Gram'] for s in ('neutral','charged')}
    coords = {s:orthogonal_coordinates(gram[s]) for s in gram}
    # ONE shared numerical common-energy representation factor. The unshifted
    # H and reconstructed ordinary common phase are both retained in receipt.
    xi = data['neutral_state']
    common = float(np.vdot(xi,data['neutral_H']@xi).real)
    h = {s:coords[s].conj().T@(data[s+'_H']-common*gram[s])@coords[s] for s in gram}
    g = {s:coords[s].conj().T@generators[s+'_G']@coords[s] for s in gram}
    initial_n = coords['neutral'].conj().T@gram['neutral']@xi
    maps = {name:coords['charged'].conj().T@gram['charged']@data[name+'_map']@coords['neutral'] for name in ('source19','SOFT')}
    norms = {name:float(np.linalg.norm(mapping@initial_n)) for name,mapping in maps.items()}
    if min(norms.values())<=0:
        raise ValueError('Actual RAW SOURCE vanished; no fresh template/filter')
    initial_c = np.column_stack([maps[name]@initial_n/norms[name] for name in maps])
    clock = math.ceil(15*math.pi*1.5/.125)
    trajectories = []
    actions = {}
    previous = None
    for count in (steps,2*steps):
        neutral = propagate(h['neutral'],g['neutral'],initial_n,clock,count)
        charged = propagate(h['charged'],g['charged'],initial_c,clock,count)
        delta = None if previous is None else dict(neutral=float(np.linalg.norm(neutral-previous[0])),
                    SOURCE={name:float(np.linalg.norm(charged[:,i]-previous[1][:,i])) for i,name in enumerate(maps)})
        trajectories.append(dict(intervals=count,
            accumulated_angle=sum(pulse_average(clock,k,count)*clock/count for k in range(count)),
            refinement_difference=delta))
        previous = neutral,charged
    nenergy,nmodes = eigh(h['neutral'])
    cenergy,cmodes = eigh(h['charged'])
    reverse_n = nmodes@(np.exp(1j*nenergy*clock)*(nmodes.conj().T@neutral))
    wrong_reverse_n = nmodes@(np.exp(1j*nenergy*clock)*(nmodes.conj().T@initial_n))
    observations = {}
    for index,name in enumerate(maps):
        free_right = cmodes@(np.exp(-1j*cenergy*clock)*(cmodes.conj().T@(maps[name]@reverse_n/norms[name])))
        relative = charged[:,index]+free_right
        wrong_right = cmodes@(np.exp(-1j*cenergy*clock)*(cmodes.conj().T@(maps[name]@wrong_reverse_n/norms[name])))
        scalar = float(np.vdot(initial_c[:,index],g['charged']@initial_c[:,index]).real-
                       np.vdot(initial_n,g['neutral']@initial_n).real)
        observations[name] = dict(native_SOURCE_norm=norms[name],actual_minus_freeword_neutral_RIGHT_defect=float(np.linalg.norm(relative)),
            actual_source_state_norm=float(np.linalg.norm(charged[:,index])),actual_freeword_RIGHT_norm=float(np.linalg.norm(free_right)),
            wrong_returned_neutral_RIGHT_defect=float(np.linalg.norm(free_right-wrong_right)),
            wrong_scalar_spin_expectation=scalar,wrong_scalar_spin_exponentiation_minus_defect=abs(np.exp(2j*math.pi*scalar)+1),
            source_ONLY_no_incoming_photon=True)
        actions[name+'_U2pi_coefficients'] = coords['charged']@charged[:,index]
        actions[name+'_freeword_actual_neutral_RIGHT'] = coords['charged']@free_right
    actions['neutral_U2pi_coefficients'] = coords['neutral']@neutral
    import mpmath as mp
    with mp.workdps(70):
        fraction = Fraction.from_float(common)
        common_phase = complex(mp.exp(-1j*mp.mpf(fraction.numerator)/fraction.denominator*clock))
    for name in actions:
        actions[name] *= common_phase
    return actions,dict(plan=PLAN,revision=REVISION,clock=clock,native_c=.125,native_ue=2.**-20,
        total_angle=2*math.pi,trajectory_refinement=trajectories,observations=observations,
        neutral_metric_norm=float(np.linalg.norm(neutral)),ordinary_unshifted_common_energy=common,
        retained_ordinary_common_phase=[common_phase.real,common_phase.imag],
        common_factor_used_for_numerics_ONLY=True,no_per_history_or_charged_floor_reset=True,
        freeword='exp(-i H_C T) S exp(+i H_N T) U_N^Omega(T) Xi / SAMEinitialSOURCEglobalZ',
        comparison_not_current_Bt_or_recaptured_basis=True,
        native_analytic_1e4_NOT_claimed=True,full_Fock_512_physical_Gauss_trajectory_NOT_claimed=True,
        debts=['nativeSOURCE19 guard/origin/screening','SOFT periodicN3 projectionzero / coordinate restrictionmissingmass',
               'generated G/H and Phi/harmonic quadrature','native weakfunction/SlaterPH/fullAux projections','finite continuous-pulse time discretization'],
        same_once_preparation=True,no_scalar_spin_generator=True,campaign_open=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-npz',type=Path,required=True)
    parser.add_argument('--cache-manifest',type=Path,required=True)
    parser.add_argument('--output-npz',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--scalar-nodes',type=int,default=128)
    parser.add_argument('--steps',type=int,default=16)
    args = parser.parse_args()
    started = time.perf_counter()
    model,data = load_shared_native_cache(args.cache_npz,args.cache_manifest)
    generators,gmeta = compile_physical_noether(model,data,args.scalar_nodes)
    actions,loop = run_shared_loop(model,data,generators,args.steps)
    matter_debt = mean_matter_complement(model,data)
    np.savez_compressed(args.output_npz,**cast(Any,generators),**cast(Any,actions))
    receipt = dict(generator=gmeta,loop=loop,mean_matter_complement=matter_debt,
        cache_npz_sha256=hashlib.sha256(args.cache_npz.read_bytes()).hexdigest(),
        cache_manifest_sha256=hashlib.sha256(args.cache_manifest.read_bytes()).hexdigest(),
        input_H_Gram_state_hashes={key:digest_array(data[key]) for key in ('neutral_H','charged_H','neutral_Gram','charged_Gram','neutral_state','source19_map','SOFT_map')},
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        export_npz_sha256=hashlib.sha256(args.output_npz.read_bytes()).hexdigest(),
        export_npz=str(args.output_npz),wall_seconds=time.perf_counter()-started)
    args.output.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
