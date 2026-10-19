#!/usr/bin/env python3
"""Original four-word SOURCE functions on the immutable initial full Slater fibre."""
from __future__ import annotations

import argparse
from copy import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import MappingProxyType
from typing import Any, Callable, Mapping, cast

if __name__ == '__main__':
    for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = '1'

import numpy as np
from scipy.linalg import expm

from improved_phi_band import ImprovedPhiLaw, selected_band
from improved_physical_compton import decode
from noncompact_screened_gauss_dressing import frame_chart
from original_charged_principal import portable
from prepared_neutral_canonical_source import LABELS, literal_coordinate_mark
from prepared_neutral_full_field import BETA
from prepared_neutral_slater_fibre import FullSlaterFibre
from prepared_neutral_spatial_charge_transfer import GAMMA5
from sea_selection import SELECTED

WORD_ORDER = ('I', 'Gamma23', 'Gamma31', 'Gamma12')
SOURCE19_SHA256 = '020fb175c9231b607fb0e53b016cf97528af9065a634dedc7ddea82bdac3556c'
CoordinatePath = Callable[[float], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]


def frozen_array(value: np.ndarray) -> np.ndarray:
    out = np.array(value, copy=True)
    out.setflags(write=False)
    return out


def source19_native_profile(law: Any, descriptor_path: Path) -> dict[str, Any]:
    """Fold the unchanged nineteen descriptor weights, never normalize them."""
    raw = descriptor_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != SOURCE19_SHA256 or law.size != 3:
        raise ValueError('Exact020fb descriptor and native N3 fold required')
    descriptor = decode(json.loads(raw))
    offsets = np.asarray(descriptor['compact_displacement'], dtype=int)
    weights = np.asarray(descriptor['source_eta'], dtype=complex)
    if offsets.shape != (19, 3) or weights.shape != (19,):
        raise ValueError('Actual nineteen weights/offsets required')
    profile = np.zeros(law.volume, complex)
    folded = (offsets + 1) % 3
    for site, weight in zip(folded, weights):
        profile[law.index[tuple(site)]] += weight
    return dict(profile=frozen_array(profile), profileHash=hashlib.sha256(profile.tobytes()).hexdigest(),
                descriptor_sha256=digest, native_fold='(d+1)%3', offsets=offsets,
                weights=weights, SOURCE_eta_unchanged=True,
                packet_geometry_phase='exp(-i k_x), separate from SOURCE eta; no rho_SOURCE translation')


def quaternion_spin_word_action() -> dict[str, Any]:
    """Derive, not fit, left S_z word multiplication in the original coordinates."""
    spin = -.5j * SELECTED[1] @ SELECTED[2]
    words = 2 * LABELS
    action = np.einsum('aij,bij->ab', words.conj(), np.einsum('ij,bjk->bik', spin, words)) / 4
    error = float(np.linalg.norm(np.einsum('ab,aij->bij', action, words) - spin @ words))
    if error > 1e-12:
        raise AssertionError('Quaternion multiplication failed')
    return dict(spin=spin, word_action=action, error=error)


class QuartetSourceFunctionBank:
    """Coordinates are template, original unit word, neutral column; no whitening."""
    def __init__(self, seed_fibre: FullSlaterFibre, profiles: Mapping[str, np.ndarray]):
        if not profiles:
            raise ValueError('At least one explicitly supplied fixed profile required')
        # Snapshot the already prepared orbitals. No constructor/eigensolve/preparation.
        seed = copy(seed_fibre)
        for name in ('mean_fields', 'mean_links', 'profile', 'quartet', 'h0', 'orbitals', 'raw_neutral', 'empty'):
            setattr(seed, name, frozen_array(getattr(seed_fibre, name)))
        for name in ('occupied', 'projectors', 'occupied_indices', 'active_indices'):
            setattr(seed, name, MappingProxyType({q: frozen_array(v) for q, v in getattr(seed_fibre, name).items()}))
        fixed = {name: frozen_array(np.asarray(value, complex)) for name, value in profiles.items()}
        if any(value.shape != (seed.law.volume,) for value in fixed.values()):
            raise ValueError('Each fixed profile must have full V entries')
        self.seed_fibre = seed
        self.profiles = MappingProxyType(fixed)
        self.template_order = tuple(fixed)
        self.orbital_sha256 = hashlib.sha256(seed.orbitals.tobytes()).hexdigest()

    def word_sources(self, fields: np.ndarray, links: np.ndarray, theta: np.ndarray,
                     a: np.ndarray) -> np.ndarray:
        """[template,word,charged,neutral-bank], actual full coefficient functions."""
        out = np.empty((len(self.profiles), 4, self.seed_fibre.charged_dimension, 2), complex)
        for t, profile in enumerate(self.profiles.values()):
            for word in range(4):
                out[t, word] = self.seed_fibre.source_vectors_at(fields, links, theta, a,
                                                                profile, np.eye(4)[word])['source_matrix']
        return out

    def source_values(self, fields: np.ndarray, links: np.ndarray, theta: np.ndarray,
                      a: np.ndarray, neutral_values: np.ndarray) -> np.ndarray:
        """[full448, templates*4*K]; neutral_values[2,K] is never renormalized."""
        neutral = np.asarray(neutral_values, complex)
        if neutral.ndim != 2 or neutral.shape[0] != 2:
            raise ValueError('Neutral bank values must have shape[2,K]')
        words = self.word_sources(fields, links, theta, a)
        return np.einsum('twcb,bk->ctwk', words, neutral).reshape(self.seed_fibre.charged_dimension, -1)

    def raw_input_map(self, beta: np.ndarray, neutral_column_count: int,
                      selected_profile: str) -> np.ndarray:
        """[2,templates*4*K,K]: sector-specific PLUS beta / MINUS beta conjugate."""
        beta = np.asarray(beta, complex)
        if beta.shape != (4,) or neutral_column_count < 1:
            raise ValueError('Literal raw quartet and positive K required')
        t = self.template_order.index(selected_profile)
        k = neutral_column_count
        out = np.zeros((2, len(self.profiles)*4*k, k), complex)
        for bank, weights in enumerate((beta, beta.conj())):
            for word, weight in enumerate(weights):
                out[bank, (4*t+word)*k:(4*t+word+1)*k] = weight*np.eye(k)
        return out

    def combine_values(self, values: np.ndarray, beta: np.ndarray,
                       neutral_column_count: int, selected_profile: str) -> np.ndarray:
        """Exact physical raw B_beta values, including mixed neutral columns."""
        maps = self.raw_input_map(beta, neutral_column_count, selected_profile)
        out = np.zeros((self.seed_fibre.charged_dimension, neutral_column_count), complex)
        for bank, charge in enumerate((1, -1)):
            rows = self.seed_fibre.charged_slices[charge]
            out[rows] = values[rows] @ maps[bank]
        return out

    def physical_source(self, moving_fibre: FullSlaterFibre, reference_frame: np.ndarray,
                        fields: np.ndarray, links: np.ndarray, theta: np.ndarray,
                        a: np.ndarray, selected_profile: str, raw_beta: np.ndarray) -> dict[str, Any]:
        """Current physical observable on supplied moving orbitals, not recapture."""
        ref = np.asarray(reference_frame, complex)
        if ref.shape != (4, 4) or np.linalg.norm(ref.conj().T@ref-np.eye(4)) > 1e-9:
            raise ValueError('Actual unitary Ref4 moving frame required')
        source = moving_fibre.source_vectors_at(fields, links, theta, a,
                                                self.profiles[selected_profile], raw_beta)
        plus = source['full108_plus_coefficients'] @ ref
        minus = ref.conj().T @ source['full108_minus_coefficients']
        return moving_fibre.source_vectors_from_marks({1: plus.reshape(moving_fibre.law.volume,4,4),
            -1: minus.reshape(4,moving_fibre.law.volume,4).transpose(1,0,2)})

    def directional_values(self, path: CoordinatePath, neutral_values: np.ndarray,
                           step: float = 2.**-16) -> dict[str, Any]:
        """Producer-only central FD; integrator adds B times polynomial derivative."""
        if step <= 0:
            raise ValueError('Positive native FD step required')
        def derivative(h: float) -> np.ndarray:
            return (self.source_values(*path(h), neutral_values)-self.source_values(*path(-h), neutral_values))/(2*h)
        coarse, fine = derivative(step), derivative(step/2)
        return dict(values=self.source_values(*path(0.), neutral_values), derivative=fine,
                    native_error_estimate=float(np.linalg.norm(fine-coarse)/3), step=step/2,
                    scope='native centralFD refinement estimate, not certified analytic remainder')

    def descriptor(self, neutral_column_count: int) -> dict[str, Any]:
        return dict(shape=[self.seed_fibre.charged_dimension, len(self.profiles)*4*neutral_column_count],
                    column_order='template outer, I/Gamma23/Gamma31/Gamma12 middle, neutral column inner',
                    template_order=self.template_order, word_order=WORD_ORDER,
                    profile_sha256={name:hashlib.sha256(value.tobytes()).hexdigest() for name,value in self.profiles.items()},
                    orbital_sha256=self.orbital_sha256, global_factor=2,
                    coefficient_map='PLUS linear beta; MINUS conjugate-linear beta; no whitening/normalization',
                    fixed_initial_functions=True, moving_physical_source_separate=True)


def full_slater_relative_spin(fibre: FullSlaterFibre, source: dict[str, Any],
                             light_spin: np.ndarray) -> dict[str, Any]:
    """Exact CAR commutator, with B Q_neutral outside RIGHT explicitly retained."""
    qcharged = fibre.onebody(light_spin)
    qneutral = fibre.onebody(light_spin, 'neutral')
    b = source['source_matrix']
    marks = source['full_SOURCE_marks']
    plus = light_spin @ source['full108_plus_coefficients']
    minus = -source['full108_minus_coefficients'] @ light_spin
    relative = fibre.source_vectors_from_marks({1:plus.reshape(fibre.law.volume,4,4),
        -1:minus.reshape(4,fibre.law.volume,4).transpose(1,0,2)})['source_matrix']
    projected_difference = qcharged@b-b@qneutral
    right_outside = np.zeros_like(b)
    right_outside[fibre.charged_slices[1],0] = (-fibre.empty.conj().T@light_spin@fibre.projectors[1]@marks[1].reshape(fibre.n,4)).T.reshape(-1)
    mm = marks[-1].transpose(1,0,2).reshape(4,fibre.n)
    right_outside[fibre.charged_slices[-1],1] = (mm@(np.eye(fibre.n)-fibre.projectors[-1])@light_spin@fibre.occupied[-1]).reshape(-1)
    error = float(np.linalg.norm(projected_difference-right_outside-relative))
    if error > 1e-10:
        raise AssertionError('Full Slater relative commutator lost neutral RIGHT')
    graph = fibre.current_rows(light_spin, source)
    return dict(relative=relative, exact_CAR_error=error,
                neutral_right_projected_norm=float(np.linalg.norm(b@qneutral)),
                neutral_right_outside_norm=float(np.linalg.norm(right_outside)),
                full_charged_spin_norm_squared=graph['full_SOURCE_response_norm_squared'],
                outside_PH_spin_norm_squared=graph['unrepresented_SOURCE_response_norm_squared'],
                neutral_background_PH=fibre.onebody_graph_diagnostic(light_spin)['neutral_particle_hole_norm_squared'])


def native_science(descriptor_path: Path) -> dict[str, Any]:
    band, frame, _ = selected_band()
    law = ImprovedPhiLaw(3, band)
    fields = np.tile(frame, (law.volume,1,1,1))
    links = np.tile(np.eye(4,dtype=complex), (law.volume,3,1,1))
    bound = source19_native_profile(law, descriptor_path)
    eta = bound['profile']
    fibre = FullSlaterFibre(law, fields, links, law.index[(0,1,1)], eta)
    bank = QuartetSourceFunctionBank(fibre, {'source19':eta, 'forward':eta})
    neutral = np.zeros((2,14), complex)
    neutral[0,:7] = np.arange(1,8)/7
    neutral[1,7:] = np.arange(1,8)/7
    rng = np.random.default_rng(222701)
    errors, variation, wrongminus = [], [], []
    theta = np.array([.21,-.13,.09])
    a = np.tile(theta/3, (law.volume,1))
    initial = bank.source_values(fields, links, theta, a, neutral)
    for scale in (0., .0007, .0013):
        phi = fields + scale*law.fields(rng.normal(size=(law.volume,4,15)))
        u = np.asarray([[expm(1j*scale*np.einsum('a,aij->ij',rng.normal(size=15),law.generators)) for _ in range(3)] for _ in range(law.volume)])
        th = theta + scale*np.array([.7,-.2,.3])
        aa = np.tile(th/3, (law.volume,1))
        aa += scale*rng.normal(size=aa.shape); aa += (th/3-aa.mean(axis=0))
        values = bank.source_values(phi,u,th,aa,neutral)
        combined = bank.combine_values(values,BETA,14,'source19')
        pure = fibre.source_vectors_at(phi,u,th,aa,eta,BETA)['source_matrix']@neutral
        errors.append(float(np.linalg.norm(combined-pure)))
        variation.append(float(np.linalg.norm(values-initial)))
        maps = bank.raw_input_map(BETA,14,'source19')
        wrongminus.append(float(np.linalg.norm(values[fibre.charged_slices[-1]]@(maps[0]-maps[1]))))
    # Recover actual source-relative spin on the uniform chart, not total physical G.
    words = bank.word_sources(fields,links,theta,a)[0]
    derived = quaternion_spin_word_action(); spin = derived['spin']; action = derived['word_action']
    chart = frame_chart(frame)[0]
    spin_full = np.kron(np.eye(law.volume), chart@spin@chart.conj().T)
    rel = full_slater_relative_spin(fibre,fibre.source_vectors_at(fields,links,theta,a,eta,BETA),spin_full)
    expected = np.zeros((448,2), complex)
    for b,q in enumerate((1,-1)):
        weights = BETA if q==1 else BETA.conj()
        expected[:,b] = np.einsum('j,ij,ic->c',weights,action,words[:,: ,b])
    recovery = float(np.linalg.norm(expected-rel['relative']))
    individual_errors = []
    for word in range(4):
        unit_source = fibre.source_vectors_at(fields,links,theta,a,eta,np.eye(4)[word])
        unit_relative = full_slater_relative_spin(fibre,unit_source,spin_full)['relative']
        expected_unit = np.einsum('i,icb->cb',action[:,word],words)
        individual_errors.append(float(np.linalg.norm(unit_relative-expected_unit)))
    v = fibre.source_vectors_at(fields,links,theta,a,eta,BETA)['total_source']
    target = rel['relative']@fibre.raw_neutral
    scalar = np.vdot(v,target)/np.vdot(v,v)
    wrongmixed = float(np.linalg.norm(target-scalar*v))
    wordspan = words.transpose(1,0,2).reshape(448,8)
    outside_word = float(np.linalg.norm(target-wordspan@np.linalg.lstsq(wordspan,target,rcond=1e-12)[0]))
    mixed_basis = fibre.source_vectors_at(fields,links,theta,a,eta,BETA)['source_matrix']
    old_mixed_outside = float(np.linalg.norm(target-mixed_basis@np.linalg.lstsq(mixed_basis,target,rcond=1e-12)[0]))
    single_ref = mixed_basis.copy()
    for charge in (1,-1):
        rows = fibre.charged_slices[charge]
        block = single_ref[rows].reshape(4,-1,2)
        block[1:] = 0
    missing_ref = float(np.linalg.norm(mixed_basis-single_ref))
    current_direction=np.zeros((law.volume,3));current_direction[:,0]=np.cos(2*np.pi*law.sites[:,1]/3)
    h=2.**-18
    current=(law.onebody(fields,links,a+h*current_direction)-law.onebody(fields,links,a-h*current_direction))/(2*h)
    current_graph=fibre.current_rows(current,fibre.source_vectors_at(fields,links,theta,a,eta,BETA))
    derivative=bank.directional_values(lambda s:(fields+s*law.fields(np.ones((law.volume,4,15))*.01),links,theta,a),neutral)
    # Moving orbitals/frame are supplied by owner; INITIAL function columns stay fixed.
    moved=copy(fibre); light=expm(-.07j*spin_full);ref=expm(.03j*spin)
    moved.orbitals=light@fibre.orbitals;moved.empty=light@fibre.empty
    moved.occupied={q:light@x for q,x in fibre.occupied.items()}
    moved.projectors={q:light@p@light.conj().T for q,p in fibre.projectors.items()}
    before=bank.source_values(fields,links,theta,a,neutral)
    physical=bank.physical_source(moved,ref,fields,links,theta,a,'source19',BETA)
    fixed_error=float(np.linalg.norm(bank.source_values(fields,links,theta,a,neutral)-before))
    if max(errors)>1e-10 or recovery>1e-10 or max(individual_errors)>1e-10 or outside_word>1e-10 or max(variation)<1e-5 or min(wrongminus)<.1 or wrongmixed<.05 or old_mixed_outside<.05 or missing_ref<.1 or fixed_error:
        raise AssertionError('Original raw-map/spin/function adversarial acceptance failed')
    if hashlib.sha256(bank.seed_fibre.orbitals.tobytes()).hexdigest()!=bank.orbital_sha256:
        raise AssertionError('Initial occupied orbitals changed')
    return dict(API=bank.descriptor(14),source19=bound,combined_pure_producer_errors=errors,
        nonconstant_function_changes=variation,wrong_minus_complex_linearity_defects=wrongminus,
        spin_word_action=action,spin_word_derivation_error=derived['error'],relative_spin=rel,
        full_fourword_relative_recovery_error=recovery,full_fourword_span_outside=outside_word,
        wrong_scalar_mixed_spin_projection_defect=wrongmixed,
        all_four_individual_word_relative_errors=individual_errors,
        old_combined_beta_static_spin_outside_norm=old_mixed_outside,
        missing_three_Ref_adverse_norm=missing_ref,
        current_full_Wick_norm_squared=current_graph['full_SOURCE_response_norm_squared'],
        current_outside_PH_norm_squared=current_graph['unrepresented_SOURCE_response_norm_squared'],
        producer_derivative_native_error=derivative['native_error_estimate'],
        fixed_initial_under_moving_frame_error=fixed_error,
        moving_physical_difference=float(np.linalg.norm(physical['source_matrix']@neutral-bank.combine_values(before,BETA,14,'source19'))),
        scope='nativeN3 full108/fixed54+R4/full448 original functions; analytic quaternion/CAR identity; not physical G/U2pi/Gauss/fullFock closure',
        packet_templates=dict(source19='unchanged eta',forward='same eta; separate photon exp(-i k_x), exp(-i omega6)'),
        debt=['physical orbital/Phi/link/EM/harmonic/Ref Noether supplied by B owner','neutral/background and charged PH outside dynamics','legal parameter/native hard-ID gaps','full field integration and exactNPZ EMvac embed owned by A','no Noether/Gauss closure promotion'])


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--descriptor',required=True,type=Path)
    args=parser.parse_args()
    started=time.perf_counter()
    root=Path(__file__).resolve().parent
    loaded={Path(cast(str,m.__file__)).name:Path(cast(str,m.__file__)).read_bytes() for m in list(sys.modules.values()) if getattr(m,'__file__',None) and Path(cast(str,m.__file__)).parent==root and Path(cast(str,m.__file__)).suffix=='.py'}
    result=native_science(args.descriptor)
    if any((root/name).read_bytes()!=content for name,content in loaded.items()):
        raise RuntimeError('Loaded scientific source changed during execution')
    result.update(source_sha256={name:hashlib.sha256(content).hexdigest() for name,content in loaded.items()},
                  loaded_sources_unchanged=True,science_seconds=time.perf_counter()-started)
    print(json.dumps(portable(result),sort_keys=True))


if __name__ == '__main__':
    main()
