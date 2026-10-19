#!/usr/bin/env python3
"""Shared exact-EM extension of the immutable A17 preparation.

The preparation coefficients remain f(H14_MC)tau.  This module owns evolution
forms, not B's physical generator, and never refilters the supplied state.
"""
from __future__ import annotations

import argparse
from copy import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import zipfile
from typing import Any, cast

if __name__ == '__main__':
    for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = '1'

import numpy as np
from scipy.linalg import eigh, lu_factor, lu_solve

from improved_phi_band import ImprovedPhiLaw, selected_band
from original_charged_principal import portable
from prepared_neutral_ambient_metric import AmbientFieldChart
from prepared_neutral_ambient_weak_state import AmbientWeakState
from prepared_neutral_full_field import BETA
from prepared_neutral_quartet_functions import QuartetSourceFunctionBank, source19_native_profile
from prepared_neutral_slater_fibre import FullSlaterFibre, slater_expectation
from prepared_neutral_weyl_hard_fock import HardFockBasis, native_packets, packet_templates, gaussian_kernel

FROZEN_STATE_SHA256 = 'eb283954e7ebb13e1b5368deb058214443d9a839b8319161c0306afb2c8b53b6'
PLAN = 'issue-222-fullfield-closedrotation-continuation-20261006'
REVISION = '57838ca23dbf1424677c1c2f4e183784463faa3f759a85ec2a2b9722ff1fd9ff'


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hermitian(matrix: np.ndarray) -> np.ndarray:
    return (matrix + matrix.conj().T) / 2


class ExtendedAmbientWeakState:
    """One fixed INITIAL functional bank; current moving SOURCE stays separate.

    Raw ordering is neutral-column then hard-Fock index. Charged base ordering
    is profile, original word, neutral column, hard-Fock index. The extra
    columns are B's physical finite-Weyl generated functions, not fake Fock
    closure. Congruences, if present, change coordinates only.
    """
    def __init__(self, original: AmbientWeakState, prepared14: np.ndarray,
                 profiles: dict[str, np.ndarray], velocity: np.ndarray):
        from prepared_neutral_noether_gaussian_partners import NoetherGaussianBank
        if tuple(profiles) != ('source19', 'SOFT'):
            raise ValueError('Exactly one SOURCE19 bank and distinct SOFT required')
        self.original = original
        self.prepared14 = np.array(prepared14, complex, copy=True)
        if self.prepared14.shape != (14,):
            raise ValueError('Immutable A17 fourteen coefficients required')
        self.prepared14.setflags(write=False)
        self.source_bank = QuartetSourceFunctionBank(original.seed_fibre, profiles)
        self.photon_basis = HardFockBasis(original.em, native_packets(original.law, original.em))
        self.noether_bank = NoetherGaussianBank(original, velocity)
        self.dimension = 214
        self.charged_dimension = 1728
        self.neutral_coordinates = np.eye(214, dtype=complex)
        self.charged_coordinates = np.eye(1728, dtype=complex)
        self.boost: dict[str,Any] = dict(Omega=0., Omega_dot=0., Y0=np.zeros(original.chart.normal_dim))
        self.cache: dict[str, Any] = {}

    def __getattr__(self, name: str) -> Any:
        # copy.copy probes an uninitialized instance before its dict is restored.
        # Bypass delegation when obtaining the owner so the missing attribute
        # raises AttributeError normally instead of recursively calling us.
        original = object.__getattribute__(self,'original')
        return getattr(original,name)

    def compilecache(self) -> dict[str, Any]:
        """Return the once compiled cache; never execute scientific compilation."""
        return self.cache

    def basis_at(self, q: np.ndarray, em: np.ndarray,
                 point: dict[str, Any] | None = None) -> dict[str, Any]:
        sample = self.original.basis_at(q, em, point)
        p, dp = self.photon_basis.polynomials(np.asarray(em)[:52])
        nb = np.einsum('bi,h->bih', sample['neutral'], p).reshape(2, 210)
        cb0 = self.source_bank.source_values(sample['point']['phi'], sample['point']['links'],
                                             sample['theta'], sample['a'], sample['neutral'])
        cb = np.einsum('ci,h->cih', cb0, p).reshape(448, 1680)
        extra = self.noether_bank.generated_values(sample, self.source_bank, self.prepared14)
        neutral = np.column_stack((nb, extra['neutral_extra']))
        charged = np.column_stack((cb, extra['charged_extra']))
        phase = np.exp(1j*self.boost['Omega']*(self.boost['Y0']@q)/self.chi)
        sample.update(neutral=phase*neutral@self.neutral_coordinates,
                      charged=phase*charged@self.charged_coordinates,
                      original_neutral=sample['neutral'], hard_polynomial=p,
                      hard_polynomial_derivative=dp, held_source_base=cb0,
                      common_boost_phase=phase, raw_neutral=neutral, raw_charged=charged)
        return sample

    sample = basis_at

    def directional_basis(self, sample: dict[str, Any], dq: np.ndarray,
                          dem: np.ndarray | None = None, step: float = 2.**-19) -> dict[str, np.ndarray]:
        dq = np.asarray(dq, float)
        dem = np.zeros(55) if dem is None else np.asarray(dem, float)
        old = dict(sample, neutral=sample['original_neutral'])
        # Original API carries full Gaussian, cutoff and Haar half-density jet.
        nd = self.original.directional_basis(old, dq, dem, step)['neutral']
        p = sample['hard_polynomial']
        dp = sample['hard_polynomial_derivative'].T@dem[:52]
        nb = (np.einsum('bi,h->bih', nd, p)
              + np.einsum('bi,h->bih', old['neutral'], dp)).reshape(2, 210)
        scale = max(1., float(np.linalg.norm(dq)), float(np.linalg.norm(dem)))
        eps = step/scale
        def path(t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            qq = sample['q']+t*dq
            pp = self.chart.embedding(qq)
            aa, tt = self.original.electromagnetic_field(sample['em']+t*dem)
            return pp['phi'], pp['links'], tt, aa
        source_jet = self.source_bank.directional_values(path, old['neutral'], eps)
        cd0 = source_jet['derivative'] + self.source_bank.source_values(*path(0.), nd)
        cb = (np.einsum('ci,h->cih', cd0, p)
              + np.einsum('ci,h->cih', sample['held_source_base'], dp)).reshape(448, 1680)
        extra = self.noether_bank.generated_directional_values(old, self.source_bank, self.prepared14,
                                                               dq, dem[52:], dem[:52])
        # Owner helper supplies FULL Gaussian/Jhalf/cutoff derivatives once.
        ne = extra['neutral_extra']
        ce = extra['charged_extra']
        phasejet = 1j*self.boost['Omega']*(self.boost['Y0']@dq)/self.chi
        phase = sample['common_boost_phase']
        return dict(neutral=phase*(np.column_stack((nb, ne))+phasejet*sample['raw_neutral'])@self.neutral_coordinates,
                    charged=phase*(np.column_stack((cb, ce))+phasejet*sample['raw_charged'])@self.charged_coordinates)

    def gauge_operators(self, coefficients: np.ndarray, sector: str) -> dict[str, Any]:
        return self.original.gauge_operators(coefficients, sector)

    def first_order_weak(self, sample: dict[str, Any], dq: np.ndarray, dem: np.ndarray,
                         gauge_velocity: np.ndarray, body: np.ndarray, reference: np.ndarray,
                         sector: str = 'charged', heavy_body: np.ndarray | None = None) -> np.ndarray:
        value = sample[sector]
        d = self.directional_basis(sample, dq, dem)[sector]
        d += 1j*self.gauge_operators(gauge_velocity, sector)['first']@value
        matter = self.fibre.onebody(body, sector)+self.original.reference_operator(reference, sector)
        if heavy_body is not None:
            matter += slater_expectation(self.heavy_projector, heavy_body)*self.fibre.overlap(sector)
        return sample['importance_weight']*(-.5j*(value.conj().T@d-d.conj().T@value)+value.conj().T@matter@value)

    def in_mean_interaction_frame(self, light: np.ndarray, reference: np.ndarray,
                                  heavy: np.ndarray | None = None) -> ExtendedAmbientWeakState:
        moved = copy(self)
        moved.original = self.original.in_mean_interaction_frame(light, reference)
        if heavy is not None:
            heavy = np.asarray(heavy, complex)
            if heavy.shape != (108,108) or np.linalg.norm(heavy.conj().T@heavy-np.eye(108)) > 1e-9:
                raise ValueError('Actual heavy108 mean unitary required')
            moved.original.heavy_orbitals = heavy@self.original.heavy_orbitals
            moved.original.heavy_projector = moved.original.heavy_orbitals@moved.original.heavy_orbitals.conj().T
        # INITIAL functional bank and Noether partners remain held, not recaptured.
        return moved

    def with_common_boost(self, Omega: float, Omega_dot: float, Y0: np.ndarray) -> ExtendedAmbientWeakState:
        moved = copy(self)
        Y0 = np.asarray(Y0, float)
        if Y0.shape != (self.chart.normal_dim,):
            raise ValueError('Full2430 common mean normal flow required')
        moved.boost = dict(Omega=float(Omega), Omega_dot=float(Omega_dot), Y0=Y0.copy())
        return moved

    def common_frame_energy(self, sample: dict[str, Any]) -> float:
        return (self.boost['Omega_dot']*(self.boost['Y0']@sample['q'])/self.chi
                -self.boost['Omega']**2*float(self.boost['Y0']@self.boost['Y0'])/(2*self.chi))

    def source_at(self, fields: np.ndarray, links: np.ndarray, theta: np.ndarray,
                  a: np.ndarray, selected_profile: str = 'source19') -> dict[str,Any]:
        return self.source_bank.physical_source(self.fibre,self.reference_frame,fields,links,theta,a,
                                               selected_profile,BETA)

    def current_source_function(self, sample: dict[str,Any], coefficients: np.ndarray,
                                selected_profile: str = 'source19') -> dict[str,Any]:
        physical = self.source_at(sample['point']['phi'],sample['point']['links'],
                                  sample['theta'],sample['a'],selected_profile)['source_matrix']@sample['neutral']@coefficients
        mapping = self.cache[selected_profile+'_map']
        held = sample['charged']@mapping@coefficients
        return dict(full448_physical_SOURCE=physical,transported_initial_SOURCE=held,
                    full448_difference_norm=float(np.linalg.norm(physical-held)),
                    SOURCE_basis_NOT_recaptured=True)

    current_source_function_residual = current_source_function


def raw_source_maps() -> dict[str, np.ndarray]:
    """Literal word/beta coefficient maps, including BOTH common partners."""
    out: dict[str, np.ndarray] = {}
    for template, name in enumerate(('source19', 'SOFT')):
        words = np.zeros((4,1728,214), complex)
        for bank in range(2):
            for word in range(4):
                for i in range(7):
                    for h in range(15):
                        n = (bank*7+i)*15+h
                        c = ((template*4+word)*14+bank*7+i)*15+h
                        words[word,c,n] = 1
                for category in range(2):
                    words[word,1680+category*16+(template*4+word)*2+bank,210+category*2+bank] = 1
        source = sum((BETA[w]*words[w]@np.diag(np.r_[np.ones(105),np.zeros(105),1,0,1,0])
                      +BETA[w].conjugate()*words[w]@np.diag(np.r_[np.zeros(105),np.ones(105),0,1,0,1])
                      for w in range(4)), np.zeros((1728,214),complex))
        out[name+'_map'] = source
        out[name+'_word_maps'] = words
    return out


def exact_base_form(left: np.ndarray, operator: np.ndarray, right: np.ndarray,
                    photon_kernel: np.ndarray) -> np.ndarray:
    return np.kron(left.conj().T@operator@right, photon_kernel)


def exact_light_base(model: ExtendedAmbientWeakState, sample: dict[str, Any],
                     values: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """ALL52 Debye BEFORE projection, literal law.terms including every leg."""
    law = model.law
    fields, links, theta = sample['point']['phi'], sample['point']['links'], sample['theta']
    zero = np.zeros((law.volume,3))
    onsite = np.zeros((108,108),complex)
    for x in range(law.volume):
        onsite[4*x:4*x+4,4*x:4*x+4] = law.onsite*fields[x,0]
    result = {s: exact_base_form(v, model.fibre.onebody(onsite,s)+model.HC[s],v,np.eye(15))
              for s,v in values.items()}
    for x,y,coefficient,path in law.terms(fields,links,zero):
        displacement = sum((model.em['legs'][3*site+axis] for site,axis in path),np.zeros(52))
        harmonic = sum((theta[axis]/law.size for _,axis in path),0.)
        body = np.zeros((108,108),complex)
        body[4*x:4*x+4,4*y:4*y+4] = coefficient*np.exp(1j*harmonic)
        # onebody is complex-linear, so conjugate term is kept separate.
        kernel = model.photon_basis.weyl_matrix(displacement)
        for sector,v in values.items():
            term = exact_base_form(v,model.fibre.onebody(body,sector),v,kernel)
            result[sector] += term+term.conj().T
    return result


def frozen_nodes(model: ExtendedAmbientWeakState) -> list[dict[str, Any]]:
    """Consume EXACT original RNG order without preparing or compiling A17."""
    rng = np.random.default_rng(222)
    qnodes = rng.standard_normal((7,2430))*model.sigma/math.sqrt(2)
    enodes = rng.standard_normal((7,55))/math.sqrt(2)
    records = []
    for q0,e0 in zip(qnodes,enodes):
        for sign in (-1,1):
            q,e = sign*q0,sign*e0
            sample = model.original.basis_at(q,e)
            factor = lu_factor(np.sqrt(model.chart.mass)[:,None]*sample['point']['jacobian'])
            directions = [lu_solve(factor,rng.standard_normal(2835)) for _ in range(4)]
            records.append(dict(sample=sample,directions=directions))
    return records


def snapshot_encode(value: Any, arrays: dict[str,np.ndarray], prefix: str) -> Any:
    """NPZ arrays plus ordinary JSON; no pickle or hidden executable cache."""
    if isinstance(value,np.ndarray):
        arrays[prefix] = value
        return {'array':prefix}
    if isinstance(value,np.generic):
        return snapshot_encode(value.item(),arrays,prefix)
    if isinstance(value,complex):
        return {'complex':[value.real,value.imag]}
    if isinstance(value,slice):
        return {'slice':[value.start,value.stop,value.step]}
    if isinstance(value,dict):
        return {'dict':[[snapshot_encode(k,arrays,prefix+'_key'+str(i)),
                         snapshot_encode(v,arrays,prefix+'_'+str(i))]
                        for i,(k,v) in enumerate(value.items())]}
    if isinstance(value,(tuple,list)):
        return {'sequence':[snapshot_encode(v,arrays,prefix+'_'+str(i)) for i,v in enumerate(value)],
                'tuple':isinstance(value,tuple)}
    if value is None or isinstance(value,(str,int,float,bool)):
        return value
    raise TypeError('Unsupported cache state '+prefix+': '+type(value).__name__)


def snapshot_decode(value: Any, arrays: dict[str,np.ndarray]) -> Any:
    if not isinstance(value,dict):
        return value
    if 'array' in value:
        return arrays[value['array']]
    if 'complex' in value:
        return complex(*value['complex'])
    if 'slice' in value:
        return slice(*value['slice'])
    if 'dict' in value:
        return {snapshot_decode(k,arrays):snapshot_decode(v,arrays) for k,v in value['dict']}
    seq = [snapshot_decode(v,arrays) for v in value['sequence']]
    return tuple(seq) if value['tuple'] else seq


def load_shared_native_cache(npz_path: Path, manifest_path: Path) -> tuple[ExtendedAmbientWeakState,dict[str,Any]]:
    """Reconstruct exact once seed WITHOUT constructor sea eigensolve/filter."""
    manifest = json.loads(manifest_path.read_bytes())
    if manifest.get('accepted') is not True:
        raise ValueError('Unaccepted scientific checkpoint is not a consumer cache')
    if sha256(npz_path) != manifest['npz_sha256']:
        raise ValueError('Shared cache hash differs from immutable manifest')
    with np.load(npz_path,allow_pickle=False) as stored:
        arrays = {name:stored[name].copy() for name in stored.files}
    seed = manifest['seed_snapshot']
    band,_,_ = selected_band()
    law = ImprovedPhiLaw(3,band)
    original = object.__new__(AmbientWeakState)
    original.__dict__.update(snapshot_decode(seed['original'],arrays))
    original.law = law
    original.chart = AmbientFieldChart(law,arrays['seed_mean_fields'],arrays['seed_mean_links'])
    from rotation_band import RotationBand
    original.heavy_law = ImprovedPhiLaw(3,RotationBand(**cast(Any,original).heavy_law_band))
    fibre = object.__new__(FullSlaterFibre)
    fibre.__dict__.update(snapshot_decode(seed['fibre'],arrays))
    fibre.law = law
    original.fibre = original.seed_fibre = fibre
    model = ExtendedAmbientWeakState(original,arrays['preparation_state14'],
                                    {'source19':arrays['source19_profile'],'SOFT':arrays['SOFT_profile']},
                                    arrays['noether_velocity'])
    model.neutral_coordinates = arrays['neutral_coordinates']
    model.charged_coordinates = arrays['charged_coordinates']
    model.dimension = arrays['neutral_H'].shape[0]
    model.charged_dimension = arrays['charged_H'].shape[0]
    data: dict[str,Any] = dict(arrays,manifest=manifest)
    model.cache = data
    return model,data


def coupled_form(left_base: np.ndarray, left_extra: np.ndarray,
                 right_base: np.ndarray, right_extra: np.ndarray,
                 operator: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Contract shared scalar-function kernels without dimension^degree arrays.

    Extra tensors are [fibre,extra-column,scalar-function]. The first15 scalar
    functions are the literal hard oscillator basis; subsequent functions
    are finite Weyl sums from the Noether owner.
    """
    h = 15
    nleft, nright = left_base.shape[1]*h, right_base.shape[1]*h
    out = np.zeros((nleft+left_extra.shape[1], nright+right_extra.shape[1]),complex)
    out[:nleft,:nright] = exact_base_form(left_base,operator,right_base,kernel[:h,:h])
    rt = np.einsum('ab,ben->aen',operator,right_extra,optimize=True)
    cross = np.einsum('ba,aen->ben',left_base.conj().T,rt,optimize=True)
    out[:nleft,nright:] = np.einsum('ben,hn->bhe',cross,kernel[:h,h:],optimize=True).reshape(nleft,-1)
    lt = np.einsum('ab,ben->aen',operator,right_base[:,:,None],optimize=True)[:,:,0]
    cross = np.einsum('aem,ab->ebm',left_extra.conj(),lt,optimize=True)
    out[nleft:,:nright] = np.einsum('ebm,mh->ebh',cross,kernel[h:,:h],optimize=True).reshape(-1,nright)
    transformed = np.einsum('aen,mn->aem',rt,kernel[h:,h:],optimize=True)
    out[nleft:,nright:] = np.einsum('aem,afm->ef',left_extra.conj(),transformed,optimize=True)
    return out


def coordinate_conditioning(gram: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    """Diagonal coordinate equilibration ONLY; no state/profile whitening."""
    diagonal = cast(Any,np.diag(gram)).real
    if np.any(diagonal <= 0):
        raise ValueError('Zero functional norm: actual null quotient required')
    scale = 1/np.sqrt(diagonal)
    coordinates = np.diag(scale).astype(complex)
    conditioned = hermitian(scale[:,None]*gram*scale[None,:])
    # A positive but tiny SOFT direction is retained, regardless of raw scale.
    eig = eigh(conditioned,eigvals_only=True)
    if eig[0] <= 0:
        raise ValueError('Nonpositive function Gram after coordinate equilibration: '+str(eig[0]))
    return coordinates,conditioned


def loaded_source_epoch() -> dict[str,str]:
    root = Path(__file__).resolve().parent
    paths = set()
    for module in list(sys.modules.values()):
        name = getattr(module,'__file__',None)
        if name:
            path = Path(name)
            if path.parent == root and path.suffix == '.py':
                paths.add(path)
    paths.add(Path(__file__).resolve())
    return {path.name:sha256(path) for path in sorted(paths)}


def source_epoch_seal(before: dict[str,str], after: dict[str,str],
                      declared: dict[str,str]) -> dict[str,Any]:
    """Accept known frozen late imports, never same-path source mutations."""
    changed = {name:dict(before=digest,after=after.get(name)) for name,digest in before.items()
               if after.get(name) != digest}
    added = {name:digest for name,digest in after.items() if name not in before}
    rejected_late = {name:dict(expected=declared.get(name),actual=digest)
                     for name,digest in added.items() if declared.get(name) != digest}
    return dict(before=before,after=after,predeclared=declared,changed=changed,added=added,
                rejected_late=rejected_late,accepted=not changed and not rejected_late,
                uninterrupted_disk_epoch_claimed=False)


def publish_scientific_cache(npz_path: Path, manifest_path: Path,
                            arrays: dict[str,np.ndarray], metadata: dict[str,Any],
                            before: dict[str,str], after: dict[str,str],
                            declared: dict[str,str], archive_path: Path) -> dict[str,Any]:
    """Private recovery checkpoint first; accepted cache only AFTER source seal."""
    checkpoint = npz_path.with_name(npz_path.stem+'.UNACCEPTED.npz')
    cast(Any,np.savez_compressed)(checkpoint,**arrays)
    epoch = source_epoch_seal(before,after,declared)
    manifest_path.with_suffix('.source-epochs.json').write_text(json.dumps(epoch,sort_keys=True,indent=2))
    private = dict(metadata,accepted=False,unaccepted_checkpoint=str(checkpoint),
                   unaccepted_checkpoint_sha256=sha256(checkpoint),source_epoch=epoch,
                   loaded_sources_archive=str(archive_path),loaded_sources_archive_sha256=sha256(archive_path))
    private_path = manifest_path.with_suffix('.UNACCEPTED.json')
    private_path.write_text(json.dumps(private,sort_keys=True,indent=2))
    if not epoch['accepted']:
        raise AssertionError('Loaded scientific source seal rejected: '+json.dumps(
            {key:epoch[key] for key in ('changed','added','rejected_late')},sort_keys=True))
    manifest = dict(metadata,accepted=True,npz_sha256=sha256(checkpoint),
        loaded_sources_before=before,loaded_sources_after=after,
        loaded_sources_unchanged=not epoch['changed'],late_imported_frozen_sources=epoch['added'],
        source_epoch=epoch,loaded_sources_archive=str(archive_path),
        loaded_sources_archive_sha256=sha256(archive_path),uninterrupted_disk_epoch_claimed=False)
    # The manifest is the accept boundary. A private checkpoint is never
    # reachable through load_shared_native_cache.
    os.replace(checkpoint,npz_path)
    manifest_path.write_text(json.dumps(manifest,sort_keys=True,indent=2))
    private_path.unlink()
    return manifest


def seed_snapshot(model: ExtendedAmbientWeakState, arrays: dict[str,np.ndarray]) -> dict[str,Any]:
    original = {k:v for k,v in vars(model.original).items()
                if k not in ('law','chart','fibre','seed_fibre','heavy_law')}
    # Band is a small immutable dataclass; reconstruct it, not any occupied state.
    original['heavy_law_band'] = dict(vars(model.heavy_law.band))
    fibre = {k:v for k,v in vars(model.original.seed_fibre).items() if k != 'law'}
    arrays['seed_mean_fields'] = model.seed_fibre.mean_fields
    arrays['seed_mean_links'] = model.seed_fibre.mean_links
    return dict(original=snapshot_encode(original,arrays,'seed_original'),
                fibre=snapshot_encode(fibre,arrays,'seed_fibre'))


def conditional_values(model: ExtendedAmbientWeakState, sample: dict[str,Any]) -> tuple[dict[str,np.ndarray],dict[str,np.ndarray]]:
    base = dict(neutral=sample['neutral'],
                charged=model.source_bank.source_values(sample['point']['phi'],sample['point']['links'],
                    sample['theta'],sample['a'],sample['neutral']))
    extra = model.noether_bank.conditional_coefficients(sample,model.source_bank,model.prepared14)
    return base,{s:extra[s][:,:,15:] for s in ('neutral','charged')}


def conditional_derivative(model: ExtendedAmbientWeakState, sample: dict[str,Any],
                           base: dict[str,np.ndarray], extra: dict[str,np.ndarray],
                           dq: np.ndarray, dtheta: np.ndarray) -> tuple[dict[str,np.ndarray],dict[str,np.ndarray]]:
    """Literal function jets with native FD debt, full Jhalf/cutoff separately."""
    scale = max(1.,float(np.linalg.norm(dq)),float(np.linalg.norm(dtheta)))
    eps = 2.**-19/scale
    def shifted(t: float) -> tuple[dict[str,np.ndarray],dict[str,np.ndarray]]:
        q,e = sample['q']+t*dq,sample['em'].copy()
        e[-3:] += t*dtheta
        point = model.chart.embedding(q)
        return conditional_values(model,model.original.basis_at(q,e,point))
    plus,pe = shifted(eps)
    minus,me = shifted(-eps)
    logder = -float(sample['q']@dq)/model.sigma**2-float(sample['theta']@dtheta)
    if np.any(dq):
        logder -= .5*model.chart.directional_jet(sample['q'],dq,sample['point'])['log_density_first']
        logder += float(sample['cutoff_gradient']@dq)/sample['cutoff']
    return ({s:(plus[s]-minus[s])/(2*eps)+logder*base[s] for s in base},
            {s:(pe[s]-me[s])/(2*eps)+logder*extra[s] for s in base})


def scalar_native_kernels(model: ExtendedAmbientWeakState, em_nodes: int) -> dict[str,Any]:
    """Explicit ALL52 Gaussian positive-form quadrature for generated H only.

    Exact conditional Gram is supplied separately. The hard15 H block is
    analytic and replaces its quadrature block, never the immutable Hprep.
    """
    rng = np.random.default_rng(222710)
    ys = rng.standard_normal((em_nodes//2,52))/math.sqrt(2)
    ys = np.concatenate((-ys,ys),axis=0)
    values,derivatives = zip(*(model.noether_bank.scalar_values(y) for y in ys))
    values = np.asarray(values)
    derivatives = np.asarray(derivatives)-ys[:,:,None]*values[:,None,:]
    omega = model.em['omega']
    kinetic = np.einsum('nim,i,nij->mj',derivatives.conj(),omega/2,derivatives,optimize=True)/len(ys)
    potential = np.einsum('nm,n,nj->mj',values.conj(),np.sum(omega*ys**2,axis=1)/2,values,optimize=True)/len(ys)
    free = hermitian(kinetic+potential)
    free[:15,:15] = np.diag(model.photon_basis.free_energy)
    return dict(ys=ys,values=values,free=free,
                generated_H_quadrature_nodes=len(ys),generated_H_quadrature_seed=222710)


def generated_weyl_kernel(scalars: dict[str,Any], displacement: np.ndarray,
                          exact_base: np.ndarray) -> np.ndarray:
    values = scalars['values']
    phase = np.exp(1j*(scalars['ys']@displacement))
    kernel = values.conj().T@(phase[:,None]*values)/len(values)
    kernel[:15,:15] = exact_base
    return kernel


def compile_extended(model: ExtendedAmbientWeakState, frozen: dict[str,np.ndarray],
                     em_nodes: int = 128) -> tuple[dict[str,np.ndarray],dict[str,Any]]:
    if em_nodes < 32 or em_nodes%2:
        raise ValueError('At least32 antithetic EM nodes required for generated H')
    records = frozen_nodes(model)
    gramkernel = model.noether_bank.scalar_gram()
    scalar = scalar_native_kernels(model,em_nodes)
    sizes = dict(neutral=214,charged=1728)
    grams = {s:np.zeros((d,d),complex) for s,d in sizes.items()}
    hs = {s:np.zeros((d,d),complex) for s,d in sizes.items()}
    node_data = []
    for index,record in enumerate(records):
        sample = record['sample']
        base,extra = conditional_values(model,sample)
        weight = sample['importance_weight']/14
        identity = {s:model.fibre.overlap(s) for s in sizes}
        gnode = {s:coupled_form(base[s],extra[s],base[s],extra[s],identity[s],gramkernel) for s in sizes}
        scalarpot = sum(model.original.classical_potential(sample['point']).values())
        hnode = {s:scalarpot*gnode[s]+coupled_form(base[s],extra[s],base[s],extra[s],identity[s],scalar['free']) for s in sizes}
        fields,links = sample['point']['phi'],sample['point']['links']
        onsite = np.zeros((108,108),complex)
        for x in range(model.law.volume):
            onsite[4*x:4*x+4,4*x:4*x+4] = model.law.onsite*fields[x,0]
        for s in sizes:
            op = model.fibre.onebody(onsite,s)+model.HC[s]
            hnode[s] += coupled_form(base[s],extra[s],base[s],extra[s],op,gramkernel)
        for x,y,coefficient,path in model.law.terms(fields,links,np.zeros((27,3))):
            displacement = sum((model.em['legs'][3*site+axis] for site,axis in path),np.zeros(52))
            harmonic = sum((sample['theta'][axis]/3 for _,axis in path),0.)
            body = np.zeros((108,108),complex)
            body[4*x:4*x+4,4*y:4*y+4] = coefficient*np.exp(1j*harmonic)
            kernel = generated_weyl_kernel(scalar,displacement,model.photon_basis.weyl_matrix(displacement))
            for s in sizes:
                term = coupled_form(base[s],extra[s],base[s],extra[s],model.fibre.onebody(body,s),kernel)
                hnode[s] += term+term.conj().T
        for direction in record['directions']:
            dq,gauge = direction[:2430],direction[2430:]
            db,de = conditional_derivative(model,sample,base,extra,dq,np.zeros(3))
            for s in sizes:
                op = model.gauge_operators(gauge,s)
                pb = -1j*db[s]+op['first']@base[s]
                pe = -1j*de[s]+np.einsum('ab,ben->aen',op['first'],extra[s],optimize=True)
                form = coupled_form(pb,pe,pb,pe,identity[s],gramkernel)
                form += coupled_form(base[s],extra[s],base[s],extra[s],op['outside_second'],gramkernel)
                hnode[s] += model.chi/8*hermitian(form)
        for axis in range(3):
            dtheta = np.eye(3)[axis]
            db,de = conditional_derivative(model,sample,base,extra,np.zeros(2430),dtheta)
            for s in sizes:
                hnode[s] += model.ue/3*coupled_form(db[s],de[s],db[s],de[s],identity[s],gramkernel)
        for s in sizes:
            grams[s] += weight*gnode[s]
            hs[s] += weight*hermitian(hnode[s])
        node_data.append(dict(index=index,theta=sample['theta'].tolist(),
                              normal_norm=float(np.linalg.norm(sample['q'])),
                              full_normal_nonzero=int(np.count_nonzero(sample['q']))))
    vacuum = np.arange(14)*15
    vacuum_gram = grams['neutral'][np.ix_(vacuum,vacuum)]
    metric_error = float(np.linalg.norm(vacuum_gram-frozen['neutral_Gram']))
    if metric_error > 1e-11:
        raise AssertionError('Retained Phi measure changed original vacuum metric')
    embedding = np.zeros((214,14),complex)
    embedding[vacuum,np.arange(14)] = 1
    state = embedding@model.prepared14
    norm = float(np.vdot(state,grams['neutral']@state).real)
    if abs(norm-1)>1e-10:
        raise AssertionError('Immutable prepared vacuum lost its metric norm')
    coords = {}
    conditioned = {}
    for s in sizes:
        coords[s],conditioned[s] = coordinate_conditioning(grams[s])
    model.neutral_coordinates,model.charged_coordinates = coords['neutral'],coords['charged']
    arrays: dict[str,np.ndarray] = {}
    for s in sizes:
        arrays[s+'_Gram'] = conditioned[s]
        arrays[s+'_H'] = coords[s].conj().T@hs[s]@coords[s]
        arrays[s+'_coordinates'] = coords[s]
        arrays[s+'_raw_Gram'] = grams[s]
    arrays['neutral_embedding'] = np.linalg.solve(coords['neutral'],embedding)
    arrays['neutral_state'] = arrays['neutral_embedding']@model.prepared14
    for name,mapping in raw_source_maps().items():
        if mapping.ndim == 2:
            arrays[name] = np.linalg.solve(coords['charged'],mapping@coords['neutral'])
        else:
            arrays[name] = np.asarray([np.linalg.solve(coords['charged'],m@coords['neutral']) for m in mapping])
    arrays['preparation_state14'] = model.prepared14
    for key,value in frozen.items():
        arrays['preparation_'+key] = value
    blockdiff = hs['neutral'][np.ix_(vacuum,vacuum)]-frozen['neutral_H']
    metadata = dict(nodes=node_data,neutral_metric_norm=norm,original_vacuum_Gram_error=metric_error,
        Hprep14_MC_vs_Hext_exactEM_vacuum_block_difference_norm=float(np.linalg.norm(blockdiff)),
        Hprep14_MC_vs_Hext_exactEM_vacuum_block_difference_max=float(np.max(abs(blockdiff))),
        new_state_refiltered=False,all52_exact_base_EM=True,generated_H_EM_nodes=em_nodes,
        generated_H_native_quadrature_debt=True,exact_conditional_EM_Gram=True,
        retained_Phi14_harmonic14_kinetic4_seed222=True,physical_input_whitening=False,
        minimum_conditioned_Gram_eigenvalues={s:float(eigh(conditioned[s],eigvals_only=True)[0]) for s in sizes},
        H_weak_Hermiticity={s:float(np.linalg.norm(arrays[s+'_H']-arrays[s+'_H'].conj().T)) for s in sizes},
        H_weak_minimum_eigenvalues={s:float(eigh(arrays[s+'_H'],eigvals_only=True)[0]) for s in sizes},
        raw_basis_dimensions=sizes,Noether_gamma=model.noether_bank.projection_receipt())
    return arrays,metadata


def export_observables(model: ExtendedAmbientWeakState, arrays: dict[str,np.ndarray]) -> dict[str,Any]:
    ncoord,ccoord = arrays['neutral_coordinates'],arrays['charged_coordinates']
    nlabels = np.r_[np.repeat((1,-1),105),1,-1,1,-1]
    clabels = np.r_[np.tile(np.repeat((1,-1),105),8),np.tile((1,-1),24)]
    arrays['neutral_charge_labels'] = nlabels
    arrays['charge_labels'] = clabels
    arrays['charged_charge_projectors'] = np.asarray([np.diag(clabels==q).astype(complex) for q in (1,-1)])
    arrays['neutral_charge_projectors'] = np.asarray([np.diag(nlabels==q).astype(complex) for q in (1,-1)])
    arrays['photon_basis_labels'] = np.asarray(model.photon_basis.labels,int)
    arrays['charged_photon_labels'] = np.r_[np.tile(np.arange(15),112),np.full(48,-1)]
    # Coordinate vectors of the source-created hard Fock sectors, not a false
    # exact number projector on the infinite-Weyl generated tails.
    photon_columns = np.zeros((15,1728,112),complex)
    for h in range(15):
        raw = np.zeros((1728,112),complex)
        raw[np.arange(112)*15+h,np.arange(112)] = 1
        photon_columns[h] = np.linalg.solve(ccoord,raw)
    arrays['photon_projector_coordinates'] = photon_columns
    arrays['packet_templates'] = packet_templates(model.law,model.photon_basis)['creator_hard_coefficients']
    arrays['noether_velocity'] = model.noether_bank.velocity
    for name,profile in model.source_bank.profiles.items():
        arrays[name+'_profile'] = profile
    state = arrays['neutral_state']
    norms: dict[str,Any] = {}
    for name in ('source19','SOFT'):
        source = arrays[name+'_map']@state
        norm = float(np.vdot(source,arrays['charged_Gram']@source).real)
        if norm <= 0:
            raise AssertionError('Actual raw SOURCE function norm vanished')
        arrays[name+'_raw_source_state'] = source
        arrays[name+'_normalized_source_state'] = source/math.sqrt(norm)
        norms[name] = dict(raw_function_norm_squared=norm,
            global_SOURCE_normalizer=math.sqrt(norm),raw_profile_norm=float(np.linalg.norm(model.source_bank.profiles[name])),
            physical_input_profile_renormalized=False,
            charge_norms={str(q):float(np.vdot(source,np.diag(clabels==q)@arrays['charged_Gram']@np.diag(clabels==q)@source).real)
                          for q in (1,-1)})
    return norms


def export_creators(model: ExtendedAmbientWeakState, arrays: dict[str,np.ndarray]) -> dict[str,Any]:
    """Exact preprojection creator graph and its same-Gram compression."""
    records = frozen_nodes(model)
    kernels = [model.noether_bank.scalar_creator_kernels(model.photon_basis.packets[:,j]) for j in range(4)]
    sizes = dict(neutral=214,charged=1728)
    weak = {s:np.zeros((4,d,d),complex) for s,d in sizes.items()}
    full = {s:np.zeros((4,d,d),complex) for s,d in sizes.items()}
    for record in records:
        sample = record['sample']
        base,extra = conditional_values(model,sample)
        weight = sample['importance_weight']/14
        for j,kernel in enumerate(kernels):
            for s in sizes:
                identity = model.fibre.overlap(s)
                weak[s][j] += weight*coupled_form(base[s],extra[s],base[s],extra[s],identity,kernel['cross'])
                full[s][j] += weight*coupled_form(base[s],extra[s],base[s],extra[s],identity,kernel['full_norm'])
    controls: dict[str,Any] = {}
    for s in sizes:
        coord = arrays[s+'_coordinates']
        gram = arrays[s+'_Gram']
        cross = np.asarray([coord.conj().T@w@coord for w in weak[s]])
        norm = np.asarray([hermitian(coord.conj().T@w@coord) for w in full[s]])
        creators = np.asarray([np.linalg.solve(gram,w) for w in cross])
        outside = np.asarray([hermitian(norm[j]-creators[j].conj().T@gram@creators[j]) for j in range(4)])
        arrays[s+'_creators'] = creators
        arrays[s+'_creator_full_norm_Grams'] = norm
        arrays[s+'_creator_outside_Grams'] = outside
        if s == 'neutral':
            state = arrays['neutral_state']
        else:
            state = arrays['source19_normalized_source_state']
        controls[s] = dict(initial_creator_norm_squared=[float(np.vdot(state,n@state).real) for n in norm],
            initial_creator_outside_norm_squared=[float(np.vdot(state,n@state).real) for n in outside])
    # Initial vacuum and SOURCE commute with every coexact creator; these
    # images have an exact coefficient representation BEFORE compression.
    errors = []
    for j in range(4):
        raw_n = np.zeros((214,214),complex)
        raw_n[:210,:210] = np.kron(np.eye(14),model.photon_basis.creation[j])
        raw_c = np.zeros((1728,1728),complex)
        raw_c[:1680,:1680] = np.kron(np.eye(112),model.photon_basis.creation[j])
        nc = np.linalg.solve(arrays['neutral_coordinates'],raw_n@arrays['neutral_coordinates'])
        cc = np.linalg.solve(arrays['charged_coordinates'],raw_c@arrays['charged_coordinates'])
        for name in ('source19','SOFT'):
            errors.append(float(np.linalg.norm((cc@arrays[name+'_map']-arrays[name+'_map']@nc)@arrays['neutral_state'])))
    controls['initial_coexact_SOURCE_creator_commutator_coefficient_errors'] = errors
    return controls


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root-lemma',type=Path,required=True)
    parser.add_argument('--frozen-state',type=Path,required=True)
    parser.add_argument('--preparation-receipt',type=Path,required=True)
    parser.add_argument('--descriptor',type=Path,required=True)
    parser.add_argument('--SOFT-profile',type=Path,required=True)
    parser.add_argument('--noether-guard',type=Path,required=True)
    parser.add_argument('--output-npz',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--em-nodes',type=int,default=128)
    args = parser.parse_args()
    started = time.perf_counter()
    if sha256(args.frozen_state) != FROZEN_STATE_SHA256:
        raise ValueError('Immutable original A17 NPZ required')
    with np.load(args.frozen_state,allow_pickle=False) as stored:
        frozen = {name:stored[name].copy() for name in stored.files}
    band,frame,_ = selected_band()
    law = ImprovedPhiLaw(3,band)
    fields = np.tile(frame,(27,1,1,1))
    links = np.tile(np.eye(4,dtype=complex),(27,3,1,1))
    profile = np.exp(-np.sum((law.sites-1)**2,axis=1)/.65**2)
    profile /= np.linalg.norm(profile)
    root_lemma = json.loads(args.root_lemma.read_bytes())
    original = AmbientWeakState(law,fields,links,law.index[(0,1,1)],profile,root_lemma)
    source19 = source19_native_profile(law,args.descriptor)
    soft = json.loads(args.SOFT_profile.read_bytes())
    if (soft['momentum_centre'] != [2e-7,0.,0.] or soft['width'] != 1e-7
            or not np.array_equal(soft['sites'],law.sites) or soft['origin'] != 1.5):
        raise ValueError('Exact unchanged B SOFT spatial restriction required')
    soft_eta = np.asarray(soft['coordinate_restriction']['real'])+1j*np.asarray(soft['coordinate_restriction']['imag'])
    guard = json.loads(args.noether_guard.read_bytes())
    model = ExtendedAmbientWeakState(original,frozen['neutral_state'],
                                    {'source19':source19['profile'],'SOFT':soft_eta},np.asarray(guard['velocity']))
    before = loaded_source_epoch()
    inputs = {name:sha256(getattr(args,name)) for name in ('root_lemma','frozen_state','preparation_receipt','descriptor','SOFT_profile','noether_guard')}
    root = Path(__file__).resolve().parent
    declared = {path.name:sha256(path) for path in root.glob('*.py')}
    start_receipt = dict(loaded_sources_before=before,predeclared_source_sha256=declared,
                         input_sha256=inputs,argv=sys.argv,cwd=str(Path.cwd()),
                         uninterrupted_disk_epoch_claimed=False)
    args.manifest.with_suffix('.start.json').write_text(json.dumps(start_receipt,sort_keys=True,indent=2))
    loaded_archive = args.output_npz.with_suffix('.loaded-sources.zip')
    with zipfile.ZipFile(loaded_archive,'w',zipfile.ZIP_DEFLATED) as archive:
        for name,digest in declared.items():
            body = (root/name).read_bytes()
            if hashlib.sha256(body).hexdigest() != digest:
                raise AssertionError('Source changed during initial snapshot: '+name)
            archive.writestr(name,body)
    arrays,science = compile_extended(model,frozen,args.em_nodes)
    science['SOURCE_norms'] = export_observables(model,arrays)
    science['creator_graphs'] = export_creators(model,arrays)
    snapshot = seed_snapshot(model,arrays)
    preparation_receipt = json.loads(args.preparation_receipt.read_bytes())
    if preparation_receipt['output_npz_sha256'] != FROZEN_STATE_SHA256:
        raise ValueError('Original preparation receipt does not bind frozen NPZ')
    raw_fn_norm = preparation_receipt['actual_neutral_norm_squared']
    raw_fn_coefficients = frozen['neutral_state']*math.sqrt(raw_fn_norm)
    science['original_A17_RAW_filtered_neutral_norm_squared'] = float(np.vdot(raw_fn_coefficients,frozen['neutral_Gram']@raw_fn_coefficients).real)
    science['original_A17_Gaussian_SOURCE_after_neutral_before_SOURCE_norm_squared'] = float(np.vdot(frozen['neutral_state'],frozen['charged_Gram']@frozen['neutral_state']).real)
    science['original_A17_SOURCE_norm_unit_error'] = abs(science['original_A17_Gaussian_SOURCE_after_neutral_before_SOURCE_norm_squared']-.8718600287632787)
    if science['original_A17_SOURCE_norm_unit_error'] > 1e-12:
        raise AssertionError('Original SOURCE normalization unit changed')
    after = loaded_source_epoch()
    metadata = dict(plan=PLAN,revision=REVISION,source=Path(__file__).name,
        source_sha256=sha256(Path(__file__)),input_sha256=inputs,
        seed_snapshot=snapshot,science=portable(science),
        archive_scope='predeclared project Python files; actual imported subset is explicit in source_epoch',
        argv=sys.argv,cwd=str(Path.cwd()),python=sys.version,wall_seconds=time.perf_counter()-started,
        preparation='EXACT original frozen f(H14_MC)tau coefficients, vacuum embedded; NO refilter/new normalization',
        evolution='same original law with exact ALL52 base Gaussian/Weyl and explicit generated H quadrature debt',
        basis_order=dict(neutral='bank/normalpoly/hard15 then even2/odd2',
            charged='profile/word/bank/normalpoly/hard15 then even16/commonodd16/relativeodd16'),
        raw_source_coefficient_map='single matrix nC x nN; PLUS beta, MINUS beta conjugate; global2 once',
        source19_two_Y_maps_identical=True,SOURCE19_fold_origin=[1,1,1],Noether_guard_origin=[1.5,1.5,1.5],
        SOFT_periodic_N3_projection_exact_zero=True,native_c=.125,native_hard_k=2*math.pi/3,
        legal_hard_k='2pi/5 plus fixed small split: distinct mode IDs, not calibrated native projection',
        photon_projector_coordinates_scope='coefficient columns for SOURCE-created hard-Fock subspaces; generated tails not falsely photon-number closed',
        debts=['native finite Phi/theta quadrature and stochastic full-cometric trace',
               'native generated-H EM quadrature and finite difference function jets',
               'infinite Weyl/Fock dynamical tails and full PH/multipair outside',
               'full auxiliary Q/all8/Root beta/Gauss-kernel evolution',
               'legal-N modes/SOFT restriction missing spatial mass/physical closed rotation/Born comparison'],
        formal_completion_not_claimed=True)
    manifest = publish_scientific_cache(args.output_npz,args.manifest,arrays,metadata,
                                       before,after,declared,loaded_archive)
    model.cache = dict(arrays,manifest=manifest)
    # Actual consumer-bound cache load is part of THIS run, never another SCI.
    recovered,data = load_shared_native_cache(args.output_npz,args.manifest)
    if not np.array_equal(recovered.prepared14,frozen['neutral_state']):
        raise AssertionError('Cache reconstruction changed frozen preparation')
    summary = dict(science=portable(science),npz_sha256=manifest['npz_sha256'],
                   manifest_sha256=sha256(args.manifest),loaded_sources_unchanged=True,
                   loaded_source_count=len(before),cache_reconstruction_without_eigensolve=True,
                   wall_seconds=time.perf_counter()-started)
    print(json.dumps(summary,sort_keys=True))


if __name__ == '__main__':
    main()
