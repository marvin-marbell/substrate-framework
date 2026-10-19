"""Complete physical64 neutral-return Gram from preserved correlated slot densities.

NEW operator consumer only. Source/copy counts unchanged. Binary64 source Gram
and operator contractions are FLOAT, not outward spectral/ground certificates.
Canonical-copy equality patterns reduce the sum AFTER signed CAR balance.
"""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
from functools import lru_cache
import hashlib
import itertools
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import neutral_return_image_control_20261011 as native

ROOT = Path(__file__).resolve().parent
PINS = {
    'connected_root_exterior_actual1_20261011_moments.npz': '4c6c6e0da22e0d9d405c37935332bb9548fbf2df5f429ec66b50aa671f8af67d',
    'neutral_return_image_control_20261011.py': '02c7e7334d8ce63fc2e25416e3f3bbdfa55718135ad46efd9f6473b50f6dc4e1',
    'neutral_return_image_factorized_20261011.py': 'c906b72557454adbe107478ccd367ecdb49599ddb410b13d83e03c7ee5170973',
    'neutral_return_image_actual4_20261011_receipt.json': '3b1454e50d6737019e7a542d43d9160e60b22d579d265d7c14521aadc2e1cf23',
    'local_so5_rep_review_recovered_corrected_freeframe_20261011_proof.json': '102966f32c49f40d79d55343925a630e30a9cf7e4a8074cf8628142ba9c5b6e8',
    'local_so5_ground_upper_complete_state_consumer_20261011_receipt.json': '2f019dd49873a5e13f130c65ef6739b9af78a16f85138dc6c6ab6e32e65c7a12',
    'neutral_return_image_API_20261011.json': '861595a1c0432fc3e965fe35472a0386217fd57b75bba7ab868918e57e4197e1',
    'neutral_return_lower_boundary_actual2_20261011_receipt.json': '7c0ae50f0790e502161e5e925e5a707b5130a15ef3648f8c9cb40ff957ac77bb',
    'root_metric_review_proof_20261011.json': '324500b502a1c0d6de3ba1108212c500f2efedd6a74bcad2c9993740723661a2',
}


def sha(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def sym_basis() -> np.ndarray:
    columns = []
    for i in range(5):
        for j in range(i, 5):
            column = np.zeros(25)
            column[5*i+j] = 1 if i == j else 1/math.sqrt(2)
            if i != j:
                column[5*j+i] = 1/math.sqrt(2)
            columns.append(column)
    return np.column_stack(columns)


def source_densities() -> tuple[dict[tuple[int, ...], np.ndarray], dict[str, float]]:
    U = sym_basis()
    with np.load(ROOT/'connected_root_exterior_actual1_20261011_moments.npz') as data:
        AA = U@data['active_pair'][:15, :15].T@U.T
        UU = U@data['auxiliary_pair'][:15, :15].T@U.T
        AU = data['mixed_pair'][:25, :25].T.copy()
    def partial(rho: np.ndarray, left: bool) -> np.ndarray:
        tensor = rho.reshape(5, 5, 5, 5)
        return np.einsum('ijkj->ik', tensor) if left else np.einsum('ijil->jl', tensor)
    A, V = partial(AA, True), partial(UU, True)
    checks = {
        'trace_AA': float(np.trace(AA)), 'trace_UU': float(np.trace(UU)),
        'trace_AU': float(np.trace(AU)),
        'mixed_active_partial_defect': float(np.linalg.norm(partial(AU, True)-A)),
        'mixed_aux_partial_defect': float(np.linalg.norm(partial(AU, False)-V)),
        'AA_partial_swap_defect': float(np.linalg.norm(partial(AA, False)-A)),
        'UU_partial_swap_defect': float(np.linalg.norm(partial(UU, False)-V)),
        'AA_product_wrongcase_defect': float(np.linalg.norm(AA-np.kron(A, A))),
        'UU_product_wrongcase_defect': float(np.linalg.norm(UU-np.kron(V, V))),
    }
    assert max(abs(checks[k]-1) for k in ('trace_AA','trace_UU','trace_AU')) < 2e-9
    assert max(checks[k] for k in ('mixed_active_partial_defect','mixed_aux_partial_defect','AA_partial_swap_defect','UU_partial_swap_defect')) < 2e-9
    assert checks['AA_product_wrongcase_defect'] > 1e-5
    assert checks['UU_product_wrongcase_defect'] > 1e-5
    UA = AU.reshape(5, 5, 5, 5).transpose(1, 0, 3, 2).reshape(25, 25)
    return {(0,):A, (1,):V, (0,0):AA, (1,1):UU, (0,1):AU, (1,0):UA}, checks


# Feature: identity(None) or (slot,p,slot,q), i.e. c_slot,p^dag c_slot,q.
# Terms: (coefficient, featureA, featureB).
def terms(M: np.ndarray, f: int, g: int | None, spins: int = 2) -> list[tuple[complex, Any, Any]]:
    n = M.shape[0]
    assert n == 2*spins
    def mode(copy: int, i: int, side: int) -> tuple[int,int]:
        return spins*copy+i//2, 2*side+i%2
    out = []
    if g is None:
        ca, cb = M@M.conj().T, M.conj().T@M
        for i,k in itertools.product(range(n), repeat=2):
            if ca[i,k] != 0: out.append((complex(ca[i,k]), (*mode(f,i,1), *mode(f,k,1)), None))
            if cb[i,k] != 0: out.append((complex(cb[i,k]), None, (*mode(f,i,0), *mode(f,k,0))))
        g = f
    for i,j,k,l in itertools.product(range(n), repeat=4):
        c = -2*M[i,j]*M[k,l].conjugate()
        if c != 0:
            out.append((complex(c), (*mode(f,i,1), *mode(g,k,1)), (*mode(g,l,0), *mode(f,j,0))))
    return out


class Kernel:
    def __init__(self, ann: list[np.ndarray], W: np.ndarray, roles: tuple[int,...], density: dict[tuple[int,...],np.ndarray]):
        self.ann, self.W, self.roles, self.density = ann, W, roles, density

    @lru_cache(maxsize=None)
    def moment(self, left: Any, right: Any) -> complex:
        word = []
        if left is not None:
            a,p,b,q = left; word.extend([(b,q,True),(a,p,False)])
        if right is not None:
            a,p,b,q = right; word.extend([(a,p,True),(b,q,False)])
        if not word: return 1+0j
        balance = Counter()
        for slot,_,creation in word: balance[slot] += 1 if creation else -1
        if any(balance.values()): return 0j
        slots = sorted(balance)
        assert len(slots) <= 2
        inv = sum(word[i][0] > word[j][0] for i in range(len(word)) for j in range(i+1,len(word)))
        projected = []
        for slot in slots:
            O = np.eye(16, dtype=complex)
            for owner,p,creation in word:
                if owner == slot: O = O@(self.ann[p].conj().T if creation else self.ann[p])
            projected.append(self.W.conj().T@O@self.W)
        operator = projected[0] if len(slots)==1 else np.kron(*projected)
        rho = self.density[tuple(self.roles[i] for i in slots)]
        return complex((-1)**inv*np.trace(operator@rho))

    def gram(self, left: list[tuple[complex,Any,Any]], right: list[tuple[complex,Any,Any]]) -> complex:
        return sum(c.conjugate()*d*self.moment(A,C)*self.moment(B,D) for c,A,B in left for d,C,D in right)

    def mean(self, values: list[tuple[complex,Any,Any]]) -> complex:
        return sum(c*self.moment(None,A)*self.moment(None,B) for c,A,B in values)


def sparse_word(v: dict[int,complex], word: list[tuple[int,bool]]) -> dict[int,complex]:
    for p,creation in reversed(word):
        out: dict[int,complex] = {}
        for mask,c in v.items():
            if bool(mask&(1<<p)) != creation:
                target = mask^(1<<p)
                out[target] = out.get(target,0)+(-1)**((mask&((1<<p)-1)).bit_count())*c
        v = out
    return v


def free_commutator(values: list[tuple[complex,Any,Any]], roles: tuple[int,...],
                    profile_h: list[np.ndarray]) -> list[tuple[complex,Any,Any]]:
    out = []
    for c,A,B in values:
        for side,feature in enumerate((A,B)):
            if feature is None:
                continue
            si,p,sk,q = feature
            hi,hk = profile_h[roles[si]],profile_h[roles[sk]]
            for r in range(4):
                for coefficient,replacement in (
                    (hi[r,p],(si,r,sk,q)),(-hk[q,r],(si,p,sk,r))):
                    if coefficient != 0:
                        out.append((c*coefficient,replacement if side==0 else A,B if side==0 else replacement))
    return out


def reduction_control(ann: list[np.ndarray], W: np.ndarray) -> dict[str,float]:
    # Independent full16-mode CAR, arbitrary correlated two-slot cell source.
    rng = np.random.default_rng(222110)
    z = rng.normal(size=25)+1j*rng.normal(size=25); z/=np.linalg.norm(z)
    R2 = np.outer(z,z.conj())
    R1a = np.einsum('ijkj->ik',R2.reshape(5,5,5,5))
    R1b = np.einsum('ijil->jl',R2.reshape(5,5,5,5))
    swap = R2.reshape(5,5,5,5).transpose(1,0,3,2).reshape(25,25)
    kernel = Kernel(ann,W,(0,1),{(0,):R1a,(1,):R1b,(0,1):R2,(1,0):swap})
    M = np.array([[2,1+1j],[3-1j,-1]],complex)
    S0,S1 = terms(M,0,None,1),terms(M,1,None,1)
    D01,D10 = terms(M,0,1,1),terms(M,1,0,1)
    reduced = kernel.gram(S0,S0)+kernel.gram(S1,S1)+2*kernel.gram(S0,S1).real+kernel.gram(D01,D01)+kernel.gram(D10,D10)
    cell = np.kron(W,W)@z
    # Numpy high-bit ordering of tensor slots converted to actual CAR labels.
    cell_dict = {(a+(b<<4)):cell[16*a+b] for a,b in itertools.product(range(16),repeat=2) if cell[16*a+b]!=0}
    source = {a+(b<<8):x*y for a,x in cell_dict.items() for b,y in cell_dict.items()}
    def hop(v: dict[int,complex], dagger: bool) -> dict[int,complex]:
        out: dict[int,complex] = {}
        for f,p,q in itertools.product(range(2),range(2),range(2)):
            am,bm = 4*f+2+p,8+4*f+q
            word = [(bm,True),(am,False)] if dagger else [(am,True),(bm,False)]
            c = M[p,q].conjugate() if dagger else M[p,q]
            for m,w in sparse_word(v,word).items(): out[m]=out.get(m,0)+c*w
        return out
    out = hop(hop(source,False),True)
    for m,w in hop(hop(source,True),False).items():out[m]=out.get(m,0)+w
    direct = sum(abs(w)**2 for w in out.values())
    full = kernel.gram(S0+S1+D01+D10,S0+S1+D01+D10)
    error = abs(direct-reduced)
    assert error < 1e-8*max(1,direct)
    assert abs(full-reduced) < 1e-8*max(1,direct)
    wrong_independent = float((kernel.gram(S0,S0)+kernel.gram(S1,S1)).real)
    assert abs(direct-wrong_independent)>1
    # Independent fullCAR H0 commutator, retaining BOTH shared cell rho squares.
    profiles=[
        np.array([[-3,2,1j,0],[2,3,0,-1j],[-1j,0,-3,-2],[0,1j,-2,3]],complex),
        np.array([[-7,5,2j,0],[5,7,0,-2j],[-2j,0,-7,-5],[0,2j,-5,7]],complex),
    ]
    def C_action(v: dict[int,complex]) -> dict[int,complex]:
        result=hop(hop(v,False),True)
        for m,w in hop(hop(v,True),False).items(): result[m]=result.get(m,0)+w
        return result
    def H_action(v: dict[int,complex]) -> dict[int,complex]:
        result: dict[int,complex]={}
        for cell_index,slot,p,q in itertools.product(range(2),range(2),range(4),range(4)):
            coefficient=profiles[slot][p,q]
            if coefficient==0: continue
            offset=8*cell_index+4*slot
            for mask,w in sparse_word(v,[(offset+p,True),(offset+q,False)]).items():
                result[mask]=result.get(mask,0)+coefficient*w
        for mask,w in v.items():
            rho=[sum((1 if p<2 else -1)/2 for slot,p in itertools.product(range(2),range(4)) if mask&(1<<(8*cell_index+4*slot+p))) for cell_index in range(2)]
            result[mask]=result.get(mask,0)+64*sum(r*r for r in rho)*w
        return result
    direct_D=H_action(out)
    for mask,w in C_action(H_action(source)).items(): direct_D[mask]=direct_D.get(mask,0)-w
    direct_D_norm=sum(abs(w)**2 for w in direct_D.values())
    direct_CD=sum(w.conjugate()*direct_D.get(mask,0) for mask,w in out.items())
    Cterms=S0+S1+D01+D10
    Dterms=free_commutator(Cterms,kernel.roles,profiles)
    Dnorm=kernel.gram(Dterms,Dterms);CD=kernel.gram(Cterms,Dterms)
    assert abs(Dnorm-direct_D_norm)<1e-8*max(1,direct_D_norm)
    assert abs(CD-direct_CD)<1e-8*max(1,abs(direct_CD))
    wrong_half_D=free_commutator(S0+S1,kernel.roles,profiles)
    assert abs(kernel.gram(wrong_half_D,wrong_half_D)-Dnorm)>1
    return {'actual_fullCAR_squared_norm':float(direct),'reduction_absolute_defect':float(error),'unreduced_Gram_defect':float(abs(full-reduced)),'drop_crosscopy_and_coherent_contact_wrongcase_difference':float(abs(direct-wrong_independent)),'fullCAR_H0_commutator_squared_defect':float(abs(Dnorm-direct_D_norm)),'fullCAR_H0_commutator_first_defect':float(abs(CD-direct_CD)),'omitted_crosscopy_commutator_wrongcase_difference':float(abs(kernel.gram(wrong_half_D,wrong_half_D)-Dnorm))}


def run(receipt: Path) -> None:
    started=time.monotonic(); own=sha(Path(__file__))
    pins={n:sha(ROOT/n) for n in PINS}; assert pins==PINS
    ann0,h4,_,W0,_=native.car()
    ann=[np.array(x.evalf(),complex) for x in ann0]; W=np.array(W0.evalf(),complex)
    controls=reduction_control(ann,W)
    density, density_checks=source_densities()
    image=json.loads((ROOT/'neutral_return_image_actual4_20261011_receipt.json').read_text())
    raw=np.asarray(image['source_observation']['actual_axis0_literal1_to2_hopping'])
    M=raw[...,0]+1j*raw[...,1]
    S0,S1=terms(M,0,None),terms(M,1,None)
    D01,D10=terms(M,0,1),terms(M,1,0)
    kernels={
        'active_self':Kernel(ann,W,(0,0),density),
        'auxiliary_self':Kernel(ann,W,(1,1),density),
        'active_auxiliary':Kernel(ann,W,(0,0,1,1),density),
        'auxiliary_auxiliary_distinct':Kernel(ann,W,(1,1,1,1),density),
    }
    A,U,AU,UU=(kernels[k] for k in ('active_self','auxiliary_self','active_auxiliary','auxiliary_auxiliary_distinct'))
    components={
        'same_active':A.gram(S0,S0), 'same_auxiliary':U.gram(S0,S0),
        'self_active_auxiliary_cross':AU.gram(S0,S1),
        'self_distinct_auxiliary_cross':UU.gram(S0,S1),
        'different_active_auxiliary':AU.gram(D01,D01),
        'different_auxiliary_active':AU.gram(D10,D10),
        'different_auxiliary_auxiliary':UU.gram(D01,D01),
    }
    mean=A.mean(S0)+63*U.mean(S0)
    total=(components['same_active']+63*components['same_auxiliary']
        +126*components['self_active_auxiliary_cross'].real
        +63*62*components['self_distinct_auxiliary_cross'].real
        +63*components['different_active_auxiliary']+63*components['different_auxiliary_active']
        +63*62*components['different_auxiliary_auxiliary'])
    projected=total.real-abs(mean)**2
    active_reference=image['source_observation']['unprojected_neutral_squared_norm_float']
    assert abs(components['same_active'].real-active_reference)<1e-8*active_reference
    assert abs(total.imag)<1e-8*total.real and projected>0
    drop_diff=(components['same_active']+63*components['same_auxiliary']+126*components['self_active_auxiliary_cross'].real+63*62*components['self_distinct_auxiliary_cross'].real).real
    assert total.real-drop_diff>1
    profiles=json.loads((ROOT/'local_so5_rep_review_recovered_corrected_freeframe_20261011_proof.json').read_text())['profiles']
    hs=[sum((float(Fraction(p[k]))*np.array(H,complex) for k,H in zip(('m_exact','c0_exact','d_exact'),h4)),np.zeros((4,4),complex)) for p in profiles]
    source_patterns=[
        (A,S0,S0,1),(U,S0,S0,63),
        (AU,S0,S1,63),(AU,S1,S0,63),
        (UU,S0,S1,3906),
        (AU,D01,D01,63),(AU,D10,D10,63),(UU,D01,D01,3906),
    ]
    # On an EXACT ground, (H0-2E0) Cg=[hfree,C]g because [rho_all,C]=0.
    # Stored source: retain eta AND the Rayleigh-to-ground reference shift.
    # Formation is already inside eta; never add it twice or use observed eta.
    relative_first=0j;relative_second=0j
    for kernel,left,right,count in source_patterns:
        DL=free_commutator(left,kernel.roles,hs)
        DR=free_commutator(right,kernel.roles,hs)
        relative_first+=count*kernel.gram(left,DR)
        relative_second+=count*kernel.gram(DL,DR)
    assert abs(relative_first.imag)<1e-7*max(1,abs(relative_first.real))
    assert abs(relative_second.imag)<1e-7*max(1,relative_second.real) and relative_second.real>0
    c0,d0=[Fraction(profiles[0][k]) for k in ('c0_exact','d_exact')]
    C_upper=131072*(c0*c0+d0*d0)
    source_caps=json.loads((ROOT/'local_so5_ground_upper_complete_state_consumer_20261011_receipt.json').read_text())
    accepted_eta=Fraction(source_caps['normalized_exact_source_residual_upper_exact'])
    reference_shift=Fraction(source_caps['Rayleigh_interval_exact'][1])-Fraction(source_caps['ground_energy_interval_exact'][0])
    assert accepted_eta==Fraction(1,65536) and reference_shift>0
    residual_source_action_cap=2*(accepted_eta+reference_shift)*C_upper
    result={
        'kind':'COMPLETE_ALL64_PHYSICAL_NEUTRAL_RETURN_FLOAT_SOURCE_GRAM_NOT_GROUND_OR_INTERVAL',
        'own_sha256':own,'input_sha256':pins,'elapsed_seconds':time.monotonic()-started,
        'controls':controls,'density_positive_controls':density_checks,
        'component_real':{k:float(v.real) for k,v in components.items()},
        'component_imaginary':{k:float(v.imag) for k,v in components.items()},
        'multiplicities':{'same_active':1,'same_auxiliary':63,'self_active_auxiliary_cross':126,'self_distinct_auxiliary_cross':3906,'different_active_auxiliary':63,'different_auxiliary_active':63,'different_auxiliary_auxiliary':3906},
        'all64_unprojected_neutral_squared_norm_FLOAT':float(total.real),
        'all64_neutral_mean_FLOAT':[float(mean.real),float(mean.imag)],
        'source_product_projector_removed_norm_FLOAT_NOT_exact_P0':float(projected),
        'drop_crosscopy_squared_error_FLOAT':float(total.real-drop_diff),
        'active_diagnostic_positive_control_defect':float(abs(components['same_active'].real-active_reference)),
        'all64_commutator_first_FLOAT_conditional_ground_equation':float(relative_first.real),
        'all64_commutator_second_FLOAT_conditional_ground_equation':float(relative_second.real),
        'complete_H0_ground_equation':'[H0_A+H0_B,C]=[hfree_A+hfree_B,C] since each initial return bilinear commutes rho_all. No HC deletion from a different operator; commutator-generated features retain full Dirac profiles. Stored source is not exact g: C(H0-2E0)Omega retained via accepted cap below. These commutator Gram numbers are FLOAT, not complete source-H variance or interval ground moments.',
        'accepted_eta_exact':'1/65536',
        'Rayleigh_to_ground_reference_shift_upper_exact':str(reference_shift),
        'all64_C_norm_upper_exact':str(C_upper),
        'missing_source_residual_action_norm_upper_exact':str(residual_source_action_cap),
        'source_to_true_ground_Q_return_vector_norm_error_upper_exact':str(C_upper/30000),
        'CAR_equality_pattern_proof':'Self terms for distinct copies overlap only after per-physical-slot number balance. Ordered distinct-copy exchange outputs have opposite canonical-copy number differences, mutually orthogonal and orthogonal to self. Each kernel retains ALL same-spin/contact/coherent histories before norm and uses actual normalized correlated one/two-slot density, not independent copies. Active/aux role counts1+63 counted after CAR.',
        'unpaid':['FLOAT extraction and moment interval enclosure','exact whole spectral P0 perturbation applied to numerical Gram','neutral Schur/resolvent response','fullcanonical cell ground','physicalPhi/lattice lift','N-uniform normalized linked stability','phase/stiffness/IR atom/sharp photon'],
        'source_unchanged':sha(Path(__file__))==own,'inputs_unchanged':all(sha(ROOT/n)==h for n,h in pins.items()),
        'complete_H_variance_claim':False,'ground_or_phase_claim':False,
    }
    assert result['source_unchanged'] and result['inputs_unchanged']
    assert not receipt.exists();receipt.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--receipt',type=Path,required=True)
    run(parser.parse_args().receipt)
