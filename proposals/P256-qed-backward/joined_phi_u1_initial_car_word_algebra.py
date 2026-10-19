"""Ordered INITIAL Slater-CAR moments and Gaussian momentum polynomials.

No interacting-time Wick, bosonic particle-hole substitution, or normalization.
Permutation signs include the actual occupied/complement orientation. Functions
accept either numpy matrices or the existing outward Ball matrix arithmetic.
"""
from __future__ import annotations
import argparse
from fractions import Fraction as F
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
import time
from typing import Any
import numpy as np
import joined_phi_u1_current_signal_interval_certificate as ar

ROOT = Path(__file__).resolve().parent
PROOF = {
    'ordered': 'For the alternating ordered string cdag(a1)c(b1)...cdag(an)c(bn), Wick contraction matrix has P(bi,aj) when i>=j and -Q(bi,aj) when i<j, Q=I-P. Determinant permutations have sign(-1)^(n-cycle_count+upward_edges). Each permutation cycle contracts to trace(A_i C_ij A_j C_jk ...), with C_ij=P if i>=j elseQ. This reproduces Tr(PAQB) with PLUS for a two-factor connected contraction, not a plus occupied substitute.',
    'copies': 'An ordered word of summed even bilinears across N identical independent auxiliary Slaters is a sum over set partitions of factor labels. Each block keeps its original relative order and lies in one copy. Distinct blocks choose distinct copies: falling factorial(N,block_count), NOT N^n for every Wick block. Active and auxiliary even algebras commute, so their subsequences factor without a fermionic sign. N63 is literal when used by the campaign.',
    'gaussian': 'Initial real129 Gaussian DENSITY variance2, Fourier momentum variance1/8. For a right-normal polynomial after e^(iqTheta), Fourier overlap shifts its Gaussian momentum mean to -q/2 and multiplies by exp(-q²). Thus linear momentum means are -q_r.q/2 and covariances q_r.q_s/8. Exact rational Gauss Gram can supply these inputs; Wick recurrence is only this initial Gaussian integral, never an evolved state claim.',
    'scope': 'Caller supplies INITIAL occupied projector(s), exact right-normal operator order, actual Ref3 constant and raw four-channel weights. These routines do not infer a future Slater, replace originalSOURCE, or price a finite-time remainder. Direct finiteCAR controls are algebra controls, not all129 evolution.'}


def matrix_trace(matrix: Any) -> Any:
    return ar.contract('ii->',matrix) if isinstance(matrix,ar.Ball) else complex(np.trace(matrix))


def ordered_car_moment(projector: Any, matrices: list[Any]) -> Any:
    """Expectation of the ordered product of even dGamma matrices at a Slater."""
    count = len(matrices)
    if not count:
        return 1
    size = projector.v.shape[0] if isinstance(projector,ar.Ball) else projector.shape[0]
    identity = ar.Ball(np.eye(size)) if isinstance(projector,ar.Ball) else np.eye(size)
    complement = identity-projector
    total: Any = 0
    for permutation in itertools.permutations(range(count)):
        seen: set[int] = set()
        value: Any = 1
        cycles = 0
        for start in range(count):
            if start in seen:
                continue
            cycles += 1
            index = start
            product = matrices[index]
            while True:
                seen.add(index)
                target = permutation[index]
                product = product@(projector if index>=target else complement)
                if target == start:
                    break
                product = product@matrices[target]
                index = target
            value = value*matrix_trace(product)
        upward = sum(i<j for i,j in enumerate(permutation))
        total = total+((-1)**(count-cycles+upward))*value
    return total


def set_partitions(labels: tuple[int,...]):
    """Every unlabeled partition once; block order follows its smallest label."""
    if not labels:
        yield ()
        return
    first, rest = labels[0], labels[1:]
    for partition in set_partitions(rest):
        yield ((first,),)+partition
        for index in range(len(partition)):
            yield partition[:index]+((first,)+partition[index],)+partition[index+1:]


def collective_car_moment(projector: Any, matrices: list[Any], copies: int) -> Any:
    if copies<0:
        raise ValueError('Negative physical species count')
    total: Any = 0
    for partition in set_partitions(tuple(range(len(matrices)))):
        if len(partition)>copies:
            continue
        weight = math.prod(copies-k for k in range(len(partition)))
        value: Any = weight
        for block in partition:
            value = value*ordered_car_moment(projector,[matrices[i] for i in sorted(block)])
        total = total+value
    return total


def role_word_moment(active: Any, auxiliary: Any, factors: list[tuple[str,Any]], copies: int=63) -> Any:
    if any(role not in ('active','auxiliary') for role,_ in factors):
        raise ValueError('Unknown physical charged role')
    a = [matrix for role,matrix in factors if role=='active']
    b = [matrix for role,matrix in factors if role=='auxiliary']
    return ordered_car_moment(active,a)*collective_car_moment(auxiliary,b,copies)


def gaussian_linear_moment(means: list[Any], covariance: list[list[Any]]) -> Any:
    """Exact Isserlis polynomial with nonzero means; no field-space truncation."""
    if len(covariance)!=len(means) or any(len(row)!=len(means) for row in covariance):
        raise ValueError('Gaussian linear statistic shape mismatch')
    def moment(indices: tuple[int,...]) -> Any:
        if not indices:
            return 1
        first, rest = indices[0], indices[1:]
        total = means[first]*moment(rest)
        for position,index in enumerate(rest):
            total = total+covariance[first][index]*moment(rest[:position]+rest[position+1:])
        return total
    return moment(tuple(range(len(means))))


def finite_car(matrix: np.ndarray) -> np.ndarray:
    """Literal finite CAR boundary used only by direct controls."""
    orbitals = matrix.shape[0]
    out = np.zeros((2**orbitals,2**orbitals),complex)
    for state in range(2**orbitals):
        for col in range(orbitals):
            if not state>>col&1:
                continue
            removed = state^(1<<col)
            ann = (-1)**((state&((1<<col)-1)).bit_count())
            for row in range(orbitals):
                if removed>>row&1:
                    continue
                created = removed|(1<<row)
                out[created,state] += ann*(-1)**((removed&((1<<row)-1)).bit_count())*matrix[row,col]
    return out


def run(output: Path) -> dict[str,Any]:
    start = time.monotonic()
    w = np.array([[1,1],[1,-1],[1,1],[1,-1]],complex)/2
    projector = w@w.conj().T
    state = np.zeros(16,complex)
    for i,j in itertools.combinations(range(4),2):
        state[(1<<i)|(1<<j)] = w[i,0]*w[j,1]-w[i,1]*w[j,0]
    matrices = [np.array([[1,2,1j,0],[2,-1,1,1j],[-1j,1,2,-2],[0,-1j,-2,0]],complex),
        np.array([[0,1j,2,1],[-1j,2,-1,0],[2,-1,-2,1j],[1,0,-1j,1]],complex),
        np.diag([2,-1,3,0]).astype(complex),
        np.array([[1,0,1,1j],[0,-2,-1j,2],[1,1j,0,1],[-1j,2,1,3]],complex)]
    direct = [finite_car(a) for a in matrices]
    observations = []
    for count in range(1,5):
        vector = state.copy()
        for matrix in reversed(direct[:count]):
            vector = matrix@vector
        expected = complex(np.vdot(state,vector))
        calculated = complex(ordered_car_moment(projector,matrices[:count]))
        if expected != calculated:
            raise ArithmeticError('Exact finiteCAR ordered moment discrepancy')
        observations.append({'order':count,'real':expected.real,'imaginary':expected.imag,'exact_gaussian_integer_defect':0})
    wrong_pair = np.trace(projector@matrices[0])*np.trace(projector@matrices[1])+np.trace(projector@matrices[0]@projector@matrices[1])
    pair = ordered_car_moment(projector,matrices[:2])
    if abs(wrong_pair-pair)<1:
        raise ArithmeticError('Occupied-versus-complement adversary blind')
    single = np.array([[.5,.5],[.5,.5]],complex)
    a = np.array([[1,2j],[-2j,-1]],complex)
    full = np.kron(np.eye(2),a)
    # Two-copy Slater coefficients are exactly1/2, avoiding sqrt roundoff.
    psi = np.zeros(16,complex)
    for i in (0,1):
        for j in (2,3):
            psi[(1<<i)|(1<<j)] = .5
    car = finite_car(full)
    copy_rows = []
    for count in range(1,5):
        vector = psi.copy()
        for _ in range(count):
            vector = car@vector
        measured = complex(np.vdot(psi,vector))
        computed = complex(collective_car_moment(single,[a]*count,2))
        if measured != computed:
            raise ArithmeticError('Identical-copy falling-factorial mismatch')
        copy_rows.append({'order':count,'real':measured.real,'imaginary':measured.imag,'exact_defect':0})
    gaussian = gaussian_linear_moment([F(-1,4)]*4,[[F(1,32)]*4 for _ in range(4)])
    if gaussian != F(19,1024):
        raise ArithmeticError('Gaussian mean/covariance momentum recurrence')
    result = {'status':'INITIAL_ORDERED_CAR_AND_GAUSSIAN_ALGEBRA_DIRECT_CONTROL_COMPLETE',
        'proof':PROOF,'ordered_controls':observations,'wrong_occupied_pairing_defect':float(abs(wrong_pair-pair)),
        'collective_two_copy_controls':copy_rows,'gaussian_fourth_exact':str(gaussian),
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'runtime':{'executable':sys.executable,'python':sys.version,'numpy':np.__version__},
        'elapsed_seconds':time.monotonic()-start}
    output.write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    print(json.dumps({'status':run(args.output)['status']}))
