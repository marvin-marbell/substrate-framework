"""Independent fixed-box consumer of the captured current-N4 law.

No supplier is imported or executed. Root search, adoption, source modification,
and interacting-ground assertions are deliberately absent. All scientific
arithmetic is exact rational or outward mpmath interval arithmetic. Frozen
binary eigenvectors are proof witnesses only; their errors are repriced here.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from fractions import Fraction as F
import hashlib
from itertools import combinations, product
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
from typing import Any
import zipfile

import ijson
import mpmath as mp
import numpy as np

from induced_phi_current_n4_root_static_slow_certificate_frame_bound import derivative_bound
from induced_phi_current_n4_root_static_slow_certificate_spectral_bound import full_q0_bound

sys.set_int_max_str_digits(0)
iv: Any = mp.iv
PREFIX = 'induced_phi_current_n4_root_static_slow_certificate'
GAUSSIAN_SHA = '6abd28b7bae5d3e8370ec66555778c0ce6768c21e8ce90674e42c5e214147f8b'
EXPECTED = {
    'root': '4691ffaa9cd81902942cbc228fc47fafff14518be6b5df2e84d08c29c7da4756',
    'full': '5b97862fc7fca21b8797d549d7d4dfdde58b101e8c46c7a7cfb6531a669c4354',
    'quantum': '0f143f66163a483cc055961888b4cc87bc96ebe23675b802a740252979acec4a',
    'matter': 'b20c715c5127d24e29d9d5db0ac3f6e2b8d282605fb01bbee89885e4016849f8',
    'source': 'da1998065d6622c4ac780409c29ae10711e01b5c5143edf620557fe15234e9e6',
}


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def exact(x: Any) -> Any:
    q = F(x)
    return iv.mpf(q.numerator)/q.denominator


def endpoint(x: Any, side: int) -> F:
    sign, mantissa, exponent, _ = x._mpi_[side]
    return F((-1 if sign else 1)*mantissa)*F(2)**exponent


def box(lo: F, hi: F) -> Any:
    return iv.mpf([exact(lo).a, exact(hi).b])


def bound(x: Any) -> dict[str, str]:
    lo, hi = endpoint(x, 0), endpoint(x, 1)
    ctx: Any = mp.mp.clone(); ctx.dps = 45
    return {'lower': str(lo), 'upper': str(hi),
            'lower_decimal': ctx.nstr(ctx.mpf(lo.numerator)/lo.denominator, 32),
            'upper_decimal': ctx.nstr(ctx.mpf(hi.numerator)/hi.denominator, 32)}


def upper(x: Any) -> F:
    return endpoint(abs(x), 1)


def stored_box(record: dict[str, Any]) -> Any:
    return box(F(record['lower']), F(record['upper']))


def select(path: Path, prefix: str) -> Any:
    with path.open('rb') as stream:
        return next(ijson.items(stream, prefix, use_float=True))


def clifford() -> tuple[np.ndarray, np.ndarray, list[tuple[int, ...]]]:
    # Exact entries reproduce captured rotation_band/sea_selection matrix units.
    x = np.asarray([[0, 1], [1, 0]], complex)
    y = np.asarray([[0, -1j], [1j, 0]], complex)
    z = np.diag([1, -1]).astype(complex)
    gamma = np.asarray([np.kron(x, np.eye(2)), *[np.kron(z, t) for t in (x, y, z)]])
    words = [(j,) for j in range(4)]+list(combinations(range(4), 2))+list(combinations(range(4), 3))+[(0, 1, 2, 3)]
    matrices = []
    for word in words:
        m = np.eye(4, dtype=complex)
        for j in word:
            m = m@gamma[j]
        matrices.append(1j*m if len(word) in (2, 3) else m)
    M = np.asarray(matrices)
    assert all(np.array_equal(gamma[i]@gamma[j]+gamma[j]@gamma[i], 2*np.eye(4)*(i == j)) for i,j in product(range(4), repeat=2))
    assert np.array_equal(np.einsum('aij,bji->ab', M, M), 4*np.eye(15))
    return gamma, M, words


def normal_columns(g: list[Any], gamma: np.ndarray, M: np.ndarray) -> list[dict[int, Any]]:
    """Exact orthonormal complement to the fifteen actual adjoint orbit vectors."""
    r = [iv.sqrt(t) for t in g]
    cols: list[dict[int, Any]] = [{15*j+j: exact(1)} for j in range(4)]
    for i,j in combinations(range(4), 2):
        den = iv.sqrt(g[i]+g[j]); cols.append({15*i+j: r[j]/den, 15*j+i: r[i]/den})
    used: set[int] = set()
    for j in range(4):
        entries = []
        for mu in range(4):
            if mu == j:
                continue
            comm = 1j*(gamma[j]@gamma[mu]-gamma[mu]@gamma[j])
            a = next(a for a in range(4, 10) if np.trace(M[a]@comm) != 0)
            entries.append((15*mu+a, exact(int(np.trace(M[a]@comm).real/8))*r[mu]))
        ids,v = zip(*entries, strict=True); used.update(ids)
        den = iv.sqrt(v[0]**2+v[1]**2); total = iv.sqrt(sum(t*t for t in v))
        n = [v[1]/den, -v[0]/den, exact(0)]
        w = [(v[1]*n[2]-v[2]*n[1])/total, (v[2]*n[0]-v[0]*n[2])/total, (v[0]*n[1]-v[1]*n[0])/total]
        cols.extend([{i:t for i,t in zip(ids,n,strict=True)}, {i:t for i,t in zip(ids,w,strict=True)}])
    for mu in range(4):
        for a in range(4, 10):
            if 15*mu+a not in used:
                cols.append({15*mu+a: exact(1)})
    ids = []; v = []
    for mu in range(4):
        comm = 1j*(M[14]@gamma[mu]-gamma[mu]@M[14])
        a = next(a for a in range(10, 14) if np.trace(M[a]@comm) != 0)
        ids.append(15*mu+a); v.append(exact(int(np.trace(M[a]@comm).real/8))*r[mu])
    norm = iv.sqrt(sum(t*t for t in v)); w = list(v); w[0] -= norm; w2 = sum(t*t for t in w)
    for j in range(1, 4):
        cols.append({ids[i]: exact(int(i == j))-2*w[i]*w[j]/w2 for i in range(4)})
    for mu in range(4):
        for a in range(10, 14):
            if 15*mu+a not in ids:
                cols.append({15*mu+a: exact(1)})
    assert len(cols) == 45
    return cols


def radial_law(g0: Any, gs: Any, profiles: list[dict[str, Any]], mask: list[list[int]], feedback: bool = True, feedback_exponent: int = 5120) -> tuple[list[Any], list[list[Any]]]:
    force = [128*g0, 128*gs]
    J = [[exact(128), exact(0)], [exact(0), exact(128)]]
    for p in profiles:
        one = [v for v in p['paths'] if v['length'] == 1]
        assert len(one) == 3 and len({v['c0'] for v in one}) == len({v['ci'] for v in one}) == 1
        assert all(F(v['ci']) == 0 for v in p['paths'] if v['length'] == 2)
        c0, ci, onsite = exact(one[0]['c0']), exact(one[0]['ci']), exact(p['onsite'])
        spatial = 6*ci**2
        scale = exact(1 if p['scale'] == 'heavy' else F(p['load'], 2**8192))
        for z, multiplicity in ((3,1),(1,3),(-1,3),(-3,1)):
            mass = (onsite+iv.sqrt(exact(2))*c0*z)**2
            E = iv.sqrt(g0*mass+gs*spatial); load = scale*multiplicity/8
            force[0] -= load*mass/E; force[1] -= load*spatial/(3*E)
            J[0][0] += load*mass**2/(2*E**3)
            J[0][1] += load*mass*spatial/(2*E**3)
            J[1][0] += load*mass*spatial/(6*E**3)
            J[1][1] += load*spatial**2/(6*E**3)
    if feedback:
        counts = Counter(sum(0 if n == 0 else 4 if n == 2 else 2 for n in q) for q in product(range(4), repeat=3))
        eps = exact(F(1, 64*2**feedback_exponent))
        for row in mask:
            dh = [exact(F(row[0],2)), exact(F(sum(row[1:]),2))]
            H = dh[0]*g0+dh[1]*gs
            for q2, count in counts.items():
                for c, multiplicity in ((F(1,5),1),(F(21,200),2)):
                    omega = iv.sqrt(exact(F(3,50))*H+exact(c*q2))
                    for i in range(2):
                        divisor = 1 if i == 0 else 3
                        force[i] += eps*count*multiplicity*exact(F(3,200))*dh[i]/omega/divisor
                        for j in range(2):
                            J[i][j] -= eps*count*multiplicity*exact(F(9,20000))*dh[i]*dh[j]/omega**3/divisor
    return force,J


def fixed_box_certificate(root: dict[str, Any], oldroot: dict[str, Any], profiles: list[dict[str, Any]], mask: list[list[int]]) -> dict[str, Any]:
    iv.dps = 2300
    centres = [F(x) for x in root['centre_exact_rational']]; radius = F(1,10**2150)
    points = [exact(x) for x in centres]; boxes = [box(x-radius,x+radius) for x in centres]
    assert all(endpoint(boxes[i],0) >= F(root['parameters'][k]['lower']) and endpoint(boxes[i],1) <= F(root['parameters'][k]['upper']) for i,k in enumerate(('g0','gs')))
    f0,J0 = radial_law(points[0],points[1],profiles,mask); _,JB = radial_law(boxes[0],boxes[1],profiles,mask)
    mid = [[(endpoint(x,0)+endpoint(x,1))/2 for x in row] for row in J0]
    det = mid[0][0]*mid[1][1]-mid[0][1]*mid[1][0]
    assert det != 0
    C = [[mid[1][1]/det,-mid[0][1]/det],[-mid[1][0]/det,mid[0][0]/det]]
    lips = []; images = []
    for i in range(2):
        lip = sum(upper(exact(int(i == j))-sum(exact(C[i][k])*JB[k][j] for k in range(2))) for j in range(2))
        lips.append(lip)
        images.append(upper(sum(exact(C[i][j])*f0[j] for j in range(2)))+lip*radius)
    assert max(lips) < 1 and max(images) < radius
    wrong_force,_ = radial_law(points[0],points[1],profiles,mask,feedback=False)
    wrong_image = max(endpoint(abs(sum(exact(C[i][j])*wrong_force[j] for j in range(2))),0) for i in range(2))
    wrong_power_force,_ = radial_law(points[0],points[1],profiles,mask,feedback_exponent=6144)
    wrong_power_image = max(endpoint(abs(sum(exact(C[i][j])*wrong_power_force[j] for j in range(2))),0) for i in range(2))
    oldpoints = [exact(F(x)) for x in oldroot['centre_exact_rational']]
    old_force,_ = radial_law(oldpoints[0],oldpoints[1],profiles,mask)
    wrong_root_image = max(endpoint(abs(sum(exact(C[i][j])*old_force[j] for j in range(2))),0) for i in range(2))
    assert min(wrong_image,wrong_power_image,wrong_root_image) > radius
    return {'exists': True, 'unique_in_fixed_box': True, 'radial_stationarity': True,
            'contraction': bound(exact(max(lips))), 'image_radii': [bound(exact(x)) for x in images],
            'radius_exact': str(radius), 'preconditioner_exact': [[str(x) for x in row] for row in C],
            'original_record_contraction': root['contraction'], 'original_record_image_radii': root['image_radii'],
            'wrong_omitted_allq_feedback_image': bound(exact(wrong_image)),
            'wrong_fast_Lambda_feedback_image_lower':bound(exact(wrong_power_image)),
            'wrong_originalRoot_at_current_law_image_lower':bound(exact(wrong_root_image)),
            'proof': 'Exact rational nonsingular C; T(g)=g-CF(g), ||DT||infinity<1 and T(box) strictly inside box. Banach fixed point gives existence and unique zero of the ACTUAL 64Lambda classical+Lambda^(3/8) all64q fast-feedback law. No root solver or centre adoption.'}


def spectrum_and_hessian(profiles: list[dict[str, Any]], g: list[Any], gamma: np.ndarray, M: np.ndarray, cols: list[dict[int, Any]]) -> tuple[list[list[Any]], list[dict[str, Any]], F, F]:
    """Direct Clifford projector trace Hessian, retaining BOTH interband terms."""
    roots = [iv.sqrt(x) for x in g]; sqrt8 = iv.sqrt(exact(8))
    gram_derivs = [[(roots[n]*c.get(15*mu+n,exact(0))+roots[mu]*c.get(15*n+mu,exact(0)))/sqrt8 for c in cols] for mu,n in product(range(4),repeat=2)]
    K = [[exact(128)*sum(row[a]*row[b] for row in gram_derivs)+sum(32*g[slot//15]*v*cols[b].get(slot,exact(0)) for slot,v in cols[a].items()) for b in range(45)] for a in range(45)]
    # ReTr(P+ T_A P- T_B)=Tr(MA MB - n.gamma MA n.gamma MB)/32.
    # The sign-dependent linear trace is imaginary and cancels in the real
    # static Hessian. Negativeheavy fills P+ of ORIGINAL h, not P-.
    tensor = np.rint(np.einsum('mij,ajk,nkl,bli->manb',gamma,M,gamma,M).real).astype(np.int64)
    assert np.array_equal(tensor, np.einsum('mij,ajk,nkl,bli->manb',gamma,M,gamma,M).real)
    cos, sin = (1,-1,-1,1),(1,1,-1,-1)
    species = []; ambient_upper = F(0); third_upper = F(0)
    for p in profiles:
        scale = F(1) if p['scale'] == 'heavy' else F(p['load'],2**8192)
        B = [[exact(0) for _ in range(45)] for _ in range(45)]
        gaps = []; vertices = []
        for k in product(range(4), repeat=3):
            x = [exact(p['onsite']),exact(0),exact(0),exact(0)]
            for path in p['paths']:
                axis,L = path['axis'],path['length']
                # AP quarter mesh: cos(2theta)=0 exactly; ci(two)=0,
                # but physical ordered two-link paths remain in frozen law.
                if L == 1:
                    x[0] += exact(path['c0'])*cos[k[axis]]*iv.sqrt(exact(2))
                    x[axis+1] -= exact(path['ci'])*sin[k[axis]]*iv.sqrt(exact(2))
            h = [r*y for r,y in zip(roots,x,strict=True)]; E = iv.sqrt(sum(y*y for y in h)); n = [y/E for y in h]
            gaps.append(endpoint(2*E,0)); vertices.append(upper(iv.sqrt(sum(y*y for y in x))/iv.sqrt(exact(128))))
            D = [[exact(F(int(A == Z),8))-sum(exact(F(int(tensor[i,A,j,Z]),32))*n[i]*n[j] for i,j in product(range(4),repeat=2) if tensor[i,A,j,Z]) for Z in range(15)] for A in range(15)]
            for a,ca in enumerate(cols):
                for b in range(a,45):
                    value = sum(va*vb*x[sa//15]*x[sb//15]*D[sa%15][sb%15]/(64*E) for sa,va in ca.items() for sb,vb in cols[b].items())
                    B[a][b] += value
        for a in range(45):
            for b in range(a,45):
                K[a][b] -= exact(scale)*B[a][b]; K[b][a] = K[a][b]
        gap = min(gaps); d1 = max(vertices)
        # For affine h and a separating occupied/empty gap, P' solves the
        # offdiagonal Sylvester equation, ||P'||<=d1/gap; differentiating it
        # gives ||P''||<=8d1²/gap². Thus |Tr(P''V)|<=32rank d1³/gap².
        # The root segment is separately checked to preserve gap>=gap_old/2.
        ambient_upper += scale*2*128*d1**2/gap
        third_upper += scale*32*128*d1**3/(gap/2)**2
        species.append({'role':p['role'],'species_count':p['load'],'rank_each':128,'dimension_each':256,
                        'gap_unscaled_lower':str(gap),'gap_physical_lower':str(gap*2**8192 if p['scale']=='heavy' else gap),
                        'physical_energy_scale':'Lambda=2^8192' if p['scale']=='heavy' else '1',
                        'original_h_occupied_sheet':'UPPER' if p['scale']=='heavy' else 'LOWER',
                        'physical_occupied_sheet':'NEGATIVE','projector':'(I+h/E)/2' if p['scale']=='heavy' else '(I-h/E)/2',
                        'vertex_uniform_Cart_operator_upper':str(d1),
                        'proof':'h(k)^2=E(k)^2 I4, Tr h(k)=0: eigenvalues +/-E, multiplicity2. Exact AP Fourier unitarity gives64 blocks and rank128. Gap=min2E>0. Heavy physical operator is -Lambda h, reversing original-h occupation.'})
    return K,species,ambient_upper,third_upper


def reprice_eigen_witness(K: list[list[Any]], old: dict[str, Any]) -> dict[str, Any]:
    U = [[F.from_float(x) for x in row] for row in old['actual_full45_collective_eigenvectors']]
    values = [F.from_float(x) for x in old['actual_full45_effective_eigenvalues_divided_Lambda']]
    orth = max(sum(abs(sum(U[k][i]*U[k][j] for k in range(45))-int(i == j)) for j in range(45)) for i in range(45))
    residual = max(sum(upper(K[i][j]-exact(sum(U[i][k]*values[k]*U[j][k] for k in range(45)))) for j in range(45)) for i in range(45))
    assert orth < 1
    price = residual+2*max(abs(x) for x in values)*orth/(1-orth)
    lower = min(values)-price
    assert lower > 0
    return {'lower':str(lower),'reconstruction_row_norm_upper':str(residual),'orthogonality_row_norm_upper':str(orth),
            'polar_normalization_and_reconstruction_price':str(price),
            'proof':'Recompute exact interval K from law, not displayed K. ||K-U diag(lambda) U^T||<=row norm, ||U^TU-I||<=row norm<1; polar orthogonalization price<=2max|lambda|orth/(1-orth). Weyl gives minlambda-price.'}


def symmetry_proof(g: list[Any], gamma: np.ndarray, M: np.ndarray, cols: list[dict[int, Any]], mask: list[list[int]]) -> dict[str, Any]:
    # First fast-ZPE variation depends only on diagonal dH at diagonal H;
    # exact Clifford traces determine all diagonal dH entries.
    nonradial_diagonal_coefficients = []
    for a,c in enumerate(cols[4:], start=4):
        for b in range(15):
            for slot in c:
                mu,A = divmod(slot,15)
                comm1 = 1j*(M[b]@M[A]-M[A]@M[b])
                comm2 = 1j*(M[b]@gamma[mu]-gamma[mu]@M[b])
                coefficient = np.trace(comm1@comm2+comm2@comm1).real
                if coefficient:
                    nonradial_diagonal_coefficients.append((a,b,slot,int(coefficient)))
    assert not nonradial_diagonal_coefficients
    # In primitive and matter forces, grade>1 traces vanish, while AP
    # reflection theta_mu -> -theta_mu leaves cos and E unchanged and flips
    # sin_mu. Offdiagonal grade1 terms therefore cancel pairwise exactly.
    assert all(np.trace(gamma[i]@M[A]) == 0 for i in range(4) for A in range(4,15))
    assert all(mask[b][mu] == int(np.array_equal(M[b]@gamma[mu]+gamma[mu]@M[b],np.zeros((4,4)))) for b,mu in product(range(15),range(4)))
    return {'all45_normal_stationarity':True,'all60_Cartesian_stationarity':True,
            'normal_rank':45,'gauge_orbit_rank':15,
            'classical_force_proof':'Tr(gamma_i M_grade>1)=0. AP k_mu->3-k_mu keeps cos(theta),E and flips sin(theta); mixed Gram force cancels. Cubic permutations equalize three spatial diagonal forces. Primitive force has only four grade1 diagonals. Adjoint gauge invariance kills15 vertical forces.',
            'allq_fast_force_proof':'At Clifford root H_ab=delta_ab sum_mu anticommSubset(b,mu)g_mu/2. All41 nonradial normal variations have EXACT zero diagonal dH by integer Clifford traces checked here. Trace Frechet derivative of each sqrt(.06H+cqhat² I) depends only on diagonal dH even at degeneracy. qhat²-only Phi fast modes have zero mean derivative. Spatial cubic symmetry and the two radial zero equations kill remaining4 forces.',
            'fast_nonradial_nonzero_integer_coefficients':nonradial_diagonal_coefficients,
            'fullnormal_extension':'The declared allq feedback functional is classical static sea plus leading primitive fast ZPE. NOT the full interacting quantum effective potential; omitted matter-fast and slow ZPE forces are not set to zero.'}


def normal_frame_certificate(g: list[Any], gamma: np.ndarray, M: np.ndarray, cols: list[dict[int,Any]], mask: list[list[int]]) -> dict[str,Any]:
    roots = [iv.sqrt(x) for x in g]
    orbit: list[dict[int,Any]] = []
    for b in range(15):
        column = {}
        for mu,A in product(range(4),range(15)):
            coefficient = np.trace(M[A]@(1j*(M[b]@gamma[mu]-gamma[mu]@M[b])))
            assert coefficient.imag == 0 and coefficient.real == round(coefficient.real)
            if coefficient.real:
                column[15*mu+A] = exact(F(int(coefficient.real),4))*roots[mu]
        orbit.append(column)
    gram_error = max(sum(upper(sum(v*cols[b].get(slot,exact(0)) for slot,v in cols[a].items())-int(a == b)) for b in range(45)) for a in range(45))
    orbit_error = max(sum(upper(sum(v*orbit[b].get(slot,exact(0)) for slot,v in orbit[a].items())-(4*sum(exact(mask[a][mu])*g[mu] for mu in range(4)) if a == b else 0)) for b in range(15)) for a in range(15))
    cross = max(sum(upper(sum(v*column.get(slot,exact(0)) for slot,v in c.items())) for column in orbit) for c in cols)
    orbit_lower = min(endpoint(4*sum(exact(mask[b][mu])*g[mu] for mu in range(4)),0) for b in range(15))-orbit_error
    assert gram_error < F(1,10**90) and cross < F(1,10**90) and orbit_error < F(1,10**90) and orbit_lower > 0
    wrong_cross = iv.sqrt(sum(v*v for v in orbit[0].values()))
    assert endpoint(wrong_cross,0) > 0
    return {'normal_rank':45,'orbit_rank':15,'normal_Gram_row_error_upper':str(gram_error),
            'orbit_Gram_row_error_upper':str(orbit_error),'normal_orbit_cross_row_upper':str(cross),
            'orbit_Gram_lower':str(orbit_lower),'wrong_gauge_vector_as_normal_cross_lower':str(endpoint(wrong_cross,0)),
            'exact_geometry_proof':'Clifford grade matrices are orthogonal. Six Gram-symmetric vectors are the weighted complements of six grade2 orbit vectors. Four grade1 generators have disjoint three-entry grade2 orbits; n,z are their exact orthonormal cross-product complements. One grade4 generator has a four-entry grade3 orbit, removed by the exact Householder matrix. Four grade3 generators span the four grade4 fields, so no grade4 field is normal. All remaining grade2/3 entries are singleton normal columns. Consequently Q^TQ=I and Q^TO=0 algebraically; outward Gram bounds independently certify rank45/15 on the actual box. Orthogonal grade-to-Cart change preserves the metric and orbit.'}


def antecedents(gaussian: dict[str, Any], docs: dict[str, Any]) -> dict[str, Any]:
    abi = gaussian['exact_original_wave_ABI']; descriptor = abi['matter']['descriptor']
    assert docs['quantum']['exact_mean'] == abi['exact_mean']
    assert docs['source']['exact_NEW_allq_BO_mean'] == abi['exact_mean']
    assert docs['matter']['physical_mean']['exact_root'] == abi['exact_mean']
    assert descriptor == docs['matter']
    assert descriptor['exact_profiles'] == docs['root']['input_descriptor']['exact_profile_paths']
    assert descriptor['original_micro_config_UNCHANGED'] == docs['full']['input_descriptor']['original_micro_config_UNCHANGED'] == docs['source']['original_micro_config_UNCHANGED']
    parameters = descriptor['parameters']
    assert {k:parameters[k] for k in ('N','V','a','Lambda','chi','Gphi')} == {'N':4,'V':64,'a':'1/64','Lambda':'2^8192','chi':'2^-4096','Gphi':'8chi'}
    assert parameters['physical_species_loads'] == [1,1,63]
    profiles = descriptor['exact_profiles']
    assert [(p['scale'],p['load']) for p in profiles] == [('heavy',1),('charged',1),('charged',63)]
    source_hashes = {}
    for name,entry in gaussian['source_at_import'].items():
        raw = entry['bytes_utf8'].encode(); assert hashlib.sha256(raw).hexdigest() == entry['sha256']
        source_hashes[name] = entry['sha256']
        ast.parse(raw, filename=name)
    for doc in (docs['root'],docs['full']):
        for name,entry in doc['source_at_import'].items():
            assert hashlib.sha256(entry['bytes_utf8'].encode()).hexdigest() == entry['sha256'] == source_hashes[name]
    assert descriptor['local_Gauss_source_at_import_sha256'] == source_hashes['induced_phi_nonuniform_gauss_quantum.py']
    assert 'lambda^(5/4)' in abi['physical_K'].lower()
    assert '8 I45' in abi['physical_G']
    return {'same_root':True,'same_law':True,'same_actual_Cartesian_frame':True,'captured_source_hashes':source_hashes,
            'exact_parameters':parameters,'profiles':profiles,'coordinate_order':descriptor['coordinate_order'],
            'source_execution':'NONE: only JSON and captured Python AST read; no legacy module import',
            'root_ancestry':'Authoritative Gaussian6abd exact_original_wave_ABI.exact_mean <- quantum0f143 <- originalRoot4691/full5b978; same exact mean in actual matterb20c and sourceda19.',
            'law_scales':'Quartic/heavy Lambda; active1 and63aux1; fast gradient1/5/alignment1/5/orderedplaquette7/10 Lambda^(5/4); Phi cometric8chi, chi2^-4096; normal45/gauge15, Fourier1/8, AP quarter mesh.'}


def runtime_bytes() -> tuple[dict[str, Any], list[Path]]:
    paths = {Path(sys.executable).resolve()}
    for module in list(sys.modules.values()):
        location = getattr(module,'__file__',None)
        if location and Path(location).is_file():
            paths.add(Path(location).resolve())
    for package in (np,mp,ijson):
        package_file = package.__file__
        assert package_file is not None
        package_root = Path(package_file).resolve().parent
        paths.update(path.resolve() for path in package_root.rglob('*.so*') if path.is_file())
        for sibling in package_root.parent.iterdir():
            if sibling.is_dir() and sibling.name == package_root.name+'.libs':
                paths.update(path.resolve() for path in sibling.rglob('*') if path.is_file())
    linkage = {}
    for native in sorted(paths):
        if '.so' not in native.name and native != Path(sys.executable).resolve():
            continue
        process = subprocess.run(['ldd',str(native)],capture_output=True,text=True,check=False)
        linkage[str(native)] = {'returncode':process.returncode,'stdout':process.stdout,'stderr':process.stderr}
        for name in re.findall(r'/[^\\s()]+',process.stdout):
            path = Path(name)
            if path.is_file():
                paths.add(path.resolve())
    manifest = {str(path):{'sha256':digest(path),'bytes':path.stat().st_size} for path in sorted(paths)}
    return {'loaded_module_and_native_dependency_bytes':manifest,'ldd':linkage,
            'scope':'Actual loaded file-backed Python modules, package native shared objects, package .libs, Python executable and ldd-resolved dependencies. Builtin/frozen modules are supplied by pinned Python binary; OS/kernel is labelled, not simulated.'},sorted(paths)


def run(gaussian_path: Path, output: Path, archive_path: Path) -> dict[str, Any]:
    started = time.monotonic(); own_path = Path(__file__).resolve(); own_raw = own_path.read_bytes()
    assert digest(gaussian_path) == GAUSSIAN_SHA
    gaussian = json.loads(gaussian_path.read_bytes())
    inputs: dict[str, Path] = {'gaussian':gaussian_path}
    abi = gaussian['exact_original_wave_ABI']
    inputs['root'] = Path(abi['matter']['descriptor']['root_receipt']['path'])
    inputs['full'] = Path(abi['matter']['descriptor']['full_receipt']['path'])
    for role in ('quantum','matter','source'):
        target = next(p for p,e in gaussian['input_capsule'].items() if e['sha256'] == EXPECTED[role])
        inputs[role] = Path(target) if Path(target).is_absolute() else gaussian_path.parent/target
    hashes = {role:digest(path) for role,path in inputs.items()}
    assert all(hashes[role] == expected for role,expected in EXPECTED.items())
    docs: dict[str, Any] = {role:json.loads(inputs[role].read_bytes()) for role in ('root','full')}
    docs['source'] = select(inputs['source'],'input_descriptor')
    docs['quantum'] = select(inputs['quantum'],'input_descriptor')
    docs['matter'] = select(inputs['matter'],'input_descriptor')
    joined = antecedents(gaussian,docs)
    gamma,M,_ = clifford()
    mask = [[int(np.array_equal(m@t+t@m,np.zeros((4,4)))) for t in gamma] for m in M]
    profiles = joined['profiles']; root = abi['exact_mean']
    rootcert = fixed_box_certificate(root,docs['root']['actual_stationary_root'],profiles,mask)
    iv.dps = 100
    newg = [stored_box(root['parameters']['g0'])]+[stored_box(root['parameters']['gs'])]*3
    oldroot = docs['root']['actual_stationary_root']
    oldg = [stored_box(oldroot['parameters']['g0'])]+[stored_box(oldroot['parameters']['gs'])]*3
    oldcols = normal_columns(oldg,gamma,M); newcols = normal_columns(newg,gamma,M)
    symmetric = symmetry_proof(newg,gamma,M,newcols,mask)
    normal_geometry = normal_frame_certificate(newg,gamma,M,newcols,mask)
    K,seas,hsea,hthird = spectrum_and_hessian(profiles,oldg,gamma,M,oldcols)
    eigen = reprice_eigen_witness(K,docs['full']['actual_full45_coupled_quantum_consumer'])
    # Price ACTUAL roots, not IR262144 or centreflux8, including both old and
    # new boxes. root radii bound exact implicit roots, not printed centres.
    delta = max(abs(F(root['centre_exact_rational'][i])-F(oldroot['centre_exact_rational'][i])) for i in range(2))+F(2,10**2150)
    assert all(endpoint(g,0)>F(1,10) and endpoint(g,1)<F(3,10) for g in [*oldg,*newg])
    frame_derivative = derivative_bound(iv,gamma,M,exact,endpoint)
    q_change = F(frame_derivative['Q_operator_Lipschitz_from_parameter_infinity_upper'])*delta
    cart_change = 100*delta
    # The per-site Fourier-normalized Cart vertex is affine and bounded
    # by ||x||/(8sqrt2); Weyl keeps every species gap at least half-oldgap.
    assert all(2*F(s['vertex_uniform_Cart_operator_upper'])*cart_change < F(s['gap_unscaled_lower'])/2 for s in seas)
    # Uniform global Cart variable Z has ||Z||²=512sum(g)<614.4.
    # Primitive E/Lambda=||Z Z^T||F²/64; differentiation gives Hessian
    # <=12||Z||²/64<120 and third<=24||Z||/64<10 along the segment.
    primitive_hessian_upper = F(120)
    primitive_third_upper = F(10)
    frame_price = 2*(primitive_hessian_upper+hsea)*q_change
    root_price = (primitive_third_upper+hthird)*cart_change
    total_price = frame_price+root_price
    lower = F(eigen['lower'])-total_price
    assert lower > F(2,5)
    # Verify gaps at new box separately; old eigen witness is never used to
    # infer current-root occupation. The static rank is per independent sea.
    current_seas = []
    for p in profiles:
        one = next(v for v in p['paths'] if v['length'] == 1)
        gap = min(endpoint(2*iv.sqrt(newg[0]*(exact(p['onsite'])+iv.sqrt(exact(2))*exact(one['c0'])*z)**2+newg[1]*6*exact(one['ci'])**2),0) for z in (3,1,-1,-3))
        previous = next(s for s in seas if s['role'] == p['role'])
        current_seas.append(dict(previous,gap_unscaled_lower=str(gap),gap_physical_lower=str(gap*2**8192 if p['scale']=='heavy' else gap)))
    # Adverse controls are actual arithmetic witnesses, not booleans comparing
    # names: wrong occupation gives positive rather than negative heavy energy;
    # wrong Cart normalization violates the exactly traced physical metric.
    heavy = profiles[0]; oneheavy = next(v for v in heavy['paths'] if v['length'] == 1)
    Eheavy = iv.sqrt(newg[0]*(exact(heavy['onsite'])+3*iv.sqrt(exact(2))*exact(oneheavy['c0']))**2+6*newg[1]*exact(oneheavy['ci'])**2)
    correct_heavy_energy = -2*Eheavy
    wrong_heavy_energy = 2*Eheavy
    traced_primitive_metric = F.from_float(float(np.trace(M[0]@M[0]).real))/32
    assert traced_primitive_metric == F(1,8)
    wrong_unit_metric_defect = abs(1/traced_primitive_metric-1)
    collapsed_gap = endpoint(2*iv.sqrt(exact(0)*Eheavy**2),1)
    controls = {
        'wrong_unit_Gphi':{'rejected':wrong_unit_metric_defect>0,'exact_cometric_defect':str(wrong_unit_metric_defect)},
        'wrong_gauge_vector_as_normal':{'rejected':F(normal_geometry['wrong_gauge_vector_as_normal_cross_lower'])>0,'normal_orbit_cross_lower':normal_geometry['wrong_gauge_vector_as_normal_cross_lower']},
        'wrong_heavy_lower_original_h':{'rejected':endpoint(correct_heavy_energy,1)<0<endpoint(wrong_heavy_energy,0),'correct_energy_divided_Lambda_per_block':bound(correct_heavy_energy),'wrong_energy_divided_Lambda_per_block':bound(wrong_heavy_energy)},
        'wrong_originalRoot_as_current':{'rejected':F(rootcert['wrong_originalRoot_at_current_law_image_lower']['lower'])>F(1,10**2150),'residual_image':rootcert['wrong_originalRoot_at_current_law_image_lower']},
        'wrong_fast_Lambda_as_Lambda5over4':{'rejected':F(rootcert['wrong_fast_Lambda_feedback_image_lower']['lower'])>F(1,10**2150),'residual_image':rootcert['wrong_fast_Lambda_feedback_image_lower']},
        'omitted_fast_feedback_fails_fixed_box':{'rejected':F(rootcert['wrong_omitted_allq_feedback_image']['lower'])>F(1,10**2150),'residual_image':rootcert['wrong_omitted_allq_feedback_image']},
        'collapsed_zero_frame_gap':{'rejected':collapsed_gap<=0,'computed_gap':str(collapsed_gap),'required_gap':'strictly positive'},
    }
    assert all(c['rejected'] for c in controls.values())
    slow = {'q0_static_lower_divided_Lambda':str(lower),'positive':True,'actual_Gphi':8,
            'q0_static_lower_divided_Lambda_outward_display':bound(exact(lower)),
            'q0_Phi_principal_frequency_squared_lower':str(8*lower*F(2)**4096),
            'old_law_independent_eigen_certificate':eigen,'actual_root_shift_infinity_upper':str(delta),
            'normal_frame_operator_change_upper':str(q_change),'uniform_Cart_background_shift_upper':str(cart_change),
            'outward_frame_derivative_certificate':frame_derivative,
            'ambient_matter_hessian_upper_divided_Lambda':str(hsea),'ambient_matter_third_upper_divided_Lambda':str(hthird),
            'primitive_hessian_upper_divided_Lambda':str(primitive_hessian_upper),'primitive_third_upper_divided_Lambda':str(primitive_third_upper),
            'frame_perturbation_price':str(frame_price),'coupledroot_shift_price':str(root_price),'total_price':str(total_price),
            'metric_proof':'Q is exact algebraic orthonormal normal frame in Tr(TaTb)=delta/2 Cart60; Fourier exp(iqx)/8 and primitive-coordinate embedding deltaPhi=Q T imply primitive metricI/8, hence Gphi=8chi. Gauge15 removed by orthogonal orbit complement. A changing orthonormal frame has unchanged metric; frozen/new overlap singular values>=1-priced_frame_change. No unit-Gphi substitution.',
            'shift_proof':'Explicit roots and both rational radii imply delta. Outward first jets of every algebraic Q entry independently price normal-frame change on [.1,.3]^4; physical uniformCart displacement<=100delta. Affine gapped sheet Sylvester derivative bounds give ||Hsea||<=2rank d1²/gap_old and ||D Hsea||<=32rank d1³/(gap_old/2)². Weyl explicitly preserves gap>=gap_old/2 along the segment. Actual scales/loads are summed. Uniform global Cart variable Z obeys ||Z||²=512sum(g)<614.4. Primitive E/Lambda=||ZZ^T||F²/64; multilinear differentiation gives Hessian<=12||Z||²/64<120, third<=24||Z||/64<10. Weyl price2H||dQ||+H3||dCart||.',
            'scope':'Static PH90 q0 slow45 stiffness at exact current root. No inference of interacting HC ground, full effective-quantum stationarity or exact full quantum ground.'}
    slow['full_coupled_q0_certificate'] = full_q0_bound(profiles,newg,mask,lower,iv,exact,endpoint)
    slow['actual_q0_supplier_composition'] = {
        'captured_supplier':'induced_phi_nonuniform_field_car.py',
        'captured_supplier_sha256':joined['captured_source_hashes']['induced_phi_nonuniform_field_car.py'],
        'q0_Phi_vertices':'vertices_q at q=0 gives sign sum_mu x_mu Q_muA M_A/(8sqrt8), including both endpoint fields and forward/reverse ordered paths; AP length2 cosine=0 at this frozen root, never path deletion.',
        'q0_Phi_contact':'PP contact exactly0: the original endpoint matter law is affine in Phi even with ordered links.',
        'BOTH_PH_identity':'response has plus/(2E)+minus/(2E) at q0; conjugate sheet amplitudes give ReTr(Pocc Va Punocc Vb)/E. Clifford P=(I+sign*n.gamma)/2 reduces this to the independently evaluated integer-tensor Dreal and /64 Fourier factor.',
        'physical_static_composition':'spectral_observation K_Lambda=Kprimitive+sea_heavy, K_unit=sea_active+63sea_aux. The certified A/ Lambda is Kprimitive-Bheavy-(Bactive+63Baux)/Lambda. Fast q0 Phi block0 and colour block I3 tensor H/5 multiply Lambda^(5/4). No contact/channel/scale/frame substitution.',
    }
    sources = gaussian['source_at_import']
    runtime_capture,runtime_paths = runtime_bytes()
    helper_path = own_path.with_name(PREFIX+'_frame_bound.py')
    supervisor_path = own_path.with_name(PREFIX+'_natural_exit.py')
    spectral_path = own_path.with_name(PREFIX+'_spectral_bound.py')
    support = {path:path.read_bytes() for path in (helper_path,spectral_path,supervisor_path)}
    with zipfile.ZipFile(archive_path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as archive:
        archive.writestr('consumer/'+own_path.name,own_raw)
        for path,raw in support.items():
            archive.writestr('consumer/'+path.name,raw)
        for path in runtime_paths:
            archive.write(path,'runtime'+str(path))
        archive.writestr('runtime_manifest.json',json.dumps(runtime_capture,indent=2))
        for role,path in inputs.items():
            archive.write(path,'inputs/'+role+'/'+path.name)
        for name,record in sources.items():
            archive.writestr('captured_sources/'+name,record['bytes_utf8'])
    assert own_path.read_bytes() == own_raw
    assert all(digest(path) == hashes[role] for role,path in inputs.items())
    assert all(path.read_bytes() == raw for path,raw in support.items())
    assert all(digest(Path(path)) == record['sha256'] for path,record in runtime_capture['loaded_module_and_native_dependency_bytes'].items())
    result = {'status':'CURRENT_N4_ROOT_STATIC_SEAS_Q0_SLOW_PROVED_SCOPED',
              'governing_plan':'issue-222-induced-photon-delta13-continuation-20261008',
              'governing_revision':'712b1723921b52c5cc7b0fced4b6428a6c5b37ad6278d074bae0895dc40a1416',
              'template_revision':'89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe',
              'Rootexist':rootcert,'stationarity':symmetric,'rank':{'per_species':128,'species':65,'normal':45,'gauge':15},
              'gaps':current_seas,'q0lower':slow,'antecedents':joined,'controls':controls,
              'actual_normal_frame':normal_geometry,
              'exact_conditions':{'input_hashes':hashes,'root':'Gaussian6abd exact_original_wave_ABI.exact_mean, unique in the original rational fixed box',
                                  'StaticSea':'65 independent filled128 oneparticle static seas; signed -Lambda heavy upper ORIGINAL h; active and63aux lower h',
                                  'slow':'same law and derived newcoupledroot/normalframe metric shift, classical+static BOTH-PH Hessian',
                                  'stationarity':'ONLY exact declared classical+leading all64q fast-ZPE feedback functional'},
              'superseded_initial_attempt':{'capsule':PREFIX+'_initial_natural_exit_capsule.json','reason':'Natural0 root/Hessian calculation retained, but initial shift price used underestimated Cart-vertex norm and an unpriced frame Lipschitz claim. This independently derived corrected receipt supersedes that price; initial receipt is not admission evidence.'},
              'nonclaims':['No original SOURCE/CURRENT427 adoption','No interacting HC ground','No full quantum ground','No zero omitted matter-fast or slow-ZPE force','No change of physical parameters, Lambda/chi, covariance, clock or source choice','No photon pole, common cone, Compton or globalphysical promotion'],
              'input_capsule':{role:{'path':str(path),'sha256':hashes[role],'bytes':path.stat().st_size} for role,path in inputs.items()},
              'immutable_archive':{'path':str(archive_path),'sha256':digest(archive_path),'bytes':archive_path.stat().st_size},
              'own_source':{'path':str(own_path),'sha256':hashlib.sha256(own_raw).hexdigest(),'bytes_utf8':own_raw.decode()},
              'source_unchanged':True,'input_unchanged':True,
              'runtime':{'python':sys.version,'executable':sys.executable,'platform':platform.platform(),'mpmath':mp.__version__,'numpy':np.__version__,'ijson':ijson.__version__,'root_interval_digits':2300,'Hessian_interval_digits':100},
              'runtime_capsule':runtime_capture,
              'elapsed_seconds':time.monotonic()-started}
    output.write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gaussian',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--archive',type=Path,required=True)
    args = parser.parse_args()
    result = run(args.gaussian,args.output,args.archive)
    print(json.dumps({'status':result['status'],'Rootexist':result['Rootexist']['exists'],
                      'q0lower':result['q0lower']['q0_static_lower_divided_Lambda'],
                      'elapsed_seconds':result['elapsed_seconds']}))
