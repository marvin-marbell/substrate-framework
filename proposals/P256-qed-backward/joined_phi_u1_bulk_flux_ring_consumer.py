"""Exact frozen-native canonical sea-to-plaquette irreducible electric kernel.

No phase, physical ground, stiffness, or pole admission. See accompanying proof.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from fractions import Fraction as F
import hashlib
import importlib.abc
import importlib.machinery
import io
from itertools import permutations, product
import json
from pathlib import Path
import sys
import time
from typing import Any
import zipfile

ROOT = Path(__file__).resolve().parent
PREFIX = 'joined_phi_u1_bulk_flux_ring_'
INPUTS = ['joined_phi_u1_ground_photon_collective_delivered_actual_20261010.npz',
          'induced_phi_uniform_selfconsistent_full_normal_quantum_final_actual_receipt.json',
          'joined_phi_u1_native_ir_mechanism_proof_corrected_20261010.json',
          'joined_phi_u1_uniform_electric_current_bridge_integration_addendum_20261010.json',
          'joined_phi_u1_unbiased_character_homology_geometry_binding_20261010.json',
          'joined_phi_u1_bulk_flux_spatial_linked_assignment_packet_20261010.json',
          PREFIX+'proof_20261010.json']
CAPTURED: dict[str, bytes] = {Path(__file__).name: Path(__file__).read_bytes()}
PINS = {
    'joined_phi_u1_ground_photon_collective_delivered_actual_20261010.npz': 'f8a0793adcbf4909402cd089471aa8c33fc74a31b66e99618f8baaab8d6dfb65',
    'induced_phi_uniform_selfconsistent_full_normal_quantum_final_actual_receipt.json': '5b97862fc7fca21b8797d549d7d4dfdde58b101e8c46c7a7cfb6531a669c4354',
    'joined_phi_u1_unbiased_character_homology_geometry_binding_20261010.json': '31a9275adabaa7e84c49b0013cca6d8ca8f32e1f6ed052d1b6dac9b9fc21028f'
}

class Loader(importlib.machinery.SourceFileLoader):
    def get_code(self, fullname: str) -> Any:
        raw = Path(self.path).read_bytes()
        CAPTURED[Path(self.path).name] = raw
        return self.source_to_code(raw, self.path)

class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is not None and spec.origin and Path(spec.origin).parent == ROOT and spec.origin.endswith('.py'):
            spec.loader = Loader(fullname, spec.origin)
        return spec

finder = Finder()
sys.meta_path.insert(0, finder)
from rotation_band import RotationBand
from local_orbital_stability import scalar_paths
sys.meta_path.remove(finder)
import numpy as np

C = tuple[F, F]
ZERO: C = (F(0), F(0))
ONE: C = (F(1), F(0))
def add(a: C, b: C) -> C:
    return a[0]+b[0], a[1]+b[1]
def mul(a: C, b: C) -> C:
    return a[0]*b[0]-a[1]*b[1], a[0]*b[1]+a[1]*b[0]
def conj(a: C) -> C:
    return a[0], -a[1]
def scale(a: C, x: F) -> C:
    return a[0]*x, a[1]*x

def car(mask: int, create: int, annihilate: int) -> tuple[int, int] | None:
    if not (mask >> annihilate) & 1:
        return None
    sign = (-1)**((mask & ((1 << annihilate)-1)).bit_count())
    mask ^= 1 << annihilate
    if (mask >> create) & 1:
        return None
    sign *= (-1)**((mask & ((1 << create)-1)).bit_count())
    return mask | (1 << create), sign

REF = sum(1 << (4*x+s) for x in range(4) for s in (0, 1))

def exact_frame(raw: bytes) -> tuple[list[list[list[C]]], F]:
    with np.load(io.BytesIO(raw), allow_pickle=False) as data:
        phi = data['Phi_binary_section']
        # Exact rational Hadamard numerator: first two columns are Gamma0=-1.
        h = [[1,0,1,0],[0,1,0,1],[-1,0,1,0],[0,-1,0,1]]
        mats = []
        for m in range(4):
            a = [[(F(float(phi[m,i,j].real)), F(float(phi[m,i,j].imag))) for j in range(4)] for i in range(4)]
            mats.append([[scale(sum_c([scale(a[k][l], F(h[k][i]*h[l][j])) for k in range(4) for l in range(4)]), F(1,2)) for j in range(4)] for i in range(4)])
        s = F(float(phi[0,0,2].real))
        assert mats[0] == [[((-s if i<2 else s) if i==j else F(0), F(0)) for j in range(4)] for i in range(4)]
        return mats, s

def sum_c(values: list[C]) -> C:
    total = ZERO
    for value in values:
        total = add(total, value)
    return total

def profiles(cfg: dict[str, Any], frame: list[list[list[C]]], s: F) -> list[dict[str, Any]]:
    result = []
    for role, count, mass in [('active',1,cfg['active_m']),('auxiliary',63,cfg['auxiliary_m'])]:
        band = RotationBand(a=cfg['a'], m=mass, r=cfg['light_r'], v=cfg['light_v'], improved=True)
        onsite, paths = scalar_paths(band)
        rows = []
        for disp, coefficients in paths:
            coeff = [(F(float(z.real)), F(float(z.imag))) for z in coefficients]
            matrix = [[sum_c([mul(coeff[m],frame[m][i][j]) for m in range(4)]) for j in range(4)] for i in range(4)]
            axis = int(np.flatnonzero(disp)[0])
            rows.append(dict(axis=axis,length=int(disp[axis]),matrix=matrix,coefficients=coeff))
        result.append(dict(role=role,count=count,onsite=F(onsite)*s,paths=rows))
    return result

# Fourier inverse graph Laplacian, only four local vertices needed. No root solve.
def local_coulomb(n: int) -> np.ndarray:
    vertices = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0]])
    q = np.array(list(product(range(n),repeat=3)))
    lam = 4*np.sum(np.sin(np.pi*q/n)**2,axis=1)
    positive = lam>0
    delta = vertices[:,None,:]-vertices[None,:,:]
    return np.sum(np.cos(2*np.pi*np.einsum('ijr,qr->ijq',delta,q[positive])/n)/lam[positive],axis=2)/n**3

def ring(p: dict[str, Any], lp: np.ndarray, omit: str = '', incoming: int = 0, one_pair: bool = False) -> dict[str, Any]:
    hops = []
    for axis, direction in [(0,1),(1,1),(0,-1),(1,-1)]:
        matrix = next(row['matrix'] for row in p['paths'] if row['axis']==axis and row['length']==1)
        if direction<0:
            matrix = [[conj(matrix[j][i]) for j in range(4)] for i in range(4)]
        e = len(hops)
        hops.append([(4*e+i,4*((e+1)%4)+j,matrix[i][j]) for i in range(4) for j in range(4) if matrix[i][j]!=ZERO])
    total = ZERO
    derivative = ZERO
    orders = []
    two_pair_weight = F(0)
    split_error = 0.
    min_gap: F | None = None
    ph_seen: set[int] = set()
    for order in permutations(range(4)):
        amps: dict[int,C] = {REF:ONE}
        d_amps: dict[int,C] = {REF:ZERO}
        k = [incoming]*4
        order_two = False
        for depth, edge in enumerate(order):
            k[edge] += 1
            updated: dict[int,C] = {}
            d_updated: dict[int,C] = {}
            for mask, amp in amps.items():
                for create, annihilate, coefficient in hops[edge]:
                    moved = car(mask,create,annihilate)
                    if moved is None:
                        continue
                    dest, sign = moved
                    updated[dest] = add(updated.get(dest,ZERO),scale(mul(amp,coefficient),F(1 if omit=='CAR' else sign)))
                    d_updated[dest] = add(d_updated.get(dest,ZERO),scale(mul(d_amps[mask],coefficient),F(1 if omit=='CAR' else sign)))
            if depth<3:
                for mask in list(updated):
                    # Q projects out the ENTIRE charged onsite sea, for ALL flows.
                    if mask == REF:
                        del updated[mask]
                        del d_updated[mask]
                        continue
                    ph = sum((mask>>(4*x+2+s))&1 for x in range(4) for s in range(2))
                    ph_seen.add(ph)
                    if ph==2:
                        order_two = True
                    if one_pair and ph>1:
                        del updated[mask]
                        del d_updated[mask]
                        continue
                    rho = np.array([((mask>>(4*x))&15).bit_count()-2 for x in range(4)])
                    div = np.array(k)-np.roll(k,1)
                    assert np.array_equal(rho,div)
                    hc = 64*float(rho@lp@rho)
                    full = F(64*sum(x*x for x in k))
                    he = float(full)-hc
                    split_error = max(split_error,abs((he+hc)-float(full)))
                    electric = full
                    if omit=='HC':
                        electric = F(float(he))
                    elif omit=='HE':
                        electric = F(float(hc))
                    denominator = 2*p['onsite']*ph+electric
                    assert denominator>0
                    min_gap = denominator if min_gap is None else min(min_gap,denominator)
                    d_updated[mask] = add(scale(d_updated[mask],-1/denominator),scale(updated[mask],-1/denominator**2))
                    updated[mask] = scale(updated[mask],-1/denominator)
            amps = updated
            d_amps = d_updated
        value = amps.get(REF,ZERO)
        total = add(total,value)
        derivative = add(derivative,d_amps.get(REF,ZERO))
        if order_two:
            two_pair_weight += value[0]
        orders.append(dict(order=list(order),real_exact=str(value[0]),imag_exact=str(value[1]),has_two_PH_intermediate=order_two))
    assert total[1]==0
    return dict(coefficient_exact=str(total[0]),coefficient=float(total[0]),imaginary_exact=str(total[1]),z_derivative_exact=str(derivative[0]),source_metric_character_exact=str(-derivative[0]),temporal_order_count=24,orders=orders,PH_counts_seen=sorted(ph_seen),orders_with_two_PH_total_exact=str(two_pair_weight),minimum_denominator_exact=str(min_gap),HE_plus_HC_fullE_max_float_error=split_error,incoming_plaquette_flow=incoming)

# Every closed four-hop CAR contraction is a directed four-cycle or two 2-cycles.
# Rotate a nonzero-character cycle to a first path touching the target face.
def native_character_enumeration(n: int) -> dict[str, Any]:
    def sid(x: tuple[int,...]) -> int:
        return (x[0]*n+x[1])*n+x[2]
    def path(x: tuple[int,...], axis: int, length: int, sign: int) -> tuple[tuple[int,...],tuple[tuple[int,int],...]]:
        y=list(x); entries=[]
        for _ in range(length):
            if sign<0:
                y[axis]=(y[axis]-1)%n
            entries.append((3*sid(tuple(y))+axis,sign))
            if sign>0:
                y[axis]=(y[axis]+1)%n
        return tuple(y),tuple(entries)
    face = {3*sid((0,0,0)):1,3*sid((1,0,0))+1:1,3*sid((0,1,0)):-1,3*sid((0,0,0))+1:-1}
    matches=[]; examined=0
    def dfs(start: tuple[int,...],x: tuple[int,...],flows: dict[int,int],words: list[tuple[int,int,int]],depth: int) -> None:
        nonlocal examined
        if depth==4:
            examined+=1
            if x==start and {e:v for e,v in flows.items() if v}==face:
                matches.append(words)
            return
        for axis,length,sign in product(range(3),(1,2),(-1,1)):
            y,entries=path(x,axis,length,sign)
            f=flows.copy()
            for e,v in entries:
                f[e]=f.get(e,0)+v
            dfs(start,y,f,words+[(axis,length,sign)],depth+1)
    for start in product(range(n),repeat=3):
        for axis,length,sign in product(range(3),(1,2),(-1,1)):
            y,entries=path(start,axis,length,sign)
            if any(e in face for e,_ in entries):
                dfs(start,y,dict(entries),[(axis,length,sign)],1)
    assert len(matches)==4 and all(all(w[1]==1 for w in words) for words in matches)
    return dict(N=n,examined_four_hop_walks=examined,matching_rotations=len(matches),matched_lengths=[[w[1] for w in words] for words in matches],two_link_paths_contributing=0,face_norm_squared=4,homology=[0,0,0],N4_four_unit_straight_winding_homology=[1,0,0],N8_four_unit_straight_is_closed=False)

def snapshot() -> dict[str,bytes]:
    captured={**CAPTURED,**{name:(ROOT/name).read_bytes() for name in INPUTS}}
    for mod in tuple(sys.modules.values()):
        filename=getattr(mod,'__file__',None)
        if filename:
            path=Path(filename)
            if path.is_file():
                captured['runtime_imports/'+str(path).lstrip('/')]=path.read_bytes()
            cached=getattr(mod,'__cached__',None)
            if cached and Path(cached).is_file():
                captured['runtime_bytecode/'+str(cached).lstrip('/')]=Path(cached).read_bytes()
    return captured

def run(output: Path) -> dict[str,Any]:
    started=time.monotonic()
    captured=snapshot()
    for name,pin in PINS.items():
        assert hashlib.sha256(captured[name]).hexdigest()==pin
    frame,s=exact_frame(captured[INPUTS[0]])
    cfg=json.loads(captured[INPUTS[1]])['input_descriptor']['original_micro_config_UNCHANGED']
    assert cfg['auxiliary_count']==63 and cfg['a']==1/64
    ps=profiles(cfg,frame,s)
    lp=local_coulomb(8)
    rows=[]
    for p in ps:
        actual=ring(p,lp)
        wrongs={label:ring(p,lp,label) for label in ('CAR','HC','HE')}
        single=ring(p,lp,one_pair=True)
        incoming=ring(p,lp,incoming=1)
        assert actual['coefficient']!=wrongs['CAR']['coefficient']
        assert actual['coefficient']!=wrongs['HC']['coefficient']
        assert actual['coefficient']!=single['coefficient']
        assert actual['coefficient']!=incoming['coefficient']
        rows.append(dict(role=p['role'],count=p['count'],onsite_positive_exact=str(p['onsite']),actual=actual,contrary_cases=wrongs,wrong_one_PH_projection=single,incoming_one_loop=incoming))
    coefficient=sum((F(row['actual']['coefficient_exact'])*row['count'] for row in rows),F(0))
    # Whole finite canonical charged sector, all one/two native translates.
    # ||c_i^dag c_j||<=1; complex absolute value<=|Re|+|Im|.
    local_bounds=[2*sum((abs(z[0])+abs(z[1]) for path in p['paths'] for row in path['matrix'] for z in row),F(0)) for p in ps]
    B=8**3*sum((p['count']*bound for p,bound in zip(ps,local_bounds)),F(0))
    delta=2*min(p['onsite'] for p in ps)
    radius=delta/B
    test=min(radius/4,abs(coefficient)*delta**4/(4*B**5))
    remainder=(test**5*B**5/delta**4)/(1-test*B/delta)
    controlled_upper=coefficient*test**4+remainder
    assert controlled_upper<0
    geometry=native_character_enumeration(8)
    assert coefficient<0 and radius<1
    # Exact projection discriminator: QVQ excludes selected-source returns.
    # Abstract method control only; not another microscopic supplier.
    irreducible=-F(1,3)
    reducible=-F(1,3-F(1,2))
    assert irreducible!=reducible
    metric=-sum((F(row['actual']['z_derivative_exact'])*row['count'] for row in rows),F(0))
    result=dict(
        kind='EXACT_FROZEN_NATIVE_BULK_UNIT_PLAQUETTE_FESHBACH_COEFFICIENT_NOT_PHASE',
        scope='Frozen archived binary Phi, identity colour, integer beta0; exact canonical onsite charged sea projection, z=0, N8 ambient. Neutral full frozen canonical ground is unchanged tensor spectator. P retains ALL electric flows. Not stationary interacting ground photon source projection.',
        coefficient_of_positive_oriented_shift_exact=str(coefficient),
        coefficient_of_positive_oriented_shift=float(coefficient),
        formal_incoming_zero_cosine_coefficient=2*float(coefficient),
        formal_incoming_zero_curvature=-2*float(coefficient),
        translation_invariant_potential=False,physical_hopping_bookkeeping_value=1,
        rows=rows,geometry=geometry,neutral_charge=0,neutral_flux_coefficient_exact='0',
        charged_species=64,
        wrong_one_active_plus_one_aux_coefficient=sum(row['actual']['coefficient'] for row in rows),
        wrong_counting_neutral_as_charged_rejected='Neutral has no exp(i k.theta), hence contributes only character0; cannot add65th copy.',
        normalization=dict(lower_order_fold_character_exact='0',
                           fourth_order_source_metric_character_exact=str(metric),
                           fourth_order_source_metric_character=float(metric),
                           definition='S(z)=I-partial_z Sigma(z); exact energy-dependent Schur kernel retains metric. No energy-independent scalar potential or dropped fourth-order normalization claim.'),
        controlled_remainder=dict(scope='Finite N8 entire charged canonical CAR x integer electric Hilbert space; neutral spectator cancels; operator-norm z0 Feshbach Neumann series, not uniform cluster theorem.',
                                  hopping_norm_upper_exact=str(B),QH0Q_gap_lower_exact=str(delta),
                                  open_analytic_radius_exact=str(radius),open_analytic_radius=float(radius),
                                  test_lambda_exact=str(test),remainder_after_order4_upper_exact=str(remainder),
                                  remainder_after_order4_upper=float(remainder),reaches_physical_one=False,
                                  test_lambda=float(test),
                                  controlled_character_upper_exact=str(controlled_upper),
                                  controlled_character_upper=float(controlled_upper),
                                  negative_character_proved_for_zero_less_lambda_le_test=True,
                                  lambda1_bound_status='FAIL: supplied global Neumann bound does not reach1; no physical sign, phase, stiffness or pole admission.'),
        reducible_memory_wrongcase=dict(label='Exact method-only Schur control, not actual-ground measurement',
                                       irreducible_exact=str(irreducible),full_reducible_substitution_exact=str(reducible)),
        arithmetic='Exact Gaussian-rational arithmetic for archived binary Phi and actual binary primitive coefficients/fullE denominators. HE/HC deletion contrary cases use observed floating Fourier Lplus converted to rational, NOT interval enclosures.',
        equations=json.loads(captured[PREFIX+'proof_20261010.json']),
        inputs_sha256={k:hashlib.sha256(captured[k]).hexdigest() for k in INPUTS},
        actual_local_imports_sha256={k:hashlib.sha256(v).hexdigest() for k,v in CAPTURED.items()},
        runtime=dict(executable=sys.executable,python=sys.version,numpy=np.__version__),
        campaign_complete=False,elapsed_seconds=time.monotonic()-started)
    assert all((ROOT/name).read_bytes()==captured[name] for name in INPUTS)
    captured.update(snapshot())
    output.write_text(json.dumps(result,indent=2)+'\n')
    with zipfile.ZipFile(output.with_suffix('.zip'),'x',zipfile.ZIP_DEFLATED) as archive:
        for name,raw in captured.items():
            archive.writestr('executed/'+name,raw)
        archive.writestr('actual.json',output.read_bytes())
    print(json.dumps({k:result[k] for k in ('kind','coefficient_of_positive_oriented_shift','formal_incoming_zero_curvature','elapsed_seconds')},indent=2))
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    run(args.output)
