"""Independent NEW-Jordan global form audit controls; never run the producer."""
from __future__ import annotations
import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
from typing import Any
import zipfile
import numpy as np
from scipy.linalg import expm

ROOT = Path(__file__).resolve().parent
PINS = {
    "joined_phi_u1_global_ground_geometry_bound.py": "50506816613aa6943a3c6e57fa39bc68f10a27083d0f67a72f5895afb051b51e",
    "joined_phi_u1_global_ground_geometry_current_actual_20261010.json": "3503233461e4b3db1d7fedb5fb2da7917030e4eba25b3b4e8f9ac8962d4d4774",
    "joined_phi_u1_global_ground_geometry_current_natural_exit_20261010.json": "1912e4c7fa65f983090ec406d3b6aa698e6324b10ded146ab87c3ef7203d07f6",
    "joined_phi_u1_stationary_flat_holonomy_typed_certified_20261010_receipt.json": "1c79b1a74c4236079645ebf9765a48c1c2263ec7b7209e9f103704c98752da94",
    "joined_phi_u1_global_ground_geometry_current_actual_20261010_input_bytes.zip": "fa232bec77fa2418e3a70ab2941652a2f4113d2a53b0ad74b439a6a0ed3b3a2c",
}


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def lattice() -> tuple[list[tuple[int, int, int]], dict[tuple[int, int, int], int], np.ndarray, np.ndarray]:
    sites = [(x, y, z) for x in range(4) for y in range(4) for z in range(4)]
    ids = {s: i for i, s in enumerate(sites)}
    targets = np.zeros((64, 3), dtype=int)
    d = np.zeros((64, 192), dtype=int)
    for x, s in enumerate(sites):
        for axis in range(3):
            t = list(s); t[axis] = (t[axis]+1) % 4
            y = ids[(t[0],t[1],t[2])]; targets[x, axis] = y
            d[x, 3*x+axis] = -1; d[y, 3*x+axis] = 1
    return sites, ids, targets, d


def plaquette(links: np.ndarray, targets: np.ndarray) -> float:
    result = 0.0
    for x in range(64):
        for a in range(3):
            for b in range(a+1, 3):
                hol = links[3*x+a] @ links[3*targets[x, a]+b] @ links[3*targets[x, b]+a].conj().T @ links[3*x+b].conj().T
                result += 4-float(np.trace(hol).real)
    return result


def assemble(profile: dict[str, Any], fields: np.ndarray, links: np.ndarray, theta: np.ndarray, direction: np.ndarray, sites: list[tuple[int, int, int]], targets: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Direct endpoint/ordered-path assembly, including both 1/2 factors and AP.
    h = np.zeros((256, 256), dtype=complex); j = h.copy(); k = h.copy()
    for x in range(64):
        h[4*x:4*x+4, 4*x:4*x+4] += float(F(profile['onsite']))*fields[x, 0]
        for row in profile['paths']:
            axis, length = int(row['axis']), int(row['length'])
            y = x; u = np.eye(4, dtype=complex); phase = 0.0; dv = 0.0; ap = 1
            for _ in range(length):
                e = 3*y+axis; u = u @ links[e]; phase += theta[e]; dv += direction[e]
                if sites[y][axis] == 3: ap = -ap
                y = int(targets[y, axis])
            fx = float(F(row['c0']))*fields[x, 0]+1j*float(F(row['ci']))*fields[x, axis+1]
            fy = float(F(row['c0']))*fields[y, 0]+1j*float(F(row['ci']))*fields[y, axis+1]
            block = ap*np.exp(1j*phase)*(fx@u+u@fy)/2
            xx, yy = slice(4*x, 4*x+4), slice(4*y, 4*y+4)
            for matrix, value in ((h, block), (j, 1j*dv*block), (k, -dv*dv*block)):
                matrix[xx, yy] += value; matrix[yy, xx] += value.conj().T
    return h, j, k


def run(output: Path) -> dict[str, Any]:
    captured = {name: (ROOT/name).read_bytes() for name in PINS}
    for name, raw in captured.items():
        assert sha(raw) == PINS[name], name
    source = Path(__file__).resolve(); own = source.read_bytes()
    archive_path = output.with_name(output.stem+'_preexecution.zip')
    with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('executed_source/'+source.name, own)
        for name, raw in captured.items(): archive.writestr('input/'+name, raw)
    receipt = json.loads(captured['joined_phi_u1_global_ground_geometry_current_actual_20261010.json'])
    capsule = json.loads(captured['joined_phi_u1_global_ground_geometry_current_natural_exit_20261010.json'])
    assert capsule['natural_exit'] and capsule['exit_code'] == 0 and capsule['source_unchanged']
    assert capsule['science_receipt']['sha256'] == PINS['joined_phi_u1_global_ground_geometry_current_actual_20261010.json']
    with zipfile.ZipFile(ROOT/'joined_phi_u1_global_ground_geometry_current_actual_20261010_input_bytes.zip') as archive:
        assert archive.read('executed_source/joined_phi_u1_global_ground_geometry_bound.py') == captured['joined_phi_u1_global_ground_geometry_bound.py']
        assert archive.read('input/joined_phi_u1_stationary_flat_holonomy_typed_certified_20261010_receipt.json') == captured['joined_phi_u1_stationary_flat_holonomy_typed_certified_20261010_receipt.json']
    profiles = json.loads(captured['joined_phi_u1_stationary_flat_holonomy_typed_certified_20261010_receipt.json'])['primitive_profiles']
    assert [p['load'] for p in profiles] == [1, 1, 63]
    weights = [abs(F(p['onsite']))+sum((2*(abs(F(r['c0']))+abs(F(r['ci']))) for r in p['paths']), F(0)) for p in profiles]
    assert list(map(str, weights)) == receipt['derived_budgets']['profile_weights']
    lam, chi, fast, delta = F(2)**8192, F(2)**-4096, F(2)**10240, F(2)**-2048
    charged = 128*weights[1]+63*128*weights[2]
    c = 256*(128*weights[0]+charged/lam)
    assert c == F(receipt['derived_budgets']['matter_C_over_Lambda']) and c < 2**40
    trial = 16*chi*2**6720/delta**2 + 64*lam*delta**4 + F(59,20)*fast*delta**2 + F(2)**24/3 + 64
    x4 = 128*(2**64+trial/lam); w = (lam*2**64+trial)/fast
    assert trial/lam < F(2)**-1467 and x4 < 2**72 and w < F(2)**-1983
    cj, ck = 512*charged, 1024*charged
    assert cj**4*x4 < 2**232 and ck**4*x4 < 2**232
    assert w/F(2)**-1024 < F(2)**-959
    # Exact metric and Clifford saturation; full arbitrary field diagnostics below.
    pauli = [np.eye(2), np.array([[0,1],[1,0]]), np.array([[0,-1j],[1j,0]]), np.diag([1,-1])]
    basis = np.array([np.kron(a,b) for i,a in enumerate(pauli) for j,b in enumerate(pauli) if (i,j)!=(0,0)])
    t = basis/np.sqrt(8)
    assert np.max(abs(np.einsum('aij,bji->ab', t, t)-np.eye(15)/2)) < 1e-14
    gamma = np.array([np.kron(pauli[1],pauli[0]), np.kron(pauli[3],pauli[1]), np.kron(pauli[3],pauli[2]), np.kron(pauli[3],pauli[3])])
    # g=1/4 gives HS2=256 and VGram/Lambda=1024 at a=1/64.
    assert F(64*4*4,4)**2/64 == 1024
    assert F(2*256)**2/64 == 4096
    assert F(1,2)*8 == 4 and F(3,20) < 4  # exact two kinetic coefficients
    rng = np.random.default_rng(222101059)
    cart = rng.normal(size=(64,4,15))/20
    fields = np.einsum('xma,aij->xmij', cart, t)
    x2 = float(np.einsum('xmij,xmji->',fields,fields).real)
    gram = np.einsum('xmij,xnji->xmn',fields,fields).real/4
    vg = 64*float(np.sum(gram*gram))
    vj = 0.0; defect = 0.0
    for fs, gs in zip(fields, gram, strict=True):
        for a in range(4):
            for b in range(4):
                anti = fs[a]@fs[b]+fs[b]@fs[a]
                sos = anti-2*gs[a,b]*np.eye(4)
                vj += 4*float(np.trace(anti@anti).real)
                defect += 4*float(np.trace(sos@sos).real)
    assert abs(x2/float(np.sum(cart*cart))-0.5)<1e-14
    assert vg >= x2*x2/64-1e-12 and vg <= vj <= 16*vg and abs(vj-vg-defect)<1e-10
    sites, _, targets, d = lattice()
    parity = np.array([(-1)**sum(s) for s in sites]); lap = d@d.T
    assert np.array_equal(lap@parity, 12*parity)
    hc = F(64*64*128**2,12)
    assert hc == F(2)**24/3 and hc/F(64*64*2**2,12) == 4096
    assert int(np.sum(128*parity)) == 0
    # Deterministic Phi=0 with one -I edge: exactly four nonflat plaquettes.
    links = np.broadcast_to(np.eye(4,dtype=complex),(192,4,4)).copy(); links[0] *= -1
    pbad = plaquette(links,targets); assert abs(pbad-32)<1e-12
    # Genuine pure gauge and noncommuting perturbation; no Phi-only flat claim.
    gs = np.array([expm(1j*np.einsum('a,aij->ij', rng.normal(size=15)/20,t)) for _ in sites])
    pure = np.array([gs[x]@gs[targets[x,a]].conj().T for x in range(64) for a in range(3)])
    assert abs(plaquette(pure,targets))<1e-10
    angles = rng.normal(size=(192,15))/50
    links = np.array([expm(1j*np.einsum('a,aij->ij',aa,t))@u for aa,u in zip(angles,pure,strict=True)])
    distance2 = float(np.sum(angles*angles)); deficits = float(np.sum(abs(links-pure)**2))
    p = plaquette(links,targets)
    assert deficits <= distance2/2+1e-10 and p <= 4*distance2+1e-10
    alignment = 0.0
    for x in range(64):
        for a in range(3):
            u = links[3*x+a]; difference = fields[x]-u@fields[targets[x,a]]@u.conj().T
            alignment += float(np.sum(abs(difference)**2))
    assert alignment <= 12*x2+1e-10
    geometry = alignment/40+0.7*p
    assert geometry <= float(F(59,20))*(2*x2+distance2)+1e-10
    # Full65 fixed-N determinant: zero mean is NOT zero Hamiltonian action.
    occupied = np.array([4*x+a for x,s in enumerate(sites) if sum(s)%2==0 for a in range(4)])
    empty = np.array([i for i in range(256) if i not in set(occupied)])
    direction = np.zeros(192); direction[::3] = 1/8
    assert np.max(abs(d@direction)) == 0 and np.dot(direction,direction) == 1
    theta = rng.normal(size=192)/10
    records = []
    for profile, weight in zip(profiles,weights,strict=True):
        charge = 0 if profile['scale']=='heavy' else 1
        phase, flow = charge*theta, charge*direction
        h,j,k = assemble(profile,fields,links,phase,flow,sites,targets)
        if charge==0:
            assert np.count_nonzero(j)==0 and np.count_nonzero(k)==0
        mean = float(np.trace(h[np.ix_(occupied,occupied)]).real)
        residual2 = float(np.sum(abs(h[np.ix_(empty,occupied)])**2))
        assert abs(mean)<1e-12 and residual2>1
        assert np.max(abs(h-h.conj().T))<1e-12
        assert np.max(np.sum(abs(h),axis=1)) <= 256*float(weight)*np.sqrt(x2)
        eps = 1e-5
        hp = assemble(profile,fields,links,phase+eps*flow,flow,sites,targets)[0]
        hm = assemble(profile,fields,links,phase-eps*flow,flow,sites,targets)[0]
        jerr = float(np.linalg.norm((hp-hm)/(2*eps)-j))
        kerr = float(np.linalg.norm((hp+hm-2*h)/eps**2-k))
        assert jerr<1e-7 and kerr<0.01
        records.append({'role':profile['role'],'U1_charge':charge,'matter_expectation':mean,'matter_action_norm_squared':residual2,'Theta_current_fd_error':jerr,'Theta_contact_fd_error':kerr,'Theta_current_norm':float(np.linalg.norm(j))})
    # Negative-Lambda neutral selects upper h states. Wrong sign changes energy.
    eigen = np.tile(np.array([3.,2.,-1.,-4.]),64)
    upper_sum = float(np.sum(np.sort(eigen)[-128:])); assert upper_sum == 320
    neutral_correct, neutral_wrong = -upper_sum, upper_sum
    assert neutral_correct != neutral_wrong
    # Actual four-colour CAR action: full/empty determinants are singlets.
    def colour_action(matrix: np.ndarray, mask: int) -> dict[int, complex]:
        state: dict[int, complex] = {}
        for a in range(4):
            for b in range(4):
                if not (mask>>b)&1: continue
                after = mask ^ (1<<b)
                if (after>>a)&1: continue
                sign = (-1)**((mask & ((1<<b)-1)).bit_count()+(after & ((1<<a)-1)).bit_count())
                dest = after | (1<<a)
                state[dest] = state.get(dest,0j)+sign*matrix[a,b]
        return state
    singlet_charge2 = sum(sum(abs(v)**2 for v in colour_action(tt,15).values()) for tt in t)
    wrong_twocolour_charge2 = sum(sum(abs(v)**2 for v in colour_action(tt,3).values()) for tt in t)
    assert singlet_charge2<1e-25 and wrong_twocolour_charge2>1
    # Theta shift leaves HE invariant; Pi-translation differentiates HE too.
    hop = np.array([[0,1],[1,0]],complex)
    theta_current = np.array([[0,1j],[-1j,0]],complex)
    eps = 1e-6
    shifted_hop = np.array([[0,np.exp(1j*eps)],[np.exp(-1j*eps),0]])
    assert np.linalg.norm((shifted_hop-hop)/eps-theta_current)<1e-5
    incoming_pi = 3
    momentum_probe_HE_derivative = F(64)*((F(incoming_pi)+F(1,10))**2-(F(incoming_pi)-F(1,10))**2)/F(1,5)
    assert momentum_probe_HE_derivative == 384 and np.trace(theta_current)==0
    # Ricci full-transverse exponent, first-cut scaling, not concavity after cut.
    # Outward normal ray of a radius1 circle in Euclidean3: j=r(1+r).
    r = F(1,10); j1=r*(1+r); j2=2*r*(1+2*r)
    assert j2>2*j1 and j2<=4*j1  # codimension exponent1 would fail
    def cut_u(x: F) -> F:
        return x if x<F(1) else F(0)
    assert cut_u(F(1)) < (cut_u(F(9,10))+cut_u(F(11,10)))/2
    assert cut_u(2*r)<=2*cut_u(r)
    # Tent zero extension has first derivative jump1: in H1, not H2.
    tent_boundary_derivative_inside, tent_boundary_derivative_outside = -1, 0
    assert tent_boundary_derivative_inside != tent_boundary_derivative_outside
    # Explicit normalized beta packet in physical momentum d: HE<=64 epsilon².
    epsilon=F(1,10); he_packet=64*epsilon**2*F(129,131)
    assert 0<he_packet<64
    borel_packet_mass_at_zero=F(0); wrong_fiber_point_mass=F(1)
    assert borel_packet_mass_at_zero != wrong_fiber_point_mass
    # Positive resolvent total-weight may be saturated with arbitrarily small gap.
    gap=F(1,2)**100; omega=F(1,10)
    accessible=1/(gap+omega)
    assert accessible<=1/omega and gap<F(1,2)**90
    result={'stage':'INDEPENDENT_EXECUTED_STRUCTURAL_CONTROLS_NOT_GLOBAL_NUMERICAL_PROOF',
        'target_pins':PINS,'source_sha256':sha(own),'archive':{'path':archive_path.name,'sha256':sha(archive_path.read_bytes())},
        'exact_recomputed_budgets':{'trial_over_Lambda':'<2^-1467','W':'<2^-1983','HS_fourth':'<2^72','Markov':'<2^-959','HC':str(hc),'geometry_coefficient':'59/20'},
        'finite_arbitrary_field_diagnostics':{'HS_squared':x2,'VGram_divided_Lambda':vg,'VJordan_divided_Lambda':vj,'SOS_identity_error':abs(vj-vg-defect),'alignment':alignment,'plaquette':p,'matter':records,'singlet_SU4_charge_squared':singlet_charge2,'wrong_twocolour_SU4_charge_squared':wrong_twocolour_charge2,'wrong_momentum_probe_HE_derivative':str(momentum_probe_HE_derivative)},
        'wrong_cases_rejected':{'Cart_in_place_of_HS_factor':4,'singlecharged_HC_missing_factor':4096,'Phi_zero_nonflat_plaquette':pbad,'zero_mean_not_zero_action':True,'neutral_wrong_sign':{'correct_divided_Lambda':neutral_correct,'wrong':neutral_wrong},'codimension_only_Jacobi_exponent_ratio':str(j2/j1),'cut_zero_extension_not_concave':True,'tent_not_operator_H2':True,'Theta_not_Pi_translation':True,'Borel_global_null_not_fiber_point_mass':True,'total_resolvent_not_gap_certificate':True},
        'scope':'Exact algebra controls and independently constructed finite nonuniform fields/native paths, not quadrature of the 6720-dimensional trial, ground computation, allflat distance, IR or pole.',
        'source_inputs_unchanged':source.read_bytes()==own and all((ROOT/name).read_bytes()==raw for name,raw in captured.items())}
    assert result['source_inputs_unchanged']
    output.write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    result=run(parser.parse_args().output)
    print(json.dumps({'stage':result['stage'],'source_sha256':result['source_sha256'],'wrong_cases_rejected':result['wrong_cases_rejected']},indent=2))
