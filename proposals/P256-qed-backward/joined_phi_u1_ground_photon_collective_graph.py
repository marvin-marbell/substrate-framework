"""Full occupation/electric graph representation and bounded spatial probe controls.

Fixed binary Phi and identity colour section are conditional intermediates only.
The countable graph is NOT replaced by its first shell; no Ritz/pole is produced.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from fractions import Fraction as F
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any
import zipfile

import numpy as np
from threadpoolctl import threadpool_limits
import induced_phi_nonuniform_field_car as field
import joined_phi_u1_stationary_full_phi_fiber as fiber
import joined_phi_u1_stationary_photon as car_source

ROOT = Path(__file__).resolve().parent
PREFIX = 'joined_phi_u1_ground_photon_collective_'
sys.set_int_max_str_digits(0)

PROOF = {
    'new_representation': 'Full site-colour occupation/electric graph, not onsite-band PH grading: a vertex is (neutral fixedN128 mask, active fixedN128 mask, all63 individually fixedN128 masks, Ref wedge4, integer129 cycle character). Every literal forward/reverse native CAR hop changes its species mask with exact CAR sign and translates its electric character by Q^T k. Diagonal is 64|Pi|^2+64rho Lplus rho+all onsite terms. Include -Lambda neutral hopping. All masks and all characters are retained by the graph definition; only a complete first action and explicit second-action escape witnesses are evaluated here.',
    'continuous_extension': 'At arbitrary Phi and SU4, replace each hopping coefficient by the literal ordered endpoint matrix, add NEW Jordan VJ, alignment/plaquette, 4chi Cartesian and 3chi/20 SU4 differential kinetic. This specifies the coupled fullPhi graph-valued differential operator, NOT its ground or evaluated resolvent. Frozen comparison section omits these differential terms and is explicitly intermediate.',
    'seed': 'Local SU4-singlet baryon occupation: active and31aux fill all4 colours on even checkerboard sites,32aux fill all4 on odd sites. Each species has128 particles, total charged density128/site gives rho=0 including Ref4. Neutral128 fills all4 on even sites. Full/empty local colour wedges are SU4 singlets. Identity-link/fixedPhi section is gauge-covariant, not a normalized fullHaar/fullPhi physical wavefunction.',
    'resolvent': 'For the fixed-section selfadjoint full graph H, P fixes only the baryon masks but includes ALL129 electric characters. PHP=D=64Pi^2 plus the seed onsite constant (zero). B=QHP contains ALL charged and neutral hops, without Fourier or occupation cutoff. For Im z=eta>0, ||P(H-z)^-1P-(D-z)^-1|| <= ||B||^2/eta^3 by two exact resolvent identities. A global cylinder bound groups by destination masks, sums rational |Re amplitude|+|Im amplitude| over every Fourier translate, then sums the squared bounds over orthogonal species destinations. Distinct Fourier shifts are orthogonal on one character but need NOT be orthogonal on arbitrary packets: residual norm is not silently used as operator norm. The universal cap2/eta may be smaller. This encloses a nonstationary seed kernel, NOT a same-ground photon measure. Lambda neutral term makes the bound uninformative at charged energies.',
    'wilson': 'F_q=1/8 sum_x exp(-2pi i q.x/4) sin(c_x.Theta), c_x oriented elementary spatial plaquette edge flow. ||F_q||<=8. q is physical spatial lattice momentum, never configuration beta. F is gauge-invariant, continuous periodic and Q-accessible. On the character seed its connected norm^2=1/2; first shell retains all129 components and every64 plaquette translate.',
    'electric_fork': 'Materially different operator: bounded electric Weyl U_q(s)=exp(i s v_q.Pi), real transverse spatial v_q formed by cosine-weighted plaquette flows. It commutes HE and HC but not native matter hopping. On the baryon character seed its connected weight is exactly zero, so move to the normalized COMPLETE charged H-action meson shell, without selecting a loop/PH channel. Electric Weyl connected weight there is 1-|sum_j w_j exp(i s v_q.Pi_j)|^2. This is an executed full-native graph shell physical probe, not a stationary state or full spectral calculation.',
    'current_contact': 'For external edge direction v, each native hop h_k(a)=h_k exp(i a v.k), current i(v.k)h_k, contact -(v.k)^2h_k, and kinetic insertions128v.Pi,128|v|^2. HC uses collective rho, never independently summed copies. These first/contact action norms are evaluated on the full graph seed. Neutral is U1 neutral and receives no edge phase.',
    'unpaid': 'FullPhi/Haar wavefunction and its differential residual, NEW Jordan ground sector, minimizing beta, actual P0, insertion/ground limits and same-ground physical pole remain OPEN. Finite first-shell output is not a finite-H replacement. No Maxwell B2, retuning, root, covariance adoption or no-go from incomingPi.',
}


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def mask_on(sites: np.ndarray, parity: int) -> int:
    return sum(15 << (4*x) for x, pos in enumerate(sites) if int(sum(pos)) % 2 == parity)


def exact_hop(path: dict[str, Any], frame: np.ndarray) -> list[list[tuple[F, F]]]:
    out = []
    for i in range(4):
        row = []
        for j in range(4):
            re, im = F(0), F(0)
            for m in range(4):
                c = complex(path['coefficients'][m]); z = complex(frame[m,i,j])
                cr,ci,zr,zi = map(F,(c.real,c.imag,z.real,z.imag))
                re += cr*zr-ci*zi; im += cr*zi+ci*zr
            row.append((int(path['seam'])*re,int(path['seam'])*im))
        out.append(row)
    return out


def shell(profile: dict[str, Any], native: dict[str, Any], mask: int, charged: bool) -> dict[str, Any]:
    # Exact rational coefficients for the explicitly archived binary Phi section.
    amplitudes: dict[tuple[int, tuple[int, ...]], tuple[F, F]] = {}
    paths = field.native_paths(profile, native)
    for path in paths:
        x,y = int(path['source']),int(path['target'])
        hop = exact_hop(path, native['frame'])
        for forward in (True,False):
            k = np.zeros(192,dtype=int)
            if charged:k[list(path['edges'])] = 1 if forward else -1
            for i in range(4):
                for j in range(4):
                    a,b = (4*x+i,4*y+j) if forward else (4*y+j,4*x+i)
                    moved = car_source.car(mask,a,b)
                    if moved is None:continue
                    dest,sign = moved
                    re,im = hop[i][j]
                    if not forward:im = -im
                    key = (dest,tuple(map(int,k)))
                    old = amplitudes.get(key,(F(0),F(0)))
                    amplitudes[key] = (old[0]+sign*re,old[1]+sign*im)
    rows = []
    norm = F(0)
    for (dest,k),amp in amplitudes.items():
        weight = amp[0]**2+amp[1]**2
        if not weight:continue
        assert dest.bit_count()==128
        flow = np.array(k,dtype=float)
        rho = np.array([((dest>>(4*x))&15).bit_count()-((mask>>(4*x))&15).bit_count() for x in range(64)])
        if charged:assert np.array_equal(native['D']@flow,rho)
        norm += weight
        rows.append(dict(mask=dest,k=flow,rho=rho,weight=weight,amp=amp))
    by_mask: dict[int, F] = defaultdict(F)
    for row in rows:by_mask[row['mask']] += abs(row['amp'][0])+abs(row['amp'][1])
    cylinder_upper = sum((value**2 for value in by_mask.values()),F(0))
    return dict(rows=rows,norm=norm,cylinder_upper=cylinder_upper,paths=len(paths),two=int(sum(len(p['edges'])==2 for p in paths)),seams=int(sum(p['seam']<0 for p in paths)))


def plaquettes(native: dict[str, Any]) -> np.ndarray:
    sites=native['sites'];lookup={tuple(map(int,x)):i for i,x in enumerate(sites)};out=[]
    for x,pos in enumerate(sites):
        xp=lookup[tuple(map(int,(pos+np.array([1,0,0]))%4))]
        yp=lookup[tuple(map(int,(pos+np.array([0,1,0]))%4))]
        c=np.zeros(192,dtype=int);c[3*x]=1;c[3*xp+1]=1;c[3*yp]=-1;c[3*x+1]=-1;out.append(c)
    return np.asarray(out)


def run(output: Path) -> dict[str, Any]:
    started=time.monotonic()
    names=['induced_phi_uniform_selfconsistent_full_normal_quantum_final_root_actual_receipt.json','induced_phi_uniform_selfconsistent_full_normal_quantum_final_actual_receipt.json','induced_phi_nonuniform_fermionic_source_unitary_resumed_actual_receipt.json','joined_phi_u1_stationary_canonical_photon_divisionfree_actual_receipt_20261010.json','joined_phi_u1_stationary_jordan_parent_packet_20261010.json','joined_phi_u1_matrix_jordan_flat_actual_20261010_receipt.json','joined_phi_u1_matrix_jordan_supplier_receipt_20261010.json']
    inputs={name:(ROOT/name).read_bytes() for name in names}
    for name in names[:3]:assert sha(inputs[name])==fiber.PINNED[name]
    pars=json.loads(inputs[names[0]])['actual_stationary_root']['parameters']
    g=[float((F(pars[k]['lower'])+F(pars[k]['upper']))/2) for k in ('g0','gs')]
    data=field.load_mean(g_override=[g[0],g[1],g[1],g[1]],mean_descriptor=json.loads(inputs[names[2]])['input_descriptor'])
    native=data['native'];qmat=native['Q'];d=native['D'];lp=native['Lplus'];pt=qmat@qmat.T
    assert qmat.shape==(192,129) and np.linalg.norm(d@qmat)<1e-10
    masks=[mask_on(native['sites'],p) for p in (0,1)]
    charged=[shell(data['profiles'][1],native,masks[0],True),shell(data['profiles'][2],native,masks[0],True),shell(data['profiles'][2],native,masks[1],True)]
    neutral=shell(data['profiles'][0],native,masks[0],False)
    loads=[1,31,32];charged_norm=sum((n*s['norm'] for n,s in zip(loads,charged)),F(0));neutral_norm=neutral['norm']
    charged_bound=sum((n*s['cylinder_upper'] for n,s in zip(loads,charged)),F(0))
    neutral_bound=neutral['cylinder_upper']
    assert all(m.bit_count()==128 for m in masks)
    rho=sum(n*np.array([((m>>(4*x))&15).bit_count()-2 for x in range(64)]) for n,m in zip([32,32],masks))
    assert np.array_equal(rho,np.zeros(64))
    loops=plaquettes(native);assert np.array_equal(d@loops.T,np.zeros((64,64)))
    incoming=.37*pt[:,0];assert np.linalg.norm(d@incoming)<1e-12
    spectral=[];fork=[];allrows=[(n,r) for n,s in zip(loads,charged) for r in s['rows']]
    assert charged_norm>0 and neutral_norm>0
    totalw=float(charged_norm)
    for qi in ((0,0,0),(1,0,0),(2,0,0)):
        phase=np.exp(-2j*np.pi*(native['sites']@np.array(qi))/4)
        energies=np.array([64*np.dot(incoming+sign*c,incoming+sign*c)-64*np.dot(incoming,incoming) for c in loops for sign in (1,-1)])
        spectral.append(dict(q_integer=list(qi),q_physical_over_inverse_a=[2*math.pi*k/4 for k in qi],norm_squared_exact='1/2',external_occupation_residual_squared_exact=str(charged_norm),external_neutral_residual_squared_over_Lambda2_exact=str(neutral_norm),global_cylinder_B_squared_upper_charged_exact=str(charged_bound),global_cylinder_B_squared_upper_neutral_over_Lambda2_exact=str(neutral_bound),kinetic_energy_transfers=energies.tolist(),compressed_nodes_are_not_poles=True,resolvent_bound_formula='min(1/eta, (charged_bound+Lambda^2*neutral_bound)/(2*eta^3)) for scalar F_q kernel; NOT ground kernel'))
        scaled_bound = neutral_bound + charged_bound/F(2**16384)
        spectral[-1]['evaluated_scaled_resolvent_enclosures'] = [
            dict(z_over_Lambda=[0,gamma],Lambda_times_seed_Wilson_resolvent_centre=[0,F(1,2*gamma).__float__()],
                 analytic_radius_exact=str(min(F(1,gamma),scaled_bound/F(2*gamma**3))+F(1024,2**8192*gamma**2)),
                 centre_kinetic_roundoff_price='Exact |energy transfer|<=1024 for |incomingPi|<1 and |plaquette|=2; additional centre radius1024/(Lambda gamma^2)',
                 scope='FixedPhi ALL-CAR full-H complex-resolvent enclosure, not low-energy pole')
            for gamma in (1,100,10000)]
        v=np.real(phase)@loops/8;v=v/(np.linalg.norm(v) or 1);assert np.linalg.norm(d@v)<1e-12
        s=1.3
        characteristic=sum(n*float(r['weight'])*np.exp(1j*s*np.dot(v,incoming+pt@r['k'])) for n,r in allrows)/totalw
        weight=1-abs(characteristic)**2
        current=sum(n*float(r['weight'])*np.dot(v,r['k'])**2 for n,r in allrows)
        contact=sum(n*float(r['weight'])*np.dot(v,r['k'])**4 for n,r in allrows)
        wrong_current=sum(n*float(r['weight'])*np.dot(v,r['k'])**2 for n,r in allrows if np.count_nonzero(r['k'])==1)
        fork.append(dict(q_integer=list(qi),Weyl_parameter=s,seed_connected_weight=0.,complete_charged_shell_connected_weight=float(weight),charged_shell_characteristic=[float(characteristic.real),float(characteristic.imag)],current_action_norm_squared=current,contact_action_norm_squared=contact,kinetic_current=128*float(v@incoming),kinetic_contact=128*float(v@v),wrong_drop_two_link_current=wrong_current,roundoff_scope='Floating structural control; q0 cosine-weighted curl sums exactly to zero, so its Weyl is identity and any tiny negative connected weight is summation roundoff.'))
    assert max(row['complete_charged_shell_connected_weight'] for row in fork)>1e-4
    # Independent adverse full129 electric + full64 Coulomb physical identity.
    a=charged[1]['rows'][0];b=next(r for r in charged[2]['rows'] if np.dot(a['k'],r['k'])!=0)
    k=a['k']+b['k'];r=a['rho']+b['rho']
    energy=64*(np.dot(incoming+pt@k,incoming+pt@k)+r@lp@r)
    fullE=incoming+pt@k+d.T@lp@r
    identity_error=abs(energy-64*np.dot(fullE,fullE))
    independent=64*(np.dot(incoming,incoming)+np.dot(pt@a['k'],pt@a['k'])+np.dot(pt@b['k'],pt@b['k'])+a['rho']@lp@a['rho']+b['rho']@lp@b['rho'])
    zeroPi=64*(np.dot(pt@k,pt@k)+r@lp@r)
    assert identity_error<1e-8 and abs(energy-independent)>1 and abs(energy-zeroPi)>1
    # Actual second-action escape: distinct auxiliary copies can undergo the
    # same hop. This has collective rho=2rho_one and Pi=incoming+2Qk.
    two_rho=2*a['rho'];two_k=2*a['k'];two_energy=64*(np.dot(incoming+pt@two_k,incoming+pt@two_k)+two_rho@lp@two_rho)
    wrong_two=64*(np.dot(incoming,incoming)+2*np.dot(pt@a['k'],pt@a['k'])+2*a['rho']@lp@a['rho'])
    assert abs(two_energy-wrong_two)>1
    double_norm=F(31*30)*a['weight']**2 # ordered distinct-copy H^2 contribution, positive witness only
    # Wilson kinetic contact checked independently from exact character shifts.
    c=loops[0];proper=64*(np.dot(incoming+c,incoming+c)-np.dot(incoming,incoming));wrong=float(64*np.dot(c,c))
    assert abs(proper-wrong)>1
    np.savez_compressed(output.with_suffix('.npz'),Phi_binary_section=native['frame'],Q=qmat,D=d,Lplus=lp,incomingPi=incoming,Wilson_64_loops=loops,sites=native['sites'])
    captured={**field.CAPTURED,Path(__file__).name:Path(__file__).read_bytes(),Path(fiber.__file__).name:Path(fiber.__file__).read_bytes(),Path(car_source.__file__).name:Path(car_source.__file__).read_bytes()}
    for module in tuple(sys.modules.values()):
        filename=getattr(module,'__file__',None)
        if filename:
            path=Path(filename).resolve()
            if path.suffix=='.py' and path.parent==ROOT:captured.setdefault(path.name,path.read_bytes())
    result=dict(kind='NEW_FULL_OCCUPATION_ELECTRIC_GRAPH_WITH_BOUNDED_SPATIAL_WILSON_AND_ELECTRIC_WEYL_FORK_NOT_GROUND',proof=PROOF,law=dict(N=4,V=64,canonical_charged_each=128,charged_copies=64,auxiliary_copies=63,neutral_count=128,Ref=4,HE=64,HC=64,Lambda='2^8192',chi='2^-4096',Jordan='NEW_NOT_ADOPTED'),scope='Fixed explicit binary comparison Phi and identity SU4 covariant section; full graph retained conceptually, complete H action evaluated. Not fullPhi physical ground.',source_sha256={k:sha(v) for k,v in captured.items()},input_sha256={k:sha(v) for k,v in inputs.items()},caps=dict(electric_modes=129,sites=64,all_charged_copies=64,full_fixedN_CAR_modes_per_species=256,occupation_cutoff=None,Fourier_cutoff=None,evaluated_actions=1,explicit_second_action_escape=True),seed_shell=dict(charged_norm_squared_exact=str(charged_norm),neutral_norm_squared_over_Lambda2_exact=str(neutral_norm),roles=[dict(role=p['role'],copies=n,first_action_destinations=len(s['rows']),all_native_paths=s['paths'],two_link_paths=s['two'],AP_seams=s['seams'],norm_squared_exact=str(s['norm'])) for p,n,s in zip([data['profiles'][1],data['profiles'][2],data['profiles'][2]],loads,charged)],neutral_first_action_destinations=len(neutral['rows']),neutral_scale='-Lambda',seed_HE=float(64*incoming@incoming),seed_HC=0,onsite_seed_energy_exact=0,ground_residual_paid='Complete fixed-section H residual squared = charged_norm + 2^16384 neutral_norm; no truncation tail at first action',ground_residual_unpaid='FullPhi and Haar differential residual: seed is not a wavefunction there.'),spatial_Wilson=spectral,tested_electric_Weyl_fork=fork,independent_adverse=dict(full_HE_HC_fullE_identity_error=identity_error,proper_collective_energy=energy,wrong_independent_copies_energy=independent,wrong_zero_incomingPi_energy=zeroPi,Wilson_incoming_transfer=proper,wrong_zeroPi_Wilson_transfer=wrong,distinct31aux_second_action_norm_witness_exact=str(double_norm),collective_doublehop_energy=two_energy,wrong_independent_doublehop_energy=wrong_two,first_shell_not_closed=True),Fourier_tail=dict(probe='ALL64 plaquette character shifts exact; none discarded',first_H_action='ALL native one/two-link character shifts exact; none discarded',higher_resolvent='No finite compression claimed. Full complex-resolvent bound above pays B globally; fails to resolve charged scale due neutral Lambda residual. No spectral external-gap bound.'),decision='Baryon Fourier-cylinder route fails ground-residual admission decisively, not a Pi no-go. Tested electric-Weyl/full-native meson fork has nonzero connected spatial weight while seed has none; hierarchy escapes after second action. Next decision is a fullPhi neutral-ground-adapted correlated graph wavefunction, not more loop Ritz or vfit.',OPEN=['fullPhi NEW Jordan stationary P0','beta minimizer and degeneracy','same-ground photon pole/dispersion','external low-energy sector and fullPhi tails'],runtime=dict(executable=sys.executable,python=sys.version,numpy=np.__version__),elapsed_seconds=time.monotonic()-started)
    result['frozen_bosonic_constant'] = 'NEW Jordan VJ and alignment/plaquette at the frozen section are common additive constants, subtracted consistently with Eseed. Cartesian/Haar dynamics have NOT been subtracted from a claimed fullPhi result.'
    assert all((ROOT/name).read_bytes()==raw for name,raw in captured.items())
    assert all((ROOT/name).read_bytes()==raw for name,raw in inputs.items())
    with zipfile.ZipFile(output.with_suffix('.zip'),'w',zipfile.ZIP_DEFLATED) as z:
        for name,raw in captured.items():z.writestr('executed_source/'+name,raw)
        for name,raw in inputs.items():z.writestr('inputs/'+name,raw)
        z.writestr('inputs/fixed_binary_section.npz',output.with_suffix('.npz').read_bytes())
    output.write_text(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    with threadpool_limits(limits=1):receipt=run(args.output)
    print(json.dumps(dict(kind=receipt['kind'],elapsed_seconds=receipt['elapsed_seconds'],charged_residual=receipt['seed_shell']['charged_norm_squared_exact'],neutral_residual_over_Lambda2=receipt['seed_shell']['neutral_norm_squared_over_Lambda2_exact'],ground='OPEN'),indent=2))
