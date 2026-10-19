"""Independent lepton B-Q-Q-Bdag experiment and primitive bridge charge flux.

Frozen pure reproject/Lanczos/numeric loader helpers are reused, not owner CLI,
protocol state, evolved checkpoints, model edits or a second H compilation.
"""
from __future__ import annotations
import itertools
import json
import math
from pathlib import Path
import sys
import time
from typing import Any
import numpy as np
from scipy import sparse
from scipy.linalg import expm
from rich_common_field_interaction_20261015_direct import NumericCache, pinned_sources, digest
from rich_common_field_interaction_20261015_transition import reproject
from rich_common_field_interaction_20261015_consume import lanczos

ROOT = Path(__file__).resolve().parent


def pair(z: complex) -> list[float]:
    return [float(np.real(z)), float(np.imag(z))]


def partitions(values: tuple[int, ...]) -> Any:
    if not values:
        yield []
        return
    for rest in partitions(values[1:]):
        yield [(values[0],)] + rest
        for k in range(len(rest)):
            yield rest[:k] + [(values[0],) + rest[k]] + rest[k+1:]


def graded_cumulant(moments: dict[tuple[int, ...], complex]) -> tuple[complex, list[Any]]:
    """All 15 set partitions, with Koszul sign, not pair-only subtraction."""
    result = 0j
    terms = []
    for blocks in partitions((0, 1, 2, 3)):
        flattened = [x for block in blocks for x in block]
        odd = [x for x in flattened if x in (0, 3)]
        sign = -1 if odd == [3, 0] else 1
        coefficient = sign * (-1)**(len(blocks)-1) * math.factorial(len(blocks)-1)
        value = coefficient * math.prod(moments[tuple(sorted(block))] for block in blocks)
        result += value
        terms.append(dict(blocks=blocks, coefficient=coefficient, value=pair(value)))
    return result, terms


def main() -> None:
    request_path, output = Path(sys.argv[1]), Path(sys.argv[2])
    req = json.loads(request_path.read_text())
    for pin in req['frozen_files']:
        if digest(ROOT/pin['file']) != pin['sha256']:
            raise ValueError('Frozen input changed: '+pin['file'])
    pinned_sources(req)
    cache = NumericCache(req)
    from rich_common_carrier_motion_20261015_model import build, fibre, law, observable, extra_operator, source_matrix
    from rich_common_frame_orbit_20261015_model import FormOperator
    from rich_frame_field_join_20261015_calculate import m_isometry
    from rich_common_carrier_motion_20261015_operators import field_observable
    start = time.monotonic()
    model = build(json.loads((ROOT/req['physical_input']).read_text()))
    result: dict[str, Any] = dict(kind='INDEPENDENT_WAVE27_BOUNDED_SAME_LAW_EXPERIMENT', input_sha256=digest(request_path),
        times=req['times'], branch='lepton', physical_M=.5, action_audits=[], rows=[],
        frozen_reuse='Pure reproject, Lanczos, NumericCache loader; original neutral_prepared_t0 only; no owner evolved states/CLI',
        angular_scope='Neutral J0 endpoints and J1 intermediate; charged Jhalf endpoints, Jhalf/Jthreehalf after first leg. After second leg Jhalf/Jthreehalf/Jfivehalf are declared; only Jhalf overlaps receiver. All multiplicities retained.')
    def save(stage: str) -> None:
        result['stage'] = stage
        result['seconds'] = time.monotonic()-start
        output.write_text(json.dumps(result, indent=2)+'\n')
        print(stage, flush=True)
    full = {name: m_isometry(model, name, 0. if name=='neutral' else .5) for name in ('neutral','lepton')}
    def cg(name: str, j: float) -> Any:
        return (full[name].conj().T@fibre(model,name,j,0. if name=='neutral' else .5)).tocsr()
    n0,n1,c0,c1 = cg('neutral',0.),cg('neutral',1.),cg('lepton',.5),cg('lepton',1.5)
    cu = sparse.hstack([c0,c1],format='csr')
    z = n0.conj().T@cache.reference
    hn,hv,hc,hu = [reproject(cache.operator(label),v) for label,v in
                  [('neutral_H_zero',n0),('neutral_H_zero',n1),('lepton_H_zero',c0),('lepton_H_zero',cu)]]
    qn = [reproject(cache.operator('neutral_Q'+str(a)),n1,n0) for a in (2,5)]
    qmean = [reproject(cache.operator('neutral_Q'+str(a)),n0) for a in (2,5)]
    qc = [reproject(cache.operator('lepton_Q'+str(a)),cu,c0) for a in (2,5)]
    sin = reproject(cache.operator('lepton_source_site0'),c0,n0)
    sout = reproject(cache.operator('lepton_source_site2'),c0,n0)
    incoming = sin@z
    result['normalization'] = dict(reference_norm2=float(np.linalg.norm(z)**2),reference_embedding_loss=float(np.linalg.norm(cache.reference-n0@z)),
        source_raw_norm2=float(np.linalg.norm(incoming)**2),receiver_raw_norm2=float(np.linalg.norm(sout@z)**2),
        convention='Raw CAR amplitudes in all moments. Neither normalized conditional source state nor response divided by source norm2; derived normalized C reported separately, not event probability.')
    result['dimensions'] = dict(neutral_J0=hn.shape[0],neutral_J1=hv.shape[0],charged_Jhalf=hc.shape[0],charged_Jhalf_Jthreehalf=hu.shape[0])
    def audit(label: str, name: str, v: Any, h: Any, x: Any) -> None:
        x = x/np.linalg.norm(x)
        actual = cache.operator(name+'_H_zero')@(v@x)
        result['action_audits'].append(dict(label=label,inside_relative=float(np.linalg.norm(v.conj().T@actual-h@x)/np.linalg.norm(actual)),outside_norm=float(np.linalg.norm(actual-v@(v.conj().T@actual)))))
    audit('reference','neutral',n0,hn,z)
    audit('source','lepton',c0,hc,incoming)
    for k in range(2):
        audit('neutral_field_'+str(k),'neutral',n1,hv,qn[k]@z)
        audit('charged_field_'+str(k),'lepton',cu,hu,qc[k]@incoming)
    save('OWN_SOURCE_AND_ALL_GENERATED_INTERMEDIATE_H_ACTIONS')
    t0,t1,t2,T = req['times']
    schedules = [(6,1),(8,1),(8,2)]
    all_rows: list[Any] = []
    states: list[Any] = []
    final_a: Any = None
    final_gen: Any = None
    for steps,split in schedules:
        receipts: list[Any] = []
        def evolve(h: Any, x: Any, duration: float) -> Any:
            for _ in range(split):
                x,rec = lanczos(h,x,duration/split,steps)
                receipts.append(dict(duration=duration/split,**rec))
            return x
        n_early,n_late,n_end = [evolve(hn,z,t) for t in (t1,t2,T)]
        a = evolve(hc,incoming,t1)
        b = evolve(hc,sout@n_end,-(T-t2))
        plain = evolve(hc,a,t2-t1)
        gen = [evolve(hu,q@a,t2-t1) for q in qc]
        ngen = [evolve(hv,q@n_early,t2-t1) for q in qn]
        rows = []
        for outer,inner in ((0,1),(1,0)):
            moments: dict[tuple[int,...],complex] = {}
            for size in range(1,5):
                for slots in itertools.combinations(range(4),size):
                    moments[slots] = 0j  # Odd-charge blocks vanish exactly by superselection.
            moments[(0,3)] = complex(np.vdot(b,plain))
            moments[(1,)] = complex(np.vdot(n_late,qmean[outer]@n_late))
            moments[(2,)] = complex(np.vdot(n_early,qmean[inner]@n_early))
            moments[(1,2)] = complex(np.vdot(n_late,qn[outer].H@ngen[inner]))
            moments[(0,1,3)] = complex(np.vdot(b,(qc[outer]@plain)[:hc.shape[0]]))
            moments[(0,2,3)] = complex(np.vdot(b,gen[inner][:hc.shape[0]]))
            moments[(0,1,2,3)] = complex(np.vdot(b,qc[outer].H@gen[inner]))
            c,terms = graded_cumulant(moments)
            rows.append(dict(C=c,moments=moments,partitions=terms))
        all_rows.append(rows)
        states.append(dict(a=a,b=b,gen=gen,receipts=receipts))
        final_a,final_gen = a,gen
        save('OWN_SCHEDULE_K'+str(steps)+'_SPLIT'+str(split)+'_COMPLETE')
    for order,label in enumerate(('TEz_late_TMz_early','TMz_late_TEz_early')):
        row = all_rows[-1][order]
        error = max(abs(row['C']-r[order]['C']) for r in all_rows[:-1])
        result['rows'].append(dict(order=label,C=pair(row['C']),magnitude=abs(row['C']),empirical_resolution=error,
            all_moments_coarse_split_max=max(abs(row['moments'][k]-all_rows[0][order]['moments'][k]) for k in row['moments']),
            all_moments_fine_split_max=max(abs(row['moments'][k]-all_rows[1][order]['moments'][k]) for k in row['moments']),
            moments={','.join(map(str,k)):pair(v) for k,v in row['moments'].items()},complete_15_graded_partitions=row['partitions'],
            source_norm2_divided_C=pair(row['C']/result['normalization']['source_raw_norm2']),
            resolved=bool(abs(row['C'])>max(10*error,1e-12)),OFF_connected_exact=[0.,0.]))
    result['state_resolution'] = dict(ket_coarse_split=float(np.linalg.norm(states[0]['a']-states[2]['a'])),ket_fine_split=float(np.linalg.norm(states[1]['a']-states[2]['a'])),
        bra_fine_split=float(np.linalg.norm(states[1]['b']-states[2]['b'])),intermediate_coarse_split=max(float(np.linalg.norm(x-y)) for x,y in zip(states[0]['gen'],states[2]['gen'])),
        intermediate_fine_split=max(float(np.linalg.norm(x-y)) for x,y in zip(states[1]['gen'],states[2]['gen'])))
    result['final_evolution_receipts'] = states[-1]['receipts']
    result['second_field_images'] = []
    for outer,inner in ((0,1),(1,0)):
        image = cache.operator('lepton_Q'+str((2,5)[outer]))@(cu@final_gen[inner])
        norms = {str(j):float(np.linalg.norm(cg('lepton',j).conj().T@image)) for j in (.5,1.5,2.5)}
        projected = sum((cg('lepton',j)@(cg('lepton',j).conj().T@image) for j in (.5,1.5,2.5)),np.zeros_like(image))
        result['second_field_images'].append(dict(outer_axis=(2,5)[outer],inner_axis=(2,5)[inner],J_norms=norms,outside_declared_J_norm=float(np.linalg.norm(image-projected)),
            endpoint_projection_difference=float(np.linalg.norm(c0.conj().T@image-qc[outer].H@final_gen[inner]))))
    save('OWN_CONNECTED_PHYSICAL_DISCRIMINATOR_COMPLETE')
    # Independent primitive CAR4 checks against explicit Kronecker Jordan-Wigner.
    eye = np.eye(16); annihilators = []
    for mode in range(4):
        a = np.zeros((16,16))
        for mask in range(16):
            if mask & (1<<mode):
                a[mask^(1<<mode),mask] = (-1)**((mask&((1<<mode)-1)).bit_count())
        annihilators.append(a)
    masks = list(range(16))
    result['CAR4'] = dict(bilinear_max_error=max(float(np.linalg.norm(extra_operator(masks,masks,[([(i,True),(j,False)],1.)]).toarray()-annihilators[i].T@annihilators[j])) for i,j in itertools.product(range(4),repeat=2)),
        canonical_anticommutator_max=max(float(np.linalg.norm(a@b.T+b.T@a-(eye if i==j else 0))) for i,a in enumerate(annihilators) for j,b in enumerate(annihilators)))
    # Primitive phase link: K transfers a q=-1 electron root1 -> added2.
    law(model,'lepton')  # Populate exact primitive phase matrices, no H compilation.
    sector = model['sectors']['lepton'];op = sector['op'];p=model['parameters']
    charge,matrices = op['hops'][2][0]
    rho = op['rho'][2]+op['rho'][3]
    terms=[]; current=[]; algebra=[]
    for axis in (None,0,1,2):
        angle=model['cache'][('phase',2,charge,tuple(np.zeros(6)),None,axis)]
        matter=(p['scalar_coupling']*matrices[0] if axis is None else 1j*p['vector_velocity']*matrices[axis+1])*model['geometry']['weights'][2]
        terms.append((matter,angle,1))
        current.extend([(-1j*charge*matter,angle,1),(1j*charge*matter.conj().T,angle.conj().T,1)])
        algebra.append(float(sparse.linalg.norm(rho@matter-matter@rho-charge*matter)))
    transfer=FormOperator(sector,sector,model['rotor'],model['radial'],terms)
    flux=FormOperator(sector,sector,model['rotor'],model['radial'],current)
    density=observable(model,'lepton',matter=rho)
    x=full['lepton']@(c0@final_a); x/=np.linalg.norm(x)
    kx=transfer@x; kdagx=transfer.H@x
    comm=1j*((transfer+transfer.H)@(density@x)-density@((transfer+transfer.H)@x))
    iphase=flux@x
    result['bridge_charge_current'] = dict(charge=charge,orientation='primitive root1 -> added2; whole added cell rho2+rho3',
        identity='[rho_added,K]=q K; i[H_bridge,rho_added]=i q(Kdagger-K)=-d H_bridge(phi)/dphi at phi=0, H(phi)=exp(i q phi)K+adjoint',
        primitive_commutator_Frobenius_max=max(algebra),source_generated_state='own incoming raw root0 source evolved to absolute .004, normalized only for this action probe',
        transfer_action_norm=float(np.linalg.norm(kx)),reverse_transfer_action_norm=float(np.linalg.norm(kdagx)),
        phase_current_action_norm=float(np.linalg.norm(iphase)),commutator_action_norm=float(np.linalg.norm(comm)),
        action_difference_norm=float(np.linalg.norm(comm-iphase)),relative_action_difference=float(np.linalg.norm(comm-iphase)/np.linalg.norm(iphase)),
        expectation=pair(complex(np.vdot(x,iphase))),
        meaning='Negative influx is negative charge entering added cell. This is charge flow through the only intercell bridge, NOT rotational momentum, Pcm, or total profile/mechanical current.',
        scope='Exact primitive matter algebra in finite charge cut and actual source-generated action. Mechanical terms are functions of commuting rho and material/field operators, hence no intercell density flow; they DO contribute -dH/da profile current. No global closure claim.')
    save('OWN_BRIDGE_CHARGE_PHASE_CURRENT_COMPLETE')
    # OFF direct operator demonstration, not duplicate long OFF propagation.
    result['OFF_direct'] = []
    he=expm(-1j*model['field']['H_E']*(t2-t1));vac=np.eye(model['field']['dimension'])[:,0]
    for name,v in [('neutral',cache.reference),('lepton',c0@final_a)]:
        off=cache.operator(name+'_H_full_transverse_off'); hfield=cache.operator(name+'_field_HE')
        q=cache.operator(name+'_Q2'); vf=v/np.linalg.norm(v)
        left=off@(q@vf)-q@(off@vf)
        right=hfield@(q@vf)-q@(hfield@vf)
        result['OFF_direct'].append(dict(sector=name,commutator_difference_norm=float(np.linalg.norm(left-right)),off_Q_commutator_norm=float(np.linalg.norm(left)),
            meaning='[H_OFF,Q_TE]=[H_E,Q_TE] on actual bounded original/source-generated vector: no residual Peierls/mechanical transverse coupling'))
    result['OFF_factorization'] = dict(cross_pair_both_orders=[pair(complex(np.vdot(vac,model['field']['Q'][a]@he@model['field']['Q'][b]@vac))) for a,b in ((2,5),(5,2))],
        field_H_offdiagonal_norm=float(np.linalg.norm(model['field']['H_E']-np.diag(np.diag(model['field']['H_E'])))),
        proof='OFF retains material/radial/charge/hops tensor EM identity plus H_E; source is EM identity and original EM vacuum. H_E leaves different TE/TM one-mode images orthogonal, so mixed pair=0 for every time; means/three-points=0. Complete cumulant therefore vanishes, same state, both profile orders. Direct commutator action is a scoped operator witness, not global theorem or numerical OFF evolution.')
    # Actual finite signed source commutators, both branches, all added sources.
    result['source_charge_signs'] = []
    for name,qcharge in [('lepton',-1.),('antilepton',1.)]:
        for site in (2,3):
            s=source_matrix(model,name,site,.5)
            qtarget=model['sectors'][name]['op']['Q'];qzero=model['sectors']['neutral']['op']['Q']
            result['source_charge_signs'].append(dict(branch=name,site=site,expected_charge=qcharge,charge_intertwiner_Frobenius=float(sparse.linalg.norm(qtarget@s-s@qzero-qcharge*s))))
    result['decision']='POSITIVE_OWN_FINITE_TWO_LEG_AND_BRIDGE_CURRENT' if all(r['resolved'] for r in result['rows']) else 'OWN_TWO_LEG_NOT_RESOLVED_IN_ALL_ORDERS'
    save('NATURAL_COMPLETION')


if __name__=='__main__':
    main()
