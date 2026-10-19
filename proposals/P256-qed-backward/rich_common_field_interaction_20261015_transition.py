"""Distinct-time same-H two-leg transition on COMPLETE connecting angular graph.

Pure small angular reprojection of published numeric Factors, NOT a second H
compilation. Jhalf -> (Jhalf,J3/2) -> Jhalf charged paths and J0 -> J1 -> J0
neutral field paths include every rank1-connected intermediate. Scalar-H
closure is checked on the actual generated vectors against full-M H action.
"""
from __future__ import annotations

import json
from pathlib import Path
import resource
import sys
import time
from typing import Any

import numpy as np
from scipy import sparse

from rich_common_field_interaction_20261015_consume import lanczos, complex_pair
from rich_common_field_interaction_20261015_direct import NumericCache, pinned_sources, digest

ROOT = Path(__file__).resolve().parent


def reproject(operator: Any, target: Any, source: Any = None) -> Any:
    """Vt^dagger Factors Vs via small channel CG matrices, no matter recompile."""
    source = target if source is None else source
    nr = operator.nr
    vt, vs = target[::nr, ::nr].tocsr(), source[::nr, ::nr].tocsr()
    layouts: dict[tuple[int, int], tuple[dict[str, Any], Any]] = {}
    def layout(old: dict[str, Any], v: Any) -> tuple[dict[str, Any], Any]:
        key = (id(old), id(v))
        if key not in layouts:
            maps = []
            reference = None
            for indices in old['indices']:
                local = v[indices]
                columns = np.unique(local.indices)
                w = local[:, columns].tocsr()
                if reference is None:
                    reference = w
                elif w.shape != reference.shape or sparse.linalg.norm(w-reference) > 1e-12:
                    raise ValueError('CG projection is not multiplicity separable')
                maps.append(columns)
            if reference is None:
                raise ValueError('Missing actual factor multiplicity')
            layouts[key] = (dict(indices=np.array(maps, dtype=int), multiplets=old['multiplets'],
                                 channels=reference.shape[1]), reference)
        return layouts[key]
    terms = []
    for left, right, a, b, c in operator.terms:
        tl, wl = layout(left, vt)
        sr, wr = layout(right, vs)
        if not tl['channels'] or not sr['channels']:
            continue
        angle = (wl.conj().T @ b @ wr).tocsr()
        if angle.nnz:
            terms.append((tl, sr, a, angle, c))
    return type(operator)(vt.shape[1], vs.shape[1], nr, terms,
                          provenance=dict(construction='Exact small angular CG reprojection of immutable numeric Factors; matter/radial data unchanged', groups=len(terms)))


def main() -> None:
    path, out = Path(sys.argv[1]), Path(sys.argv[2])
    request: dict[str, Any] = json.loads(path.read_text())
    pinned_sources(request)
    cache = NumericCache(request)
    from rich_common_carrier_motion_20261015_model import build, fibre
    from rich_frame_field_join_20261015_calculate import m_isometry
    start = time.monotonic()
    model = build(json.loads((ROOT/request['physical_input']).read_text()))
    full_iso = {name:m_isometry(model,name,0. if name=='neutral' else .5) for name in ('neutral','lepton','antilepton')}
    def cg(name: str, j: float) -> Any:
        return (full_iso[name].conj().T @ fibre(model,name,j,0. if name=='neutral' else .5)).tocsr()
    v0, v1 = cg('neutral',0.), cg('neutral',1.)
    z = v0.conj().T @ cache.reference
    h0 = reproject(cache.operator('neutral_H_zero'),v0)
    h1 = reproject(cache.operator('neutral_H_zero'),v1)
    q10 = [reproject(cache.operator('neutral_Q'+str(axis)),v1,v0) for axis in (2,5)]
    q00 = [reproject(cache.operator('neutral_Q'+str(axis)),v0) for axis in (2,5)]
    proof = ROOT/request['off_factorization']['file']
    if digest(proof) != request['off_factorization']['sha256']:
        raise ValueError('Changed exact physical-OFF witness')
    t1,t2,total = request['field_times'][0],request['field_times'][1],request['receiver_time']
    if not 0 < t1 < t2 < total:
        raise ValueError('Genuinely distinct strictly ordered leg times required')
    result: dict[str, Any] = dict(kind='ACTUAL_DISTINCT_TIME_CAR36_CONNECTED_SPATIAL_TWO_FIELD_LEG_TRANSITION',
        request_sha256=digest(path),source_time=0.,field_times=[t1,t2],receiver_time=total,
        source_site=request['source_site'],receiver_site=request['receiver_site'],
        preparation=dict(meaning='Original nonstationary prepared t0 only',
                         norm_squared=float(np.linalg.norm(z)**2),
                         original_M0_to_J0_loss=float(np.linalg.norm(cache.reference-v0@z))),
        angular_graph=dict(neutral_reference_J=[0.],neutral_field_intermediate_J=[1.],
                           charged_endpoint_J=[.5],charged_field_intermediate_J=[.5,1.5],
                           physical_M0_reference_Mhalf_source=.5,
                           meaning='Every rank1 connecting channel and multiplicity retained; final orthogonal J cannot reach the actual J0/Jhalf receiver bra'),
        numeric_cache=request['numeric_cache'],control=dict(kind='EXACT_SAME_STATE_COMPLETE_TRANSVERSE_OFF_FACTORIZATION',
            proof=request['off_factorization'],connected=[0.,0.],
            meaning='Actual complete mechanical+Peierls OFF factorization with Coulomb/charges/hops retained; analytic control, never a fabricated numerical evolution'),
        action_audits=[],rows=[],decision='NOT_YET_OBSERVED')
    def save(stage: str) -> None:
        result['stage']=stage
        result['elapsed_seconds']=time.monotonic()-start
        out.with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
        print(stage,flush=True)
    def audit(label: str, full: Any, reduced: Any, v: Any, x: Any) -> None:
        if np.linalg.norm(x)==0:
            return
        x=x/np.linalg.norm(x)
        before=time.monotonic()
        actual=full@(v@x)
        inside=v.conj().T@actual
        projected=reduced@x
        row: dict[str, Any]=dict(label=label,full_M_action_seconds=time.monotonic()-before,
                 inside_action_relative_error=float(np.linalg.norm(inside-projected)/max(np.linalg.norm(inside),np.finfo(float).tiny)),
                 outside_generated_J_action_norm=float(np.linalg.norm(actual-v@inside)))
        result['action_audits'].append(row)
        if row['inside_action_relative_error']>1e-10 or row['outside_generated_J_action_norm']>1e-9:
            save('ACTUAL_ANGULAR_CLOSURE_OR_FACTOR_ACTION_FAILED')
            raise ValueError('Actual generated path fails scalar-H closure/equality')
    audit('neutral_original_J0',cache.operator('neutral_H_zero'),h0,v0,z)
    for axis,q in zip((2,5),q10):
        audit('neutral_Q'+str(axis)+'_J1',cache.operator('neutral_H_zero'),h1,v1,q@z)
    result['dimensions']=dict(neutral_J0=h0.shape[0],neutral_J1=h1.shape[0])
    save('ACTUAL_GENERATED_ANGULAR_H_ACTIONS')
    schedules=[(request['coarse_steps'],1),(request['fine_steps'],1),(request['fine_steps'],2)]
    def evolve(h: Any, x: Any, duration: float, steps: int, split: int, receipts: list[Any]) -> Any:
        for _ in range(split):
            x,record=lanczos(h,x,duration/split,steps)
            receipts.append(dict(duration=duration/split,**record))
        return x
    neutral=[]
    for steps,split in schedules:
        receipts: list[Any]=[]
        n1=evolve(h0,z,t1,steps,split,receipts)
        n2=evolve(h0,z,t2,steps,split,receipts)
        end=evolve(h0,z,total,steps,split,receipts)
        middle=[evolve(h1,q@n1,t2-t1,steps,split,receipts) for q in q10]
        pairs=[complex(np.vdot(n2,q10[a].H@middle[b])) for a,b in ((0,1),(1,0))]
        means1=[complex(np.vdot(n1,q@n1)) for q in q00]
        means2=[complex(np.vdot(n2,q@n2)) for q in q00]
        neutral.append(dict(endpoint=end,pairs=pairs,means1=means1,means2=means2,evolution=receipts))
    result['neutral_resolution']=dict(endpoint_coarse_split_L2=float(np.linalg.norm(neutral[0]['endpoint']-neutral[2]['endpoint'])),
        endpoint_fine_split_L2=float(np.linalg.norm(neutral[1]['endpoint']-neutral[2]['endpoint'])))
    save('ACTUAL_DISTINCT_TIME_NEUTRAL_FIELD_MARGINAL')
    for branch in ('lepton','antilepton'):
        vh,vthree=cg(branch,.5),cg(branch,1.5)
        vu=sparse.hstack([vh,vthree],format='csr')
        fullh=cache.operator(branch+'_H_zero')
        hh,hu=reproject(fullh,vh),reproject(fullh,vu)
        sin=reproject(cache.operator(branch+'_source_site'+str(request['source_site'])),vh,v0)
        sout=reproject(cache.operator(branch+'_source_site'+str(request['receiver_site'])),vh,v0)
        qs=[reproject(cache.operator(branch+'_Q'+str(axis)),vu,vh) for axis in (2,5)]
        incoming=sin@z
        audit(branch+'_raw_source_Jhalf',fullh,hh,vh,incoming)
        for axis,q in zip((2,5),qs):
            audit(branch+'_Q'+str(axis)+'_Jhalf_plus_Jthreehalf',fullh,hu,vu,q@incoming)
        result['dimensions'][branch]=dict(endpoint_Jhalf=hh.shape[0],intermediate_Jhalf_plus_Jthreehalf=hu.shape[0])
        save(branch+'_ACTUAL_GENERATED_ANGULAR_H_ACTIONS')
        outputs=[]
        states=[]
        for index,(steps,split) in enumerate(schedules):
            receipts=[]
            a=evolve(hh,incoming,t1,steps,split,receipts)
            b=evolve(hh,sout@neutral[index]['endpoint'],-(total-t2),steps,split,receipts)
            plain=evolve(hh,a,t2-t1,steps,split,receipts)
            generated=[evolve(hu,q@a,t2-t1,steps,split,receipts) for q in qs]
            g=complex(np.vdot(b,plain))
            rows=[]
            for ordering,(outer,inner) in enumerate(((0,1),(1,0))):
                full=complex(np.vdot(b,qs[outer].H@generated[inner]))
                tout=complex(np.vdot(b,(qs[outer]@plain)[:hh.shape[0]]))
                tin=complex(np.vdot(b,generated[inner][:hh.shape[0]]))
                qout,qin=neutral[index]['means2'][outer],neutral[index]['means1'][inner]
                d=neutral[index]['pairs'][ordering]
                disconnected=g*d+tout*qin+tin*qout-2*g*qout*qin
                rows.append(dict(full=full,disconnected=disconnected,connected=full-disconnected,
                                 fermion_pair=g,field_pair=d,three_out=tout,three_in=tin,
                                 field_mean_out=qout,field_mean_in=qin))
            outputs.append(rows)
            states.append(dict(a=a,b=b,generated=generated,evolution=receipts))
        for ordering,label in enumerate(('TEz_late_TMz_early','TMz_late_TEz_early')):
            final=outputs[2][ordering]
            error=max(abs(outputs[index][ordering]['connected']-final['connected']) for index in (0,1))
            magnitude=abs(final['connected'])
            result['rows'].append(dict(branch=branch,physical_frame_spin_M=.5,profile_time_order=label,
                moments={key:complex_pair(value) for key,value in final.items()},
                source_retained_raw_norm_squared=float(np.linalg.norm(incoming)**2),
                receiver_retained_raw_norm_squared=float(np.linalg.norm(sout@z)**2),
                resolution=dict(connected_empirical_error=error,
                    all_moments_coarse_split_max=max(abs(outputs[0][ordering][key]-value) for key,value in final.items()),
                    all_moments_fine_split_max=max(abs(outputs[1][ordering][key]-value) for key,value in final.items()),
                    ket_coarse_split_L2=float(np.linalg.norm(states[0]['a']-states[2]['a'])),
                    ket_fine_split_L2=float(np.linalg.norm(states[1]['a']-states[2]['a'])),
                    intermediate_coarse_split_L2_max=max(float(np.linalg.norm(x-y)) for x,y in zip(states[0]['generated'],states[2]['generated'])),
                    intermediate_fine_split_L2_max=max(float(np.linalg.norm(x-y)) for x,y in zip(states[1]['generated'],states[2]['generated']))),
                magnitude_against_exact_physical_OFF=magnitude,
                resolved=magnitude>request['resolution_margin']*error and magnitude>request['absolute_floor'],
                evolution=states[2]['evolution']))
        result['decision']='POSITIVE_FINITE_DISTINCT_TIME_SAME_H_TWO_LEG_MECHANISM' if any(row['resolved'] for row in result['rows']) else 'NO_RESOLVED_DISTINCT_TIME_MECHANISM_IN_COMPLETED_CHANNELS'
        save('FIRST_ACTUAL_DISTINCT_TIME_PHYSICAL_DISCRIMINATOR' if len(result['rows'])==2 else 'BOTH_SIGNED_BRANCHES_PHYSICALLY_COMPLETE')
        synthesis=dict(kind='ACTUAL_DISTINCT_TIME_SAME_CAR36_TWO_FIELD_LEG_PROVISIONAL',decision=result['decision'],
            source_site=request['source_site'],receiver_site=request['receiver_site'],source_time=0.,field_times=[t1,t2],receiver_time=total,
            physical_frame_spin_M=.5,observed_branches=sorted({row['branch'] for row in result['rows']}),
            rows=[{key:row[key] for key in ('branch','profile_time_order','magnitude_against_exact_physical_OFF','resolved','resolution')} for row in result['rows']],
            maximum_inside_action_error=max(row['inside_action_relative_error'] for row in result['action_audits']),
            maximum_outside_generated_J_action_norm=max(row['outside_generated_J_action_norm'] for row in result['action_audits']),
            control=result['control'],limits=['Finite nonstationary unamputated transition, not asymptotic photons/event probabilities',
                'Distinct leg times; no coincident projected-coordinate commutator promoted to physics',
                'All connecting J1 and J3/2 angular channels/multiplicities retained and actual generated-H closure checked',
                'Physical M+.5 directly resolved; minus-M is not independently evolved or fabricated',
                'Empirical coarse/fine/two-substep resolution, finite radial/EM/native-charge/material cuts remain'])
        out.with_name(out.name+'_provisional_synthesis.json').write_text(json.dumps(synthesis,indent=2)+'\n')
    result['peak_RSS_KiB']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    save('NATURAL_COMPLETION')


if __name__=='__main__':
    main()
