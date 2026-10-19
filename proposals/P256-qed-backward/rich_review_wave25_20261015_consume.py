"""Independent spatial consumer of frozen pure Gauss/CAR sector construction.
No producer CLI, evolved state, ground state or producer measurement API is used.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse, special
from scipy.sparse.linalg import expm_multiply
from rich_photon_fullq_20261014_model import Parameters, Sector

ROOT = Path(__file__).resolve().parent
PREFIX = 'rich_review_wave25_20261015_'


def maxabs(matrix: Any) -> float:
    return float(np.max(np.abs(matrix.data))) if matrix.nnz else 0.0


def comm(a: Any, b: Any) -> Any:
    return a @ b - b @ a


def ev(op: Any, states: Any) -> Any:
    return np.sum(states.conj() * (op @ states), axis=0).real


def chebyshev(h: Any, initial: Any, duration: float) -> tuple[Any, dict[str, Any]]:
    """Independent global polynomial action with analytic absolute Bessel tail.

    |J_n(z)| <= (z/2)^n/n! exp(z^2/[4(n+1)]); the ratio of these
    majorants is <= z/[2(n+1)]. Roundoff is NOT included in this tail.
    """
    diag = h.diagonal()
    radius = np.asarray(abs(h).sum(axis=1)).ravel() - abs(diag)
    lo, hi = float(np.min(diag-radius)), float(np.max(diag+radius))
    center, scale = (lo+hi)/2, (hi-lo)/2
    z = scale * duration
    degree = max(1, int(z))
    while True:
        n = degree + 1
        logfirst = n*math.log(z/2) - special.gammaln(n+1) + z*z/(4*(n+1))
        ratio = z/(2*(n+1))
        tail = math.inf if ratio >= 1 else 2*math.exp(logfirst)/(1-ratio)
        if tail < 2e-12:
            break
        degree += 1
    scaled = (h-center*sparse.eye(h.shape[0], format='csr'))/scale
    previous = initial.copy()
    current = scaled @ initial
    answer = special.jv(0, z)*previous + 2*(-1j)*special.jv(1, z)*current
    for k in range(2, degree+1):
        nxt = 2*(scaled @ current)-previous
        answer += 2*((-1j)**k)*special.jv(k, z)*nxt
        previous, current = current, nxt
    return np.exp(-1j*center*duration)*answer, dict(enclosure=[lo, hi], degree=degree,
             analytic_tail=tail, roundoff_included=False)


def run(cutoff: int, config: dict[str, Any]) -> dict[str, Any]:
    started = time.monotonic()
    s = Sector(Parameters(geometry='cube', cutoff=cutoff, hopping=config['hopping'],
               repulsion=config['repulsion'], electric_inertia=config['electric_inertia'],
               reference_particles=2, anchor=0, background='integer_ions'), 1, 1)
    shape = s.H.shape
    zero = sparse.csr_matrix(shape)
    positions = s.positions
    left = np.flatnonzero(positions[:, 0] == 0)
    right = np.flatnonzero(positions[:, 0] == 1)
    left_edges = [e for e, (a,b) in enumerate(s.edges) if positions[a,0] == positions[b,0] == 0]
    right_edges = [e for e, (a,b) in enumerate(s.edges) if positions[a,0] == positions[b,0] == 1]
    bridges = [e for e in range(len(s.edges)) if e not in left_edges+right_edges]
    # Explicit physical yz perimeter, independent of producer face helper.
    vertices = [int(np.flatnonzero(np.all(positions == coord, axis=1))[0])
                for coord in [(0,0,0),(0,1,0),(0,1,1),(0,0,1)]]
    cycle = np.zeros(len(s.edges), dtype=int)
    for a,b in zip(vertices, vertices[1:]+vertices[:1]):
        if (a,b) in s.edges:
            cycle[s.edges.index((a,b))] = 1
        else:
            cycle[s.edges.index((b,a))] = -1
    # Complete physical product projected ONCE; no compressed single-link product.
    shift = cycle[s.chords]
    mapping = np.array([s.flux_index.get(tuple(m+shift),-1) for m in s.flux])
    valid = np.flatnonzero(mapping >= 0)
    cols = np.concatenate([fi*s.nflux+valid for fi in range(len(s.fock))])
    rows = np.concatenate([fi*s.nflux+mapping[valid] for fi in range(len(s.fock))])
    w = sparse.csr_matrix((np.ones(len(rows)),(rows,cols)),shape=shape)
    source = (w-w.T)/(2j)
    density = np.repeat(s.occupations, s.nflux, axis=0)
    rho = density-s.background
    electric = s.electric
    currents = [-1j*(t-t.T) for t in s.transfers]
    weights = np.array([(positions[a,0]+positions[b,0])/2 for a,b in s.edges])
    doubles = np.array([[(bits >> (2*i) & 1)*(bits >> (2*i+1) & 1)
                         for i in range(s.vertices)] for bits in s.fock])
    he = np.sum(electric**2,axis=1)/(2*config['electric_inertia'])
    hu = config['repulsion']*np.repeat(doubles.sum(axis=1),s.nflux)
    hoff = sparse.diags(he+hu,format='csr')
    hronly = sparse.diags(np.sum(electric**2*weights,axis=1)/(2*config['electric_inertia'])
                 + config['repulsion']*np.repeat(doubles[:,right].sum(axis=1),s.nflux),format='csr')
    hr = hronly + sum((weights[e]*(t+t.T) for e,t in enumerate(s.transfers)),zero)
    hl = s.H-hr
    er = sparse.diags(np.sum(electric[:,right_edges]**2,axis=1)/(2*config['electric_inertia']),format='csr')
    cut = hoff+sum((s.transfers[e]+s.transfers[e].T for e in left_edges+right_edges),zero)
    rcut = hronly+sum((s.transfers[e]+s.transfers[e].T for e in right_edges),zero)
    lcut = cut-rcut
    # Strong all-retained-basis physical and projected-locality discriminators.
    algebra: dict[str, Any] = dict(
        face_closed=float(np.max(abs(s.incidence @ cycle))),
        face_physical_support=float(np.max(abs(electric[rows]-electric[cols]-cycle))),
        face_matter_support=float(np.max(abs(density[rows]-density[cols]))),
        gauss=float(np.max(abs(electric @ s.incidence.T+rho))),
        H_reconstruction=maxabs(s.H-hoff-sum((t+t.T for t in s.transfers),zero)),
        ownership=maxabs(hr+hl-s.H),
        source_right_E=max(maxabs(comm(source,sparse.diags(electric[:,e]))) for e in right_edges),
        source_right_density=max(maxabs(comm(source,sparse.diags(density[:,v]))) for v in right),
        source_right_transfer=max(max(maxabs(comm(source,s.transfers[e])),maxabs(comm(source,s.transfers[e].T))) for e in right_edges),
        source_right_H=maxabs(comm(source,hr)),
        off_ER_conserved=maxabs(comm(hoff,er)),
        off_HR_conserved=maxabs(comm(hoff,hronly)),
        source_off_HR=maxabs(comm(source,hronly)),
        cut_removed_exact=maxabs(s.H-cut-sum((s.transfers[e]+s.transfers[e].T for e in bridges),zero)),
        cut_L_R_comm=maxabs(comm(lcut,rcut)),
        cut_L_ER_comm=maxabs(comm(lcut,er)),
        source_cut_R_comm=maxabs(comm(source,rcut)))
    algebra['dotE_minus_primitive_J'] = max(maxabs(1j*comm(s.H,sparse.diags(electric[:,e]))-currents[e]) for e in range(len(s.edges)))
    algebra['density_continuity'] = max(maxabs(1j*comm(s.H,sparse.diags(density[:,v]))
         +sum((s.incidence[v,e]*currents[e] for e in range(len(s.edges))),zero)) for v in range(s.vertices))
    hop_support = []
    for e,t in enumerate(s.transfers):
        rr,cc = t.nonzero()
        unit = np.zeros(len(s.edges));unit[e] = 1
        a,b = s.edges[e]
        charge = np.zeros(s.vertices);charge[a] = -1;charge[b] = 1
        hop_support.append([float(np.max(abs(electric[rr]-electric[cc]-unit))),
                            float(np.max(abs(density[rr]-density[cc]-charge)))])
    algebra['primitive_hop_E_and_density_support'] = hop_support
    # Newly constructed ion-matched state at EVERY cutoff; no L1 state embedding.
    seed = np.zeros(s.dimension,dtype=complex)
    up = int(np.flatnonzero(np.all(positions == (0,0,0),axis=1))[0])
    down = int(np.flatnonzero(np.all(positions == (1,0,0),axis=1))[0])
    bits = (1 << (2*up)) | (1 << (2*down+1))
    seed_index = s.fock_index[bits]*s.nflux+s.flux_index[(0,)*len(s.chords)]
    seed[seed_index] = 1
    pulse = expm_multiply(-1j*config['pulse_alpha']*source,seed,traceA=0)
    initial = np.column_stack((seed,pulse))
    initial_matter = [initial[:,k].reshape(len(s.fock),s.nflux) for k in range(2)]
    initial_matter_distance = float(np.linalg.norm(initial_matter[1] @ initial_matter[1].conj().T-initial_matter[0] @ initial_matter[0].conj().T))
    times = np.linspace(0,config['duration'],config['intervals']+1)
    trajectory = expm_multiply(-1j*s.H, initial, start=0, stop=config['duration'],num=len(times),traceA=-1j*float(s.H.diagonal().sum()))
    boundary_mask = np.tile(np.any(abs(s.flux) == cutoff,axis=1),len(s.fock))
    records = []
    for t,states in zip(times,trajectory):
        prob = abs(states)**2
        records.append(dict(time=float(t), norm=prob.sum(axis=0).tolist(),H=ev(s.H,states).tolist(),
             ER=ev(er,states).tolist(),HR=ev(hr,states).tolist(),HL=ev(hl,states).tolist(),
             current=np.array([ev(j,states) for j in currents]).T.tolist(),
             density=(prob.T @ density).tolist(),electric=(prob.T @ electric).tolist(),
             right_power=ev(1j*comm(s.H,hr),states).tolist(),
             boundary_ANYchord=prob[boundary_mask].sum(axis=0).tolist()))
    alternate,resolution = chebyshev(s.H,initial,config['duration'])
    resolution.update(state_l2_per_column=np.linalg.norm(alternate-trajectory[-1],axis=0).tolist(),
           ER_per_column_difference=(ev(er,alternate)-ev(er,trajectory[-1])).tolist(),
           HR_per_column_difference=(ev(hr,alternate)-ev(hr,trajectory[-1])).tolist())
    work = float(np.diff(ev(s.H,initial))[0])
    final = records[-1]
    delta_er,delta_hr = float(np.diff(final['ER'])[0]),float(np.diff(final['HR'])[0])
    result = dict(cutoff=cutoff, dimension=s.dimension, positions=positions.tolist(),edges=s.edges,
        tree=s.tree,chords=s.chords,ions=s.background.tolist(),source_cycle=cycle.tolist(),
        source_edges=left_edges,receiver_edges=right_edges,removed_bridge_edges=bridges,
        seed_index=seed_index,seed_Emax=float(np.max(abs(electric[seed_index]))),
        seed_H_variance=float(np.vdot(s.H @ seed,s.H @ seed).real-ev(s.H,initial)[0]**2),
        initial_matter_distance=initial_matter_distance,algebra=algebra,records=records,resolution=resolution,
        pulse_work=work,final_delta_ER=delta_er,final_delta_HR=delta_hr,
        final_delta_ER_over_own_pulse_work=delta_er/work,
        final_delta_HR_over_own_pulse_work=delta_hr/work,
        max_boundary_ANYchord=np.max([r['boundary_ANYchord'] for r in records],axis=0).tolist(),
        max_H_drift=np.max(abs(np.array([r['H'] for r in records])-records[0]['H']),axis=0).tolist(),
        max_norm_drift=np.max(abs(np.array([r['norm'] for r in records])-1),axis=0).tolist(),
        same_state_controls='Exact operator proof: Hoff conserves its own ER/HRoff; cut H=Lcut+Rcut, [Lcut,Rcut]=[Lcut,ER]=[source,Rcut]=[source,ER]=0. Identical newly prepared pulse/baseline pair therefore has identically zero receiver excess for both controls at every time. No coupled-H energy used as control observable.',
        boundary_meaning='Sampled probability ANY independent chord is at retained boundary; not omitted norm, not all physical links, not a continuum error bound.',
        elapsed_seconds=time.monotonic()-started)
    np.savez_compressed(ROOT/f'{PREFIX}L{cutoff}_states.npz',initial=initial,endpoint=trajectory[-1],
                         independent_endpoint=alternate,times=times,states=trajectory)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('cutoff',type=int,choices=(1,2))
    args = parser.parse_args()
    config = json.loads(Path(args.input).read_text())
    result = run(args.cutoff,config)
    (ROOT/f'{PREFIX}L{args.cutoff}_result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:result[key] for key in ['cutoff','dimension','pulse_work','final_delta_ER',
         'final_delta_HR','final_delta_ER_over_own_pulse_work','final_delta_HR_over_own_pulse_work',
         'max_boundary_ANYchord','resolution','elapsed_seconds']},indent=2),flush=True)


if __name__ == '__main__':
    main()
