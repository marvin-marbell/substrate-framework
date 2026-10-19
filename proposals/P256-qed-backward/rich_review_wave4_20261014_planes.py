"""NEW three-dimensional physical planes with local far-end causal receivers."""
from __future__ import annotations
from dataclasses import replace
import json
from pathlib import Path
import sys
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import expm_multiply
from rich_spin_flux_spatial_20261014_3d_model import Parameters, Sector, LAW
from rich_review_wave4_20261014_tools import gram_frame, complex_pair, save


def probe(p: Parameters, inputs: dict) -> dict:
    sector = Sector(p)
    energy, ground, _ = sector.ground()
    times = np.linspace(0,inputs['time_stop'],inputs['time_count'])
    sources = np.column_stack([sector.F[index]@ground for index in [0,1]])
    receivers = np.column_stack([sector.F[index]@ground for index in [2,3]])
    sources -= ground[:,None]*(ground.conj()@sources)[None,:]
    receivers -= ground[:,None]*(ground.conj()@receivers)[None,:]
    incoming, gin = gram_frame(sources)
    outgoing, gout = gram_frame(receivers)
    waves = sector.evolve(incoming,times,energy)
    amplitude = np.einsum('di,tdj->tij',outgoing.conj(),waves)
    probability = np.sum(np.abs(amplitude)**2,axis=1)
    raw_chi = sector.retarded(ground,energy,[sector.F[index] for index in [0,1,2,3]],.39+.12j)
    kicks = []
    diagonal_receivers = {'far_xy_rung10_energy':sector.electric[:,10]**2/(2*p.inertia),
                          'far_xz_rung11_energy':sector.electric[:,11]**2/(2*p.inertia),
                          'far_active_charge':sector.active_density[:,6:9].sum(axis=1)}
    for source_index in [0,1]:
        prepared = expm_multiply(1j*inputs['kick']*sector.F[source_index],ground)
        states = sector.evolve(prepared,times,energy)
        probabilities = np.abs(states)**2
        observations = {}
        for name,diagonal in diagonal_receivers.items():
            values = probabilities@diagonal
            baseline = float(np.abs(ground)**2@diagonal)
            observations[name] = {'initial_change':float(values[0]-baseline),'max_absolute_change':float(np.max(np.abs(values-baseline))),'change_series':(values-baseline).tolist()}
        fluxes = np.array([[np.vdot(state,f@state).real for f in sector.F[2:]] for state in states])
        initial_flux = np.array([np.vdot(ground,f@ground).real for f in sector.F[2:]])
        kicks.append({'source_plane':source_index,'local_far_receivers':observations,'right_loop_flux_max_change':np.max(np.abs(fluxes-initial_flux),axis=0).tolist(),
                      'norm_error':float(np.max(np.abs(probabilities.sum(axis=1)-1)))})
    commutators = [[float(np.linalg.norm((sector.F[right]@sector.F[left]-sector.F[left]@sector.F[right])@ground)) for left in [0,1]] for right in [2,3]]
    supports = [np.flatnonzero(face).tolist() for face in sector.faces]
    return {'law':LAW,'parameters':vars(p),'dimension':sector.dimension,'ground_energy':energy,'ground_residual':float(np.linalg.norm(sector.H@ground-energy*ground)),
            'constraints':sector.constraints(),'cut_edges':sector.cut_edges,'Wilson_face_edge_supports':supports,
            'incoming_gram_eigenvalues':gin.tolist(),'outgoing_gram_eigenvalues':gout.tolist(),
            'incoming_frame_error':float(np.linalg.norm(incoming.conj().T@incoming-np.eye(incoming.shape[1]))),
            'outgoing_frame_error':float(np.linalg.norm(outgoing.conj().T@outgoing-np.eye(outgoing.shape[1]))),
            'normalized_transition_initial':probability[0].tolist(),'normalized_transition_max':np.max(probability,axis=0).tolist(),
            'raw_remote_chi':[[complex_pair(x) for x in row] for row in raw_chi[2:,:2]],'raw_remote_chi_max':float(np.max(np.abs(raw_chi[2:,:2]))),
            'same_state_interplane_Wilson_commutator_source_norms':commutators,'physical_kicks':kicks}


def main() -> None:
    inputs = json.loads(Path(sys.argv[1]).read_text());p=Parameters(**inputs['parameters'])
    result = {name:probe(parameters,inputs) for name,parameters in [('connected',p),('rail_cut',replace(p,cut=True)),('no_spin_torque',replace(p,torque=0.))]}
    save('planes',{'kind':'INDEPENDENT_NEW_3D_PLANES_AND_STRICT_FAR_RECEIVER','inputs':inputs,'cases':result,
                   'scope':'Two actual spatial Wilson planes, not multiplied D tensors or proven transverse photons. Adjacent Wilson source/out loops share boundary rungs; separately measure far-end10/11 electric energies and active charge at6/7/8. Railcut3/4/5 tested physically, not declared by name. No v2 G imported or v3 G claim. Added AFTER once final19-file diagnostics: this new consumer/3D source has no final Pyright claim.'})

if __name__=='__main__':
    main()
