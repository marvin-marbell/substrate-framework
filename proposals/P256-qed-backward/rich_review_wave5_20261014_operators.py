"""NEW actual coordinate/source/current consumer of Wave5's saved finite events."""
import json
from pathlib import Path
from dataclasses import replace
import numpy as np
from scipy.linalg import expm
from rich_frame_gauss_20261014_model import frame_gauss, spatial_consumer
from rich_quantum_collision_20261014_model import Sector, Parameters
from rich_review_wave5_20261014_consumer import encode, norm

ROOT=Path(__file__).resolve().parent
PREFIX='rich_review_wave5_20261014'


def main():
    request=json.loads((ROOT/(PREFIX+'_input.json')).read_text())
    p=json.loads((ROOT/'rich_frame_gauss_20261014_input.json').read_text())
    frame=frame_gauss(); vacuum=np.eye(4)[:,0]
    pauli=[np.array([[0,1],[1,0]])/2,np.array([[0,-1j],[1j,0]])/2,np.diag([-.5,.5])]
    # Match ascending m convention without assuming a physical charge-spin operator.
    pauli[1]=-pauli[1]
    u=expm(-.37j*pauli[2])@expm(-.81j*pauli[1])@expm(.22j*pauli[2])
    epsilon=2e-6; derivative=[]
    for source in frame['raw_source']:
        state=source@frame['P']@vacuum
        coefficients=state.reshape(14,4)[1:5].reshape(2,2,4)
        for axis,a in enumerate(pauli):
            plus=expm(1j*epsilon*a)@u; minus=expm(-1j*epsilon*a)@u
            finite=1j*np.einsum('mnf,mn->f',coefficients,(plus-minus)/(2*epsilon))
            expected=np.einsum('mnf,mn->f',np.einsum('ab,bnf->anf',-a.T,coefficients),u)
            derivative.append(norm(finite-expected))
    # Full 700-state physical source and Cartesian current threepoint, NEW z/k.
    spatial=spatial_consumer(p,request['frame_wavevector'])
    stored=np.load(ROOT/(PREFIX+'_frame_states.npz'))
    v=np.zeros((len(spatial['H']),2),complex); v[:len(stored['source_vectors'])]=stored['source_vectors']
    en=1.5*p['field_frequency']; z=complex(*request['frame_z']); zout=1.13+.09j
    rin=np.linalg.solve(z*np.eye(len(v))-spatial['H']+en*np.eye(len(v)),v)
    rout=np.linalg.solve(zout*np.eye(len(v))-spatial['H']+en*np.eye(len(v)),np.eye(len(v)))
    threepoint=np.array([v.conj().T@rout@j@rin for j in spatial['current']])
    h=spatial['H']; rho=spatial['rho']; q=np.array(request['frame_wavevector'])
    # q-contracted actual threepoint matches the same-H density commutator insertion.
    contracted=np.einsum('a,amn->mn',q,threepoint)
    commutator=-v.conj().T@rout@(h@rho-rho@h)@rin
    collision_saved=np.load(ROOT/(PREFIX+'_collision_states.npz'))
    collision_result=json.loads((ROOT/(PREFIX+'_result.json')).read_text())['collision']
    params=Parameters(**json.loads((ROOT/'rich_quantum_collision_20261014_input.json').read_text())['parameters'])
    models={'full':Sector(params),'full_cut':Sector(replace(params,cut=True)),'axial':Sector(params,vector=False),'current_off':Sector(replace(params,active_hop=0.,supplier_hop=0.))}
    measurements={}
    for name,s in models.items():
        final=collision_saved[name+'_final']
        current={sp:np.array([(final.conj()*(j@final)).sum(axis=0).real for j in js]) for sp,js in s.currents.items()}
        spin_current=np.array([(final.conj()*(ops[2]@final)).sum(axis=0).real for ops in s.spin_currents])
        spin=np.array(collision_result['cases'][name]['fixed_Sz'])
        delta=spin-spin[:,[0]]
        motion=np.abs(final)**2
        x=motion.T@(s.active_density@s.positions[:,0])
        t=s.transfers['active'][0]; step=1e-4
        hp=s.H+(np.exp(-1j*step)-1)*t+(np.exp(1j*step)-1)*t.getH()
        hm=s.H+(np.exp(1j*step)-1)*t+(np.exp(-1j*step)-1)*t.getH()
        jdiff=(hp-hm)/(2*step)-s.currents['active'][0]
        kdiff=(hp+hm-2*s.H)/step**2-s.contacts['active'][0]
        measurements[name]={'final_active_supplier_currents':current,'final_actual_fixed_Sz_spin_current':spin_current,'causal_spin_change_from_own_evolved_baseline':delta,'max_unitary_causal_spin_change':np.max(abs(delta[:,1])),'final_charge_X':x,'max_unitary_current_change':max(np.max(abs(a[:,1]-a[:,0])) for a in current.values()),'current_derivative_max':np.max(abs(jdiff.data)) if jdiff.nnz else 0.,'contact_derivative_max':np.max(abs(kdiff.data)) if kdiff.nnz else 0.}
    result={'coordinate_Noether_derivative_residuals':derivative,'full_spatial_source_G':v.conj().T@rin,'full_Cartesian_unamputated_threepoint':threepoint,'current_density_threepoint_Ward_residual':norm(contracted-commutator),'collision':measurements,'meaning':'Coordinate left derivative acts on actual field U; no primitive physical spin on scalar charged colour modes. Threepoint unamputated. Actual CAR Sz and current measured on NEW reviewer events, not outgoing source-index labels.'}
    (ROOT/(PREFIX+'_operators_result.json')).write_text(json.dumps(encode(result),indent=2))
    print(json.dumps(encode({'coordinate_max':max(derivative),'threepoint_Ward':result['current_density_threepoint_Ward_residual'],'causal_spin':{k:v['max_unitary_causal_spin_change'] for k,v in measurements.items()}})))


if __name__=='__main__':
    main()
