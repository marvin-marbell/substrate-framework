"""NEW actual Phi momentum displaced sea: moving raw source metric/current.
No imported evolved output. exp(i eta Phi) applied to declared original Omega.
"""
import gc
import json
from pathlib import Path
import resource
import time
import numpy as np
from scipy.linalg import eigh, solve_sylvester
import rich_composite_geometry_20261015_model as model
from rich_composite_geometry_20261015_science import load
from rich_composite_geometry_20261015_runtime import primitive_for
from rich_composite_geometry_20261015_cloud_probe import neutral_state,inner
from rich_composite_geometry_20261015_source_eom import source_operator,bare_state,commutator_terms,apply_terms
from rich_review_wave36_20261016_consume import combine,phi_action,norm
ROOT=Path(__file__).resolve().parent
P='rich_review_wave36_20261016_'
O='rich_composite_geometry_20261015_'


def main():
    started=time.time();config=json.loads((ROOT/(O+'correlated_source_eom1_input.json')).read_text())
    data=load(ROOT/config['neutral_basis_snapshot']);primitive=primitive_for(data);p=data['p'];coupling=load(ROOT/config['neutral_coupling_snapshot']);eigen=np.load(ROOT/config['neutral_states_snapshot'])
    omega=neutral_state(data,coupling,eigen['states'][:,0]);scalar,rotor=model.field_and_rotor(p);del data,coupling,eigen;gc.collect()
    eta=.37;phi=np.diag(scalar['grid']);phase=np.diag(np.exp(1j*eta*scalar['grid']));hphi=np.asarray(scalar['H'])
    comm=hphi@phi-phi@hphi
    momentum=-1j*np.asarray(scalar['P'])/(p['scalar_metric']*p['scalar_volume_ratio'])
    momentum_error=float(np.linalg.norm(comm-momentum))
    if momentum_error>1e-12:raise AssertionError('Primitive momentum commutator mismatch')
    operator=source_operator(config['source_site'],config['source_kind'],config['source_twice_m'],False)
    bare=bare_state(primitive,omega,operator)
    groups=apply_terms(primitive,omega,commutator_terms(primitive,operator),scalar,rotor);keys=list(groups);states=[groups[key] for key in keys];base=combine(states,np.ones(len(states)))
    kicked=phi_action(bare,phase);kicked_base=phi_action(base,phase)
    sources=[kicked,phi_action(kicked,phi)]
    correct=combine([phi_action(kicked_base,phi),phi_action(kicked,momentum)],[1,1]);images=[kicked_base,correct]
    g=np.array([[inner(a,b) for b in sources] for a in sources]);k=np.array([[inner(a,b) for b in images] for a in sources]);gdot=1j*(k-k.conj().T)
    # Independent finite-field time product rule, not K-Hermitization:
    # Only changed scalar momentum terms contribute relative to un-kicked sea.
    old_sources=[bare,phi_action(bare,phi)];old_images=[base,combine([phi_action(base,phi),phi_action(bare,momentum)],[1,1])]
    old_k=np.array([[inner(a,b) for b in old_images] for a in old_sources]);old_dot=1j*(old_k-old_k.conj().T)
    direct=np.zeros((2,2),complex)
    for i in range(2):
        for j in range(2):
            coordinate=np.linalg.matrix_power(phi,i+j);product_comm=hphi@coordinate-coordinate@hphi
            direct[i,j]=1j*(inner(kicked,phi_action(kicked,product_comm))-inner(bare,phi_action(bare,product_comm)))+old_dot[i,j]
    difference=float(np.linalg.norm(gdot-direct))
    if difference>1e-10:raise AssertionError('Actual metric derivative violates independent scalar product rule')
    if np.linalg.norm(gdot)<1e-3:raise AssertionError('Bad fixed-metric case never reached nonstationary consumer')
    eig,u=eigh(g);s=(u*np.sqrt(eig))@u.conj().T;si=(u*(1/np.sqrt(eig)))@u.conj().T;sdot=solve_sylvester(s,s,gdot)
    moving=si@k@si+1j*sdot@si;wrong_no_connection=si@k@si
    q=np.array([2/3 if flavour==0 else -1/3 for edge,flavour in keys]);chord=np.array([edge==3 for edge,flavour in keys])
    current=phi_action(combine(states,-1j*q*chord),phase);dressed_current=phi_action(current,phi)
    contact=phi_action(combine(states,-q*q*chord),phase);dressed_contact=phi_action(contact,phi)
    np.savez(ROOT/(P+'metric_raw.npz'),G=g,K=k,Gdot=gdot,independent_product_Gdot=direct,sqrtG=s,dot_sqrtG=sdot,correct_moving_generator=moving,wrong_no_connection_generator=wrong_no_connection,primitive_momentum_commutator=momentum)
    result=dict(kind='ACTUAL_ORIGINAL_OMEGA_QUANTUM_SCALAR_MOMENTUM_DISPLACEMENT_NOT_STATIONARY_VACUUM',eta=eta,reference_norm2=norm(omega),source_norms2=[norm(a) for a in sources],Phi_source_EOM_norm2=norm(correct),Gdot_real=gdot.real.tolist(),Gdot_imag=gdot.imag.tolist(),Gdot_norm=float(np.linalg.norm(gdot)),independent_product_derivative_error=difference,Hphi_Phi_vs_primitive_minus_iP_error=momentum_error,actual_sqrtG_connection_norm=float(np.linalg.norm(sdot@si)),correct_moving_generator_nonhermiticity=float(np.linalg.norm(moving-moving.conj().T)),wrong_no_connection_nonhermiticity=float(np.linalg.norm(wrong_no_connection-wrong_no_connection.conj().T)),wrong_fixed_G_Hermitization_discarded_metric_flux=float(np.linalg.norm(gdot)),dressed_same_Peierls_current_norm2=norm(dressed_current),dressed_same_Peierls_contact_norm2=norm(dressed_contact),elapsed_seconds=time.time()-started,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,scope='Unitary quantum Phi phase imprint on original declared hot sea; not replacement solved ground. [Hphi,Phi]=-iP/(metric*volume), full finitePhi7/R10/PW19 fields. Selected source family has actual changing G; bad fixed-G Hermitization/no-connection rejected by independent raw product rule. Chosen J action unchanged by scalar phase. No time-domain mobility, decay, mass, stationary pole, or R/U dressed-EOM claim.')
    (ROOT/(P+'metric_result.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
