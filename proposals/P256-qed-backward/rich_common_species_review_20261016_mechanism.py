"""Actual cross commutator actions distinguish kinetics from projected geometry."""
from pathlib import Path
import json
import numpy as np
from scipy import sparse
import rich_composite_geometry_20261015_model as model
from rich_composite_geometry_20261015_science import load
from rich_composite_geometry_20261015_runtime import primitive_for
from rich_composite_geometry_20261015_cloud_probe import neutral_state,inner
from rich_composite_geometry_20261015_source_eom import bare_state
from rich_common_native_species_20261016_sources import operator_terms,algebra_receipt
from rich_common_species_review_20261016_math import pair,pair_operator
ROOT=Path(__file__).resolve().parent
P='rich_common_species_review_20261016_'


def original_source_audit(carrier,joint):
    config=json.loads((ROOT/'rich_common_native_species_20261016_joint1_input.json').read_text());data=load(ROOT/config['neutral_basis_snapshot']);primitive=primitive_for(data);coupling=load(ROOT/config['neutral_coupling_snapshot']);eigen=np.load(ROOT/config['neutral_states_snapshot']);omega=neutral_state(data,coupling,eigen['states'][:,0]);norm=float(inner(omega,omega).real)
    descriptors=[dict(sector=number,site=0,twice_J=1,family='bare') for number in (33,31)]+[dict(sector=36,B_site=0,L_site=2,twice_J=2,family='bare')];rows=[]
    for descriptor in descriptors:
        meta=joint if descriptor['sector']==36 else carrier;pr=meta['primitive']
        if any(vector!=pr.local.vectors[i] for i,vector in enumerate(primitive.local.vectors)):raise AssertionError('Original/native local coordinate prefix mismatch')
        bare=bare_state(pr,omega,operator_terms(descriptor));strength=float(inner(bare,bare).real);probe=next(probe for probe in meta['probes'] if probe['descriptor']==descriptor)
        if abs(strength-probe['raw_strength'])>2e-11:raise AssertionError('Original CAR source strength differs from declared native source')
        rows.append(dict(descriptor=descriptor,actual_raw_strength=strength,declared_raw_strength=probe['raw_strength'],Noether_operator=algebra_receipt(descriptor)))
    complement=rows[0]['actual_raw_strength']+rows[1]['actual_raw_strength']-norm
    if abs(complement)>2e-11:raise AssertionError('Original same-orbital CAR addition/removal complement failed')
    return dict(omega_norm=norm,sources=rows,addition_plus_removal_minus_omega_norm=complement,source_order='Joint canonical Bdag0 Ldag2 Omega, L acts first. N33/N31 opposite original source order in DIFFERENT number sectors, not an intersector current.')


def field_and_CAR_algebra(meta):
    primitive=meta['primitive'];scalar,rotor=model.field_and_rotor(meta['p']);maps=primitive.maps
    qterms=[operations for (edge,flavour,white,component,dagger),terms in maps.items() if edge==0 and not white for operations,coefficient,field in terms]
    wterms=[operations for (edge,flavour,white,component,dagger),terms in maps.items() if edge==2 and white for operations,coefficient,field in terms]
    contractions=sum(int(q[1][0]==w[0][0])+int(w[1][0]==q[0][0]) for q in qterms for w in wterms)
    if contractions:raise AssertionError('Chosen species even bilinears unexpectedly share a CAR mode')
    axes=np.asarray(primitive.graph['axes']);rotations=[];pointwise=[]
    for edge in (0,2):
        rotations.append([sum((axes[edge,a]*sparse.csr_matrix(rotor['coordinate'][i][a]).toarray() for a in range(3)),np.zeros((10,10),complex)) for i in range(3)])
        pointwise.append(np.einsum('qia,a->qi',rotor['R'],axes[edge]))
    comm=np.array([[rotations[1][i]@rotations[0][j]-rotations[0][j]@rotations[1][i] for j in range(3)] for i in range(3)])
    norms=np.linalg.norm(comm,axis=(2,3));pointwise_comm=np.array([[pointwise[1][:,i]*pointwise[0][:,j]-pointwise[0][:,j]*pointwise[1][:,i] for j in range(3)] for i in range(3)])
    result=dict(actual_quark_even_bilinears=len(qterms),actual_white_even_bilinears=len(wterms),cross_CAR_contraction_count=contractions,projected_R_commutator_Frobenius=norms.tolist(),maximum_projected_R_commutator=float(norms.max()),unprojected_pointwise_rotation_product_commutator=float(np.linalg.norm(pointwise_comm)),derivation='For actual quark/white maps [c_a†c_b,c_c†c_d]=delta_bc c_a†c_d-delta_da c_c†c_b: all cross deltas vanish because colour0..2 versus white3. Full Phi and R coordinate multiplication commute, as do independent PW links; chosen quark receiver edge0 is TREE and white source edge2 has no U at all. Thus full-multiplication DeltaH_white and J_quark commute before field/body P (and reverse). Finite P_R R P_R already need not commute; source/body Galerkin P can additionally produce noncommutation even in scalar channels. This identifies a legitimate finite matrix coupling, not automatically a defect or proof for the untruncated law.')
    np.savez(ROOT/(P+'field_algebra_raw.npz'),projected_R_operators=np.array(rotations),projected_R_cross_commutators=comm,unprojected_rotation_coordinate_values=np.array(pointwise),pointwise_commutators=pointwise_comm)
    return result


def cross_action(space,state,config,source):
    receiver='quark' if source=='white' else 'white';source_edge=config[source+'_probe_edge'];receiver_edge=config[receiver+'_probe_edge'];d=pair(space,source,source_edge,0,config['theta']);j=pair(space,receiver,receiver_edge,1,0.);zero=sparse.csr_matrix(space['g'].shape,dtype=complex)
    dparts=[pair_operator(space,(d[0],zero)),pair_operator(space,(zero,d[1]))];jparts=[pair_operator(space,(j[0],zero)),pair_operator(space,(zero,j[1]))];names=[];images=[]
    for a in range(2):
        for b in range(2):
            names.append(('vector' if a==0 else 'scalar')+'_'+('vector' if b==0 else 'scalar'));images.append(dparts[a]@(jparts[b]@state)-jparts[b]@(dparts[a]@state))
    delta=pair_operator(space,d);current=pair_operator(space,j);actual=delta@(current@state)-current@(delta@state);combined=sum(images);gram=np.array([[np.vdot(a,b) for b in images] for a in images]);expect=np.array([np.vdot(state,image) for image in images]);leading=float((1j*np.vdot(state,actual)).real)
    direct=float(2*np.vdot(current@state,-1j*(delta@state)).real);charge_comm=[]
    z=state.reshape(space['n'],space['nf']);dv=(delta@state).reshape(z.shape)
    for site in range(4):
        q=space['data']['Q'+receiver][:,site,None];charge_comm.append(float(np.linalg.norm(delta@(z*q).reshape(-1)-(dv*q).reshape(-1))))
    if np.linalg.norm(actual-combined)>2e-10 or abs(leading-direct)>2e-10:raise AssertionError('Actual metric-aware cross-commutator action/product rule mismatch')
    result=dict(source=source,receiver=receiver,source_edge=source_edge,receiver_edge=receiver_edge,source_DeltaH_action_norm2=float(np.vdot(delta@state,delta@state).real),cross_commutator_action_norm2=float(np.vdot(actual,actual).real),cross_commutator_expectation_real=float(np.vdot(state,actual).real),cross_commutator_expectation_imag=float(np.vdot(state,actual).imag),initial_delta_receiver_current_slope=leading,independent_state_derivative_slope=direct,component_names=names,component_action_norms2=np.diag(gram).real.tolist(),component_initial_slopes=(1j*expect).real.tolist(),coherent_component_Gram_sum_real=float(gram.sum().real),composition_error=float(np.linalg.norm(actual-combined)),receiver_charge_cross_commutator_action_norms=charge_comm,scope='Operators act in independent sqrtG orthonormal coordinates: weak product is represented with full G inverse, not weak-matrix multiplication. DeltaH/J contain actual vector+Phi hops and finite theta; commutator has no Hphi/Hrot/electric kinetic term. Norm of sum includes ALL interference, not sum of piece norms.')
    np.savez(ROOT/(P+'cross_'+source+'_raw.npz'),commutator_action=actual,component_actions=np.array(images),component_Gram=gram,component_expectations=expect,source_DeltaH_action=delta@state,receiver_J_action=current@state)
    return result
