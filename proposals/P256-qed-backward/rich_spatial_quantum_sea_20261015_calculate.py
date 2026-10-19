"""NEW physical calculation, not a replay or test harness."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import time
import numpy as np
from scipy import sparse
from scipy.linalg import eigh_tridiagonal
from scipy.sparse.linalg import expm_multiply
from rich_spatial_quantum_sea_20261015_model import (geometry, fields, configurations, sector,
    one_body, source_packet, source_operator, field_operator, leakage)


def real(value):
    return float(np.real(value))


def lanczos(h, vector, shift, steps):
    norm=float(np.linalg.norm(vector))
    if norm < 1e-13:return dict(energies=[],weights=[],norm_squared=norm**2,steps=0,last_beta=0.)
    q=vector/norm; previous=np.zeros_like(q); beta=0.; alphas=[];betas=[];basis=[]
    for iteration in range(steps):
        basis.append(q)
        r=h@q-shift*q-beta*previous
        alpha=real(np.vdot(q,r));r-=alpha*q
        # Full reorthogonalization preserves residues under degeneracy/roundoff.
        for old in basis:r-=old*np.vdot(old,r)
        newbeta=float(np.linalg.norm(r));alphas.append(alpha)
        if newbeta < 1e-11 or iteration==steps-1:break
        betas.append(newbeta);previous,q=q,r/newbeta;beta=newbeta
    energies,vectors=eigh_tridiagonal(np.array(alphas),np.array(betas))
    weights=norm**2*abs(vectors[0])**2
    return dict(energies=energies.tolist(),weights=weights.tolist(),norm_squared=norm**2,
                steps=len(alphas),last_beta=newbeta,
                meaning='Gauss-Lanczos spectral quadrature of FULL projected variational H, not exact full-CAR spectrum; last beta not an error bound')


def retarded(spectrum, frequencies, eta, fermion=False, removal=None):
    e=np.array(spectrum['energies']);w=np.array(spectrum['weights']);z=np.array(frequencies)+1j*eta
    result=np.sum(w[None,:]/(z[:,None]-e[None,:]),axis=1)
    if fermion:
        er=np.array(removal['energies']);wr=np.array(removal['weights'])
        result+=np.sum(wr[None,:]/(z[:,None]+er[None,:]),axis=1)
    else:result-=np.sum(w[None,:]/(z[:,None]+e[None,:]),axis=1)
    return [[real(x),float(x.imag)] for x in result]


def matter_observables(sec,geo):
    u=geo['orbitals'];matrices=[]
    for site in range(27):
        local=u[4*site:4*site+4]
        matrices.append(local.conj().T@local)
    spin=np.kron(np.eye(27),np.kron(np.eye(2),np.diag([.5,-.5])))
    matrices.append(u.conj().T@spin@u)
    return one_body(sec['masks'],np.array(matrices))


def observe(state, sec, field, matter, currents, contacts, vacuum_density):
    nf=len(sec['masks']);nb=len(field['coordinates']);a=state.reshape(nf,nb)
    density=np.array([real(np.vdot(a, op@a)) for op in matter[:27]])
    probability=np.sum(abs(a)**2,axis=0)
    rho_field=a.conj().T@a
    purity=real(np.trace(rho_field@rho_field))
    phi=field['coordinates'][:,0]
    fermion_prob=np.sum(abs(a)**2,axis=1)
    pairs=np.array([(mask >> 54).bit_count() for mask in sec['masks']])
    return dict(norm=real(np.vdot(state,state)),density=density.tolist(),charge_excess=(density-vacuum_density).tolist(),
        total_particles=float(np.sum(density)),fixed_frame_spin_z=real(np.vdot(a,matter[27]@a)),
        phi_mean=float(probability@phi),phi_variance=float(probability@(phi**2)-(probability@phi)**2),
        field_means=(probability@field['coordinates'][:,1:]).tolist(),field_purity=purity,
        field_linear_entropy=1-purity,positive_band_occupancy=float(fermion_prob@pairs),
        currents=[real(np.vdot(state,j@state)) for j in currents],
        contacts=[real(np.vdot(state,d@state)) for d in contacts],
        field_probabilities=probability.tolist())


def trajectories(sec,field,initial,kicked,config,matter,currents,contacts,vacuum_density,h=None):
    times=np.linspace(0,config['duration'],config['time_points']);ham=sec['h'] if h is None else h
    trace=ham.diagonal().sum()
    base=expm_multiply(-1j*ham,initial,start=0,stop=times[-1],num=len(times),traceA=-1j*trace)
    pulse=expm_multiply(-1j*ham,kicked,start=0,stop=times[-1],num=len(times),traceA=-1j*trace)
    baseline=[observe(s,sec,field,matter,currents,contacts,vacuum_density) for s in base]
    kicked_result=[observe(s,sec,field,matter,currents,contacts,vacuum_density) for s in pulse]
    return dict(times=times.tolist(),baseline=baseline,kicked=kicked_result,
                density_difference_l2=[float(np.linalg.norm(np.array(a['density'])-b['density'])) for a,b in zip(kicked_result,baseline)],
                field_probability_difference_l1=[float(np.sum(abs(np.array(a['field_probabilities'])-b['field_probabilities']))) for a,b in zip(kicked_result,baseline)]),base,pulse


def run_rank(config,geo,field,rank,full):
    masks,pairs,pairweight=configurations(geo,field,config,rank)
    print('Building rank',rank,'fermion dimensions',[len(m) for m in masks],flush=True)
    sectors=[sector(m,geo,field,config) for m in masks]
    neutral,positive,negative=sectors;nb=len(field['coordinates']);packet=source_packet(geo,config)
    source_plus=source_operator(neutral,positive,packet,True)
    source_minus=source_operator(neutral,negative,packet,False)
    bplus=(source_plus@neutral['ground'].reshape(len(masks[0]),nb)).ravel()
    bminus=(source_minus@neutral['ground'].reshape(len(masks[0]),nb)).ravel()
    occupation=one_body(neutral['masks'],np.outer(packet,packet.conj()))[0]
    a0=neutral['ground'].reshape(len(masks[0]),nb)
    raw_minus=real(np.vdot(a0,occupation@a0));raw_plus=1-raw_minus
    plusnorm=real(np.vdot(bplus,bplus));minusnorm=real(np.vdot(bminus,bminus))
    initial=bplus/np.sqrt(plusnorm)
    phase=np.exp(1j*config['kick_strength']*field['coordinates'][:,1])
    kicked=(initial.reshape(len(masks[1]),nb)*phase).ravel()
    matter0=matter_observables(neutral,geo);matter=matter_observables(positive,geo)
    vacuum_density=np.array([real(np.vdot(a0,op@a0)) for op in matter0[:27]])
    currents=[-field_operator(positive,d) for d in field['derivative']]
    contacts=[field_operator(positive,d) for d in field['contact']]
    neutral_currents=[-field_operator(neutral,d) for d in field['derivative']]
    neutral_contacts=[field_operator(neutral,d) for d in field['contact']]
    neutral_obs=observe(neutral['ground'],neutral,field,matter0,neutral_currents,neutral_contacts,vacuum_density)
    result=dict(rank=rank,selected_pairs=pairs,fermion_dimensions=[len(m) for m in masks],
        quantum_dimensions=[s['h'].shape[0] for s in sectors],particles=[54,55,53],
        gram=dict(type='Exact orthonormal fixed-orbital Slater determinant Gram',minimum=1.,maximum=1.,whitening='identity, explicitly retained'),
        energies=[s['energy'] for s in sectors],gaps=[positive['energy']-neutral['energy'],negative['energy']-neutral['energy']],
        source=dict(raw_creation_norm2=raw_plus,raw_removal_norm2=raw_minus,raw_CAR_sum=raw_plus+raw_minus,
                    retained_creation_norm2=plusnorm,retained_removal_norm2=minusnorm,
                    omitted_creation_weight=raw_plus-plusnorm,omitted_removal_weight=raw_minus-minusnorm,
                    preparation='Normalized projected charged source; explicitly disclosed source loss. Subsequent field kick has NO projection or conditioning.'),
        neutral=neutral_obs,
        neutral_projection=leakage(neutral,geo,field,neutral['ground']),
        eigensolver_residuals=[float(np.linalg.norm(s['h']@s['ground']-s['energy']*s['ground'])) for s in sectors])
    trajectory,base,pulse=trajectories(positive,field,initial,kicked,config,matter,currents,contacts,vacuum_density)
    result['motion_and_field_reaction']=trajectory
    delta=np.array(trajectory['baseline'][-1]['density'])-trajectory['baseline'][0]['density']
    result['actual_spatial_motion']=dict(density_change_l2=float(np.linalg.norm(delta)),
        per_axis_marginal_change_l2=[float(np.linalg.norm(np.bincount(geo['cells'][:,axis],weights=delta,minlength=3))) for axis in range(3)],
        t0_body_difference_l2=trajectory['density_difference_l2'][0],
        phase_kick_norm_error=abs(float(np.linalg.norm(kicked))-1))
    if full:
        print('Computing projected spectra and frozen-coupling contrast',flush=True)
        spectra_plus=lanczos(positive['h'],bplus,neutral['energy'],config['lanczos_steps'])
        spectra_minus=lanczos(negative['h'],bminus,neutral['energy'],config['lanczos_steps'])
        result['source_G']=dict(addition=spectra_plus,removal=spectra_minus,
            retarded=retarded(spectra_plus,config['frequencies'],config['eta'],True,spectra_minus))
        current_spectra=[];field_spectra=[]
        for j in range(4):
            vector=neutral_currents[j]@neutral['ground'];vector-=neutral['ground']*np.vdot(neutral['ground'],vector)
            spectrum=lanczos(neutral['h'],vector,neutral['energy'],config['lanczos_steps'])
            current_spectra.append(dict(spectrum=spectrum,retarded=retarded(spectrum,config['frequencies'],config['eta'])))
            vector=(a0*field['coordinates'][:,j+1]).ravel();vector-=neutral['ground']*np.vdot(neutral['ground'],vector)
            spectrum=lanczos(neutral['h'],vector,neutral['energy'],config['lanczos_steps'])
            field_spectra.append(dict(spectrum=spectrum,retarded=retarded(spectrum,config['frequencies'],config['eta'])))
        result['current_retarded']=current_spectra;result['field_retarded']=field_spectra
        center=np.argmin(np.sum((field['coordinates']-np.array([config['phi_center'],0,0,0,0]))**2,axis=1))
        frozen=sparse.kron(sparse.eye(len(masks[1])),field['boson'],format='csr')
        frozen+=sparse.kron(one_body(masks[1],geo['onsite'])[0],sparse.eye(nb),format='csr')
        for op,coefficient in zip(positive['operators'],field['coefficients'][:,center]):
            frozen+=coefficient*sparse.kron(op,sparse.eye(nb),format='csr')
        frozen-=np.sum(geo['reference_values'][:54])*sparse.eye(frozen.shape[0],format='csr')
        contrast,_,_=trajectories(positive,field,initial,kicked,config,matter,currents,contacts,vacuum_density,h=frozen)
        result['frozen_field_coupling_control']=contrast
        result['control_meaning']='Same initial charged state and field-only kick; H coupling coefficients frozen at phi_center,A=0. Field kinetic retained, quantum matter-field feedback deliberately removed. Current entries here remain full-law diagnostic operators, not frozen-H currents.'
    arrays=dict(neutral_ground=neutral['ground'],charged_initial=initial,charged_kicked=kicked,
                baseline_states=base,kicked_states=pulse,coordinates=field['coordinates'],orbitals=geo['orbitals'],
                neutral_masks=np.array([str(m) for m in masks[0]]),positive_masks=np.array([str(m) for m in masks[1]]),negative_masks=np.array([str(m) for m in masks[2]]))
    return result,arrays


def main():
    start=time.time();inputpath=Path(sys.argv[1]);output=Path(sys.argv[2]);config=json.loads(inputpath.read_text())
    geo=geometry(config);field=fields(config)
    main_result,arrays=run_rank(config,geo,field,config['pair_rank'],True)
    control,_=run_rank(config,geo,field,config['control_pair_rank'],False)
    result=dict(kind='CONDITIONAL_3D_FILLED_SEA_QUANTUM_VARIATIONAL_CANDIDATE_NOT_NATURE_DERIVATION',
        input=config,geometry=dict(cells=geo['cells'].tolist(),single_particle_modes=108,filled_negative_modes=54,
            q=[0,0,2*np.pi/3],polarizations=['x_cos','x_sin','y_cos','y_sin'],canonical_inertias=[config['volume']*config['scalar_inertia']]+[config['volume']/2]*4,
            reference_negative_values=geo['reference_values'][:54].tolist(),reference_positive_values=geo['reference_values'][54:].tolist()),
        main=main_result,rank_control=control,
        limitations=['Ordered mirror/quaternion Clifford ansatz from actual common_root constructor, frozen frame orientation and primitive CAR/charge/statistics are inputs, not substrate/NoEther derivations.',
            'Full 108-mode CAR determinant signs and actual 54 occupied negative modes. Finite selected single/double pair configurations are NOT complete all-pair Fock space; neutral omitted-H residual and source losses measured.',
            'Periodic Fourier electric kinetic on finite Q boxes with multiplication-coordinate coupling; boundary discontinuity and 3-point field cutoff are severe regulator choices, not converged noncompact canonical fields.',
            'Quantum scalar amplitude has supplied restoring potential about ordered nonzero phi_center. No bare transverse B2/oscillator restoring potential. Full matrix/frame fluctuations absent.',
            'Longitudinal nonuniform interaction is set to zero in this conditional transverse law; uniform ionic background only. Not full Maxwell Gauss completion.',
            'One common nonzero q with both polarizations and sin/cos partners is NOT localized incoming/outgoing photon packet or bulk photon claim.',
            'Variational energy differences are not rigorous gap bounds; fixed internal spin is not full physical Noether spin; no Lorentz/strong/composite/proton/QCD claim.',
            'Lanczos quadratures use both signs and the same projected H, with raw spectral norms retained. Finite Krylov truncation is not complete full-CAR spectral proof.'],elapsed_seconds=time.time()-start)
    np.savez_compressed(output.with_suffix('.npz'),**arrays)
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(result=str(output),elapsed=result['elapsed_seconds'],gaps=main_result['gaps'],motion=main_result['actual_spatial_motion'],projection=main_result['neutral_projection'])),flush=True)


if __name__=='__main__':main()
