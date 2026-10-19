"""New exact mirror-source cyclic physical EM and quantum-field event consumer."""
import json
from math import comb
from pathlib import Path
import sys
import numpy as np
from scipy import sparse
from scipy.linalg import eigh, expm
from rich_full_colour_pairs_20261015_model import build, law, spin_spectrum, retarded, rotate_to_original


def mean(state,operator):
    return float(np.vdot(state,operator @ state).real)


def cj(value):
    value = np.asarray(value)
    return dict(real=value.real.tolist(),imag=value.imag.tolist())


def evolve(state,energies,vectors,times):
    return np.array([vectors @ (np.exp(-1j*energies*t)*(vectors.conj().T @ state)) for t in times])


def rotated_reference_overlap(model,reference,old):
    sector = model['sectors']['neutral']; old_lookup = {int(mask):i for i,mask in enumerate(old['neutral_masks'])}
    rr,cc,vv = [],[],[]
    for j,mask in enumerate(sector['masks']):
        for original,coefficient in rotate_to_original(mask).items():
            if original in old_lookup: rr.append(old_lookup[original]);cc.append(j);vv.append(coefficient)
    rotation = sparse.csr_matrix((vv,(rr,cc)),shape=(len(old_lookup),len(sector['masks'])))
    cross = old['neutral_P'].conj().T @ sparse.kron(rotation,sparse.eye(19)) @ sector['P']
    cross = np.kron(np.asarray(cross),np.eye(len(model['scalar']['grid'])))
    return float(abs(np.vdot(old['reference'],cross @ reference))**2)


def neutral_other_sector_bounds(model):
    p = model['parameters']; field = model['scalar'];loop = model['loop'];grid = field['grid']
    # D_f=I+r Re(exp(iqA)U) >= (1-r)I, since PW19 U is
    # orthogonal compression of the unitary multiplication matrix U(g).
    # Relative to filled negative/positive X sea, k defects mean 2k or
    # 2(12-k) positive particle/hole forms. The bound is a MATRIX inequality
    # at each phi; the complete unchanged scalar kinetic is retained.
    baseline = np.kron(np.eye(19),field['H'])+np.kron(p['electric']*loop['C']/2-p['plaquette']*loop['B'],np.eye(len(grid)))
    bounds = {}
    for k in range(1,12):
        h = baseline.copy()
        for a,phi in enumerate(grid):
            defects = 2*(k if phi >= 0 else 12-k)
            potential = -p['scalar_coupling']*abs(phi)*(12*np.eye(19)+4*p['loop_route']*loop['B']-defects*(1-p['loop_route'])*np.eye(19))
            h[np.ix_(np.arange(19)*len(grid)+a,np.arange(19)*len(grid)+a)] += potential
        bounds[str(k)] = float(eigh(h,eigvals_only=True)[0])
    umat = np.block([[loop['U'][a,b] for b in range(3)] for a in range(3)])
    return dict(bounds=bounds,compressed_U_norm=float(np.linalg.norm(umat,2)),single_particle_D_minimum=float(eigh(np.eye(57)+p['loop_route']*(umat+umat.conj().T)/2,eigvals_only=True)[0]),meaning='Rigorous finite-PW19 operator lower bounds for ALL other neutral rotated total occupations, including all spin/flavour sectors, with the same scalar kinetic; not bounds on charged disconnected sectors.')


def main():
    inp,out = Path(sys.argv[1]),Path(sys.argv[2]); p = json.loads(inp.read_text());model = build(p)
    nf = len(model['scalar']['grid']); stored = {};sectors = {};spectra = {};observables = {}
    for name,sector in model['sectors'].items():
        op = model['operators'][name];h,j,k = model['laws'][name];spin = np.kron(op['S2'],np.eye(nf))
        e,v,s = spin_spectrum(h,spin);spectra[name] = e,v,s
        original_pairs = 6. if name == 'neutral' else 4.5
        obs = dict(current=j,contact=k,spin_squared=spin,loop_C=np.kron(op['C'],np.eye(nf)),loop_B=np.kron(op['B'],np.eye(nf)),matter_octet=np.kron(op['octet'],np.eye(nf)),loop_boundary=np.kron(op['boundary'],np.eye(nf)),phi=np.kron(np.eye(sector['dimension']),model['scalar']['phi']),phi2=np.kron(np.eye(sector['dimension']),model['scalar']['phi']**2),scalar_energy=np.kron(np.eye(sector['dimension']),model['scalar']['H']),scalar_boundary=np.kron(np.eye(sector['dimension']),model['scalar']['boundary']))
        # Original positive number has no off-diagonal matrix element inside a
        # fixed-X occupation branch: it is N/2 times identity. Pair probability
        # is nevertheless broad and is stored explicitly below, not cut away.
        obs['original_pair_number'] = original_pairs*np.eye(len(e));observables[name] = obs
        sectors[name] = dict(number={'neutral':12,'baryon':15,'antibaryon':9}[name],charge={'neutral':0,'baryon':1,'antibaryon':-1}[name],dimension=len(e),gauge_dimension=sector['dimension'],raw_dimension=sector['raw_dimension'],ground_energy=float(e[0]),all_energies=e.tolist(),all_spins=s.tolist(),ground_residual=float(np.linalg.norm(h @ v[:,0]-e[0]*v[:,0])),gauss_error=sector['gauge_error'],route_gauss_commutators=op['route_gauss_commutators'],ground={key:mean(v[:,0],operator) for key,operator in obs.items()})
        for key,value in dict(H=h,J=j,K=k,energies=e,vectors=v,spins=s,P=sector['P'].toarray(),masks=np.array(sector['masks'],np.uint32),**obs).items():stored[name+'_'+key] = value
    reference = spectra['neutral'][1][:,0];e0 = spectra['neutral'][0][0];stored['reference'] = reference
    cn,cb = model['Cneutral'],model['Cbaryon'];stored['Cneutral'] = cn;stored['Cbaryon'] = cb
    hn,jn,kn = model['laws']['neutral'];hb,jb,kb = model['laws']['baryon'];ha,ja,ka = model['laws']['antibaryon']
    cchecks = dict(neutral_unitarity=float(np.linalg.norm(cn.conj().T @ cn-np.eye(len(cn)))),charged_unitarity=float(np.linalg.norm(cb.conj().T @ cb-np.eye(cb.shape[1]))),neutral_H=float(np.linalg.norm(hn @ cn-cn @ hn)),neutral_J=float(np.linalg.norm(jn @ cn+cn @ jn)),neutral_K=float(np.linalg.norm(kn @ cn-cn @ kn)),H=float(np.linalg.norm(ha @ cb-cb @ hb)),J=float(np.linalg.norm(ja @ cb+cb @ jb)),K=float(np.linalg.norm(ka @ cb-cb @ kb)),neutral_reference_C_overlap=float(abs(np.vdot(reference,cn @ reference))**2))
    sources = {};receivers = dict(neutral=reference)
    for name,matrices in [('baryon',model['sources']),('antibaryon',model['anti_sources'])]:
        e,v,spins = spectra[name];sources[name] = {}
        for spin,source in matrices.items():
            state = source @ reference;weights = abs(v.conj().T @ state)**2;bright = np.flatnonzero(weights>1e-11);pole = abs(e-e[bright[0]])<1e-8
            sources[name][spin] = dict(norm_squared=float(np.vdot(state,state).real),sumrule=float(weights.sum()),lowest_bright_gap=float(e[bright[0]]-e0),pole_residue=float(weights[pole].sum()),normalized_pole_residue=float(weights[pole].sum()/weights.sum()),all_gaps=(e-e0).tolist(),all_weights=weights.tolist(),spin_residual=float(np.linalg.norm(observables[name]['spin_squared'] @ state-float(spin)*(float(spin)+1)*state)))
            stored[name+'_source_'+spin] = source;stored[name+'_source_state_'+spin] = state
            if spin == '0.5':
                receiver = v[:,pole] @ (v[:,pole].conj().T @ state);receivers[name] = receiver/np.linalg.norm(receiver)
    responses = {};events = {};times = np.array(p['times'])
    for name,state in receivers.items():
        h,j,k = model['laws'][name];e,v,_ = spectra[name]
        response = retarded(e,v,j,state,p['frequencies'],p['eta'],k)
        responses[name] = {key:(cj(value) if key in ['first','second','retarded','kernel'] else value.tolist() if isinstance(value,np.ndarray) else value) for key,value in response.items()}
        responses[name].update(virtual_states=len(e),direct_current_norm_squared=mean(state,j @ j),spectral_sum=float(response['weights'].sum()),current_eigenstate_residual=float(np.linalg.norm(h @ state-response['energy']*state)))
        angle = p['pulse_angle']*(-1 if name == 'antibaryon' else 1);hp,jp,kp = law(model['operators'][name],model['scalar'],p,angle)
        preparations = dict(EMflux=expm(-1j*p['pulse_duration']*hp) @ state,fieldonly=np.kron(np.eye(model['sectors'][name]['dimension']),expm(-1j*p['field_kick']*model['scalar']['P'])) @ state)
        baseline = evolve(state,e,v,times);base = {key:[mean(x,operator) for x in baseline] for key,operator in observables[name].items()}
        events[name] = {}
        for prep,kicked in preparations.items():
            states = evolve(kicked,e,v,times);data = {key:[mean(x,operator) for x in states] for key,operator in observables[name].items()}
            outcomes = abs(v.conj().T @ kicked)**2
            events[name][prep] = dict(times=times.tolist(),baseline=base,event=data,excess={key:(np.array(data[key])-np.array(base[key])).tolist() for key in data},initial_changes={key:mean(kicked,operator)-mean(state,operator) for key,operator in observables[name].items()},norm_error=float(abs(np.linalg.norm(kicked)-1)),energy_injection=mean(kicked,h)-mean(state,h),inclusive_probabilities=outcomes.tolist(),inclusive_sum=float(outcomes.sum()))
            stored[name+'_'+prep+'_states'] = states;stored[name+'_'+prep+'_prepared'] = kicked
        stored[name+'_baseline_states'] = baseline;stored[name+'_receiver'] = state
        delta = 1e-4;hp,_,_ = law(model['operators'][name],model['scalar'],p,delta);hm,_,_ = law(model['operators'][name],model['scalar'],p,-delta)
        responses[name]['Wilson_derivative_errors'] = dict(current=float(np.linalg.norm((hp-hm)/(2*delta)+j)),contact=float(np.linalg.norm((hp-2*h+hm)/delta**2-k)))
    oldjson = json.loads((inp.parent/p['frozen_result']).read_text());old = np.load(inp.parent/p['frozen_arrays'])
    comparison = dict(frozen_reference_overlap_squared=rotated_reference_overlap(model,reference,old),frozen_paircut1_energies={name:oldjson['sectors'][name]['energy'] for name in sectors},new_minus_frozen_energy={name:sectors[name]['ground_energy']-oldjson['sectors'][name]['energy'] for name in sectors},frozen_gaps=oldjson['positive_excitation_gaps'],frozen_scalar_only_gap=oldjson['full_pair_scalar_only_control']['baryon_gap'],frozen_reference_pair_boundary=oldjson['sectors']['neutral']['ground']['pair_number'])
    bounds = neutral_other_sector_bounds(model);bounds['lowest_other_sector_bound'] = min(bounds['bounds'].values());bounds['certifies_extremal_neutral_branches_ground'] = bounds['lowest_other_sector_bound']>float(e0)
    distributions = {name:{str(k):comb(12 if name == 'neutral' else 9,k)/2**(12 if name == 'neutral' else 9) for k in range(13 if name == 'neutral' else 10)} for name in sectors}
    result = dict(schema=1,parameters=p,sectors=sectors,sources=sources,C=cchecks,current_response=responses,events=events,positive_selected_source_cyclic_gaps={name:sectors[name]['ground_energy']-e0 for name in ['baryon','antibaryon']},original_pair_distributions=distributions,comparison=comparison,neutral_full_fixedN_bound=bounds,
        closure=dict(representation='a_plus=(c_positive+c_negative)/sqrt2; a_minus=(c_positive-c_negative)/sqrt2. H(A),J(A),K(A) all conserve each rotated mirror/flavour/spin occupation. Reference fills ALL a_minus modes. Physical original c_positive^dagger source reduces exactly to a_plus^dagger/sqrt2 on that reference.',neutral='N12: a_minus12,a_plus0; full Gauss loop singlets and quantum scalar. The mirror-reflected phi-parity branch is exactly degenerate and gives the same response spectrum.',baryon='N15: a_minus12,a_plus3, flavour uud and M=+1/2, ALL Gauss/spin sectors. Includes all original hole-pair numbers0..9, not original paircut0.',antibaryon='N9: actual same-CAR C image of N15, a_minus9,a_plus0. No independent hole species.',omitted_current_connected_complement='EXACTLY ZERO: all H(A)/J(A)/K(A) preserve these rotated occupations; all spectral virtuals retained. Physical original pair number operator has off-branch components, so its distribution is calculated in the full original CAR basis rather than claiming the branch projector is its complete measurement algebra.',missing_disconnected_complement='Other rotated-occupation blocks at fixed N15/N9 and every other total N are not solved. No selected charged globalground/all-number stability assertion. Original negative-mirror determinant is NOT the new interacting reference; the same-law filled-reference ground is recomputed, with exact neutral lower bounds recorded.'),scope='Finite same-coupled-law exact source/current cyclic sectors; original full-pair physics retained by exact mirror basis. PW19 and scalar7DVR unchanged. No added mass/penalty/tuned parameter, no scalar-only substitution, no dynamic photon/COM/QCD/nature/global charged stability claim.')
    out.write_text(json.dumps(result,indent=2)+'\n');np.savez_compressed(out.with_suffix('.npz'),**stored)
    print(json.dumps(dict(dimensions={name:data['dimension'] for name,data in sectors.items()},gaps=result['positive_selected_source_cyclic_gaps'],C=cchecks,neutral_other_bound=bounds['lowest_other_sector_bound'],neutral_ground=float(e0),comparison=comparison)))


if __name__ == '__main__':
    main()
