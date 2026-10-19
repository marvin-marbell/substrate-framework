"""Discriminating new field-frame/source/current/quantum feedback calculation."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
from scipy.linalg import eigh, expm
from rich_frame_gauss_20261014_model import frame_gauss, carrier, spatial_consumer, charge_geometry, angular


def norm(a):
    return float(np.linalg.norm(a))


def encode(a):
    a = np.asarray(a)
    return {'real':a.real.tolist(),'imag':a.imag.tolist()}


def run(p, destination):
    frame = frame_gauss(); pp = frame['P']; nn = frame['N']; src = frame['source']
    i0 = np.flatnonzero(abs(nn)<1e-8); i1 = np.flatnonzero(abs(nn-1)<1e-8); i2 = np.flatnonzero(abs(nn-2)<1e-8)
    physical_gauss = max(norm(g@pp) for g in frame['G'])
    bare_leak = [norm((np.eye(56)-pp@pp.conj().T)@c@pp[:,i0]) for c in frame['bare']]
    dressed_leak = [norm((np.eye(56)-pp@pp.conj().T)@c@pp) for c in frame['raw_source']]
    car = [[norm(a.conj().T@b+b@a.conj().T-(np.eye(4) if x==y else np.zeros((4,4)))) for y,b in enumerate(src)] for x,a in enumerate(src)]
    car_same = [[norm(a@b+b@a) for b in src] for a in src]
    charge_comm = [norm(np.diag(p['charge']*nn)@a-a@np.diag(p['charge']*nn)-p['charge']*a) for a in src]
    noether = sum(s@s for s in frame['S'])
    rotation2 = expm(-2j*np.pi*frame['S'][2]); rotation4 = expm(-4j*np.pi*frame['S'][2])
    raw_j = [np.kron(a,np.eye(4)) for a in frame['left']]
    comm_bare = max(norm(j@c-c@j) for j in raw_j for c in frame['bare'])
    comm_dressed = [norm(raw_j[2]@c-c@raw_j[2]) for c in frame['raw_source']]
    # Actual differential pullback on U, independent of a physical Pauli assignment.
    theta = 1e-6
    derivative = (expm(-1j*theta*frame['S'][2])-expm(1j*theta*frame['S'][2]))/(-2j*theta)
    # Independent group-coordinate derivative at an actual nonidentity field U.
    u = expm(-.73j*angular(.5)[0])@expm(.41j*angular(.5)[1])
    coordinate_derivative = (expm(1j*theta*angular(.5)[2])@u-expm(-1j*theta*angular(.5)[2])@u)/(-2j*theta)
    coordinate_residual = norm(coordinate_derivative+angular(.5)[2]@u)
    sectors = []
    states = {}
    for occupation in (0,1,2):
        for l in range(p['lmax']+1):
            body = carrier(p,l,occupation); vals, vec = eigh(body['H'])
            states[(occupation,l)] = (body,vals,vec)
            total2 = sum(a@a for a in body['total'])
            ground = vec[:,vals < vals[0]+1e-8]
            sectors.append({'N':occupation,'charge':p['charge']*occupation,'l':l,'K_multiplicity':2*l+1,'block_dimension':len(vals),'complete_dimension':len(vals)*(2*l+1),'energy_min':float(vals[0]),'ground_multiplicity_per_K':ground.shape[1],'physical_total_J2_ground':np.linalg.eigvalsh(ground.conj().T@total2@ground).tolist(),'rotation_commutator':max(norm(body['H']@a-a@body['H']) for a in body['total']),'energies':vals.tolist()})
    neutral,v0,u0 = states[(0,0)]; odd,v1,u1 = states[(1,0)]; pair,v2,u2 = states[(2,0)]
    e0 = float(v0[0]); nb = odd['nb']; vacuum = u0[:,0]
    source_vectors = np.column_stack([np.kron(s[np.ix_(i1,i0)],np.eye(nb))@vacuum for s in src])
    overlaps = u1.conj().T@source_vectors
    z = p['frequency']+1j*p['broadening']
    g = overlaps.conj().T@((1/(z-(v1-e0)))[:,None]*overlaps)
    weights = np.sum(abs(overlaps)**2,axis=1)
    dominant = [{'energy':float(v1[k]-e0),'weight':float(weights[k])} for k in np.flatnonzero(weights>1e-7)]
    # Adverse bare source never enters the physical Gauss sector.
    bare_gram = [[float(np.vdot(pp[:,i1].conj().T@a@pp[:,i0],pp[:,i1].conj().T@b@pp[:,i0]).real) for b in frame['bare']] for a in frame['bare']]
    ground = u1[:,v1 < v1[0]+1e-8]
    magnetic_gram = np.column_stack([a@ground[:,0] for a in odd['mu']])
    magnetic_gram = magnetic_gram.conj().T@magnetic_gram
    qops = [np.kron(np.eye(2),a) for a in odd['field']['Q']]
    qvectors = np.column_stack([a@ground[:,0] for a in qops])
    qover = u1.conj().T@qvectors
    d = qover.conj().T@((1/(z-(v1-v1[0])))[:,None]*qover)
    occupation_operator = np.kron(np.eye(2),np.diag([sum(n) for n in odd['field']['basis']]))
    field_number = float(np.trace(ground.conj().T@occupation_operator@ground).real/ground.shape[1])
    field_vector = [float(np.vdot(ground[:,0],a@ground[:,0]).real) for a in qops]
    # Magnetic contact comes from the very same minimal substitution.
    contact = odd['C']**2*(1/p['common_inertia']+1/p['relative_inertia'])*np.eye(len(v1))
    bprobe = p['probe_field']; hz = odd['H']-bprobe*odd['mu'][2]+.5*bprobe**2*contact
    hminus = odd['H']+bprobe*odd['mu'][2]+.5*bprobe**2*contact
    finite_difference_current = norm((hminus-hz)/(2*bprobe)-odd['mu'][2])
    finite_difference_contact = norm((hz+hminus-2*odd['H'])/bprobe**2-contact)
    # Restore ALL K copies and higher l for the actual finite-q spatial current.
    spatial = spatial_consumer(p,p['wavevector']); hh=spatial['H']; rho=spatial['rho']; jj=spatial['current']
    ward = sum(q*a for q,a in zip(p['wavevector'],jj))+hh@rho-rho@hh
    ward_relative = norm(ward)/max(norm(hh@rho-rho@hh),1e-14)
    # l=0 source into the full orientation basis, field vacuum and its dressed eigenstate.
    nw = len(spatial['spatial']['basis']); full_source = np.zeros((nw*2*nb,2),complex)
    full_source[:2*nb,:] = source_vectors
    # This is the common-H finite-q resolvent/current consumer, not a stitched G/D.
    sv, su = eigh(hh)
    prepared = su[:,sv < sv[0]+1e-8][:,0]
    legs = np.column_stack([a@prepared for a in jj])
    leg_over = su.conj().T@legs
    current_gram = legs.conj().T@legs
    current_response = leg_over.conj().T@((1/(z-(sv-sv[0])))[:,None]*leg_over)
    source_overlap = su.conj().T@full_source
    resolvent_source = su@((1/(z-(sv-e0)))[:,None]*source_overlap)
    outgoing_z = z+.2
    outgoing_bra = (source_overlap.conj().T*(1/(outgoing_z-(sv-e0)))[None,:])@su.conj().T
    same_source_vertex = np.stack([full_source.conj().T@su@((1/(z-(sv-e0)))[:,None]*(su.conj().T@a@full_source)) for a in jj])
    threepoint = np.stack([outgoing_bra@a@resolvent_source for a in jj])
    ward_leg = ward@prepared
    topfield = np.kron(np.eye(2),np.diag([sum(n)==p['field_cutoff'] for n in odd['field']['basis']]))
    top_probability = float(np.trace(ground.conj().T@topfield@ground).real/ground.shape[1])
    electric_energy, magnetic_tensor = charge_geometry(p)
    k = np.array(p['wavevector']); k2 = np.dot(k,k)
    electric_field = [-1j*component*rho/(p['permittivity']*k2) for component in k]
    electric_gauss = norm(1j*sum(component*a for component,a in zip(k,electric_field))-rho/p['permittivity'])
    projected_mu = [ground.conj().T@a@ground for a in odd['mu']]
    projected_total = [ground.conj().T@a@ground for a in odd['total']]
    mu_ratio = float(sum(np.trace(a.conj().T@b).real for a,b in zip(projected_total,projected_mu))/sum(np.trace(a.conj().T@a).real for a in projected_total))
    # A physical material comparator: leave frame/Gauss untouched, suppress
    # common inverse metric. Halfspin survives; charge's spin magnetism does not.
    uncrossed_parameters = dict(p,common_inertia=1e12)
    uncrossed = carrier(uncrossed_parameters,0,1)
    uv, uu = eigh(uncrossed['H']); ug = uu[:,uv < uv[0]+1e-8]
    uncrossed_mu = max(norm(ug.conj().T@a@ug) for a in uncrossed['mu'])
    # Integrate x cross j/2 over the SAME six Gaussian clouds. No Noether=
    # magnetization identity is assumed: the actual moment of inertia tensor
    # weights the charge velocity derived from the inverse scalar-field metric.
    points = p['radius']*np.concatenate((np.eye(3),-np.eye(3)))
    tensor_from_current = p['charge']/12*sum((np.dot(x,x)+2*p['cloud_width']**2)*np.eye(3)-np.outer(x,x) for x in points)
    moment_from_current = [sum(tensor_from_current[a,b]*(odd['da'][b]/p['common_inertia']+odd['db'][b]/p['relative_inertia']) for b in range(3)) for a in range(3)]
    current_moment_residual = max(norm(a-b) for a,b in zip(moment_from_current,odd['mu']))
    vector_residual = 0.
    for a in range(3):
        for b in range(3):
            expected = np.zeros_like(odd['H'])
            if a != b:
                caxis = 3-a-b
                sign = 1 if (a,b,caxis) in ((0,1,2),(1,2,0),(2,0,1)) else -1
                expected = 1j*sign*odd['mu'][caxis]
            vector_residual = max(vector_residual,norm(odd['total'][a]@odd['mu'][b]-odd['mu'][b]@odd['total'][a]-expected))
    result = {
      'candidate':'new_scalar_colour_shell_plus_SU2_fieldframe_right_Gauss_positive_cross_metric',
      'parameters':p,'units':'hbar=1; charge q=-1; no identified light speed or rest/inertial relation',
      'Gauss':{'raw_dimension':56,'physical_dimension':4,'sector_dimensions':[len(i0),len(i1),len(i2)],'physical_residual':physical_gauss,'bare_creation_leakage':bare_leak,'bare_physical_gram':bare_gram,'dressed_leakage':dressed_leak,'dressed_CAR_residuals':car,'vacuum_source_gram':encode(source_vectors.conj().T@source_vectors),'pair_source_gram':encode(np.column_stack([s[np.ix_(i2,i1)].conj().T.ravel() for s in src]).conj().T@np.column_stack([s[np.ix_(i2,i1)].conj().T.ravel() for s in src]))},
      'physical_Noether':{'generator':'left differential on U; primitive scalar colour source commutes with it','J2_by_N':{str(n):np.linalg.eigvalsh(noether[np.ix_(ix,ix)]).tolist() for n,ix in ((0,i0),(1,i1),(2,i2))},'rotation_2pi':encode(rotation2),'rotation_4pi_residual':norm(rotation4-np.eye(4)),'proper_derivative_residual':norm(derivative-frame['S'][2]),'bare_Noether_commutator':comm_bare,'dressed_Noether_commutator_norms':comm_dressed},
      'complete_sectors':sectors,
      'charged_consumer':{'neutral_reference_energy':e0,'odd_charge_creation_energy':float(v1[0]-e0),'even_pair_creation_energy':float(v2[0]-e0),'uncoupled_frame_creation':.75/(2*p['frame_inertia'])+.75/(2*p['common_inertia']),'source_resolvent_G':encode(g),'source_poles':dominant,'current_vertex_3x2x2':encode(same_source_vertex),'unamputated_threepoint_3x2x2':encode(threepoint),'incoming_z':[float(z.real),float(z.imag)],'outgoing_z':[float(outgoing_z.real),float(outgoing_z.imag)],'finite_q_current_gram':encode(current_gram),'finite_q_current_gram_eigenvalues':np.linalg.eigvalsh(current_gram).tolist(),'finite_q_current_response':encode(current_response),'spatial_Ward_relative_residual':ward_relative,'spatial_Ward_source_residual':norm(ward_leg),'full_spatial_dimension':len(hh),'physical_positions':'six Gaussian clouds centred x=W(+/-radius e_a), width sigma, each occupation charge qN/6'},
      'same_current_field_feedback':{'magnetic_gram':encode(magnetic_gram),'field_resolvent_D':encode(d),'field_number_in_odd_ground':field_number,'field_Q_in_selected_odd_ground':field_vector,'top_field_cutoff_probability':top_probability,'same_current_contact':float(contact[0,0].real),'finite_difference_current_residual':finite_difference_current,'finite_difference_contact_residual':finite_difference_contact,'magnetic_probe_ground_energy':float(eigh(hz,eigvals_only=True)[0]),'zero_probe_ground_energy':float(v1[0])},
      'longitudinal_electric_Gauss':{'law':'epsilon div E=rho, epsilon/2 integral E^2; supplied electric law, exact static constrained solution','energy_N1':electric_energy,'energy_by_N':[electric_energy*n*n for n in (0,1,2)],'charge_by_N':[p['charge']*n for n in (0,1,2)],'Fourier_Gauss_residual':electric_gauss,'source_charge_commutator_residuals':charge_comm,'pair_minus_two_odd_creation_energy':float(v2[0]+e0-2*v1[0]),'source_dressing':'N to N+1 shifts this same constrained E(W,x) by one charge cloud; eliminated longitudinal field is not an independent field vacuum'},
      'independent_discriminators':{'same_creation_CAR_residuals':car_same,'direct_group_coordinate_Noether_derivative_residual':coordinate_residual,'magnetic_moment_per_physical_total_J':mu_ratio,'uncrossed_metric_magnetic_ground_norm':uncrossed_mu,'uncrossed_metric_halfspin_J2':.75,'charge_magnetic_tensor':magnetic_tensor,'actual_current_integral_moment_residual':current_moment_residual,'physical_current_vector_rotation_residual':vector_residual,'tensor_from_actual_charge_positions':tensor_from_current.tolist()},
      'open_inputs':['primitive scalar fermionic CAR and colour fundamental representation','SU2 frame cover and centre sewing of associated bundle','rigid Gaussian octahedral charged cloud and neutral marked SO3 carrier; not self-bound','positive angular field metric and radii/inertias','three supplied vector magnetic oscillators, not spatial photons or derived Maxwell stiffness','supplied electric quadratic law; longitudinal Gauss integrated exactly, no dynamical electric or free COM','finite lmax and field occupation cutoff; actual retained-boundary probability reported','no positron: N2 is two same-sign charges, N0 is neutral vacuum','no E=mc2, Lorentz invariance, g=2 or infrared electron claim']}
    destination.write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(destination.with_suffix('.npz'),physical_projector=pp,physical_number=nn,physical_Noether=np.array(frame['S']),source=np.array(src),odd_H=odd['H'],odd_eigenvalues=v1,odd_eigenvectors=u1,source_vectors=source_vectors,spatial_H=hh,spatial_current=np.array(jj),spatial_density=rho,current_legs=legs,spatial_state=prepared,field_Q=np.array(qops),magnetic_source=np.array(odd['mu']),contact=contact)
    print(json.dumps({'Gauss':result['Gauss'],'physical_Noether':result['physical_Noether'],'charged_consumer':result['charged_consumer'],'same_current_field_feedback':result['same_current_field_feedback']}))


if __name__ == '__main__':
    run(json.loads(Path(sys.argv[1]).read_text()),Path(sys.argv[2]))
