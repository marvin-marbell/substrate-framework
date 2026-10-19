"""Independent NEW source/receiver contrasts; no author CLI replay or tensor join."""
from __future__ import annotations
import json
from dataclasses import asdict
from pathlib import Path
from typing import cast
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import expm_multiply
import rich_spatial_20261014_model as spatial
import rich_spinor_rot_20261014_model as angular
import rich_spinor_gauss_20261014_model as gauss
import rich_composite_rot_20261014_consumer as composite

ROOT = Path(__file__).resolve().parent

def norm(op):
    return float(np.linalg.norm(op.data if sparse.issparse(op) else op))

def spatial_probe():
    """Fresh remote electric receiver, loop/electric sources and a TRUE middle cut."""
    rows = []
    times = np.linspace(0., 4.3, 44)
    for name, hopping, rail, cut, number in [
        ('connected', 1., 1., False, 2), ('outer_rails_removed', 1., 0., False, 2),
        ('all_middle_links_cut', 1., 1., True, 2), ('no_motion', 0., 1., False, 2),
        ('bare_electric_vacuum', 1., 1., False, 0)]:
        s = spatial.Sector(spatial.Parameters(cells=3, hopping=hopping, rail_hopping=rail), number//2, number//2)
        if cut:
            for edge in (3, 4, 5):
                s.H -= s.transfers[edge]+s.transfers[edge].T
                s.transfers[edge] = sparse.csr_matrix(s.H.shape)
                s.currents[edge] = sparse.csr_matrix(s.H.shape, dtype=complex)
        if number:
            e, v = s.solve(4); g = v[:, 0]; energy = float(e[0])
        else:
            energy = 0.; g = np.zeros(s.dimension, complex)
            g[s.index[(0, tuple(np.zeros(len(s.chords), int)))]] = 1.
        w0 = s.wilson(s.faces[0]); w2 = s.wilson(s.faces[4])
        f0 = (w0-w0.T)/(2j); f2 = (w2-w2.T)/(2j)
        e0 = sparse.diags(s.electric[:, 0]); e2 = sparse.diags(s.electric[:, 6])
        chi, sources = spatial.time_response(s, g, energy, [f0, e0, f2, e2], times, [0, 1])
        kicked = expm_multiply(-.17j*f0, g)
        waves = expm_multiply(-1j*s.H, kicked, start=0., stop=4.3, num=len(times), traceA=-1j*float(s.H.diagonal().sum()))
        shift = np.abs(waves)**2 @ (s.electric[:, 6]**2/2) - np.abs(g)**2 @ (s.electric[:, 6]**2/2)
        rows.append(dict(name=name, parameters=asdict(s.p), number=number, dimension=s.dimension,
                         cut_edges=[3,4,5] if cut else [], energy=energy,
                         remote_chi_flux_from_flux=float(np.max(np.abs(chi[:, 2, 0]))),
                         remote_chi_E_from_flux=float(np.max(np.abs(chi[:, 3, 0]))),
                         remote_chi_E_from_E=float(np.max(np.abs(chi[:, 3, 1]))),
                         local_flux_response=float(np.max(np.abs(chi[:, 0, 0]))),
                         remote_electric_energy_kick_shift=float(np.max(np.abs(shift))),
                         source_weights=np.sum(np.abs(sources)**2, axis=0).tolist(),
                         boundary_weight=float(np.abs(g)**2 @ np.any(abs(s.electric)==1, axis=1)),
                         gauss=s.gauss_error, continuity=s.continuity_error()))
    # Different literal bounded-link roots; this is not an ordinary-rotor convergence claim.
    roots = []
    for bound in (1,2):
        s = spatial.Sector(spatial.Parameters(cells=1, electric_bound=bound))
        e,v=s.solve(4); g=v[:,0]; w=s.wilson(s.faces[0]); f=(w-w.T)/(2j)
        chi,_=spatial.time_response(s,g,float(e[0]),[f],times,[0])
        roots.append(dict(bound=bound, dimension=s.dimension, energy=float(e[0]),
                          boundary_weight=float(np.abs(g)**2 @ np.any(abs(s.electric)==bound,axis=1)),
                          local_flux_chi_max=float(np.max(abs(chi)))))
    return dict(times=times.tolist(), rows=rows, bounded_root_contrast=roots,
                scope='Actual link/time receivers on open anchored ladder. Outer-rail removal is NOT complete spatial disconnection. No Bloch momentum, massless speed or ordinary rotor convergence asserted.')

def angular_probe():
    rows=[]
    qin=np.array([.31,-.27,.44]); qout=np.array([-.19,.38,.23]); ein=np.array([1.,.2,-.1]); eout=np.array([.1,1.,-.3])
    for coupling in (0.,.45):
        m=angular.solve(angular.build(angular.Parameters(charged_lmax=2, coupling=coupling, field_cutoff=2)))
        h=m['H']; e=m['energies']; v=m['vectors']; src=angular.vacuum_sources(m); z=.37+.09j
        spectral=angular.charged_green(m,z.real,z.imag)
        direct=src.conj().T @ np.linalg.solve(z*np.eye(len(h))-h,src)
        # API index order is tested against actual operator resolvent, not a diagonal-only control.
        jin=angular.current(m,qin,ein); jout=angular.current(m,qout,eout)
        co=angular.contact(m,qout,eout,qin,ein); i=0; f=1; omega=.63; eta=.09
        full=angular.amplitude(m,i,f,jin,jout,co,omega,eta)
        low=angular.amplitude(m,i,f,jin,jout,co,omega,eta,virtual=np.arange(4))
        a=v[:,i]; b=v[:,f]; wo=omega+e[i]-e[f]
        actual=b.conj()@co@a + b.conj()@jout.conj().T@np.linalg.solve((e[i]+omega+1j*eta)*np.eye(len(h))-h,jin@a) + b.conj()@jin@np.linalg.solve((e[i]-wo+1j*eta)*np.eye(len(h))-h,jout.conj().T@a)
        nc=len(m['charged']['basis']); nn=len(m['neutral']['basis']); nf=m['field_dimension']
        bare=np.kron(np.kron(np.kron(angular.one_current(m['charged'],m['params'],qin,ein),np.eye(2)),np.eye(nn)),np.eye(nf))
        weights=v.conj().T@src
        rows.append(dict(coupling=coupling, parameters=m['parameters'], dimension=len(h), G_shape=list(spectral.shape),
                         G_direct_error=float(np.linalg.norm(spectral-direct)), G_transpose_direct_error=float(np.linalg.norm(spectral.T-direct)),
                         source_isometry_error=float(np.linalg.norm(src.conj().T@src-np.eye(2*nc))),
                         full_receiver_sum_rule_error=float(np.linalg.norm(weights.conj().T@weights-np.eye(2*nc))),
                         lowest_four_source_weight=float(np.sum(abs(weights[:4])**2)), total_source_weight=float(np.sum(abs(weights)**2)),
                         amplitude_direct_error=float(abs(full-actual)), four_state_amplitude_error=float(abs(full-low)),
                         full_amplitude=[float(full.real),float(full.imag)],
                         omitted_field_current_source_norm=float(np.linalg.norm((jin-bare)@a)),
                         rotation_H_commutator_norms=[norm(h@j-j@h) for j in m['angular']]))
    return dict(rows=rows, scope='Addition-only empty-reference G. Supplied isotropic vector cavity, not Gauss field or massless photons. Full finite receiver compared to resolvent and premature four-state receiver.')

def gauss_probe():
    rows=[]
    for ref in (0.,.3):
        s=gauss.Sector(gauss.Parameters(flux_reference=ref),1); e,v=s.solve(6); g=v[:,1]
        profile=np.linspace(-.19,.23,len(s.edges)); phase=.41
        # D(E)=exp(i phase profile.E): D T_l Ddag=exp(-i phase profile_l) T_l.
        diagonal=np.repeat(np.exp(1j*phase*(s.e@profile)),4)
        translated=sparse.diags(diagonal)@s.H@sparse.diags(diagonal.conj())
        phased=s.H.copy()
        for f,t in zip(profile,s.transfers):
            phased+=(np.exp(-1j*phase*f)-1)*t+(np.exp(1j*phase*f)-1)*t.getH()
        flux=s.wilson([0,2,1])-s.wilson([0,3,1]); m=s.magnetic(2); sz=s.spin[2]
        # Same E-unitary must also translate EXTERNAL Wilson restoring reference to be isospectral.
        rows.append(dict(parameters=asdict(s.p), dimension=s.dimension,
                         hopping_only_phase_vs_full_E_unitary_norm=norm(phased-translated),
                         original_state_phase_rotated_energy=float(np.vdot(diagonal*g,translated@(diagonal*g)).real), original_energy=float(e[1]),
                         total_Peierls_current=float(np.vdot(g,m@g).real), flux_mean=float(np.vdot(g,flux@g).real),
                         spin_flux_connected=float((np.vdot(g,sz@flux@g)-np.vdot(g,sz@g)*np.vdot(g,flux@g)).real),
                         Wilson_loop_source_H_change_norm=norm(.0017*flux),
                         flux_response_at_new_z=gauss.retarded(s,g,float(e[1]),[flux,sz],.43+.11j).tolist(),
                         constraints=s.constraint()))
    return dict(rows=rows, scope='Complete electric-coordinate unitary family and hopping-only external-phase source differ with Kref. Loop-source H-lambdaF is physical compact-flux source; neither this nor a source-current derivative defines continuum B.')

def composite_probe():
    p=json.loads((ROOT/'rich_composite_rot_20261014_magnetic_request.json').read_text())
    h,m=composite.build(p); data=np.load(ROOT/'rich_composite_rot_20261014_magnetic1_receipt.npz'); g=data['ground_plus']
    current,_=composite.current_contact(m,[.29,-.41,0.],[1.,.3,0.]); magnetic,_=composite.current_contact(m,[0,0,0],[0,0,0],magnetic=True)
    positions = cast(np.ndarray, m['positions'])
    charge=np.exp(1j*(composite.COORDS[positions]@np.array([.29,-.41,0.])))@m['charges']
    rho=sparse.diags(np.repeat(charge,m['ns']*m['nf']))
    comm=[h@j-j@h for j in m['J']]
    return dict(inputs=p, state_input='magnetic1_receipt.npz ground_plus', dimension=len(g), state_norm=float(np.vdot(g,g).real),
                state_eigen_residual=float(np.linalg.norm(h@g-np.vdot(g,h@g)*g)),
                J2=composite.quadratic_generator(g,m['J']), Jz=composite.expectation(g,m['J'][2]),
                H_internal_J_commutator_source_norms=[float(np.linalg.norm(c@g)) for c in comm],
                current_internal_J_commutator_norms=[norm(j@current-current@j) for j in m['J']],
                magnetic_internal_J_commutator_norms=[norm(j@magnetic-magnetic@j) for j in m['J']],
                density_internal_J_commutator_norms=[norm(j@rho-rho@j) for j in m['J']],
                magnetic_plus=composite.expectation(g,magnetic),
                scope='Internal J rotates primitive spins and local orientors, not square positions. Bare physical current/density are scalars under internal J at spin_orbit0. Axial H conserves only Jz; this is not free spatial Noether spinhalf or a spatial vector-current rotation claim.')

def encode(value):
    if isinstance(value,complex): return [value.real,value.imag]
    if isinstance(value,np.generic): return value.item()
    raise TypeError(type(value).__name__)

def main():
    result={}
    for name,probe in [('spatial',spatial_probe),('angular',angular_probe),('gauss',gauss_probe),('composite',composite_probe)]:
        result[name]=probe(); print(json.dumps({name:result[name]},default=encode),flush=True)
    (ROOT/'rich_review_wave3_20261014_result.json').write_text(json.dumps(result,default=encode,indent=2)+'\n')

if __name__=='__main__':
    main()
