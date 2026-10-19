"""NEW literal64-cell allowed-momentum occupied sea and full virtual pair receiver."""
from __future__ import annotations
from itertools import product
import json
from pathlib import Path
import sys
import numpy as np
from scipy.linalg import eigh
from rich_filled_symmetry_20261014_model import Parameters, build, hamiltonian, spectrum, vertex, contact, pair_creation
from rich_review_wave4_20261014_tools import complex_pair, save


def main() -> None:
    inputs=json.loads(Path(sys.argv[1]).read_text());p=Parameters(**inputs['parameters']);model=build(p)
    grid_axis=np.pi*np.array([-1.,-.5,0.,.5]);grid=np.array(list(product(grid_axis,repeat=3)))
    spectra=np.array([eigh(hamiltonian(model,k),eigvals_only=True,check_finite=False) for k in grid])
    number=576//2;sea_energy=float(spectra[:,:number].sum())
    kh=np.zeros(3);qin=np.array([np.pi/2,0.,0.]);qout=np.array([0.,np.pi/2,0.]);kp=kh+qin-qout
    epsin=np.array([0.,1.,0.]);epsout=np.array([0.,0.,1.]);omega=inputs['omega'];eta=inputs['eta']
    eh,vh=spectrum(model,kh);ep,vp=spectrum(model,kp)
    hid=np.flatnonzero(eh<0)[-1];pid=np.flatnonzero(ep>0)[2]
    h=vh[:,hid];particle=vp[:,pid];gap=ep[pid]-eh[hid];wo=omega-gap
    direct=particle.conj()@contact(model,kp,kh,qin,epsin,qout,epsout)@h
    terms=[];unblocked_terms=[];intermediates=[]
    for first_q,first_eps,w,second_eps in [(qin,epsin,omega,epsout),(-qout,epsout,-wo,epsin)]:
        ka=kh+first_q;ea,va=spectrum(model,ka)
        kb=kp-first_q;eb,vb=spectrum(model,kb)
        pa=particle.conj()@vertex(model,kp,ka,second_eps)@va
        ah=va.conj().T@vertex(model,ka,kh,first_eps)@h
        pb=particle.conj()@vertex(model,kp,kb,first_eps)@vb
        bh=vb.conj().T@vertex(model,kb,kh,second_eps)@h
        particle_channels=ea>0;hole_channels=eb<0
        particle_term=np.sum(pa[particle_channels]*ah[particle_channels]/(w-(ea[particle_channels]-eh[hid])+1j*eta))
        hole_term=-np.sum(pb[hole_channels]*bh[hole_channels]/(w-(ep[pid]-eb[hole_channels])+1j*eta))
        terms.append(particle_term+hole_term)
        unblocked_terms.append(np.sum(pa*ah/(w-(ea-eh[hid])+1j*eta))-np.sum(pb*bh/(w-(ep[pid]-eb)+1j*eta)))
        intermediates.append({'particle_momentum':ka.tolist(),'hole_removed_momentum':kb.tolist(),'particle_channels':int(particle_channels.sum()),'hole_channels':int(hole_channels.sum()),'particle_term':complex_pair(particle_term),'hole_term':complex_pair(hole_term)})
    amplitude=direct+sum(terms)
    supplied=pair_creation(model,kh,qin,epsin,qout,epsout,omega,particle=2,hole=-1,eta=eta)
    required=[kh,kp,kh+qin,kp-qin,kh-qout,kp+qout]
    allowed=[bool(np.min(np.linalg.norm(grid-momentum,axis=1))<1e-10) for momentum in required]
    save('filled_grid',{'kind':'INDEPENDENT_NEW_LITERAL_PERIODIC_SEA_CONSUMER','inputs':inputs,'cells':len(grid),'one_particle_modes_per_cell':len(eh),'sea_particles':int(len(grid)*number),
                        'sea_energy':sea_energy,'neutralizing_background_per_cell':number,'all_probe_momenta_present':allowed,
                        'literal_occupied_Nplus1_energy':float(ep[pid]),'literal_occupied_Nminus1_energy':float(-eh[hid]),
                        'literal_pair_energy_difference_from_same_sea':float(gap),'incoming_frequency':omega,'outgoing_frequency':float(wo),
                        'contact_amplitude':complex_pair(direct),'full_Fock_virtual_time_orders':intermediates,'pair_amplitude':complex_pair(amplitude),
                        'supplier_complete_pair_error':float(abs(supplied['amplitude']-amplitude)),
                        'wrong_no_Pauli_blocking_error':float(abs(direct+sum(unblocked_terms)-amplitude)),
                        'scope':'All reference and excitation/intermediate momenta on literal periodic4^3 grid. Same full576-mode law at every grid momentum; no off-grid Esea splice. Complete Pauli-allowed particle/hole channels and both time orders. External-source finite pair response, not normalized photon scattering. This corrective physical consumer was added AFTER late12-file diagnostics and has no inherited checker claim.'})

if __name__=='__main__':
    main()
