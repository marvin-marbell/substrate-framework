"""Actual canonical-source Hamiltonian differentiation, all angular products first."""
import json
from pathlib import Path
from itertools import product
import numpy as np
from scipy import sparse as sp
from scipy.sparse.linalg import expm_multiply
import rich_coupled_frame_photon_20261016_model as law
import rich_gauge_band_frame_20261016_model as band

ROOT=Path(__file__).resolve().parent;P='rich_review_wave45_20261016_'


def main():
    p=dict(law.DEFAULTS,field_total_cut=0);m=law.Model(5,p);hm=m.matter;ref=band.Model(4)
    neutral=expm_multiply(-.13j*ref.H,ref.preparation());x=hm.source_from(ref,0,True)@neutral;x/=np.linalg.norm(x)
    a=band.angular((0,1,2));nf,ng,nm,nr=hm.shape;ne=len(a['labels'])
    ir=[a['labels'].index(z) for z in hm.material['labels']]
    er=sp.csr_matrix(([1.]*nm,(ir,range(nm))),shape=(ne,nm));embedding=band.tensor(sp.eye(nf),sp.eye(ng),er,sp.eye(nr))
    fi,gi,ri=sp.eye(nf),sp.eye(ng),sp.eye(nr)
    HR=band.tensor(fi,gi,a['casimir']/(2*p['material_inertia']),ri)
    H=HR+band.tensor(sp.diags(np.sum(hm.sector.electric**2,axis=1)/(2*p['electric_inertia'])),gi,sp.eye(ne),ri)
    H+=band.tensor(fi,gi,sp.eye(ne),hm.Hradial[:nr,:nr])
    H+=band.tensor(fi,hm.g['casimir'],sp.eye(ne),sp.diags(1/(2*p['radial_mass']*hm.rnodes**2)))
    H+=p['root_coupling']*band.tensor(hm.sector.bilinear(hm.geo['beta']),gi,sp.eye(ne),sp.diags(hm.rnodes))
    hopping=sp.csr_matrix(H.shape,dtype=complex)
    for i,j,l in product(range(3),repeat=3):hopping+=-2*np.cos(p['k'][j])*band.tensor(hm.sector.bilinear(hm.geo['gamma'][i]),hm.g['O'][l][i],a['O'][l][j],ri)
    H+=hopping;D=m.dipoles(a)[0];z=embedding@x;vertices,contacts=m.source_vertices()
    def action(eps,h):
        y=expm_multiply(-1j*eps*D,z);return embedding.conj().T@expm_multiply(1j*eps*D,h@y)
    base=action(0.,H);records=[];raw=dict(source=x,extended_source=z)
    for eps in (.002,.001):
        plus=action(eps,H);minus=action(-eps,H)
        first=(plus-minus)/(2*eps);second=(plus+minus-2*base)/eps**2
        wrongfirst=(action(eps,H-HR)-action(-eps,H-HR))/(2*eps)
        records.append(dict(step=eps,full_first_derivative_error=float(np.linalg.norm(first-vertices[0]@x)),
            full_contact_derivative_error=float(np.linalg.norm(second-contacts[0,0]@x)),
            actual_missing_mechanical_H_source_error=float(np.linalg.norm(wrongfirst-vertices[0]@x))))
        raw.update({f'first_{eps}':first,f'contact_{eps}':second,f'wrong_{eps}':wrongfirst})
    result=dict(kind='OWN_CANONICAL_PHYSICAL_SOURCE_HAMILTONIAN',dimension=hm.dimension,extended_R_dimension=H.shape[0],Gauss_error=hm.sector.gauss_error,records=records,
        convention='V=d_AHv=+i[D,H40]; physical J=-V; vary A at fixed canonical momentum, never fixed mechanical momentum.',
        scope='Own fresh N5 source and actual unitary-conjugated source H in R l<=2 image regulator. First/second derivatives are full-before-P through required order; finite-A curve itself not infinite-angular exact. Wrong case actually deletes HR from source Hamiltonian, not an observable relabel.')
    np.savez_compressed(ROOT/(P+'coupled_source_raw.npz'),**raw)
    (ROOT/(P+'coupled_source_result.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
