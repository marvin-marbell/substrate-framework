"""Only invalidated field-angular generator and full versus projected pulse layers."""
import json
from pathlib import Path
import numpy as np
from scipy import sparse as sp
from scipy.linalg import expm
import rich_coupled_frame_photon_20261016_model as law
from rich_review_wave45_20261016_collective import vertices_probe

ROOT=Path(__file__).resolve().parent;P='rich_review_wave45_20261016_'


def main():
    result=dict(kind='OWN_EVOLVED_PHYSICAL_STATES_FINAL_GENERATOR_RECHECK',branches=[],updated_Frechet_API_probe=vertices_probe())
    for number in (5,3):
        m=law.Model(number);raw=np.load(ROOT/(P+f'coupled_N{number}_raw.npz'));x=raw['initial'];records=[]
        ext=law.field_basis(m.parameters['field_total_cut']+2)
        qe=[law.field_operator(ext,m.basis,a,'q') for a in range(3)];pe=[law.field_operator(ext,m.basis,a,'p') for a in range(3)]
        independent=[qe[(a+1)%3].conj().T@pe[(a+2)%3]-qe[(a+2)%3].conj().T@pe[(a+1)%3] for a in range(3)]
        for key in ('initial','free','final0','final1'):
            z=raw[key]
            comm=[float(np.linalg.norm(m.H@(j@z)-j@(m.H@z))) for j in m.J]
            old=[sp.kron(m.matter.J[a],m.fi)+sp.kron(m.identity,m.q[(a+1)%3]@m.p[(a+2)%3]-m.q[(a+2)%3]@m.p[(a+1)%3]) for a in range(3)]
            wrong=[float(np.linalg.norm(m.H@(j@z)-j@(m.H@z))) for j in old]
            records.append(dict(state=key,full_product_J_commutator=comm,prematurely_projected_J_commutator=wrong))
        pulse=[]
        # Full oscillator impulse applied to the actual entangled incoming field packet.
        # Two extensions measure the numerical tail of this source-image operation.
        for axis in (0,1):
            previous=None
            for cut in (7,8):
                fb=law.field_basis(cut);embed=np.eye(len(fb))[:,[fb.index(v) for v in m.basis]]
                q=law.field_operator(fb,fb,axis,'q').toarray();U=expm(.045j*q)
                image=x.reshape(m.shape)@(U@embed).T;keep=[fb.index(v) for v in m.basis]
                retained=image[:,keep];norm=np.linalg.norm(retained);tail=max(0.,np.linalg.norm(image)**2-norm**2)
                state=retained.ravel()/norm
                pulse.append(dict(axis=axis,extension=cut,discarded_probability=float(tail),projected_impulse_difference=float(np.linalg.norm(state-raw[f'kick{axis}'])),extension_step_difference=None if previous is None else float(np.linalg.norm(state-previous))))
                previous=state
        result['branches'].append(dict(number=number,independent_generator_matrix_error=max(float(sp.linalg.norm(a-b)) for a,b in zip(m.Jfield,independent)),states=records,full_vs_projected_impulse=pulse,
            limits='Hamiltonian/source/dynamics unchanged; only old bad J observable rechecked on own actual evolved states. Original protocols used exp(i epsilon PqP), a declared numerical field-only impulse; full impulse image/tail separately quantified. Global total J closure is not localized spin/cloud transport.'))
        (ROOT/(P+'coupled_final_result.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
