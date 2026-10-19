"""Independent fresh eight-field states: full U0/W wavefunctions, position integrals.
This is a free-field/source-kernel review, not new coupled-body evolution.
"""
from pathlib import Path
import json
import numpy as np
from scipy.linalg import expm
import rich_spatial_frame_photon_20261016_fields as frozen
from rich_review_wave49_20261016_exact_picture import wave_fourier,momentum_moment

ROOT=Path(__file__).resolve().parent;P='rich_review_wave49_20261016_'


def physical_tables(maximum,time,inertia,impulse):
    z,w=np.polynomial.hermite.hermgauss(72);width=np.sqrt(1+(time/inertia)**2)
    x=width*z+time/inertia*impulse;weight=width*w*np.exp(z*z)
    f=wave_fourier(x,maximum,time,inertia,impulse)
    df=wave_fourier(x,maximum,time,inertia,impulse,derivative=True)
    return dict(gram=f.conj().T@(weight[:,None]*f),q=f.conj().T@((weight*x)[:,None]*f),p=-1j*f.conj().T@(weight[:,None]*df),p2=df.conj().T@(weight[:,None]*df))


def assemble(labels,tables,axis=None,kind='gram'):
    out=np.ones((len(labels),len(labels)),complex)
    for a,table in enumerate(tables):
        op=table[kind if a==axis else 'gram']
        out*=np.array([[op[n[a],m[a]] for m in labels] for n in labels])
    return out


def measure(labels,psi,time,inertia,impulse):
    maximum=max(map(max,labels));cache={};tables=[]
    for value in impulse:
        if value not in cache:cache[value]=physical_tables(maximum,time,inertia,float(value))
        tables.append(cache[value])
    norm=float(np.vdot(psi,assemble(labels,tables)@psi).real)
    read=lambda kind:np.array([np.vdot(psi,assemble(labels,tables,a,kind)@psi).real/norm for a in range(8)])
    return dict(norm2=norm,Pi=read('p'),q=read('q'),physical_kinetic_energy=float(sum(read('p2'))/(2*inertia)))


def run():
    rng=np.random.default_rng(490049);time=.31;inertia=2.;records={};arrays={}
    for cut in (1,2):
        labels=frozen.basis(cut);d=len(labels);psi=rng.normal(size=d)+1j*rng.normal(size=d);psi/=np.linalg.norm(psi)
        baseline=measure(labels,psi,0,inertia,np.zeros(8));free=measure(labels,psi,time,inertia,np.zeros(8))
        Pi=np.array([momentum_moment(labels,a,1) for a in range(8)]);H0=sum(momentum_moment(labels,a,2) for a in range(8))/(2*inertia)
        badfree=expm(-1j*time*H0)@psi;badPi=np.einsum('i,aij,j->a',badfree.conj(),Pi,badfree).real
        protocols=[]
        for incoming,outgoing in ((0,6),(2,4)):
            eps=np.zeros(8);eps[incoming]=.19
            kicked0=measure(labels,psi,0,inertia,eps);kicked=measure(labels,psi,time,inertia,eps)
            work=kicked0['physical_kinetic_energy']-baseline['physical_kinetic_energy'];prediction=float(np.dot(eps,baseline['Pi'])/inertia+np.dot(eps,eps)/(2*inertia))
            initial=kicked0['Pi']-baseline['Pi'];later=kicked['Pi']-free['Pi']
            errors=dict(full_W_impulse=float(np.max(abs(initial-eps))),physical_free_Pi_invariance=float(np.max(abs(free['Pi']-baseline['Pi']))),full_W_other_mode_Pi_temporal_invariance=float(np.max(abs(later-eps))),ballistic_qdot_Pi=float(np.max(abs(free['q']-baseline['q']-time/inertia*baseline['Pi']))),full_W_ballistic_q_shift=float(np.max(abs(kicked['q']-free['q']-time/inertia*eps))),full_W_work=abs(work-prediction),physical_H0_conservation=abs(kicked['physical_kinetic_energy']-kicked0['physical_kinetic_energy']),normalization=max(abs(row['norm2']-1) for row in (baseline,free,kicked0,kicked)))
            assert max(errors.values())<2e-11
            protocols.append(dict(incoming=incoming,outgoing=outgoing,initial_reference_all8_Pi=baseline['Pi'].tolist(),initial_full_W_kicked_all8_Pi=kicked0['Pi'].tolist(),final_reference_all8_Pi=free['Pi'].tolist(),final_full_W_kicked_all8_Pi=kicked['Pi'].tolist(),initial_outgoing_preparation_shift=float(initial[outgoing]),final_outgoing_crossprotocol_difference=float(later[outgoing]),temporal_outgoing_crossprotocol_difference=float(later[outgoing]-initial[outgoing]),physical_work=float(work),errors=errors))
        bad_drift=float(np.linalg.norm(badPi-baseline['Pi']));assert bad_drift>1e-5
        records[str(cut)]=dict(field_dimension=d,protocols=protocols,WRONG_cut_H0_evolution_Pi_drift_norm=bad_drift)
        arrays.update({f'cut{cut}_fresh_coefficients':psi,f'cut{cut}_labels':np.array(labels),f'cut{cut}_initial_Pi':baseline['Pi'],f'cut{cut}_exact_free_Pi':free['Pi'],f'cut{cut}_WRONG_cut_free_Pi':badPi,f'cut{cut}_initial_q':baseline['q'],f'cut{cut}_exact_free_q':free['q']})
    np.savez_compressed(ROOT/(P+'exact_picture_means_raw.npz'),**arrays)
    result=dict(kind='OWN_FULL_UNBOUNDED_FREE_QUANTUM_FIELD_EVOLUTION_VIA_INDEPENDENT_FOURIER_WAVEFUNCTIONS',time=time,inertia=inertia,records=records,meaning='All Pi/q/energy/norm readouts are independent position integrals of full U0 W Hermite wavefunctions and their spatial derivatives, not literal zero fills, not coefficient-space vdot across protocols, not exp(-it P H0 P). No body/source state from author48 and no coupled-body response or continuum closure claim.')
    (ROOT/(P+'exact_picture_means_result.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':run()
