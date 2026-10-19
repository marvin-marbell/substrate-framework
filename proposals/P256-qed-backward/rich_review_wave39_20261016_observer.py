"""Correct only reviewer EOM units/sign; measure new matter and +/-q quadratures from OWN raw fresh amplitudes."""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
P='rich_review_wave39_20261016_'

def main():
    with np.load(ROOT/(P+'dynamics_raw.npz')) as z:
        V=z['V'];HV=z['HV'];JV=z['JV'];fs=z['features'];G=z['gram'];W=z['W'];H=z['H']
    eig,U=np.linalg.eigh(H);initial=W.conj().T@G[:,[0,36,37]];initial/=np.linalg.norm(initial,axis=0)
    field=(fs[:,[4,6,12,14]]+1j*fs[:,[5,7,13,15]])/16
    matter=np.column_stack([8*fs[:,1],fs[:,8]+1j*fs[:,9],fs[:,16]+1j*fs[:,17],fs[:,10]+1j*fs[:,11],fs[:,18]+1j*fs[:,19]])
    records=[];arrays=[]
    for t in (0.,.015,.04,.08):
        c=U@(np.exp(-1j*t*eig)[:,None]*(U.conj().T@initial));coef=W@c
        psi=V@coef;hp=HV@coef;ph=V@W@H@c;norm=np.sum(abs(psi)**2,axis=0)
        current=np.einsum('ns,nks->sk',psi.conj(),np.einsum('nka,as->nks',JV[:,8:],coef))/norm[:,None]
        eom=1j*(np.einsum('ns,nk,ns->sk',hp.conj(),field,psi)-np.einsum('ns,nk,ns->sk',psi.conj(),field,hp))/norm[:,None]
        eomproj=1j*(np.einsum('ns,nk,ns->sk',ph.conj(),field,psi)-np.einsum('ns,nk,ns->sk',psi.conj(),field,ph))/norm[:,None]
        obs=np.einsum('ns,nk,ns->sk',psi.conj(),field,psi)/norm[:,None]
        density=np.einsum('ns,nk,ns->sk',psi.conj(),matter,psi)/norm[:,None]
        arrays.append((obs,density,eom,eomproj,current))
        records.append(dict(time=t,full_sample_physical_EOM_plus_current_over16=np.linalg.norm(eom+current/16,axis=1).tolist(),physical_projected_vs_full_EOM=np.linalg.norm(eomproj-eom,axis=1).tolist(),doublon=density[:,0].real.tolist(),rho_q_abs=abs(density[:,1:3]).tolist()))
    quadratures=np.array([a[0] for a in arrays]);densities=np.array([a[1] for a in arrays])
    result=dict(correction='Original dynamics_result full_EOM_minus_current_over_I mislabeled raw E as physical E/16 and wrong sign. Superseded ONLY that field. Here physical d<E/16>/dt=-<J>/16. Finite-sample full-H weak estimator not exact identity; Galerkin full-EOM defect remains explicit.',records=records,physical_field_change=float(np.linalg.norm(quadratures[-1]-quadratures[0])),matter_change=float(np.linalg.norm(densities[-1]-densities[0])),matter_pulse_minus_background_change=float(np.linalg.norm((densities[-1,1:]-densities[-1,0])-(densities[0,1:]-densities[0,0]))),quadrature_layout='time,state(background,xy,xz),4 modes(q0T0,q0T1,q1T0,q1T1); complex real/imag retained; qminus=complex conjugate')
    np.savez_compressed(ROOT/(P+'observer_raw.npz'),Eplus=quadratures,Eminus=quadratures.conj(),matter=densities,full_EOM=np.array([a[2] for a in arrays]),projected_EOM=np.array([a[3] for a in arrays]),current=np.array([a[4] for a in arrays]))
    (ROOT/(P+'observer_result.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    assert result['matter_change']>1e-5 and result['matter_pulse_minus_background_change']>1e-6
if __name__=='__main__':main()
