"""Analyze ONLY this consumer's new raw thermal paths; no producer-output replay."""
import hashlib
import json
from pathlib import Path
import numpy as np
import rich_unitary_rotor_photon_20261016_thermal as rotor
from rich_collective_photon_response_20261016_run import summarize

ROOT=Path(__file__).resolve().parent
P='rich_collective_photon_response_20261016_'


def loop_fields(theta,pos,arcs):
    """Two physical transverse compact curls sin Fxy / sin Fxz, no new energy."""
    fields=[];flux=[]
    for pol in (1,2):
        a=3*np.arange(len(pos));b=a+pol
        F=theta[...,a]+theta[...,3*arcs[a,1]+pol]-theta[...,3*arcs[b,1]]-theta[...,b]
        flux.append(F)
    for q in (np.pi/2,np.pi):
        phase=np.exp(1j*q*pos[:,0])/np.sqrt(len(pos))
        for F in flux:fields.append(np.sin(F)@phase)
    return np.array(fields).transpose(1,0,2),np.mean(np.cos(np.array(flux)),axis=(0,2,3))


def scalar_stat(x):
    blocks=np.array([b.mean() for b in np.array_split(np.asarray(x),8) if len(b)])
    return dict(mean=float(np.mean(x)),block_SEM=float(blocks.std(ddof=1)/np.sqrt(len(blocks))))


def main():
    pos,arcs,D=rotor.geometry(4);cases=[]
    for path in sorted(ROOT.glob(P+'b*_raw.npz')):
        if path.name.endswith('_field_raw.npz'):continue
        name=path.name[len(P):-len('_raw.npz')]
        data=np.load(path);meta=json.loads((ROOT/(P+name+'_result.json')).read_text())['metadata']
        theta=data['theta'];N,M,_=theta.shape;dt=meta['dt'];beta=meta['beta']
        B,plaquette=loop_fields(theta,pos,arcs)
        C=np.array([np.mean(B*np.roll(B.conjugate(),lag,axis=2),axis=2) for lag in range(M)]).transpose(1,2,0)
        Cn=np.fft.ifft(C,axis=2)*M;DB=dt*Cn
        DE=dt*data['EE_matsubara'];current=data['interval_current_matsubara']
        Jtau=np.fft.fft(current,axis=2)/M
        # Keep uncentered physical two-point and separately label finite-sample
        # connected correction; never erase the actual electric zero sector.
        meanB=B.mean(axis=(0,2));connected=C-abs(meanB)[None,:,None]**2
        delta=np.roll(theta,-1,axis=1)-theta;delta[:,-1]+=data['alpha']@D
        wrapped=(delta+np.pi)%(2*np.pi)-np.pi
        coeff=np.array([np.exp(1j*q*pos[arcs[:,0],0])/np.sqrt(len(pos))*(arcs[:,2]==pol) for q in (np.pi/2,np.pi) for pol in (1,2)])
        transverse_winding=wrapped.sum(axis=1)@coeff.T/(2*np.pi)
        np.savez_compressed(ROOT/(P+name+'_field_raw.npz'),B=B,B_tau=C,B_matsubara=Cn,D_B=DB,
                            B_connected_finite_sample=connected,D_E=DE,interval_J_tau=Jtau,
                            transverse_temporal_winding=transverse_winding,
                            plaquette=plaquette,omega=2*np.pi*np.arange(M)/beta,
                            omega_hat=2*np.sin(np.pi*np.arange(M)/M)/dt,
                            q_labels=data['labels'][0],source_raw_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        modes=[]
        for p in range(4):
            source=data['source_hessian'][:,p];one=data['source_one'][:,p]
            ward0=scalar_stat(source[:,0]);den=max(1e-300,ward0['block_SEM'])
            modes.append(dict(q=float(data['labels'][0,p,0]),polarization=int(data['labels'][0,p,1]),
                              D_E_n0=scalar_stat(DE[:,p,0]),D_E_n1=scalar_stat(DE[:,p,1]),
                              D_E_n2=scalar_stat(DE[:,p,2]),EE_tau0=scalar_stat(data['EE_tau'][:,p,0].real),
                              EE_tauhalf=scalar_stat(data['EE_tau'][:,p,M//2].real),
                              direct_EE_tauhalf=scalar_stat(data['direct_EE_tau'][:,p,M//2].real),
                              interval_contact=scalar_stat(data['interval_contact'][:,p]),
                              interval_J_tau0=scalar_stat(Jtau[:,p,0].real),interval_J_tauhalf=scalar_stat(Jtau[:,p,M//2].real),
                              B_tau0=scalar_stat(C[:,p,0].real),B_tauhalf=scalar_stat(C[:,p,M//2].real),
                              D_B_n0=scalar_stat(DB[:,p,0].real),D_B_n1=scalar_stat(DB[:,p,1].real),
                              static_source_Ward=ward0,static_Ward_mean_over_blockSEM=ward0['mean']/den,
                              source_one_n1_mean_abs=float(abs(one[:,1].mean())),
                              source_feedback_n1=float((source[:,1]-data['logdet_hessian'][:,p,1]).mean()),
                              max_tau_imaginary_mean=float(abs(data['EE_tau'][:,p].mean(axis=0).imag).max()),
                              first_half_D_E_n1=float(DE[:N//2,p,1].mean()),second_half_D_E_n1=float(DE[N//2:,p,1].mean())))
        cases.append(dict(name=name,metadata=meta,modes=modes,plaquette=scalar_stat(plaquette),
                          maximum_selected_step_halving_difference=float(abs(data['derivative_step_difference']).max()),
                          step_halving_samples=int(data['step_comparison_exercised'].sum()),
                          maximum_source_condition=None if meta['stabilized'] else float(data['max_source_matrix_condition'].max()),
                          maximum_transverse_temporal_winding=float(abs(transverse_winding).max()),
                          raw_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                          full_fields=dict(B_tau=summarize(C),D_B=summarize(DB),interval_J_tau=summarize(Jtau))))
    output=dict(kind='ACTUAL_NEW_SAME_PATH_ELECTRIC_AND_COMPACT_CURL_FIELD_RESPONSE',cases=cases,
                meaning='D_E=dt sum exp(+iwn lag)<E_q(lag)E_-q(0)>; D_B likewise for sin plaquette curl. Full annealed response, no irreducible kernel or pole fit.',
                limits=['Independent electric zero frequency retained','Magnetic sin-plaquette measured, no B² term added',
                        'Finite chains and block SEM do not certify equilibrium','Cold centered stable derivative is numerical with selected step-halving',
                        'B one-point finite-sample subtraction separately labelled; no forced pathwise parity',
                        'Heavy-rotor slow coordinate retention also occurs without magnetic restoring, so retention alone is not a photon'])
    (ROOT/(P+'response_summary.json')).write_text(json.dumps(output,indent=2)+'\n')
    for case in cases:
        print(json.dumps(dict(name=case['name'],modes=case['modes'],plaquette=case['plaquette'])),flush=True)


if __name__=='__main__':main()
