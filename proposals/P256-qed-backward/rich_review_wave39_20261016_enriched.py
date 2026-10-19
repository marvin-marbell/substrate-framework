"""Only newly changed coherent pair amplitude/fresh common guide and 76-state projection."""
import json
from pathlib import Path
import numpy as np
import rich_bulk_photon_20261016_state as b
import rich_flux_phase_photon_20261016_state as flux
import rich_pi_flux_dynamics_20261016_state as wave
import rich_pi_flux_dynamics_20261016_pair_enriched_state as pair
from rich_review_wave39_20261016_dynamics import independent_image
ROOT=Path(__file__).resolve().parent
P='rich_review_wave39_20261016_'

def main():
    F,G,codes,_=flux.kernel(True,'pair',.8);F2=flux.kernel(True,'pair',1.2)[0];geo=wave.geometry();eps=.35
    chunks=[];samples=[]
    for seed in (392041,392079):
        u,d,E,acc,prop=pair.sample(seed,192,14000,160,F,F2,G)
        chunks.append(pair.measure(u,d,E,F,F2,G,codes,*geo,eps));samples.append((u,d,E))
    V,HV,JV,EN,weights=[np.concatenate([c[k] for c in chunks]) for k in range(5)]
    images=[];guides=[]
    for k in (9,71):
        u,d,E=[smp[k].copy() for smp in samples[0]]
        env=b.envelope(u,d,E,.45,.15,.08)
        det=[np.linalg.det(b.submatrix(A,u,d)) for A in (F,F2,G)]
        guide=env**2*(det[0]**2+det[1]**2+.08*det[2]**2+2e-24)
        guides.append(float(abs(guide-weights[k])/guide))
        actual=np.concatenate([independent_image(u.copy(),d.copy(),E.copy(),A,codes,*geo[:5],eps) for A in (F,F2)])/np.sqrt(guide)
        images.append(float(np.linalg.norm(actual-HV[k])/np.linalg.norm(actual)))
    Es=np.concatenate([sample[2] for sample in samples]);field=Es@geo[3].T/16
    doublon=np.array([len(set(u)&set(d)) for us,ds,es in samples for u,d in zip(us,ds)])
    observables=np.column_stack([field,doublon]);rows={};raw={}
    for label,size in (('original_subspace',38),('coherent_pairs',76)):
        A=V[:,:size];HA=HV[:,:size];N=len(V);normal=np.mean(abs(A[:,0])**2)
        gram=A.conj().T@A/N/normal;hraw=A.conj().T@HA/N/normal
        eg,U=np.linalg.eigh(gram);keep=eg>1e-8*eg.max();W=U[:,keep]/np.sqrt(eg[keep]);horth=W.conj().T@hraw@W;H=(horth+horth.conj().T)/2
        eig,Q=np.linalg.eigh(H);initial=W.conj().T@gram[:,[0,36,37]];initial/=np.linalg.norm(initial,axis=0)
        records=[];traces=[]
        for t in (0.,.04,.08):
            c=Q@(np.exp(-1j*t*eig)[:,None]*(Q.conj().T@initial));coef=W@c;psi=A@coef;hp=HA@coef;norm=np.sum(abs(psi)**2,axis=0)
            obs=np.einsum('ns,nk,ns->sk',psi.conj(),observables,psi)/norm[:,None];traces.append(obs)
            records.append(dict(time=t,energy=np.real(np.sum(psi.conj()*hp,axis=0)/norm).tolist(),full_H_residual=(np.linalg.norm(hp-A@W@H@c,axis=0)/np.sqrt(norm)).tolist(),second_pair_coeff_norm_not_probability=np.linalg.norm(coef[38:],axis=0).tolist()))
        traces=np.array(traces);raw[label]=traces
        rows[label]=dict(rank=int(keep.sum()),raw_H_asymmetry=float(np.linalg.norm(horth-horth.conj().T)/np.linalg.norm(horth)),records=records,physical_field_change=float(np.linalg.norm(traces[-1,:,:4]-traces[0,:,:4])),doublon_change=(traces[-1,:,4]-traces[0,:,4]).real.tolist())
    result=dict(samples=len(V),seeds=[392041,392079],independent_H_images_relative=images,common_guide_relative=guides,comparison=rows,paired_subspace_observable_difference=float(np.linalg.norm(raw['coherent_pairs'][-1]-raw['original_subspace'][-1])),scope='New coherent delta.8/delta1.2 pair amplitudes, same H; shared pair kernels not own solved ground. Whole-H Galerkin and small finite MC only. Comparing original subspace on SAME common-guide samples, not rerunning prior38 evidence. Nonorthogonal pair coefficient norms are NOT probabilities.')
    np.savez_compressed(ROOT/(P+'enriched_raw.npz'),**raw,V=V,HV=HV,weights=weights)
    (ROOT/(P+'enriched_result.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    assert max(images)<1e-8 and max(guides)<1e-12
    assert result['paired_subspace_observable_difference']>1e-6
if __name__=='__main__':main()
