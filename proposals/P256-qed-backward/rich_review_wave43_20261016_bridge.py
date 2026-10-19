"""Own physical free-rotor imaginary-time response control for temporal mixing."""
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
P='rich_review_wave43_20261016_'


def main():
    cfg=json.loads((ROOT/(P+'bridge_input.json')).read_text())
    rng=np.random.default_rng(cfg['seed']);N=cfg['samples'];M=cfg['slices'];beta=cfg['beta'];I=cfg['I'];dt=beta/M
    # Loop bridge in real lifts: iid Gaussian increments conditioned to sum zero.
    increments=rng.normal(0,np.sqrt(dt/I),(N,M));increments-=increments.mean(axis=1,keepdims=True)
    score=-increments/(dt/I)
    EE=np.array([-np.mean(score*np.roll(score,lag,axis=1),axis=1) for lag in range(M)]).T
    EE[:,0]+=I/dt
    blocks=np.array([x.mean(axis=0) for x in np.array_split(EE,20)])
    mean=EE.mean(axis=0);sem=blocks.std(axis=0,ddof=1)/np.sqrt(len(blocks))
    integer=np.arange(-80,81);weights=np.exp(-beta*integer**2/(2*I));truth=float(np.sum(integer**2*weights)/weights.sum())
    # Wrapped w!=0 loop branches are suppressed; this is a declared quantitative tail, not E cutoff.
    winding_bound=float(2*np.exp(-I*(2*np.pi)**2/(2*beta))/(1-np.exp(-3*I*(2*np.pi)**2/(2*beta))))
    all_equal=np.zeros(M);all_equal[0]=I/dt
    result=dict(kind='OWN_FREE_FULL_ROTOR_TEMPORAL_RESPONSE_CONTROL',input=cfg,
        exact_integer_spectrum_EE=truth,bridge_EE=mean.tolist(),block_standard_error=sem.tolist(),
        standardized_deviation=((mean-truth)/sem).tolist(),all_equal_path_wrong_EE=all_equal.tolist(),
        omitted_winding_probability_upper_bound=winding_bound,
        classification='independent Gaussian bridges for free rotor control; NOT interacting V64 state, no thermal-convergence claim for producer',
        physical_interpretation='E conserved by HE, so unequal-imaginary-time EE must equal equal-time <E²>; wrong equal-slice initialization fails this quantum observable')
    np.savez_compressed(ROOT/(P+'bridge_raw.npz'),blocks=blocks,EE_mean=mean,EE_sem=sem)
    (ROOT/(P+'bridge_result.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)
    assert max(abs((mean-truth)/sem))<5
    assert abs(all_equal[M//2]-truth)>1
    assert winding_bound<1e-100


if __name__=='__main__':main()
