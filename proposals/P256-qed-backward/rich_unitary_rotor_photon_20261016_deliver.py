"""Reduce own observed arrays; package exact artifacts without pole assertions."""
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np

ROOT=Path(__file__).resolve().parent
P='rich_unitary_rotor_photon_20261016_'


def connected(arr):
    E=arr['E'].mean(axis=0);J=arr['J'].mean(axis=0)
    EE=arr['EE'].mean(axis=0).copy();JJ=arr['JJ'].mean(axis=0).copy();EJ=arr['EJ'].mean(axis=0).copy()
    for p in range(4):
        partner=p+2 if p<2 else p-2
        EE[p]-=E[p].mean()*E[partner].mean()
        JJ[p]-=J[p]*J[partner,0]
        EJ[p]-=E[p]*J[partner,0]
    return EE,JJ,EJ


def continuity_shape(JJ,contact,dt):
    """Exact continuum H identity, evaluated by explicit finite-dt quadrature.
    CEE''=CJJ and CEE'(0+)=-contact/2; no absolute CEE or pole inferred.
    """
    M=JJ.shape[1];out=np.zeros_like(JJ)
    for lag in range(1,M):
        tau=lag*dt;u=np.arange(lag+1)*dt
        out[:,lag]=-.5*contact*tau+np.trapezoid((tau-u)*JJ[:,:lag+1],u,axis=1)
    return out


def main():
    result=json.loads((ROOT/(P+'result.json')).read_text());science_exit=json.loads((ROOT/(P+'exit.json')).read_text())
    reduction={};numbers=[]
    for case in result['cases']:
        name=case['name'];meta=case['metadata'];beta=meta['beta'];M=meta['M'];dt=beta/M
        with np.load(ROOT/(P+name+'_raw.npz')) as saved:arr={k:saved[k] for k in saved.files}
        EE,JJ,EJ=connected(arr);phase=np.exp(2j*np.pi*np.arange(M)[None,:]*np.arange(M)[:,None]/M)
        mats=np.array([dt*x@phase.T for x in (EE,JJ,EJ)])
        blocks=np.array_split(np.arange(len(arr['EE'])),8)
        estimates=np.array([connected({k:v[b] for k,v in arr.items() if len(v)==len(arr['EE'])}) for b in blocks])
        errors=estimates.std(axis=0,ddof=1)/np.sqrt(len(blocks))
        # Finite-dt current kernel only, not exact Trotter source Hessian.
        contact=arr['contact'].mean(axis=(0,2));kernel=contact-mats[1,:,0]
        shape=continuity_shape(JJ,contact,dt)
        block_shapes=np.array([continuity_shape(estimates[k,1],arr['contact'][b].mean(axis=(0,2)),dt) for k,b in enumerate(blocks)])
        shape_error=block_shapes.std(axis=0,ddof=1)/np.sqrt(len(blocks))
        np.savez_compressed(ROOT/(P+name+'_response.npz'),EE_connected=EE,JJ_connected=JJ,EJ_connected=EJ,
                            block_standard_error=errors,matsubara=mats,current_riemann_kernel=kernel,
                            continuity_delta_EE=shape,continuity_delta_EE_block_error=shape_error,
                            tau=np.arange(M)*dt,omega=2*np.pi*np.arange(M)/beta)
        entry=dict(EE_tau0_real=EE[:,0].real.tolist(),EE_tau_half_real=EE[:,M//2].real.tolist(),
                   EE_tauhalf_block_error=errors[0,:,M//2].tolist(),JJ_tau0_real=JJ[:,0].real.tolist(),
                   JJ_tauhalf_real=JJ[:,M//2].real.tolist(),contact_real=contact.real.tolist(),
                   EJ_tauhalf_real=EJ[:,M//2].real.tolist(),EJ_tauhalf_imag=EJ[:,M//2].imag.tolist(),
                   current_riemann_kernel_real=kernel.real.tolist(),plaquette_cosine=float(arr['plaquette'].mean()),
                   continuity_delta_EE_tauhalf_real=shape[:,M//2].real.tolist(),
                   continuity_delta_EE_tauhalf_block_error=shape_error[:,M//2].tolist(),
                   max_propagator_condition=float(arr['max_propagator_condition'].max()),
                   scope='Finite-beta/dt exploratory thermal imaginary-time response; 8 block errors do not certify effective sample size; Riemann current kernel is not exact finite-dt source Hessian.')
        reduction[name]=entry
        numbers.append(f"{name}: EE(0)y={EE[0,0].real:.6g}, EE(beta/2)y={EE[0,M//2].real:.6g} +/- {errors[0,0,M//2]:.3g}; JJ(0)y={JJ[0,0].real:.6g}, contact_y={contact[0].real:.6g}, <cos plaquette>={arr['plaquette'].mean():.6g}.")
        numbers.append(f"  JJ(beta/2)y={JJ[0,M//2].real:.6g}; continuum-identity/trapezoid DeltaEE(beta/2)y={shape[0,M//2].real:.6g} +/- {shape_error[0,M//2]:.3g}; NOT direct field resolution or exact finite-dt Ward claim.")
    (ROOT/(P+'response_summary.json')).write_text(json.dumps(reduction,indent=2)+'\n')
    small=result['small']
    lines=['NEW FULL COMPACT U1 ROTOR: ACTUAL V64 THERMAL STATE AND RESPONSE',
           'Scope: separate declared law, not an S1 repair or theorem of S1 failure.',
           'L4/V64/192 arcs, t1 U4 I16 ions1, original two SAME-charge flavours; no bare B/Maxwell.',
           'E in Z unbounded, bilateral U unitary; full compact heat kernel, no electric cutoff.',
           'Exact down PH gives nu-nh charge, attractive centered Hubbard +1/site; constants/linear terms retained.',
           'Compact local temporal boundary holonomies project FULL Gauss, not mean half filling.',
           'Gauss-only originalN64 thermal spin mixture differs from old canonical32up/32down.',
           'Positive scalar*heatkernels*|det|² proven in derivation and tested on actual physical two-site H.',
           f"Own two-site projected M1 trace rel.error={small['relative_trace_error']:.4g}; wrong unconjugated holonomy fails by {small['wrong_phase_relative_error']:.4g}.",
           f"Own exact-HS rank1 logratio error={small['HS_rank_one_logratio_error']:.4g}; primitive J derivative error={small['primitive_current_derivative_error']:.4g}.",
           'Actual V64 MC uses independent fresh angle/HS inputs, full determinants/virtuals and heatkernel electric sum.',
           'Static-link, full temporal compact bridge, joint-holonomy/linear-path, uniform-holonomy and exact HS slice sweeps.',
           'First fixed-time-initialization attempt preserved and rejected for insufficient temporal mixing; material bridge correction used.',
           'Each case4800 moves,1600 burn,200 saved states. Independent seeds at beta.5/M4 and beta1/M8.',
           'Two transverse y/z modes at q=(+/-pi/2,0,0); EE/JJ/EJ/contact and Matsubara arrays delivered.',
           'Current source J=-dH/dA; contact=d²H/dA² from same primitive Peierls H, no Maxwell addition.',
           'E=-iKprime/K, coincident E²=-Kdoubleprime/K; current Wick response uses all Gaussian fermion states.',
           *numbers,
           'These are measured finite-beta/Trotter imaginary-time responses, NOT real-time photon poles.',
           'Direct electric estimates remain noise/seed-limited; beta1/M8 EE(beta/2)y differs -0.695 vs40.09. No resolved direct field dispersion.',
           'Continuum H identity CEEsecond=CJJ, CEEprime(0+)=-contact/2 gives current-linked field SHAPE only; finite-dt quadrature explicitly labelled.',
           'Independent own exact2site full-H: EE0 .0676442 -> EEhalf .0258995, JJ0 1.35808 -> .386351, contact .539538, Edot=J exact.',
           'Same small-case Trotter trace errors M1 14.55%,M4 1.154%,M8 .2913%,M16 .0730%; same-Trotter positivity is not exact-H equality.',
           'Final selected Pyright/LSP/AST evidence is in language_final; environment versions in environment.json.',
           'dt=.125/.25 comparisons and finite independent seeds are diagnostics, not an extrapolation or mixing proof.',
           'Eight-block errors are exploratory and can underestimate slow angle/holonomy autocorrelation.',
           'Direct dense products used at beta<=1; no high-beta stabilization or exact vacuum claim.',
           'Full heatkernel represented by wrapped Poisson images -2..2; omitted images begin5pi, bound in small_result.',
           'No E-law cap; finite MC does not establish visits to every flux/winding sector.',
           'Own-H bare defeat: i[H,E]=J nonzero at t1, so conserved-electric-only dynamics is not this H.',
           'Contact−dt sumJJ reported only as finite-dt Riemann kernel, not an exact source-Hessian Ward theorem.',
           'Band40 bilateral convention matches; finite-tree band frame and this V64 thermal law are NOT yet joined.',
           f"Actual science natural exit {science_exit['natural_exit']}; exact command/source hashes/streams in execution_input,source_capsule,exit.",
           'No photon/Nature/c/Lorentz/groundstate assertion. Parent alone owns joined-direction and publication.']
    assert len(lines)<=80
    (ROOT/(P+'synthesis.txt')).write_text('\n'.join(lines)+'\n')
    files=sorted(p for p in ROOT.glob(P+'*') if p.is_file() and not p.name.endswith(('delivery_capsule.zip','manifest.json','delivery_receipt.json')))
    manifest={p.name:dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size) for p in files}
    (ROOT/(P+'manifest.json')).write_text(json.dumps(manifest,indent=2)+'\n')
    with zipfile.ZipFile(ROOT/(P+'delivery_capsule.zip'),'w',zipfile.ZIP_DEFLATED) as z:
        for p in files+[ROOT/(P+'manifest.json')]:z.write(p,p.name)
    capsule=ROOT/(P+'delivery_capsule.zip')
    receipt=dict(capsule=capsule.name,sha256=hashlib.sha256(capsule.read_bytes()).hexdigest(),bytes=capsule.stat().st_size,science_natural_exit=science_exit['natural_exit'],synthesis_lines=len(lines))
    (ROOT/(P+'delivery_receipt.json')).write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
