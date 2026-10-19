"""Separate actual t0 current calibration: physical Q shift versus finite-cut remainder.
Does not mutate/replay the primary experiment or any predecessor producer.
"""
from typing import Any
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
import numpy as np
from scipy import sparse
from scipy.linalg import expm
from rich_spatial_helicity_20261015_model import build,field_structure,field_drive,apply_terms
ROOT=Path(__file__).resolve().parent


def shifted_derivatives(model: Any,law: Any,dq: Any)->Any:
    shape=model['field']['boson'].shape
    result=[[sparse.csr_matrix(shape) for _ in range(16)] for _ in range(4)];index=2
    for axis in range(3):
        for level in (range(3) if axis<2 else [0]):
            profile=np.zeros(4)
            if axis<2:profile[2*axis:2*axis+2]=law['charge']*np.array([np.cos(2*np.pi*level/3),np.sin(2*np.pi*level/3)])
            angle=float(profile@dq);c,s=model['field']['coupling'][index:index+2]
            shifted_cos=np.cos(angle)*c-np.sin(angle)*s;shifted_sin=np.sin(angle)*c+np.cos(angle)*s
            for a in range(4):result[a][index]=-profile[a]*shifted_sin;result[a][index+1]=profile[a]*shifted_cos
            index+=2
    return result


def main()->None:
    started=time.time();inp=Path(sys.argv[1]);config:Any=json.loads(inp.read_text());output=Path(sys.argv[2])
    lawpath=ROOT/config['law_input'];cachepath=ROOT/config['cache_states'];cached_result_path=ROOT/config['cache_result'];law:Any=json.loads(lawpath.read_text())
    model:Any=build(law);structure=field_structure(model,law);cache=np.load(cachepath);initial=cache['charged_initial'];neutral=cache['neutral_ground']
    cached_result:Any=json.loads(cached_result_path.read_text());hcheck=float(np.linalg.norm(model['neutral']['h']@neutral-cached_result['energies']['neutral']*neutral))
    sec=model['positive'];nm=sum(sec['sizes']);nphi=model['field']['nphi'];nf=model['field']['shell_dimension']
    def apply_em(state: Any,u: Any)->Any:return np.einsum('ij,mpj->mpi',u,state.reshape(nm,nphi,nf)).ravel()
    def currents(state: Any,derivatives: Any)->Any:return np.array([-float(np.real(np.vdot(state,apply_terms(sec,d,state)))) for d in derivatives])
    rho0=initial.reshape(nm,-1)@initial.reshape(nm,-1).conj().T;baseline=currents(initial,model['field']['derivatives']);arms={}
    for sigma in [1,-1]:
        dq=np.array([config['drive_Q_amplitude'],0,0,sigma*config['drive_Q_amplitude']]);dp=np.array([0,config['drive_P_amplitude'],-sigma*config['drive_P_amplitude'],0])
        up=expm(1j*sum(c*q for c,q in zip(dp,structure['q'])));uq=expm(-1j*sum(c*p for c,p in zip(dq,structure['p'])))
        phase=apply_em(initial,up);shift=apply_em(initial,uq);full,drive=field_drive(model,structure,initial,sigma,config['drive_Q_amplitude'],config['drive_P_amplitude'])
        shifted=shifted_derivatives(model,law,dq);jshift=currents(initial,shifted);jshift_phase=currents(phase,shifted);jfull=currents(full,model['field']['derivatives'])
        phase_remainder=jshift_phase-jshift;shift_remainder=jfull-jshift_phase;physical=jshift-baseline
        matter={}
        for name,state in [('phase_only',phase),('shift_only',shift),('full_drive',full)]:
            a=state.reshape(nm,-1);matter[name]=float(np.linalg.norm(a@a.conj().T-rho0))
        arms[str(sigma)]=dict(drive=drive,baseline_current=baseline.tolist(),finite_phase_only_current=currents(phase,model['field']['derivatives']).tolist(),
            finite_shift_only_current=currents(shift,model['field']['derivatives']).tolist(),finite_full_drive_current=jfull.tolist(),
            before_cut_shifted_source_current=jshift.tolist(),before_cut_shifted_source_on_phase_state=jshift_phase.tolist(),
            legitimate_coordinate_shift_response=physical.tolist(),finite_Q_phase_at_shift_remainder=phase_remainder.tolist(),
            finite_Pi_shift_product_remainder=shift_remainder.tolist(),total_finite_minus_ambient_shift_remainder=(jfull-jshift).tolist(),
            decomposition_closure_error=float(np.linalg.norm((jfull-baseline)-(physical+phase_remainder+shift_remainder))),
            complete_matter_reduced_state_change_norm=matter,
            interpretation='Ambient U=exp(-i dQ.Pi)exp(i dPi.Q): Udag J(Q)U=J(Q+dQ). BEFORE-CUT shifted-source Peierls operator computed directly. Finite full-drive initial J change has a legitimate coordinate-shift part; remaining Q-phase/CCR/shift-product error explicitly separated, not all labelled cutoff.')
    result=dict(kind='ACTUAL_SAME_H_INITIAL_CURRENT_COORDINATE_SHIFT_CALIBRATION',input=config,
        bindings={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [inp,lawpath,cachepath,cached_result_path]},cached_neutral_H_state_residual=hcheck,arms=arms,
        formula='Jfinite(full)-J0=[Jambient_shift(psi)-J0]+[Jambient_shift(Up psi)-Jambient_shift(psi)]+[Jfinite(Uq Up psi)-Jambient_shift(Up psi)].',
        limits='Exact finite same-shell/current-source calibration, not cutoff convergence. Q/P drives remain field-only and all matter reduced-state changes are reported. Fixed-axis polarization angular momentum remains NOT identified photon helicity.',
        elapsed_seconds=time.time()-started,peak_resident_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(result=str(output),elapsed=result['elapsed_seconds'],peak_resident_KiB=result['peak_resident_KiB'],arms=arms)),flush=True)


if __name__=='__main__':main()
