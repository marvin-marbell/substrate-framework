#!/usr/bin/env python3
"""Pure-Abelian SAME-C all-occupation determinant and even-macro remainder.

This is an independent scientific continuation of the captured bare-electric
core, not a modification of its Phi/colour/global Gauss construction.  The
closedwalk determinant is admitted only after actual scalar*Gamma cyclic
character checks and C inverse-intertwiner probes.  Short windings remain
explicit adverse controls; macro statements require all extents > 4 and even.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
import mpmath as mp
import numpy as np
from scipy.linalg import eigh
import physical_c_bare_electric_loops as loops
core=loops.core;alpha=core.alpha
ROOT=Path(__file__).resolve().parent
SOURCE_BYTES=Path(__file__).read_bytes()
LOOPS_BYTES=loops.SOURCE_BYTES
CORE_BYTES=loops.CORE_BYTES


def hopping(phases: np.ndarray,targets: np.ndarray,nt: int) -> np.ndarray:
    """Actual four-dimensional Wilson H; antiperiodic temporal seam."""
    v=len(targets);gamma=alpha.gamma_frame();identity=np.eye(4)
    H=np.zeros((4*v*nt,4*v*nt),complex)
    for t in range(nt):
        for x in range(v):
            sx=slice(4*(t*v+x),4*(t*v+x)+4)
            for j in range(3):
                y=int(targets[x,j]);sy=slice(4*(t*v+y),4*(t*v+y)+4)
                u=np.exp(1j*phases[t,x,j])
                H[sx,sy]+=(identity-gamma[j+1])*u
                H[sy,sx]+=(identity+gamma[j+1])*u.conjugate()
            y=((t+1)%nt)*v+x;sy=slice(4*y,4*y+4)
            u=-1 if t==nt-1 else 1
            H[sx,sy]+=u*(identity-gamma[0])
            H[sy,sx]+=u*(identity+gamma[0])
    return H


def cyclic_character(law,background,phases: np.ndarray) -> dict:
    v=len(background['targets']);nt=len(phases);P=np.eye(4*v,dtype=complex);scalar=0.
    for theta in phases:
        base=law._base(theta,background['phi'],background['links'])
        P=base['T']@P;scalar+=base['scalar']
    sign,log=np.linalg.slogdet(np.eye(4*v)+P)
    H=hopping(phases,background['targets'],nt)
    dsign,dlog=np.linalg.slogdet(np.eye(len(H))-law.kappa*H)
    scaled=scalar+float(log)+4*v*nt*math.log(2*law.kappa)
    return dict(logdet_actual_D=float(dlog),scaled_whole_scalar_Gamma_logtrace=scaled,
        equality_absolute_error=abs(float(dlog)-scaled),D_determinant_phase=[float(dsign.real),float(dsign.imag)],
        Gamma_determinant_phase=[float(sign.real),float(sign.imag)],onebody_dimensions=[4*v,len(H)])


def operator_probes(b: dict) -> list[dict]:
    rng=np.random.default_rng(222804);law=loops.make_law(.06,b);rows=[]
    for index in range(4):
        phases=rng.normal(0,.25,b['a'].shape)
        first=rng.normal(0,.15,phases.shape);second=rng.normal(0,.15,phases.shape)
        p=law._base(phases,b['phi'],b['links']);reverse=law._base(-phases,b['phi'],b['links'])
        c=law.potential(phases,b['phi'],b['links'])
        inverse_residual=np.linalg.norm(p['T']@law.C@reverse['T'].T@law.C.conj().T-np.eye(len(p['T'])))
        jets={}
        if index==0:
            basejet=law._base(phases,b['phi'],b['links'],(first,second))
            cjet=law.potential(phases,b['phi'],b['links'],(first,second))
            jets={key:float(np.linalg.norm(cjet[key]-basejet[key])) for key in ('h','hx','hy','hxy')}
            jets['scalar_pref_difference']=abs(cjet['pref']-basejet['scalar'])
            jets['scalar_mixed_difference']=abs(cjet['prefxy']-basejet['scalar_xy'])
            CH=np.rint(law.CH.real)+1j*np.rint(law.CH.imag)
            gamma=alpha.gamma_frame()
            jets['exact_beta_C_anticommutation']=float(np.linalg.norm(CH@gamma[0].T@CH.conj().T+gamma[0]))
            jets['exact_spatial_C_transpose_intertwining']=max(float(np.linalg.norm(CH@g.T@CH.conj().T-g)) for g in gamma[1:])
        shifts=[]
        for j in range(2):
            U=np.zeros((len(phases),len(phases)),complex)
            for x in range(len(phases)):U[x,b['targets'][x,j]]=np.exp(1j*phases[x,j])
            shifts.append(U)
        rows.append(dict(index=index,h_C_base_difference=float(np.linalg.norm(c['h']-p['h'])),
            scalar_difference=abs(c['pref']-p['scalar']),inverse_intertwiner_residual=float(inverse_residual),
            spatial_shift_commutator=float(np.linalg.norm(shifts[0]@shifts[1]-shifts[1]@shifts[0])),mixed_jets=jets))
    return rows


def character_and_parity() -> dict:
    result={}
    for shape in ((4,3,1),(4,4,2)):
        b=loops.background(shape);rows=[]
        for k in (.08,.04,.02):
            law=loops.make_law(k,b);nv=2*len(b['targets']);vac=[]
            for amplitude in (0.,.15,-.15):
                state=loops.float_state(law,b,amplitude,'C');vac.append(state['loglambda'])
            cb=-(vac[1]+vac[2]-2*vac[0])/(2*.15**2*b['normB'])
            rows.append(dict(k=k,cB_C_vac_over_k4=cb/k**4))
        # Genuine mixed fields, not a power of a scalar eigenvalue.
        rng=np.random.default_rng(222810);law=loops.make_law(.06,b)
        characters=[]
        for nt in (1,2,4):
            phases=rng.normal(0,.12,(nt,len(b['targets']),3))
            characters.append(dict(nt=nt,measurement=cyclic_character(law,b,phases),phases=phases.tolist()))
        result[str(shape)]=dict(spatial_coefficients=rows,cyclic_characters=characters,
            short_windings='odd-y3 and zself1' if shape==(4,3,1) else 'even all axes but z-winding2; NOT macro geometry',
            max_line_toron=b['max_line_toron'])
    return result


def spin_character() -> dict:
    gamma=alpha.gamma_frame();eye=np.eye(4);rows=[]
    for mu in range(4):
        for nu in range(mu+1,4):
            product=(eye-gamma[mu])@(eye-gamma[nu])@(eye+gamma[mu])@(eye+gamma[nu])
            rows.append(dict(mu=mu,nu=nu,oriented_plaquette_spintrace=[float(np.trace(product).real),float(np.trace(product).imag)],
                backtrack_norm=float(np.linalg.norm((eye-gamma[mu])@(eye+gamma[mu])))))
    return dict(rows=rows,closedwalk_count='4 starting corners * 2 orientations; Tr H^4 per unoriented plaquette=-64 cos(F).',
        actual_logdet_term='-k^4 Tr(H^4)/4=+16 k^4 cos(F); action difference=16 k^4(1-cos(F)), quadratic c=8 k^4, beta=16 k^4.')


def macro_bound() -> dict:
    with mp.workdps(100):
        k=mp.mpf('1e-250');N=mp.mpf('6.25e1148');V=mp.mpf(2)**92;r=8*k
        percell=(mp.mpf(4)/3)*r**6/(1-r*r)
        total=N*V*percell;beta=16*N*k**4
        hessian=4*r**6*(6/(1-r*r)+2*r*r/(1-r*r)**2)
        qminimum2=4*mp.sin(mp.pi/(mp.mpf(2)**24))**2
        fundamental_relative_bound=N*hessian/(beta*qminimum2/2)
        # On even tori, all odd walks are absent.  This all-phase bound does
        # not delete temporal winding or any Fock occupation.
        return dict(assumptions=['Phi exactly zero; every SU4 link identity; arbitrary time-dependent Abelian spatial link phases in temporal gauge A0=0.',
            'C inverse-intertwiner identity and scalar determinant character derived below.',
            'Every spatial extent L=2^24 and temporal extent Nt=2^20 even and >4; V4=2^92.',
            'Temporal antiperiodic fermion boundary; finiteNt fullFock cyclic trace, not occupied-vacuum truncation.',
            'kaux=1e-250, Naux=6.25e1148; no explicit magnetic plaquette inserted.'],
            norm_proof='Each direction Hmu=(I-gamma_mu)Umu+(I+gamma_mu)Umu† has norm2: orthogonal spin projectors commute with scalar unitary shifts. Thus ||H||<=8 uniformly in all Abelian phases.',
            convergence='logdet(I-kH)=-sum_n k^n Tr(H^n)/n, absolute convergence for8k<1. |TrH^n|<=4V4*8^n.',
            parity='Bipartite even torus makes odd closedwalk traces vanish, independent of U1 phases and temporal seam signs. TrH²=0 by Wilson projector backtracking; no winding of length<=4 on macro extents.',
            remainder_difference_per_cell='|[logdetD(A)-logdetD(0)]-16k^4 sumplaquettes[cos(F)-1]|/V4 <= (4/3)*(8k)^6/[1-(8k)^2].',
            remainder_per_cell=mp.nstr(percell,60),Naux_times_V4_remainder_absolute=mp.nstr(total,60),
            actual_derived_beta=mp.nstr(beta,60),Naux_times_remainder_per_cell=mp.nstr(N*percell,60),
            nonlinear_plaquette_control='|16Naux*k^4*(1-cosF)-8Naux*k^4 F²| <= (2/3)*Naux*k^4*|F|^4 per plaquette, all real F.',
            derivative_remainder_proof='For any link direction with sup|delta_theta|<=1, ||Hprime||,||Hsecond||<=8. Product rule gives |d² TrH^n|<=4V4*n²*8^n. Odd terms vanish identically. Thus |d²R|/V4<=4*r^6*[6/(1-r²)+2r²/(1-r²)^2], r=8k.',
            Naux_Hessian_remainder_per_cell=mp.nstr(N*hessian,60),
            Naux_Hessian_remainder_total=mp.nstr(N*V*hessian,60),
            fundamental_transverse_plane_wave_relative_Hessian_bound=mp.nstr(fundamental_relative_bound,60),
            fundamental_plane_wave_normalization='Unit-peak cos mode, q=2pi/L, polarization transverse; average A²=1/2, leading Hessian/cell=beta*qHat²/2. Bound applies at trivial Abelian background, temporal gauge. Exact global/current/source poles are not claimed.',
            alloccupation='det(I+product T) is the exact sum of every exterior occupation grade0..4V, multiplied by every scalar prefactor; no occupation0 or temporal winding omission.',
            debts=['Extension to actual Phi normals and noncommuting SU4 links.','Global Gauss/centre projection and source-conditioned current Ward identity.','Integration of this determinant into the complete core field measure and its actual photon/source consumer.'],
            bareelectric_dispersion='At leading quadratic order betaB=16Naux*k^4; betaE=16Naux*k^4+1/(2UE), UE=.02. Derived omegaHat²=betaB/betaE*qHat², not a speed fit.',
            scope='Uniform spatial/time-dependent allphase fullFock determinant theorem for SAME C only in pure Abelian Phi0/SU4I temporal-gauge submanifold. Temporal Gauss/gauge insertions are not part of the proved cyclic-character map. NOT full interacting Issue222 closure.')


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument('--capsule',required=True);args=parser.parse_args();start=time.time()
    b=loops.background((4,4,2));probes=operator_probes(b);characters=character_and_parity();spin=spin_character()
    # The exact algebra is material: C anti-transpose maps the original L†DL
    # to its inverse at reversed phases, irrespective of [B,F].  Verify rather
    # than assuming commutativity from the initial sinusoidal profile.
    derivation=dict(block_form='In beta +/- basis: T=[[B^-1,-B^-1 F],[-F†B^-1,B+F†B^-1F]], B Hermitian; no [B,F]=0 assumption.',
        inverse='T^-1=[[B+F B^-1 F†,F B^-1],[B^-1 F†,B^-1]].',
        pure_Abelian_C_identity='C T(-theta)^T C†=T(theta)^-1 follows from C gamma/spin transpose intertwining and U(-theta)^T=U(theta)†. Therefore C logT(-theta)^T C†=-logT(theta), hC=hbase and prefC=scalar (Tr logT=0).',
        qualification='Identity is specific to zero Phi and scalar SU4I U1 links; the executable probes include noncommuting spatial shifts and mixed derivatives.',
        cyclic_character='detD4D=(2k)^(4V Nt)*exp(sum_t scalar_t)*det(I+T_(Nt-1)...T_0), temporal APBC. Obtained by beta-projector temporal block elimination; order of scalar factors is immaterial, matrix order is retained.',
        finiteNt='This determinant is the exact fullFock character, not sum of highest h eigenvalues or a scalar lambda product.')
    derivation['exact_spin_algebra']='CH gamma0^T CH†=-gamma0; CH gammaj^T CH†=gammaj, j=1..3, verified exactly in the integer/i Clifford frame. U(-theta)^T=U(theta)†. Hence B reverse-transpose is B, spatial-current reverse-transpose changes sign, and C swaps beta projectors; substitution into the displayed inverse blocks proves the identity without commuting covariant shifts.'
    derivation['temporal_scope']='Cyclic determinant map proved/exercised for actual mixed spatial fields with temporal A0=0 and APBC. Norm/closedwalk algebra also extends to scalar temporal phases, but mapping these to actual core Gauss/centre insertions is NOT claimed.'
    passed=all(p['h_C_base_difference']<1e-8 and p['inverse_intertwiner_residual']<1e-8 for p in probes)
    passed=passed and all(c['measurement']['equality_absolute_error']<1e-7 for group in characters.values() for c in group['cyclic_characters'])
    passed=passed and all(value<1e-8 for value in probes[0]['mixed_jets'].values())
    passed=passed and all(row['oriented_plaquette_spintrace']==[-8.,0.] and row['backtrack_norm']==0. for row in spin['rows'])
    report=dict(schema='actual-C-even-closedwalk-v1',operator_identity=derivation,operator_probes=probes,characters_and_parity=characters,
        spin_character=spin,macro_bound=macro_bound(),actual_identity_checks_passed=passed)
    execution=dict(source_sha256=hashlib.sha256(SOURCE_BYTES).hexdigest(),source_bytes_at_IMPORT=SOURCE_BYTES.decode(),
        loops_source_sha256=hashlib.sha256(LOOPS_BYTES).hexdigest(),loops_source_bytes_at_IMPORT=LOOPS_BYTES.decode(),
        core_sha256=hashlib.sha256(CORE_BYTES).hexdigest(),core_bytes_at_IMPORT=CORE_BYTES.decode(),frozen_dependencies=loops.IMPORT_EPOCH,
        argv=sys.argv,python=sys.executable,python_version=platform.python_version(),numpy=np.__version__,mpmath=mp.__version__,
        thread_environment={name:os.environ.get(name) for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')},elapsed_seconds=time.time()-start)
    Path(args.capsule).write_text(json.dumps(dict(report=report,execution=execution),indent=2)+'\n')
    print(json.dumps(dict(capsule=args.capsule,passed=passed,operator_probes=probes,
        characters=[dict(shape=shape,errors=[c['measurement']['equality_absolute_error'] for c in group['cyclic_characters']],spatial=group['spatial_coefficients']) for shape,group in characters.items()],macro=report['macro_bound']),indent=2),flush=True)
    if not passed:raise ArithmeticError('Actual C/whole-scalar character identity falsified; macro bound is not admitted')


if __name__=='__main__':main()
