"""Exact mixed-word countercase and native CAR/fullE flow algebra; not dynamics."""
from __future__ import annotations
import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
from typing import Any, cast
import numpy as np
import sympy as sp

ROOT = Path(__file__).resolve().parent


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def car_hop(mask: int, row: int, col: int) -> tuple[int, int] | None:
    if not (mask >> col) & 1 or (mask >> row) & 1:
        return None
    sign = -1 if (mask & ((1 << col) - 1)).bit_count() % 2 else 1
    removed = mask ^ (1 << col)
    sign *= -1 if (removed & ((1 << row) - 1)).bit_count() % 2 else 1
    return removed | (1 << row), sign


def native(N: int) -> dict[str, Any]:
    V = N ** 3
    def site(p: tuple[int, int, int]) -> int:
        return (p[0] * N + p[1]) * N + p[2]
    parity = np.fromfunction(lambda x, y, z: 1 - 2 * ((x+y+z) % 2), (N,N,N), dtype=int).astype(np.int64)
    base = sum(15 << (4 * site((x,y,z))) for x in range(N) for y in range(N) for z in range(N) if parity[x,y,z] == 1)
    mask = sum(base << (4 * V * species) for species in range(65))
    A, B, C, L = (0,2,2), (1,2,2), (2,2,2), (0,3,2)
    prepared = car_hop(mask, 4*site(L), 4*site(A))
    assert prepared is not None
    mask, _ = prepared
    def number_rho(state: int) -> np.ndarray:
        sector_mask = (1 << (4*V))-1
        assert all(((state >> (4*V*s)) & sector_mask).bit_count() == 2*V for s in range(65))
        rho = np.zeros((N,N,N),dtype=np.int64)
        for x in range(N):
            for y in range(N):
                for z in range(N):
                    i=site((x,y,z))
                    rho[x,y,z] = sum(((state >> (4*V*s+4*i)) & 15).bit_count()-2 for s in range(64))
        return rho
    def divergence(e: np.ndarray) -> np.ndarray:
        return cast(np.ndarray, sum(e[...,axis]-np.roll(e[...,axis],1,axis=axis) for axis in range(3)))
    e3 = np.zeros((N,N,N,3),dtype=np.int64)
    e3[...,0] = 192*parity
    e3[A+(1,)] -= 3  # actual A -> L CAR preparation
    # Incoming closed face of strength 1/3; never reset before either word leg.
    e3[A+(0,)] += 1
    e3[B+(1,)] += 1
    e3[(0,3,2,0)] -= 1
    e3[A+(1,)] -= 1
    assert np.array_equal(divergence(e3),3*number_rho(mask))
    def apply(state: dict[int,int], row: tuple[int,int,int], col: tuple[int,int,int]) -> dict[int,int]:
        out: dict[int,int]={}
        for basis, coefficient in state.items():
            for colour in range(4):
                transition=car_hop(basis,4*site(row)+colour,4*site(col)+colour)
                if transition is not None:
                    new, sign=transition
                    out[new]=out.get(new,0)+coefficient*sign
        return {basis:coefficient for basis,coefficient in out.items() if coefficient}
    first=apply({mask:1},B,C)
    word=apply(first,A,B)
    direct=apply({mask:1},A,C)
    assert word == direct and len(word)==1
    outmask=next(iter(word))
    out=e3.copy();out[A+(0,)]+=3;out[B+(0,)]+=3
    assert np.array_equal(divergence(out),3*number_rho(outmask))
    rho_diff=number_rho(outmask)-number_rho(mask)
    assert rho_diff[A]==1 and rho_diff[C]==-1 and rho_diff[B]==0
    D=(1,3,2)
    blocked=car_hop(mask,4*site(B),4*site(D));assert blocked is not None
    occupied_mask,_=blocked
    occupied_word=apply(apply({occupied_mask:1},B,C),A,B)
    occupied_direct=apply({occupied_mask:1},A,C)
    assert occupied_word != occupied_direct
    assert sum(value**2 for value in occupied_word.values()) == 3
    assert sum(value**2 for value in occupied_direct.values()) == 1
    def electric(e: np.ndarray) -> F:
        return F(64*sum(int(value)**2 for value in e.flat),9)
    delta=electric(out)-electric(e3)
    inc=np.zeros_like(e3);inc[A+(0,)]=3;inc[B+(0,)]=3
    exact_shift=F(64*(2*sum(int(a)*int(b) for a,b in zip(e3.flat,inc.flat))+sum(int(b)**2 for b in inc.flat)),9)
    assert delta==exact_shift and delta!=128
    # Conditional binary64 orthogonal Gauss decomposition is a separate control.
    rho=number_rho(mask).astype(float)
    eigen=np.zeros((N,N,N))
    for axis in range(3):
        shape=[1,1,1];shape[axis]=N
        eigen+=4*np.sin(np.pi*np.arange(N)/N).reshape(shape)**2
    transformed=np.fft.fftn(rho);potential_hat=np.zeros_like(transformed)
    nonzero=eigen>0;potential_hat[nonzero]=transformed[nonzero]/eigen[nonzero]
    potential=np.real(np.fft.ifftn(potential_hat))
    longitudinal=np.stack([potential-np.roll(potential,-1,axis=axis) for axis in range(3)],axis=-1)
    transverse=e3.astype(float)/3-longitudinal
    full=e3.astype(float)/3
    defect=abs(float(np.sum(full**2)-np.sum(transverse**2)-np.sum(longitudinal**2)))
    assert defect < 1e-6
    return {'N':N,'species':{'charged':64,'neutral':1,'canonical_each':2*V,'Ref4_static':True},
            'native_colour_summed_word_equals_direct_on_empty_central_channel':True,
            'occupied_central_channel_rejects_unconditional_two_link_replacement':True,
            'once_shared_integer_flow_final_central_charge':int(rho_diff[B]),
            'fullE_energy_increment_exact':str(delta),'bare_self_only_increment_128_rejected':True,
            'orthogonal_fullPi_collective_HC_defect_float':defect,
            'point_probe_not_fullPhi_SU4_quantum_source':True,'fullH_action_or_GNS_dynamics_proved':False}


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument('--request',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();request=json.loads(args.request.read_text())
    assert all(digest(ROOT/name)==pin for name,pin in request['pins'].items())
    H=sp.Matrix([[0,0,0],[0,2,1],[0,1,2]])
    U=sp.Matrix([[1,0,0],[0,0,sp.I],[0,sp.I,0]])  # exp(-i pi H/2)
    A=sp.Matrix([[0,1,0],[1,0,0],[0,0,0]])
    P=sp.diag(1,1,0);omega=sp.Matrix([1,0,0]);At=U.H*A*U
    assert U.H*U==sp.eye(3) and H*omega==sp.zeros(3,1)
    full=(omega.H*At*At*omega)[0];cropped=(omega.H*P*At*P*At*P*omega)[0]
    assert full==1 and cropped==0
    assert ((sp.eye(3)-P)*At*omega).norm()**2==1
    assert (omega.H*A*H*A*omega)[0]==2 and (omega.H*A*H*H*A*omega)[0]==5
    result={'kind':'EXACT_MIXED_WORD_AND_NATIVE_CAR_FULL_E_BOUNDARY_CONTROL_NOT_FULL_DYNAMICS',
            'source_sha256':digest(Path(__file__)),'request_sha256':digest(args.request),
            'feature_countercase':{'positive_ground_spectrum':[0,1,3],'A_measure':'(delta_1+delta_3)/2',
                'first_moment':2,'second_moment':5,'full_double_word':str(full),'compressed_double_word':str(cropped),
                'timezero_GNS_invariance_from_positive_feature_rejected':True,'method_only_not_native':True},
            'native_CAR_fullE': [native(N) for N in (4,6,8)],
            'full_local_GNS_or_boundary_state_identity':False,'physical1_phase_stiffness_photon':False,
            'campaign_complete':False,'old_producer_solver_root_checker_rerun':False}
    assert all(digest(ROOT/name)==pin for name,pin in request['pins'].items())
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'mixed_word_wrongcase_rejected':True,'N':[4,6,8],'full_dynamics':False}))

if __name__=='__main__':
    main()
