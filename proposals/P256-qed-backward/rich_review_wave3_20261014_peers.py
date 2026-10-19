"""Fresh fixed-Hilbert supplier-mobility and mobile full-receiver consumer."""
from __future__ import annotations
from dataclasses import replace, asdict
import json
from pathlib import Path
import numpy as np
from scipy.linalg import eigh
import rich_induced_spin_20261014_model as induced
import rich_mobile_spinor_20261014_model as mobile

ROOT=Path(__file__).resolve().parent

def induced_probe():
    rows=[]
    for hopping in (0.,.4,1.4,2.4):
        s=induced.Sector(induced.Parameters(medium_hop=hopping))
        e,v=eigh(s.H.toarray()); block=v[:,abs(e-e[0])<1e-9]
        def mean(op): return float(np.trace(block.conj().T@(op@block)).real/block.shape[1])
        chi=sum((s.retarded(g,float(e[0]),[s.F,s.Ia,s.Im],.37+.09j) for g in block.T))/block.shape[1]
        rows.append(dict(parameters=asdict(s.p),dimension=s.dimension,ground_rank=block.shape[1],energy=float(e[0]),
                         flux=mean(s.F),active=mean(s.Ia),supplier=mean(s.Im),total=mean(s.It),
                         source_flux_variance=mean(s.F@s.F)-mean(s.F)**2,
                         response=chi.tolist(),constraints=s.constraints()))
    return dict(rows=rows,scope='Fixed active/supplier count, Gauss background and E cutoff; only supplier hopping changes. Ground projector trace used for frozen-supplier degeneracy, not arbitrary eigsh column. Supplier is a distinct charged species, NOT derived CAR electron or a photon medium.')

def mobile_probe():
    m=mobile.build(); k=np.array([.23,-.16,.09]); qin=np.array([-.13,.22,.17]); qout=np.array([.19,-.08,.11]); ei=np.array([.4,1.,-.2]); eo=np.array([1.,.3,.1]); kout=k+qin-qout
    h=mobile.hamiltonian(m,k); z=2.18+.08j; src=m['source']
    green=mobile.charged_green(m,k,z.real,z.imag); direct=src.conj().T@np.linalg.solve(z*np.eye(len(h))-h,src)
    ev,v=mobile.spectrum(m,k); ef,vf=mobile.spectrum(m,kout); omega=.48; eta=.08; i=0; f=1
    result=mobile.scattering(m,k,qin,ei,qout,eo,omega,i,f,eta)
    low=mobile.scattering(m,k,qin,ei,qout,eo,omega,i,f,eta,virtual=np.arange(4))
    a=v[:,i]; b=vf[:,f]; jout=mobile.vertex(m,kout,k+qin,eo.conj()); jin=mobile.vertex(m,k+qin,k,ei)
    je=mobile.vertex(m,k-qout,k,eo.conj()); ji=mobile.vertex(m,kout,k-qout,ei)
    co=mobile.contact(m,kout,k,qin,ei,qout,eo)
    full=b.conj()@co@a + b.conj()@jout@np.linalg.solve((ev[i]+omega+1j*eta)*np.eye(len(h))-mobile.hamiltonian(m,k+qin),jin@a) + b.conj()@ji@np.linalg.solve((ev[i]-result['outgoing_energy']+1j*eta)*np.eye(len(h))-mobile.hamiltonian(m,k-qout),je@a)
    q=kout-k; ward=mobile.vertex(m,kout,k,q)-m['params'].charge*(mobile.hamiltonian(m,kout)-h)
    source_amplitudes=v.conj().T@src
    density=np.eye(len(h))*m['params'].charge
    return dict(parameters=m['parameters'],dimension=len(h),momenta=dict(kin=k.tolist(),qin=qin.tolist(),qout=qout.tolist(),kout=kout.tolist()),
                Green_direct_error=float(np.linalg.norm(green-direct)),addition_source_sumrule=float(np.linalg.norm(source_amplitudes.conj().T@source_amplitudes-np.eye(src.shape[1]))),
                low_four_source_weight=float(np.sum(abs(source_amplitudes[:4])**2)),full_source_weight=float(np.sum(abs(source_amplitudes)**2)),
                finite_q_macrocell_Ward_norm=float(np.linalg.norm(ward)),amplitude_direct_error=float(abs(full-result['amplitude'])),
                low_four_amplitude_error=float(abs(full-low['amplitude'])),amplitude=complex(full),recoil=float(result['recoil_energy']),
                macrocell_density_internal_rotation_commutator=float(np.linalg.norm(m['J'][0]@density-density@m['J'][0])),
                scope='Same all162 Bloch law/state and exact macrocell Peierls line source. Full addition G and nonforward virtual receiver; source omits intra-shell EM density/current, no physical quantum field legs. Only lattice/material rotational symmetries expected.')

def encode(x):
    if isinstance(x,complex):return [x.real,x.imag]
    if isinstance(x,np.generic):return x.item()
    raise TypeError(type(x).__name__)

def main():
    result={'induced':induced_probe(),'mobile':mobile_probe()}
    (ROOT/'rich_review_wave3_20261014_peers_result.json').write_text(json.dumps(result,default=encode,indent=2)+'\n')
    print(json.dumps(result,default=encode),flush=True)

if __name__=='__main__':main()
