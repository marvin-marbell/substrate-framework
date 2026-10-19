#!/usr/bin/env python3
"""NEW literal fullq peer consumer: charged-state G, two flux D, J/C scattering.
The one-plaquette predecessor is NOT imported or used as a tensor source.
"""
import os
for variable in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[variable]='1'
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import zipfile
from typing import TypeAlias, TypedDict
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
import rich_photon_fullq_20261014_model as supplier

CONSUMER='rich-charged-fullq-twoflux-interaction-v1-20261014'

RealData: TypeAlias = float | list["RealData"]


class ComplexData(TypedDict):
    real: RealData
    imag: RealData


class InelasticResult(TypedDict):
    final_index: int
    transfer: float
    omega_out: float
    matrix: ComplexData
    physical_weight: float


class InteractionResult(TypedDict):
    qin: object
    qout: object
    omega: float
    elastic: ComplexData
    physical_elastic: ComplexData
    ground_only: ComplexData
    ground_only_difference_norm: float
    inelastic: list[InelasticResult]
    selected_final_state_scope: str


def encode(x) -> ComplexData:
    a=np.asarray(x);return ComplexData(real=a.real.tolist(),imag=a.imag.tolist())


def polarization_frame(q):
    q=np.asarray(q,float);q=q/np.linalg.norm(q)
    seed=np.eye(3)[np.argmin(abs(q))];first=np.cross(q,seed);first/=np.linalg.norm(first)
    return np.column_stack((first,np.cross(q,first)))


def profiles(sector,q):
    frame=polarization_frame(q);q=np.asarray(q)
    return np.array([((sector.positions[b]-sector.positions[a])@frame)
        *np.exp(1j*q@((sector.positions[a]+sector.positions[b])/2))
        *np.sinc(q@(sector.positions[b]-sector.positions[a])/(2*np.pi)) for a,b in sector.edges])


def spin_links(sector):
    """Split the ACTUAL supplied CAR transfer by removed physical-spin bit."""
    currents=[];contacts=[]
    for edge,(origin,_) in enumerate(sector.edges):
        t=sector.transfers[edge].tocoo();bits=np.asarray(sector.fock,dtype=np.int64)[t.col//sector.nflux]
        target_bits=np.asarray(sector.fock,dtype=np.int64)[t.row//sector.nflux]
        spinup=((bits^target_bits)&(1<<(2*origin)))!=0
        c=[];k=[]
        for spin in range(2):
            mask=spinup if spin==0 else ~spinup
            split=sparse.csr_matrix((t.data[mask],(t.row[mask],t.col[mask])),shape=t.shape)
            c.append(-1j*(split-split.T));k.append(-(split+split.T))
        currents.append(c);contacts.append(k)
    return currents,contacts


def vertices(sector,link_currents,q):
    r=profiles(sector,q)
    return [sum((r[e,pol]*link_currents[e][spin] for e in range(len(sector.edges))),
        sparse.csr_matrix(sector.H.shape,dtype=complex)) for spin in range(2) for pol in range(2)]


def contact_element(sector,link_contacts,qout,qin,left,right):
    a,b=profiles(sector,qout).conj(),profiles(sector,qin)
    out=np.zeros((4,4),complex)
    for spin in range(2):
        values=np.array([np.vdot(left,k[spin]@right) for k in link_contacts])
        out[2*spin:2*spin+2,2*spin:2*spin+2]=np.einsum('ep,eq,e->pq',a,b,values)
    return out


def physical(matrix):return matrix.reshape(2,2,2,2).sum(axis=(0,2))


def orthogonal_selected(sector,count):
    energy,states=sector.solve(count)
    start=0
    while start<len(energy):
        end=start+1
        while end<len(energy) and abs(energy[end]-energy[start])<1e-8:end+=1
        states[:,start:end]=np.linalg.qr(states[:,start:end])[0]
        start=end
    return energy,states


def interaction(sector,g,E0,selected_energy,selected_states,qin,qout,omega,eta,receivers) -> InteractionResult:
    links,contacts=spin_links(sector)
    ji=vertices(sector,links,qin);jo=[x.conj().T for x in vertices(sector,links,qout)]
    incoming=np.column_stack([x@g for x in ji]);outgoing=np.column_stack([x@g for x in jo])
    identity=sparse.eye(sector.dimension)
    first=splu(((E0+omega+1j*eta)*identity-sector.H).tocsc()).solve(incoming)
    def response(f,transfer):
        outgoing_energy=omega-transfer
        second=splu(((E0-outgoing_energy+1j*eta)*identity-sector.H).tocsc()).solve(outgoing)
        matrix=contact_element(sector,contacts,qout,qin,f,g)
        for a in range(4):
            matrix[a,:]+=(f.conj()@(jo[a]@first))
            for b in range(4):matrix[a,b]+=np.vdot(f,ji[b]@second[:,a])
        return matrix
    elastic=response(g,0.)
    inelastic: list[InelasticResult]=[]
    for index in np.flatnonzero((selected_energy-E0>1e-7)&(selected_energy-E0<omega))[:receivers]:
        transfer=float(selected_energy[index]-E0);matrix=response(selected_states[:,index],transfer)
        inelastic.append(InelasticResult(final_index=int(index),transfer=transfer,omega_out=omega-transfer,
            matrix=encode(matrix),physical_weight=float(np.linalg.norm(physical(matrix))**2)))
    c=contact_element(sector,contacts,qout,qin,g,g)
    ground_in=np.array([np.vdot(g,x@g) for x in ji]);ground_out=np.array([np.vdot(g,x@g) for x in jo])
    frozen_intermediate=c+np.outer(ground_out,ground_in)*(1/(omega+1j*eta)+1/(-omega+1j*eta))
    return InteractionResult(qin=qin,qout=qout,omega=omega,elastic=encode(elastic),physical_elastic=encode(physical(elastic)),
        ground_only=encode(frozen_intermediate),ground_only_difference_norm=float(np.linalg.norm(physical(elastic-frozen_intermediate))),
        inelastic=inelastic,selected_final_state_scope='partial low receiver spectrum; every intermediate is full sparse H resolvent')


def run(request,output):
    p=supplier.Parameters(**request.get('parameters',{}))
    charged=supplier.Sector(p,2,1);plus=supplier.Sector(p,3,1);minus=supplier.Sector(p,1,1)
    energies,states=orthogonal_selected(charged,request.get('selected_states',10));g=states[:,0];E0=energies[0]
    Eplus=plus.solve(2)[0][0];Eminus=minus.solve(2)[0][0]
    hamiltonian=charged.H.tocsr()
    state_id=hashlib.sha256(hamiltonian.data.tobytes()+hamiltonian.indices.tobytes()+g.tobytes()).hexdigest()
    loops=[charged.flux_operator(face) for face in charged.faces]
    field=[(x-x.T)/(2j) for x in loops]
    additions=[];removals=[];source_reports=[]
    for site in range(charged.vertices):
        a,ar=supplier.charged_source(charged,plus,site,0,True)
        b,br=supplier.charged_source(charged,minus,site,0,False)
        additions.append(a@g);removals.append(b@g);source_reports.append(dict(site=site,addition=ar,removal=br))
    A=np.column_stack(additions);B=np.column_stack(removals)
    identity=sparse.eye(charged.dimension);green=[]
    eta=request.get('eta',.05)
    for w in request.get('frequencies',[-1.,.5,1.2]):
        z=w+1j*eta
        # Literal peer pure G call: same charged state, source paths and full H.
        G=supplier.charged_green(charged,plus,minus,g,E0,0.,z)
        D=supplier.retarded_matrix(charged,g,E0,field,z)
        green.append(dict(omega=w,G_spinup=encode(G),D_twoflux=encode(D)))
    probes=request.get('probes',[dict(qin=[.7,.2,.45],qout=[-.25,.65,.3],omega=1.7),
        dict(qin=[-.4,.6,.3],qout=[.55,-.3,.7],omega=2.4),dict(qin=[.2,-.7,.6],qout=[-.6,-.2,.4],omega=3.3)])
    scattering=[interaction(charged,g,E0,energies,states,x['qin'],x['qout'],x['omega'],eta,request.get('receivers',3)) for x in probes]
    spin_squared=supplier.spin_squared(charged,g)
    result=dict(consumer=CONSUMER,law=supplier.LAW_ID,parameters=asdict(p),reference=dict(Nup=2,Ndown=1,dimension=charged.dimension,
        ground_energy=float(E0),state_id=state_id,Sz=.5,S2=spin_squared,state=charged.reduced_field(g)),
        spinup_adjacent_sectors=dict(addition_counts=[3,1],removal_counts=[1,1],addition_dimension=plus.dimension,
            removal_dimension=minus.dimension,mu_plus=float(Eplus-E0),mu_minus=float(E0-Eminus),
            channel_gap=float(Eplus+Eminus-2*E0),scope='spin-up-accessible adjacent sector gap, NOT global N4 charge gap because N4 spin-singlet sector is not computed'),
        source_equal_time=encode(A.conj().T@A+(B.conj().T@B).T),source_reports=source_reports,
        spectral=green,scattering=scattering,selected_energies=energies.tolist(),
        scope='Literal fullq double-face same charged H/state join; two genuine graph flux channels, not massless propagating branches; spin-up G6x6 plus spin-resolved J/C; anchored recoil and static countercharge explicit, no oneplaquette/old oscillator tensors imported')
    np.savez_compressed(output.with_suffix('.npz'),ground=g,selected_energy=energies,selected_states=states,
        H_data=hamiltonian.data,H_indices=hamiltonian.indices,H_indptr=hamiltonian.indptr,
        electric=charged.electric,addition_sources=A,removal_sources=B)
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(consumer=CONSUMER,E0=float(E0),S2=spin_squared,
        dimension=charged.dimension,ground_intermediate_differences=[x['ground_only_difference_norm'] for x in scattering],
        inelastic_max=[max([f['physical_weight'] for f in x['inelastic']],default=0.) for x in scattering])))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--receipt',required=True);parser.add_argument('--worker',action='store_true');a=parser.parse_args()
    if a.worker:run(json.loads(Path(a.input).read_text()),Path(a.output));return
    start=time.time();cmd=[sys.executable,str(Path(__file__).resolve()),'--input',a.input,'--output',a.output,'--receipt',a.receipt,'--worker']
    result=subprocess.run(cmd,capture_output=True,text=True)
    files=[Path(__file__),Path(supplier.__file__),Path(supplier.__file__).with_name('rich_photon_20261014_loop.py'),Path(a.input)]
    for f in (Path(a.output),Path(a.output).with_suffix('.npz')):
        if f.exists():files.append(f)
    receipt=dict(command=cmd,natural_exit=result.returncode,start_unix=start,end_unix=time.time(),stdout=result.stdout,
        stderr=result.stderr,python=sys.executable,python_version=platform.python_version(),numpy=np.__version__,
        sha256={str(f.resolve()):hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
    Path(a.receipt).write_text(json.dumps(receipt,indent=2)+'\n')
    with zipfile.ZipFile(Path(a.receipt).with_suffix('.zip'),'w',zipfile.ZIP_DEFLATED) as capsule:
        for f in files+[Path(a.receipt)]:capsule.write(f,arcname=f.name)
    print(result.stdout,end='');print(result.stderr,end='',file=sys.stderr);sys.exit(result.returncode)
if __name__=='__main__':main()
