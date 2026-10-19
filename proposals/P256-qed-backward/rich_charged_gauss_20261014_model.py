#!/usr/bin/env python3
"""Anchored charged-sector completion of Astra's compact electric Hubbard plaquette.
No bare magnetic term, old oscillator G, or cubic Gaussian D is used.
"""
from __future__ import annotations
import os
for variable in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[variable]='1'
import argparse
from dataclasses import asdict
import hashlib
import itertools
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import zipfile
import numpy as np
from scipy.linalg import eigh
import rich_interaction_20261014_coupled as neutral_supplier
from rich_photon_20261014_loop import Parameters, hop, plaquette_embedding, transverse_frame

LAW_ID='rich-charged-gauss-anchored-wilson-plaquette-v1-20261014'


def complex_data(value):
    a=np.asarray(value);return dict(real=a.real.tolist(),imag=a.imag.tolist())


def car(bits,mode,create):
    if bool((bits>>mode)&1)==create:return None
    return bits^(1<<mode),(-1)**((bits&((1<<mode)-1)).bit_count())


class ChargedGaussMolecule(neutral_supplier.CoupledMolecule):
    """Same local law plus sector-labelled opposite anchor charge c=N-2.
    rho_s=n_s-.5-c*delta_s,anchor; E_e-E_e-1=-rho_e.
    Anchor is an explicit external reservoir, with zero supplied charging energy.
    N2 restriction is exactly the selected neutral supplier Hamiltonian.
    """
    def __init__(self,number=3,cutoff=4,parameters=Parameters(),anchor=0,frozen=False):
        if parameters.link_charge!=1 or parameters.sites_per_loop!=4 or parameters.particles_per_loop!=2:
            raise ValueError('Neutral base has e=1, four sites, Nref2')
        self.number,self.anchor=number,anchor
        self.parameters,self.cutoff,self.frozen=parameters,cutoff,frozen
        self.countercharge=number-2
        self.basis=[]
        for occupied in itertools.combinations(range(8),number):
            bits=sum(1<<i for i in occupied)
            rho2=[2*(((bits>>(2*s))&1)+((bits>>(2*s+1))&1))-1
                  -2*self.countercharge*int(s==anchor) for s in range(4)]
            if frozen:self.basis.append((bits,(0,0,0,0)));continue
            for circulation in range(-cutoff-2,cutoff+2):
                previous=2*circulation+1;fields=[]
                for site in range(4):
                    previous-=rho2[site];fields.append(previous)
                if max(abs(e) for e in fields)<=2*cutoff+1:self.basis.append((bits,tuple(fields)))
        self.index={key:i for i,key in enumerate(self.basis)};d=len(self.basis)
        self.transfer=np.zeros((4,2,d,d),complex)
        self.electric=np.array([np.asarray(fields)/2 for _,fields in self.basis])
        self.double=np.array([sum(((b>>(2*s))&1)*((b>>(2*s+1))&1) for s in range(4)) for b,_ in self.basis])
        self.electric_energy=np.sum(self.electric**2,axis=1)/(2*parameters.electric_inertia)
        for column,(bits,fields) in enumerate(self.basis):
            for edge in range(4):
                for spin in range(2):
                    result=hop(bits,2*((edge+1)%4)+spin,2*edge+spin)
                    if result is None:continue
                    shifted=list(fields)
                    if not frozen:shifted[edge]+=2
                    row=self.index.get((result[0],tuple(shifted)))
                    if row is not None:self.transfer[edge,spin,row,column]=-parameters.hopping*result[1]
        adjoint=self.transfer.conj().transpose(0,1,3,2)
        self.current=-1j*(self.transfer-adjoint);self.contact=-(self.transfer+adjoint)
        self.H=np.diag(parameters.repulsion*self.double+self.electric_energy)+np.sum(self.transfer+adjoint,axis=(0,1))
        self.energies,self.states=eigh(self.H)
        self.gaps=self.energies-self.energies[0]
        sz=np.array([sum(((b>>(2*s))&1)-((b>>(2*s+1))&1) for s in range(4))/2 for b,_ in self.basis])
        self.sz=sz
        deg=np.flatnonzero(self.gaps<1e-9);frame=self.states[:,deg]
        spin,rotation=eigh(frame.conj().T@(sz[:,None]*frame))
        self.ground=frame@rotation[:,-1]
        self.state_id=hashlib.sha256(self.H.tobytes()+self.ground.tobytes()).hexdigest()

    def spin_properties(self,state=None):
        plus=np.zeros((len(self.basis),len(self.basis)),complex)
        for column,(bits,fields) in enumerate(self.basis):
            for site in range(4):
                result=hop(bits,2*site,2*site+1)
                if result is not None:
                    row=self.index.get((result[0],fields))
                    if row is not None:plus[row,column]=result[1]
        g=self.ground if state is None else state
        return dict(Sz=float(np.sum(abs(g)**2*self.sz)),
            S2=float(np.sum(abs(g)**2*self.sz**2)+.5*(np.linalg.norm(plus@g)**2+np.linalg.norm(plus.conj().T@g)**2)),
            raising_norm=float(np.linalg.norm(plus@g)),lowering_norm=float(np.linalg.norm(plus.conj().T@g)),
            fermion_parity=(-1)**self.number)

    def contact_element(self,qout,qin,left,right):
        a=plaquette_embedding(qout,0,1)@transverse_frame(qout)
        b=plaquette_embedding(qin,0,1)@transverse_frame(qin)
        elements=np.einsum('i,esij,j->es',left.conj(),self.contact,right,optimize=True)
        result=np.zeros((4,4),complex)
        for spin in range(2):result[2*spin:2*spin+2,2*spin:2*spin+2]=np.einsum('ep,eq,e->pq',a.conj(),b,elements[:,spin])
        return result

    def response(self,qout,qin,z):
        incoming=self.vertex(qin);outgoing=self.vertex(qout).conj().transpose(0,2,1)
        v,g=self.states,self.ground
        a=np.einsum('i,aij,jn->an',g.conj(),outgoing,v,optimize=True)
        b=np.einsum('ni,bij,j->bn',v.conj().T,incoming,g,optimize=True)
        c=np.einsum('i,bij,jn->bn',g.conj(),incoming,v,optimize=True)
        d=np.einsum('ni,aij,j->an',v.conj().T,outgoing,g,optimize=True)
        # Include full eigenspace; zero-gap commutator contributions cancel exactly,
        # without deleting another spin member of the ground multiplet by index.
        return self.contact_element(qout,qin,g,g)+np.einsum('an,bn,n->ab',a,b,1/(z-self.gaps))-np.einsum('bn,an,n->ab',c,d,1/(z+self.gaps))

    def raman(self,qout,qin,omega,eta,final,ground_only=False):
        incoming=self.vertex(qin);outgoing=self.vertex(qout).conj().transpose(0,2,1)
        g,f=self.ground,self.states[:,final]
        intermediate=g[:,None] if ground_only else self.states
        gaps=np.array([0.]) if ground_only else self.gaps
        a=np.einsum('i,aij,jn->an',f.conj(),outgoing,intermediate,optimize=True)
        b=np.einsum('ni,bij,j->bn',intermediate.conj().T,incoming,g,optimize=True)
        c=np.einsum('i,bij,jn->bn',f.conj(),incoming,intermediate,optimize=True)
        d=np.einsum('ni,aij,j->an',intermediate.conj().T,outgoing,g,optimize=True)
        omega_out=omega-self.gaps[final]
        return self.contact_element(qout,qin,f,g)+np.einsum('an,bn,n->ab',a,b,1/(omega-gaps+1j*eta))+np.einsum('bn,an,n->ab',c,d,1/(-omega_out-gaps+1j*eta))

    def elastic(self,qout,qin,omega,eta,ground_only=False):
        incoming=self.vertex(qin);outgoing=self.vertex(qout).conj().transpose(0,2,1)
        g=self.ground;v=g[:,None] if ground_only else self.states
        gaps=np.array([0.]) if ground_only else self.gaps
        a=np.einsum('i,aij,jn->an',g.conj(),outgoing,v,optimize=True)
        b=np.einsum('ni,bij,j->bn',v.conj().T,incoming,g,optimize=True)
        c=np.einsum('i,bij,jn->bn',g.conj(),incoming,v,optimize=True)
        d=np.einsum('ni,aij,j->an',v.conj().T,outgoing,g,optimize=True)
        return self.contact_element(qout,qin,g,g)+np.einsum('an,bn,n->ab',a,b,1/(omega-gaps+1j*eta))+np.einsum('bn,an,n->ab',c,d,1/(-omega-gaps+1j*eta))

    def field_response(self,z):
        """Full electric-field D_EE[4,4] in THIS same reference state, not A-A."""
        amp=self.states.conj().T@(self.electric*self.ground[:,None])
        return np.einsum('na,nb,n->ab',amp.conj(),amp,1/(z-self.gaps))-np.einsum('nb,na,n->ab',amp.conj(),amp,1/(z+self.gaps))


def charged_source(origin,target,mode,create=True):
    """Gauge-admissible Wilson/CAR operator target <- origin, unit amplitude.
    Anchor→site follows positive edges; create raises each path E by1, removes
    lower them. Countercharge changes simultaneously by +/-1 at the anchor.
    Cutoff projection is literal and its spectral weight is not renormalized.
    """
    difference=1 if create else -1
    if target.number-origin.number!=difference or target.anchor!=origin.anchor:
        raise ValueError('Source requires N change and identical anchor')
    if target.parameters!=origin.parameters or target.cutoff!=origin.cutoff or target.frozen!=origin.frozen:
        raise ValueError('Source requires same microscopic law/cutoff')
    out=np.zeros((len(target.basis),len(origin.basis)),complex)
    site=mode//2;path=[];edge=origin.anchor
    while edge!=site:path.append(edge);edge=(edge+1)%4
    for column,(bits,fields) in enumerate(origin.basis):
        result=car(bits,mode,create)
        if result is None:continue
        shifted=list(fields)
        if not origin.frozen:
            for edge in path:shifted[edge]+=2*difference
        row=target.index.get((result[0],tuple(shifted)))
        if row is not None:out[row,column]=result[1]
    return out


def green_spectral(reference,plus,minus):
    addition=np.column_stack([plus.states.conj().T@(charged_source(reference,plus,a,True)@reference.ground) for a in range(8)])
    removal=np.column_stack([minus.states.conj().T@(charged_source(reference,minus,a,False)@reference.ground) for a in range(8)])
    return dict(addition=addition,removal=removal,
        addition_poles=plus.energies-reference.energies[0],
        removal_poles=reference.energies[0]-minus.energies,
        reference_state_id=reference.state_id,plus_state_id=plus.state_id,minus_state_id=minus.state_id)


def retarded_green(data,z):
    """Full 8x8 spin/site G, both stored poles use denominator z-pole."""
    a,b=data['addition'],data['removal']
    return np.einsum('na,nb,n->ab',a.conj(),a,1/(z-data['addition_poles']))+np.einsum('nb,na,n->ab',b.conj(),b,1/(z-data['removal_poles']))


def physical(tensor):return tensor.reshape(2,2,2,2).sum(axis=(0,2))


def packet_motion(neutral,charged,mode=4):
    """Create an actual Wilson/CAR packet above neutral ground, then evolve H3.
    This packet is a declared different state from the N3 ground used by G/J/D.
    """
    raw=charged_source(neutral,charged,mode,True)@neutral.ground
    weight=float(np.vdot(raw,raw).real);initial=raw/np.sqrt(weight)
    coefficients=charged.states.conj().T@initial
    density=np.array([[((b>>(2*s))&1)+((b>>(2*s+1))&1) for s in range(4)] for b,_ in charged.basis])
    times=np.linspace(0.,12.,49);rows=[];currents=[]
    for t in times:
        state=charged.states@(np.exp(-1j*charged.gaps*t)*coefficients)
        prob=abs(state)**2
        rows.append([float(t),*list(prob@density),float(prob@charged.electric_energy),
            float(prob@charged.double),float(np.vdot(state,charged.H@state).real),float(prob@charged.sz)])
        currents.append(np.einsum('i,esij,j->es',state.conj(),charged.current,state,optimize=True).real)
    rows=np.array(rows)
    return dict(mode=mode,source_norm_squared=weight,initial_spin=charged.spin_properties(initial),
        ground_overlap=float(abs(np.vdot(charged.ground,initial))**2),
        columns=['time','n0','n1','n2','n3','electric_energy','double_occupancy','total_H','Sz'],
        history=rows.tolist(),currents=np.array(currents).tolist(),
        density_ranges=np.ptp(rows[:,1:5],axis=0).tolist(),
        electric_energy_range=float(np.ptp(rows[:,5])),double_occupancy_range=float(np.ptp(rows[:,6])),
        conserved_H_range=float(np.ptp(rows[:,7])),conserved_Sz_range=float(np.ptp(rows[:,8])),
        max_number_current=float(np.max(np.abs(currents))),
        reference='Wilson/CAR mode4 above N2 ground, evolving full H3; not ground-reference G state')


def run(request,output):
    cutoff=request.get('cutoff',4);anchor=request.get('anchor',0);p=Parameters(**request.get('parameters',{}))
    sectors={n:ChargedGaussMolecule(n,cutoff,p,anchor) for n in (1,2,3,4)}
    neutral,charged=sectors[2],sectors[3]
    frozen=ChargedGaussMolecule(3,cutoff,p,anchor,frozen=True)
    spectral_neutral=green_spectral(neutral,sectors[3],sectors[1])
    spectral_charged=green_spectral(charged,sectors[4],sectors[2])
    packet=packet_motion(neutral,charged)
    frequencies=request.get('frequencies',[-2.,-.5,.5,1.1,2.,4.])
    eta=request.get('eta',.04);responses=[]
    probes=request.get('probes',[dict(qin=[.7,.2,.45],qout=[-.25,.65,.3],omega=2.1),dict(qin=[-.4,.6,.3],qout=[.55,-.3,.7],omega=3.3),dict(qin=[.2,-.7,.6],qout=[-.6,-.2,.4],omega=4.3)])
    for probe in probes:
        qi,qo,w=np.array(probe['qin']),np.array(probe['qout']),probe['omega']
        elastic=charged.elastic(qo,qi,w,eta);bare=frozen.elastic(qo,qi,w,eta);groundonly=charged.elastic(qo,qi,w,eta,True)
        available=np.flatnonzero((charged.gaps>1e-7)&(charged.gaps<w))
        inelastic=[]
        for f in available[:request.get('max_receivers',12)]:
            matrix=charged.raman(qo,qi,w,eta,int(f))
            inelastic.append(dict(final_index=int(f),transfer=float(charged.gaps[f]),omega_out=float(w-charged.gaps[f]),matrix=complex_data(matrix),physical_weight=float(np.linalg.norm(physical(matrix))**2)))
        responses.append(dict(**probe,elastic=complex_data(elastic),physical_elastic=complex_data(physical(elastic)),
            frozen_elastic=complex_data(bare),ground_only_elastic=complex_data(groundonly),
            feedback_difference_norm=float(np.linalg.norm(physical(elastic-bare))),
            ground_only_difference_norm=float(np.linalg.norm(physical(elastic-groundonly))),
            available_receiver_count=len(available),selected_receivers=inelastic))
    greens=[]
    for w in frequencies:
        z=w+1j*eta
        greens.append(dict(omega=w,G_neutral=complex_data(retarded_green(spectral_neutral,z)),
            G_charged=complex_data(retarded_green(spectral_charged,z)),
            D_EE_charged=complex_data(charged.field_response(z)),
            current_response_charged=complex_data(charged.response(np.array(probes[0]['qout']),np.array(probes[0]['qin']),z))))
    summary=[]
    for n,m in sectors.items():
        q=np.abs(m.ground)**2
        summary.append(dict(N=n,countercharge=m.countercharge,dimension=len(m.basis),E0=float(m.energies[0]),
            state_id=m.state_id,spin=m.spin_properties(),electric_energy=float(q@m.electric_energy),double_occupancy=float(q@m.double)))
    result=dict(law=LAW_ID,parent_law=neutral_supplier.LAW_ID,parameters=asdict(p),cutoff=cutoff,anchor=anchor,
        source='c_dagger(site,spin) times integer Wilson anchor-to-site electric raising; simultaneous opposite anchor charge; removal is reverse map',
        sectors=summary,neutral_charge_gap=float(sectors[3].energies[0]+sectors[1].energies[0]-2*sectors[2].energies[0]),
        neutral_mu_plus=float(sectors[3].energies[0]-sectors[2].energies[0]),neutral_mu_minus=float(sectors[2].energies[0]-sectors[1].energies[0]),
        charged_charge_gap=float(sectors[4].energies[0]+sectors[2].energies[0]-2*sectors[3].energies[0]),
        charged_mu_plus=float(sectors[4].energies[0]-sectors[3].energies[0]),charged_mu_minus=float(sectors[3].energies[0]-sectors[2].energies[0]),
        source_equal_time_neutral=complex_data(spectral_neutral['addition'].conj().T@spectral_neutral['addition']+spectral_neutral['removal'].T@spectral_neutral['removal'].conj()),
        source_equal_time_charged=complex_data(spectral_charged['addition'].conj().T@spectral_charged['addition']+spectral_charged['removal'].T@spectral_charged['removal'].conj()),
        matrices=greens,scattering=responses,
        charged_packet=packet,
        scope='Anchored countercharge completion; physical fermion spin-half charged N3 state, no free isolated electron/COM/relativistic positron claim. One dynamical flux cycle, not two propagating photon branches. D_EE is exact electric spectral tensor, not fabricated A-A. Static reservoir charging energy supplied zero.')
    arrays={}
    for n,m in sectors.items():
        arrays.update({f'N{n}_H':m.H,f'N{n}_energies':m.energies,f'N{n}_states':m.states,f'N{n}_ground':m.ground,
            f'N{n}_basis_bits':np.array([x[0] for x in m.basis]),f'N{n}_electric':m.electric})
    arrays.update(J_charged=charged.current,C_charged=charged.contact,
        neutral_addition=spectral_neutral['addition'],neutral_removal=spectral_neutral['removal'],
        neutral_addition_poles=spectral_neutral['addition_poles'],neutral_removal_poles=spectral_neutral['removal_poles'],
        charged_addition=spectral_charged['addition'],charged_removal=spectral_charged['removal'],
        charged_addition_poles=spectral_charged['addition_poles'],charged_removal_poles=spectral_charged['removal_poles'])
    np.savez_compressed(output.with_suffix('.npz'),**arrays)
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(law=LAW_ID,neutral_charge_gap=result['neutral_charge_gap'],charged_charge_gap=result['charged_charge_gap'],
        charged_spin=charged.spin_properties(),feedback=[x['feedback_difference_norm'] for x in responses],
        ground_intermediate_difference=[x['ground_only_difference_norm'] for x in responses])))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--receipt',required=True);parser.add_argument('--worker',action='store_true');args=parser.parse_args()
    if args.worker:run(json.loads(Path(args.input).read_text()),Path(args.output));return
    start=time.time();command=[sys.executable,str(Path(__file__).resolve()),'--input',args.input,'--output',args.output,'--receipt',args.receipt,'--worker']
    execution=subprocess.run(command,capture_output=True,text=True)
    files=[Path(__file__),Path(neutral_supplier.__file__),Path(neutral_supplier.__file__).with_name('rich_photon_20261014_loop.py'),Path(args.input)]
    for path in (Path(args.output),Path(args.output).with_suffix('.npz')):
        if path.exists():files.append(path)
    receipt=dict(command=command,natural_exit=execution.returncode,start_unix=start,end_unix=time.time(),
        stdout=execution.stdout,stderr=execution.stderr,python=sys.executable,python_version=platform.python_version(),
        numpy=np.__version__,sha256={str(f.resolve()):hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
    Path(args.receipt).write_text(json.dumps(receipt,indent=2)+'\n')
    capsule=Path(args.receipt).with_suffix('.zip')
    with zipfile.ZipFile(capsule,'w',zipfile.ZIP_DEFLATED) as archive:
        for path in files+[Path(args.receipt)]:archive.write(path,arcname=path.name)
    print(execution.stdout,end='');print(execution.stderr,end='',file=sys.stderr);sys.exit(execution.returncode)
if __name__=='__main__':main()
