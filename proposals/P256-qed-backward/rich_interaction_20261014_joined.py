#!/usr/bin/env python3
"""Run NEW own two-mode construction and literal RichElectron same-H consumer.
This does NOT combine independent laws' propagators. Each law/state is separate.
"""
import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
import numpy as np
from scipy.linalg import eigh
import rich_interaction_20261014_model as own
import rich_electron_20261014_model as electron


def solve_peer(p):
    sector=electron.build_sector(p,5)
    energies,vectors=eigh(sector['H'])
    deg=np.flatnonzero(energies-energies[0]<1e-9)
    ground_frame=vectors[:,deg]
    spins,rotation=eigh(ground_frame.conj().T@sector['spin'][2]@ground_frame)
    ground=ground_frame@rotation[:,-1]
    return dict(sector=sector,energy=energies,states=vectors,ground=ground,E0=energies[0],
                spin_z=float(spins[-1]))


def joined_peer(request):
    p=electron.Parameters(**request.get('electron_params',{}))
    full=solve_peer(p); frozen=solve_peer(dataclasses.replace(p,coupling=0.))
    v,E,g=full['states'],full['energy'],full['ground']; e0=E[0]; sector=full['sector']
    spectrum=[]
    for omega in request.get('frequencies',[.4,.75,1.05,1.4]):
        z=omega+1j*.035
        D=own.retarded(full,[sector['Q']],z)
        D0=own.retarded(frozen,[frozen['sector']['Q']],z)
        spectrum.append(dict(omega=omega,D=D,D0=D0,Pi_exact=np.linalg.inv(D0)-np.linalg.inv(D),
            full_edge_current_response=own.retarded(full,sector['edge_current'],z)))
    amplitudes=[]
    probes=[([0.,.7,0.],[1.,0.,0.],[.65,0.,0.],[0.,1.,0.],1.1),
            ([.5,.5,0.],[1/np.sqrt(2),-1/np.sqrt(2),0.],[-.8,.3,0.],[.3/np.sqrt(.73),.8/np.sqrt(.73),0.],1.6),
            ([.3,.6,.4],[2/np.sqrt(5),-1/np.sqrt(5),0.],[-.6,-.2,.5],[1/np.sqrt(10),-3/np.sqrt(10),0.],2.2)]
    for kin,epsin,kout,epsout,omega in probes:
        incoming=own.plane_wave_edges(kin,epsin); outgoing=own.plane_wave_edges(kout,epsout).conj()
        ji=electron.current(sector,incoming); jo=electron.current(sector,outgoing)
        c=electron.contact(sector,outgoing,incoming)
        first=v@((v.conj().T@ji@g)/(e0+omega-E+1j*.035))
        receivers=[]
        for f in range(min(32,len(E))):
            transfer=E[f]-e0; outomega=omega-transfer
            if outomega<=0: continue
            second=v@((v.conj().T@jo@g)/(e0-outomega-E+1j*.035))
            amplitude=np.vdot(v[:,f],c@g+jo@first+ji@second)
            receivers.append(dict(final_index=f,energy_transfer=float(transfer),omega_out=float(outomega),
                amplitude=amplitude,weight=abs(amplitude)**2))
        elastic=np.vdot(g,c@g+jo@first+ji@(v@((v.conj().T@jo@g)/(e0-omega-E+1j*.035))))
        amplitudes.append(dict(kin=kin,kout=kout,epsin=epsin,epsout=epsout,omega=omega,
            elastic=elastic,receivers=receivers))
    # Literal peer carrier Green function uses its own N4 reference, clearly separate state.
    sectors=electron.solve(p)
    green=electron.retarded_green(electron.spectral_data(sectors,p,0.),.6,.035)
    return dict(law='RICH_ELECTRON_EXACT_DIMER_RING_QUANTUM_PEIERLS_CAVITY',
        law_source=str(Path(electron.__file__).resolve()),params=dataclasses.asdict(p),
        state='spin-up member of exact N5 lowest-energy eigenspace',spin_z=full['spin_z'],
        ground_energy=e0,ground_pair_expectation=float(np.vdot(g,(sector['positive_band_number']-np.eye(len(g)))@g).real),
        field_response=spectrum,nonforward_response=amplitudes,
        peer_G_N4_reference=dict(state='peer lowest N4 state, not same-state splice into charged N5 response',G=green),
        geometry='square positions supplied here: straight edges length1; oriented ring peer link order identical',
        difference_from_own_law='one uniform loop-phase cavity versus two spatial standing-wave link modes')


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--input',required=True); parser.add_argument('--output',required=True)
    parser.add_argument('--receipt',required=True); parser.add_argument('--worker',action='store_true')
    args=parser.parse_args()
    if not args.worker:
        started=time.time()
        command=[sys.executable,str(Path(__file__).resolve()),'--input',args.input,'--output',args.output,'--receipt',args.receipt,'--worker']
        result=subprocess.run(command,capture_output=True,text=True)
        files=[Path(__file__),Path(own.__file__),Path(electron.__file__),Path(args.input)]
        bindings={str(f.resolve()):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
        out=Path(args.output)
        if out.exists(): bindings[str(out.resolve())]=hashlib.sha256(out.read_bytes()).hexdigest()
        receipt=dict(command=command,natural_exit=result.returncode,started_unix=started,finished_unix=time.time(),
            stdout=result.stdout,stderr=result.stderr,sha256=bindings,python=sys.executable)
        Path(args.receipt).write_text(json.dumps(receipt,indent=2)+'\n')
        print(result.stdout,end='');print(result.stderr,end='',file=sys.stderr)
        sys.exit(result.returncode)
    request=json.loads(Path(args.input).read_text())
    results=dict(own_same_state_law=own.calculate(request),literal_peer_same_H_consumer=joined_peer(request),
        no_splice='each response computed from its own full H and exact same-law state; photon peer induced law is not joined')
    Path(args.output).write_text(json.dumps(results,default=own.encode,indent=2)+'\n')
    ownresult=results['own_same_state_law']
    print(json.dumps(dict(own_pair=ownresult['ground_pair_expectation'],
        field_changes=[x['relative_field_change'] for x in ownresult['full_feedback']],
        scattering_changes=[s['elastic_change_fraction'] for s in ownresult['scattering']],
        peer_pair=results['literal_peer_same_H_consumer']['ground_pair_expectation']),default=own.encode))
if __name__=='__main__': main()
