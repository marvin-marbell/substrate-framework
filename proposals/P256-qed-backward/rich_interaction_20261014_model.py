#!/usr/bin/env python3
"""Exact finite CAR/link-photon candidate, not continuum QED or a Ward-only probe.
All sectors, photons, currents and resolvents use the same Peierls Hamiltonian.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
if __name__ == '__main__':
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[name] = '1'
import numpy as np
from scipy.linalg import eigh, expm

LAW = 'SPINFUL_SQUARE_TWO_QUANTIZED_LINK_MODES_V1'
POSITIONS = np.array([[0.,0.,0.],[1.,0.,0.],[1.,1.,0.],[0.,1.,0.]])
EDGES = ((0,1),(1,2),(2,3),(3,0))
DEFAULT = dict(t=[1.,.65,1.,.65], omega=[.8,.95], g=.45, photon_levels=4,
               mode_momenta=[[0.,.9,0.],[.9,0.,0.]],
               mode_polarizations=[[1.,0.,0.],[0.,1.,0.]])

def encode(x):
    if isinstance(x, np.ndarray):
        return np.stack((x.real,x.imag),axis=-1).tolist() if np.iscomplexobj(x) else x.tolist()
    if isinstance(x, complex): return [x.real,x.imag]
    if isinstance(x, np.generic): return x.item()
    raise TypeError(type(x).__name__)

def plane_wave_edges(momentum, polarization):
    """Exact oriented straight-link integral of eps exp(i q.x), link charge=1."""
    q, eps = np.asarray(momentum,float), np.asarray(polarization,float)
    return np.array([np.dot(eps,POSITIONS[b]-POSITIONS[a])
        *np.exp(1j*np.dot(q,(POSITIONS[a]+POSITIONS[b])/2))
        *np.sinc(np.dot(q,POSITIONS[b]-POSITIONS[a])/(2*np.pi)) for a,b in EDGES])

def basis(up,down):
    return [sum(1<<(2*i) for i in u)+sum(1<<(2*i+1) for i in d)
            for u in itertools.combinations(range(4),up)
            for d in itertools.combinations(range(4),down)]

def car(mask,orbital,create):
    occupied=(mask>>orbital)&1
    if occupied == int(create): return None
    return mask^(1<<orbital), (-1)**((mask&((1<<orbital)-1)).bit_count())

def hop(masks,target,source):
    out=np.zeros((len(masks),len(masks)),complex); index={m:i for i,m in enumerate(masks)}
    for j,m in enumerate(masks):
        first=car(m,source,False)
        if first is None: continue
        second=car(first[0],target,True)
        if second is not None and second[0] in index:
            out[index[second[0]],j]=first[1]*second[1]
    return out

def one_body(masks,matrix):
    out=np.zeros((len(masks),len(masks)),complex)
    for a in range(8):
        for b in range(8):
            if abs(matrix[a,b])>1e-14: out+=matrix[a,b]*hop(masks,a,b)
    return out

def build_sector(params,up=3,down=2):
    """Returns exact finite H, J[spin,edge], C[spin,edge], Q and CAR state basis.
    External source is phi_edge, inserted as exp(i(phi_edge+g A_edge(Q))).
    J=-dH/dphi, C=d²H/dphi². Field truncation is declared, not a fake continuum.
    """
    p={**DEFAULT,**params}; n=int(p['photon_levels']); masks=basis(up,down)
    phi=np.asarray(p.get('external_phase',[0.,0.,0.,0.]),float)
    a=np.diag(np.sqrt(np.arange(1,n)),1); eye=np.eye(n)
    aa=[np.kron(a,eye),np.kron(eye,a)]; fdim=n*n
    Q=[(x+x.T)/np.sqrt(2) for x in aa]
    Hfield=sum(w*(x.T@x+np.eye(fdim)/2) for w,x in zip(p['omega'],aa))
    H=np.kron(np.eye(len(masks)),Hfield).astype(complex)
    J=np.zeros((2,4,H.shape[0],H.shape[1]),complex); C=np.zeros_like(J)
    coefficients=np.array([plane_wave_edges(q,e).real for q,e in zip(p['mode_momenta'],p['mode_polarizations'])]).T
    h=np.zeros((8,8),complex)
    for edge,(s,t) in enumerate(EDGES):
        link=expm(1j*p['g']*sum(coefficients[edge,m]*Q[m] for m in range(2)))
        for spin in range(2):
            term=-p['t'][edge]*np.exp(1j*phi[edge])*np.kron(hop(masks,2*t+spin,2*s+spin),link)
            H+=term+term.conj().T
            J[spin,edge]=-1j*(term-term.conj().T)
            C[spin,edge]=-(term+term.conj().T)
            h[2*t+spin,2*s+spin]=h[2*s+spin,2*t+spin]=-p['t'][edge]
    ev,vec=eigh(h); positive=vec[:,ev>0]@vec[:,ev>0].conj().T
    pair=np.kron(one_body(masks,positive)-max(up+down-4,0)*np.eye(len(masks)),np.eye(fdim))
    return dict(H=H,J=J,C=C,Q=[np.kron(np.eye(len(masks)),x) for x in Q],
        pair=pair,masks=masks,field_dimension=fdim,params=p,band_energies=ev,
        up=up,down=down,link_mode_coefficients=coefficients)

def current(model,direction,spin=None):
    """Pure exact current vertex; complex directions denote positive-frequency sources."""
    j=model['J'].sum(axis=0) if spin is None else model['J'][spin]
    return np.einsum('e,eij->ij',np.asarray(direction),j)

def contact(model,left,right,spin=None):
    c=model['C'].sum(axis=0) if spin is None else model['C'][spin]
    return np.einsum('e,eij->ij',np.asarray(left)*np.asarray(right),c)

def diagonalize(model):
    energy,states=eigh(model['H']); return dict(model=model,energy=energy,states=states,
        ground=states[:,0],E0=energy[0],gap=energy[1]-energy[0])

def retarded(solution,operators,z):
    """Full exact Lehmann tensor -i theta(t)<[Oa(t),Ob(0)]>, both signs retained."""
    v=solution['states']; g=solution['ground']; delta=solution['energy']-solution['E0']
    amp=np.array([v.conj().T@op@g for op in operators])
    return (amp.conj()/(z-delta))@amp.T-(amp/(z+delta))@amp.conj().T

def scattering(solution,kin,epsin,kout,epsout,omega,eta=.035):
    """Full Kramers-Heisenberg operator from differentiated same H, not imported QED.
    Reports selected inelastic receiver states with omega_out=omega-(Ef-Ei).
    Outgoing current uses complex-conjugate spatial profile. Photon/matter recoil
    is actual manybody final-energy transfer on a finite anchored square.
    """
    model=solution['model']; v=solution['states']; E=solution['energy']; g=solution['ground']; E0=E[0]
    left=plane_wave_edges(kout,epsout).conj(); right=plane_wave_edges(kin,epsin)
    Jin=current(model,right); Jout=current(model,left); C=contact(model,left,right)
    first=v@((v.conj().T@Jin@g)/(E0+omega-E+1j*eta))
    amplitudes=[]
    for f in range(min(32,len(E))):
        transfer=float(E[f]-E0); outgoing=omega-transfer
        if outgoing<=0: continue
        second=v@((v.conj().T@Jout@g)/(E0-outgoing-E+1j*eta))
        total=C@g+Jout@first+Jin@second
        amplitude=np.vdot(v[:,f],total)
        amplitudes.append(dict(final_index=f,energy_transfer=transfer,omega_out=outgoing,
            amplitude=amplitude,weight=abs(amplitude)**2,
            final_pair_expectation=float(np.vdot(v[:,f],model['pair']@v[:,f]).real)))
    # Full spin-current channels including cross spin terms, not fictitious spin flips.
    spin_matrix=np.zeros((2,2),complex)
    for outspin in range(2):
        jo=current(model,left,outspin)
        for inspin in range(2):
            ji=current(model,right,inspin)
            a=v@((v.conj().T@ji@g)/(E0+omega-E+1j*eta))
            b=v@((v.conj().T@jo@g)/(E0-omega-E+1j*eta))
            cc=contact(model,left,right,inspin) if inspin==outspin else np.zeros_like(C)
            spin_matrix[outspin,inspin]=np.vdot(g,cc@g+jo@a+ji@b)
    # Ground-only virtual intermediate is an explicit inadequate control.
    elastic_frozen=np.vdot(g,C@g)+np.vdot(g,Jout@g)*np.vdot(g,Jin@g)*(1/(omega+1j*eta)+1/(-omega+1j*eta))
    return dict(kin=kin,kout=kout,epsin=epsin,epsout=epsout,omega=omega,
        spin_current_channel_matrix=spin_matrix,elastic=sum(spin_matrix.ravel()),
        ground_only_intermediate_elastic=elastic_frozen,receivers=amplitudes)

def annihilation(source,target,orbital):
    index={m:i for i,m in enumerate(target['masks'])}
    a=np.zeros((len(target['masks']),len(source['masks'])),complex)
    for j,m in enumerate(source['masks']):
        result=car(m,orbital,False)
        if result is not None and result[0] in index: a[index[result[0]],j]=result[1]
    return np.kron(a,np.eye(source['field_dimension']))

def charged_G(solution,omega,eta=.035):
    """Exact spin-up site 4x4 retarded electron G: N+1 addition and N-1 holes."""
    m=solution['model']; p=m['params']; g=solution['ground']; z=omega+1j*eta
    plus=diagonalize(build_sector(p,m['up']+1,m['down']))
    minus=diagonalize(build_sector(p,m['up']-1,m['down']))
    addition=np.array([plus['states'].conj().T@annihilation(plus['model'],m,2*i).conj().T@g for i in range(4)])
    removal=np.array([minus['states'].conj().T@annihilation(m,minus['model'],2*i)@g for i in range(4)])
    result=(addition.conj()/(z-(plus['energy']-solution['E0'])))@addition.T
    result+=(removal/(z+(minus['energy']-solution['E0'])))@removal.conj().T
    return dict(G=result,addition_energies=plus['energy']-solution['E0'],
        hole_energies=minus['energy']-solution['E0'],
        equal_time_anticommutator=addition.conj()@addition.T+removal@removal.conj().T)

def static_amputated_vertex(params,omega,direction,eta=.035,step=1e-4):
    """Gamma=-dG^{-1}/ds at phi=s*direction, including reference-state relaxation.
    This is the static-source 4x4 spin-up vertex, NOT an uncomputed dynamic Gamma.
    The finite derivative has O(step²) error; every shifted G uses its own exact
    same-law ground and full addition/removal/photon sectors.
    """
    greens=[]
    for s in (-step,step):
        p={**params,'external_phase':(s*np.asarray(direction)).tolist()}
        solution=diagonalize(build_sector(p))
        greens.append(charged_G(solution,omega,eta)['G'])
    return -(np.linalg.inv(greens[1])-np.linalg.inv(greens[0]))/(2*step)

def calculate(request):
    p={**DEFAULT,**request.get('params',{})}; full=diagonalize(build_sector(p))
    uncoupled=diagonalize(build_sector({**p,'g':0.}))
    observations=[]
    frequencies=request.get('frequencies',[.4,.75,1.05,1.4])
    for w in frequencies:
        z=w+1j*request.get('eta',.035)
        D=retarded(full,full['model']['Q'],z); D0=retarded(uncoupled,uncoupled['model']['Q'],z)
        # Exact implicit photon self-energy in the retained field space; no RPA claimed.
        Pi=np.linalg.inv(D0)-np.linalg.inv(D)
        edgeJ=list(full['model']['J'].sum(axis=0))
        observations.append(dict(omega=w,D=D,D0=D0,Pi_exact=Pi,
            current_tensor=retarded(full,edgeJ,z),
            relative_field_change=float(np.linalg.norm(D-D0)/np.linalg.norm(D0))))
    directions=request.get('scattering',[dict(kin=[0.,.7,0.],epsin=[1.,0.,0.],kout=[.65,0.,0.],epsout=[0.,1.,0.],omega=1.1),
        dict(kin=[.5,.5,0.],epsin=[1/np.sqrt(2),-1/np.sqrt(2),0.],kout=[-.8,.3,0.],epsout=[.3/np.sqrt(.73),.8/np.sqrt(.73),0.],omega=1.6),
        dict(kin=[.3,.6,.4],epsin=[2/np.sqrt(5),-1/np.sqrt(5),0.],kout=[-.6,-.2,.5],epsout=[1/np.sqrt(10),-3/np.sqrt(10),0.],omega=2.2)])
    scatter=[]
    for item in directions:
        rich=scattering(full,**item); frozen=scattering(uncoupled,**item)
        rich['uncoupled_photon_elastic']=frozen['elastic']
        rich['elastic_change_fraction']=float(abs(rich['elastic']-frozen['elastic'])/max(abs(frozen['elastic']),1e-14))
        scatter.append(rich)
    g=full['ground']; m=full['model']
    green=charged_G(full,request.get('G_omega',.6))
    green0=charged_G(uncoupled,request.get('G_omega',.6))
    sigma=np.linalg.inv(green0['G'])-np.linalg.inv(green['G'])
    vertex_direction=request.get('vertex_direction',[.2,.4,-.3,.1])
    gamma=static_amputated_vertex(p,request.get('G_omega',.6),vertex_direction)
    receipt=dict(law=LAW,params=p,state='lowest full H in Nup=3,Ndown=2; N=5 sea+one charged electron',
        dimensions=dict(matter=len(m['masks']),photon=m['field_dimension'],coupled=len(g)),
        ground_energy=full['E0'],uncoupled_ground_energy=uncoupled['E0'],gap=full['gap'],
        valence_band_energies=m['band_energies'],
        ground_pair_expectation=float(np.vdot(g,m['pair']@g).real),
        ground_photon_coordinates=[float(np.vdot(g,q@g).real) for q in m['Q']],
        full_feedback=observations,scattering=scatter,
        G_spinup=green,G0_spinup=green0,Sigma_exact_spinup=sigma,
        static_source_vertex=dict(direction=vertex_direction,Gamma=gamma,step=1e-4,
            semantics='-dG_inverse/ds with phi=s*direction; full ground-state relaxation; O(step²)'),
        scope={'spin':'physical CAR spin up/down; spin independent hopping conserves total spin',
        'pairs':'exact finite interband electron-hole pairs relative to filled negative hopping bands; NOT derived relativistic positrons',
        'photon':'two dynamical quantized standing-wave link coordinates; primitive restoring energy, not derived Maxwell law',
        'recoil':'actual coupled final-state energy; anchored finite square lacks exact continuous momentum conservation',
        'truncation':'finite photon basis; full CAR Fock sector, not frozen valence or ground intermediate'})
    return receipt

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--input',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args(); raw=Path(args.input).read_bytes(); request=json.loads(raw)
    result=calculate(request)
    result['binding']=dict(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        input_sha256=hashlib.sha256(raw).hexdigest(),python=os.sys.executable)
    Path(args.output).write_text(json.dumps(result,default=encode,indent=2)+'\n')
    print(json.dumps(dict(law=LAW,output=args.output,ground_pair=result['ground_pair_expectation'],
        max_field_change=max(x['relative_field_change'] for x in result['full_feedback']),
        scattering_changes=[s['elastic_change_fraction'] for s in result['scattering']]),default=encode))
if __name__=='__main__': main()
