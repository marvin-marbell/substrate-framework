#!/usr/bin/env python3
"""Full finite compact electric/Hubbard graph, exact Gauss sector, no B^2.

This extends the interacting loop mechanism to connected shared-link geometry.
It is NOT the frozen isolated-ring/Gaussian law. Truncation is on chord electric
integers; all physical tree electric fields and CAR matter fluctuations remain.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass, asdict
import hashlib
import itertools
import json
from pathlib import Path
import platform
import sys
import time
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh, splu
from rich_photon_20261014_loop import hop

LAW_ID = 'rich-photon-fullq-connected-gauge-hubbard-v1-20261014'
ROOT = Path(__file__).resolve().parent

@dataclass(frozen=True)
class Parameters:
    geometry: str = 'double_face'
    cutoff: int = 3
    hopping: float = 1.0
    repulsion: float = 4.0
    electric_inertia: float = 1.0
    reference_particles: int = 2
    anchor: int = 0
    background: str = 'integer_ions'


def geometry(name):
    if name == 'double_face':
        positions = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[1,0,1],[0,0,1]], float)
        edges = [(0,1),(1,2),(3,2),(0,3),(1,4),(5,4),(0,5)]
        tree = [0,1,3,4,6]
        faces = [np.array([1,1,-1,-1,0,0,0]), np.array([1,0,0,0,1,-1,-1])]
    elif name == 'cube':
        positions = np.array(list(itertools.product((0,1), repeat=3)), float)
        edges = [(i,j) for i,x in enumerate(positions) for j,y in enumerate(positions)
                 if np.sum(np.abs(y-x)) == 1 and np.sum(y-x) == 1]
        reached, tree = {0}, []
        while len(reached) < len(positions):
            for edge,(a,b) in enumerate(edges):
                if (a in reached) != (b in reached):
                    tree.append(edge); reached.update((a,b)); break
        faces = []
        for i,j in ((0,1),(0,2),(1,2)):
            other = ({0,1,2}-{i,j}).pop()
            for side in (0,1):
                vertices = []
                for x,y in ((0,0),(1,0),(1,1),(0,1)):
                    coordinate = np.zeros(3); coordinate[i]=x;coordinate[j]=y;coordinate[other]=side
                    vertices.append(int(np.flatnonzero(np.all(positions==coordinate,axis=1))[0]))
                cycle = np.zeros(len(edges), int)
                for a,b in zip(vertices,vertices[1:]+vertices[:1]):
                    edge = edges.index((a,b)) if (a,b) in edges else edges.index((b,a))
                    cycle[edge] = 1 if edges[edge] == (a,b) else -1
                faces.append(cycle)
    else:
        raise ValueError(name)
    incidence = np.zeros((len(positions),len(edges)))
    for edge,(a,b) in enumerate(edges):
        incidence[a,edge]=1;incidence[b,edge]=-1
    chords = [edge for edge in range(len(edges)) if edge not in tree]
    return positions, edges, tree, chords, incidence, faces


def fock_basis(vertices, up, down):
    return sorted(sum(1<<(2*i) for i in u)+sum(1<<(2*i+1) for i in d)
                  for u in itertools.combinations(range(vertices),up)
                  for d in itertools.combinations(range(vertices),down))


class Sector:
    def __init__(self, parameters: Parameters, up: int, down: int):
        self.parameters,self.up,self.down=parameters,up,down
        self.positions,self.edges,self.tree,self.chords,self.incidence,self.faces=geometry(parameters.geometry)
        self.vertices=len(self.positions)
        self.fock=fock_basis(self.vertices,up,down)
        self.fock_index={bits:i for i,bits in enumerate(self.fock)}
        self.flux=np.array(list(itertools.product(range(-parameters.cutoff,parameters.cutoff+1),repeat=len(self.chords))),int)
        self.flux_index={tuple(m):i for i,m in enumerate(self.flux)}
        self.nflux=len(self.flux);self.dimension=len(self.fock)*self.nflux
        self.occupations=np.array([[(bits>>(2*i)&1)+(bits>>(2*i+1)&1) for i in range(self.vertices)] for bits in self.fock])
        if parameters.background=='integer_ions':
            background=np.zeros(self.vertices)
            background[parameters.anchor]=1
            second=int(np.flatnonzero(np.all(self.positions==np.array([1,0,0]),axis=1))[0])
            background[second]+=1
        elif parameters.background=='uniform_fractional':
            background=np.full(self.vertices,parameters.reference_particles/self.vertices)
        else:
            raise ValueError(parameters.background)
        background[parameters.anchor]+=up+down-parameters.reference_particles
        self.background=background
        self.rho=self.occupations-background
        # B E+rho=0; the deleted anchor row fixes the redundant Gauss equation.
        rows=[i for i in range(self.vertices) if i!=parameters.anchor]
        tree_inverse=np.linalg.inv(self.incidence[np.ix_(rows,self.tree)])
        tree_base=(-self.rho[:,rows])@tree_inverse.T
        cycle_tree=-self.incidence[np.ix_(rows,self.chords)].T@tree_inverse.T
        electric=np.zeros((len(self.fock),self.nflux,len(self.edges)))
        electric[:,:,self.tree]=tree_base[:,None,:]+(self.flux@cycle_tree)[None,:,:]
        electric[:,:,self.chords]=self.flux[None,:,:]
        self.electric=electric.reshape(self.dimension,len(self.edges))
        self.gauss_error=float(np.max(np.abs(self.electric@self.incidence.T+np.repeat(self.rho,self.nflux,axis=0))))
        self.integer_electric_error=float(np.max(np.abs(self.electric-np.rint(self.electric))))
        self.transfers=[]
        for edge,(a,b) in enumerate(self.edges):
            row,col,data=[],[],[]
            shift=np.zeros(len(self.chords),int)
            if edge in self.chords:shift[self.chords.index(edge)]=1
            mapping=np.array([self.flux_index.get(tuple(m+shift),-1) for m in self.flux])
            valid=np.flatnonzero(mapping>=0)
            for fi,bits in enumerate(self.fock):
                for spin in (0,1):
                    result=hop(bits,2*b+spin,2*a+spin)
                    if result is None:continue
                    newbits,sign=result
                    row.extend((self.fock_index[newbits]*self.nflux+mapping[valid]).tolist())
                    col.extend((fi*self.nflux+valid).tolist())
                    data.extend([-parameters.hopping*sign]*len(valid))
            self.transfers.append(sparse.csr_matrix((data,(row,col)),shape=(self.dimension,self.dimension)))
        double=np.array([sum((bits>>(2*i)&1)*(bits>>(2*i+1)&1) for i in range(self.vertices)) for bits in self.fock])
        self.electric_energy=np.sum(self.electric**2,axis=1)/(2*parameters.electric_inertia)
        self.H=sparse.diags(self.electric_energy+parameters.repulsion*np.repeat(double,self.nflux),format='csr')
        self.H+=sum((t+t.T for t in self.transfers),sparse.csr_matrix(self.H.shape))
        self.currents=[-1j*(t-t.T) for t in self.transfers]
        self.contacts=[-(t+t.T) for t in self.transfers]

    def continuity_error(self):
        errors=[]
        for site in range(self.vertices):
            density=sparse.diags(np.repeat(self.rho[:,site],self.nflux))
            difference=1j*(self.H@density-density@self.H)
            for edge,(a,b) in enumerate(self.edges):
                difference+=self.currents[edge]*((1 if site==a else 0)-(1 if site==b else 0))
            errors.append(float(np.max(np.abs(difference.data))) if difference.nnz else 0.)
        return max(errors)

    def solve(self, count=18):
        values,vectors=eigsh(self.H,k=min(count,self.dimension-2),which='SA',tol=2e-10,
                             v0=np.full(self.dimension,1/np.sqrt(self.dimension)))
        order=np.argsort(values)
        return values[order],vectors[:,order]

    def flux_operator(self, cycle):
        shift=np.asarray(cycle)[self.chords].astype(int)
        mapping=np.array([self.flux_index.get(tuple(m+shift),-1) for m in self.flux])
        valid=np.flatnonzero(mapping>=0)
        rows=np.concatenate([fi*self.nflux+mapping[valid] for fi in range(len(self.fock))])
        cols=np.concatenate([fi*self.nflux+valid for fi in range(len(self.fock))])
        return sparse.csr_matrix((np.ones(len(rows)),(rows,cols)),shape=self.H.shape)

    def boundary_weight(self,state):
        mask=np.any(np.abs(self.flux)==self.parameters.cutoff,axis=1)
        return float(np.sum(np.abs(state.reshape(len(self.fock),self.nflux)[:,mask])**2))

    def reduced_field(self,state):
        wave=state.reshape(len(self.fock),self.nflux)
        probabilities=np.linalg.eigvalsh(wave@wave.conj().T)
        probabilities=probabilities[probabilities>1e-14]
        return {'matter_field_entropy':float(-np.sum(probabilities*np.log(probabilities))),
                'matter_purity':float(np.sum(probabilities**2)),
                'boundary_flux_probability':self.boundary_weight(state)}


def edge_profile(sector,q,polarization):
    q=np.asarray(q);polarization=np.asarray(polarization)
    return np.array([np.dot(sector.positions[b]-sector.positions[a],polarization)*
                     np.cos(np.dot(q,(sector.positions[a]+sector.positions[b])/2))
                     for a,b in sector.edges])


def current(sector,profile):
    return sum((coefficient*j for coefficient,j in zip(profile,sector.currents)),sparse.csr_matrix(sector.H.shape,dtype=complex))


def contact(sector,left,right):
    return sum((a*b*c for a,b,c in zip(left,right,sector.contacts)),sparse.csr_matrix(sector.H.shape))


def retarded_matrix(sector,ground,energy,operators,z):
    """Full finite-H resolvent, not truncated selected-state Lehmann output."""
    source=np.column_stack([o@ground-np.vdot(ground,o@ground)*ground for o in operators])
    h=sector.H-energy*sparse.eye(sector.dimension)
    minus=splu((z*sparse.eye(sector.dimension)-h).tocsc()).solve(source)
    plus=splu((z*sparse.eye(sector.dimension)+h).tocsc()).solve(source)
    # Operators here are Hermitian; reciprocity second term reverses endpoints.
    return source.conj().T@minus-(source.conj().T@plus).T


def path(sector,source,target):
    queue=[source];previous={source:None}
    for vertex in queue:
        if vertex==target:break
        for edge,(a,b) in enumerate(sector.edges):
            if vertex in (a,b):
                neighbor=b if vertex==a else a
                if neighbor not in previous:previous[neighbor]=(vertex,edge,1 if vertex==a else -1);queue.append(neighbor)
    cycle=np.zeros(len(sector.edges),int);vertex=target
    while previous[vertex] is not None:
        parent,edge,sign=previous[vertex];cycle[edge]=sign;vertex=parent
    return cycle


def charged_source(neutral,charged,site,spin,addition=True,dressed=True,string_override=None):
    """Physical CAR source with explicit static countercharge and Wilson string.

    Undressed identity-electric source is actually projected onto the target
    Gauss sector; it vanishes at nonanchor sites rather than acquiring a hidden
    tree string by simply keeping the chord labels unchanged.
    """
    string=(path(neutral,neutral.parameters.anchor,site) if string_override is None else np.asarray(string_override))*(1 if addition else -1)
    row,col,data=[],[],[];max_gauss=0.;outside=0;bare_rejections=0
    for fi,bits in enumerate(neutral.fock):
        orbital=2*site+spin
        occupied=bool(bits&(1<<orbital))
        if occupied==addition:continue
        sign=(-1)**((bits&((1<<orbital)-1)).bit_count())
        newbits=bits^(1<<orbital);newfi=charged.fock_index[newbits]
        for mi,m in enumerate(neutral.flux):
            desired=neutral.electric[fi*neutral.nflux+mi]+(string if dressed else 0)
            newm=tuple(np.rint(desired[neutral.chords]).astype(int))
            target=charged.flux_index.get(newm)
            if target is None:outside+=1;continue
            newindex=newfi*charged.nflux+target
            discrepancy=float(np.max(np.abs(charged.electric[newindex]-desired)))
            if discrepancy>1e-9:
                bare_rejections+=1;max_gauss=max(max_gauss,discrepancy);continue
            row.append(newindex);col.append(fi*neutral.nflux+mi);data.append(sign)
    operator=sparse.csr_matrix((data,(row,col)),shape=(charged.dimension,neutral.dimension))
    return operator,{'required_string_edges':string.tolist(),'applied_string_edges':(string if dressed else np.zeros_like(string)).tolist(),'electric_source_discrepancy':max_gauss,
                     'cutoff_excluded_basis_sources':outside,'Gauss_rejected_basis_sources':bare_rejections,
                     'nonzero_matrix_elements':operator.nnz}


def spin_squared(sector,state):
    target=fock_basis(sector.vertices,sector.up+1,sector.down-1) if sector.down else []
    index={bits:i for i,bits in enumerate(target)}
    raised=np.zeros((len(target),sector.nflux),complex)
    wave=state.reshape(len(sector.fock),sector.nflux)
    for fi,bits in enumerate(sector.fock):
        for site in range(sector.vertices):
            result=hop(bits,2*site,2*site+1)
            if result is not None:
                newbits,sign=result;raised[index[newbits]]+=sign*wave[fi]
    sz=(sector.up-sector.down)/2
    return float(sz*(sz+1)+np.sum(np.abs(raised)**2))


def charged_green(neutral,plus,minus,ground,energy,mu,z):
    """Full spin-up site-matrix causal G with physical strings and both sectors."""
    additions=np.column_stack([charged_source(neutral,plus,site,0,True)[0]@ground for site in range(neutral.vertices)])
    removals=np.column_stack([charged_source(neutral,minus,site,0,False)[0]@ground for site in range(neutral.vertices)])
    particle=splu(((z+energy+mu)*sparse.eye(plus.dimension)-plus.H).tocsc()).solve(additions)
    hole=splu(((z-energy+mu)*sparse.eye(minus.dimension)+minus.H).tocsc()).solve(removals)
    return additions.conj().T@particle+(removals.conj().T@hole).T


def frozen_matter(sector,profiles=(),frequencies=()):
    """Executed no-field-feedback comparator with literal A=0 hopping."""
    n=len(sector.fock);interaction=np.zeros(n);transfers=[]
    for fi,bits in enumerate(sector.fock):
        interaction[fi]=sector.parameters.repulsion*sum((bits>>(2*i)&1)*(bits>>(2*i+1)&1) for i in range(sector.vertices))
    for a,b in sector.edges:
        t=np.zeros((n,n))
        for fi,bits in enumerate(sector.fock):
            for spin in (0,1):
                result=hop(bits,2*b+spin,2*a+spin)
                if result is not None:
                    newbits,sign=result;t[sector.fock_index[newbits],fi]+=-sector.parameters.hopping*sign
        transfers.append(t)
    h=np.diag(interaction)+sum((t+t.T for t in transfers),np.zeros((n,n)))
    values,vectors=np.linalg.eigh(h);ground=vectors[:,0]
    currents=[sum((-1j*d*(t-t.T) for d,t in zip(profile,transfers)),np.zeros((n,n),complex)) for profile in profiles]
    response=[]
    sources=np.column_stack([operator@ground for operator in currents]) if currents else np.zeros((n,0),complex)
    for omega in frequencies:
        z=complex(omega,.05);delta=h-values[0]*np.eye(n)
        chi=sources.conj().T@np.linalg.solve(z*np.eye(n)-delta,sources)-(sources.conj().T@np.linalg.solve(z*np.eye(n)+delta,sources)).T
        response.append(complex_json(chi))
    return values,response


def complex_json(value):
    return {'real':np.asarray(value).real.tolist(),'imag':np.asarray(value).imag.tolist()}


def run(p,name):
    start=time.time();neutral=Sector(p,1,1);values,vectors=neutral.solve();ground=vectors[:,0]
    loops=[neutral.flux_operator(face) for face in neutral.faces]
    # Double-face flux channels are xy and xz, i.e. two transverse directions
    # relative to shared x link. Cube chooses its xy/xz faces at x-compatible axes.
    selected=[0,1] if p.geometry=='double_face' else [0,2]
    fields=[(loops[i]-loops[i].T)/(2j) for i in selected]
    electric_fields=[sparse.diags(neutral.electric@neutral.faces[i]/np.dot(neutral.faces[i],neutral.faces[i])) for i in selected]
    q=np.array([.7,0.,0.]);profiles=[edge_profile(neutral,q,e) for e in ([0,1,0],[0,0,1])]
    currents=[current(neutral,d) for d in profiles]
    density=[sparse.diags(np.repeat(neutral.rho@np.cos(neutral.positions@q),neutral.nflux)),
             sparse.diags(np.repeat(neutral.rho@np.sin(neutral.positions@q),neutral.nflux))]
    operators=fields+electric_fields+currents+density
    spectral=[]
    for operator in operators:
        source=operator@ground-np.vdot(ground,operator@ground)*ground
        amplitudes=vectors[:,1:].conj().T@source
        spectral.append({'exact_source_norm':float(np.vdot(source,source).real),
                         'selected_low_weights':np.abs(amplitudes).tolist(),
                         'selected_low_residues':(np.abs(amplitudes)**2).tolist(),
                         'selected_weight_sum':float(np.sum(np.abs(amplitudes)**2))})
    response=[]
    for omega in (0.,.3,.7,1.2,2.0):
        chi=retarded_matrix(neutral,ground,values[0],operators,complex(omega,.05))
        response.append({'omega':omega,'eta':.05,'chi_full8':complex_json(chi),
                         'D_flux_2x2':complex_json(-chi[:2,:2]),'rho_J_blocks':complex_json(chi[4:,4:])})
    contacts=np.array([[np.vdot(ground,contact(neutral,a,b)@ground) for b in profiles] for a in profiles])
    low_flux=[sum(spectral[i]['selected_low_residues'][j] for i in (0,1)) for j in range(len(values)-1)]
    active=np.flatnonzero(np.asarray(low_flux)>1e-8)
    charged=Sector(p,2,1);cv,cw=charged.solve(count=12)
    removed=Sector(p,0,1);rv,rw=removed.solve(count=12)
    site=4 if p.geometry=='double_face' else neutral.vertices-1
    addition,ad=charged_source(neutral,charged,site,0,True,True)
    bare,bd=charged_source(neutral,charged,site,0,True,False)
    removal,rd=charged_source(neutral,removed,site,0,False,True)
    additions=addition@ground;removals=removal@ground
    aw=np.abs(cw.conj().T@additions)**2;rwgt=np.abs(rw.conj().T@removals)**2
    active_add=np.flatnonzero(aw>1e-8)
    active_remove=np.flatnonzero(rwgt>1e-8)
    mu=float((cv[0]-rv[0])/2)
    chemical_interval=[float(values[0]-rv[0]),float(cv[0]-values[0])]
    alternate=path(neutral,p.anchor,site)-neutral.faces[selected[1]]
    alt_source,alt_data=charged_source(neutral,charged,site,0,True,True,string_override=alternate)
    alt_vector=alt_source@ground
    alt_weights=np.abs(cw.conj().T@alt_vector)**2
    green_grid=[{'omega':omega,'eta':.05,'G_spin_up':complex_json(charged_green(neutral,charged,removed,ground,values[0],mu,complex(omega,.05)))}
                for omega in (-2.,-.7,0.,.7,2.)]
    # Exact real-time full-H evolution by spectral exponential action (not low-state reconstruction).
    from scipy.sparse.linalg import expm_multiply
    kick=fields[0]@ground;norm=np.linalg.norm(kick);kick/=norm
    motion=expm_multiply(-1j*neutral.H,kick,start=0.,stop=8.,num=33,endpoint=True)
    times=np.linspace(0,8,33);energy=[];exchange=[]
    matter_H=neutral.H-sparse.diags(neutral.electric_energy)
    for t,state in zip(times,motion):
        energy.append(float(np.vdot(state,neutral.H@state).real))
        exchange.append({'time':float(t),'electric_energy':float(np.sum(np.abs(state)**2*neutral.electric_energy)),
                         'matter_hopping_energy':float(np.vdot(state,matter_H@state).real),
                         'current_y':float(np.vdot(state,currents[0]@state).real),'current_z':float(np.vdot(state,currents[1]@state).real)})
    control=Sector(Parameters(**{**asdict(p),'hopping':0.0}),1,1)
    tv,tw=control.solve(count=4)
    cosine=[float(np.vdot(ground,(u+u.T)@ground).real/2) for u in loops]
    control_cosine=[float(np.vdot(tw[:,0],(control.flux_operator(face)+control.flux_operator(face).T)@tw[:,0]).real/2) for face in control.faces]
    frozen,frozen_response=frozen_matter(neutral,profiles,(0.,.3,.7,1.2,2.))
    nofeedback_current_change=[]
    for full,frozen_chi in zip(response,frozen_response):
        actual=np.asarray(full['chi_full8']['real'])+1j*np.asarray(full['chi_full8']['imag'])
        comparator=np.asarray(frozen_chi['real'])+1j*np.asarray(frozen_chi['imag'])
        nofeedback_current_change.append(float(np.linalg.norm(actual[4:6,4:6]-comparator)/np.linalg.norm(comparator)))
    receipt={'law':LAW_ID,'parameters':asdict(p),'geometry':{'positions':neutral.positions.tolist(),'edges':neutral.edges,'tree':neutral.tree,'chords':neutral.chords,'faces':[f.tolist() for f in neutral.faces]},
             'dimensions':{'neutral':neutral.dimension,'addition':charged.dimension,'removal':removed.dimension},
             'ground_energy':float(values[0]),'low_gaps':(values[1:]-values[0]).tolist(),
             'field_active_low_gaps':[(float(values[i+1]-values[0]),float(low_flux[i])) for i in active],
             'loop_cosine_expectations':cosine,'field_state':neutral.reduced_field(ground),
             'operator_continuity_max':neutral.continuity_error(),
             'electric_integer_spectrum_error':neutral.integer_electric_error,
             'static_background':neutral.background.tolist(),
             'Gauss_diagonal_max':neutral.gauss_error,'neutral_spin_squared':spin_squared(neutral,ground),
             'charged_ground_spin_squared':spin_squared(charged,cw[:,0]),'charged_ground_Sz':.5,
             'charged_ground_energy':float(cv[0]),'removal_ground_energy':float(rv[0]),
             'charge_pair_half_gap_with_explicit_static_references':float((cv[0]+rv[0]-2*values[0])/2),
             'adjacent_sector_neutral_mu_interval':chemical_interval,'selected_mu':mu,
             'grandcanonical_addition_gap':float(cv[0]-values[0]-mu),
             'grandcanonical_removal_poles':(values[0]-rv-mu).tolist(),
             'G_full_spin_up':green_grid,
             'alternate_string_source':{'source_data':alt_data,'norm':float(np.vdot(alt_vector,alt_vector).real),
                                        'selected_addition_residues':alt_weights.tolist(),
                                        'original_vs_alternate_state_distance':float(np.linalg.norm(additions-alt_vector))},
             'source':{'site':site,'anchor':p.anchor,'addition':ad,'removal':rd,'wrong_bare':bd,
                       'addition_norm':float(np.vdot(additions,additions).real),'bare_norm':float(np.vdot(bare@ground,bare@ground).real),
                       'removal_norm':float(np.vdot(removals,removals).real),
                       'addition_poles':(cv-values[0]).tolist(),'addition_residues':aw.tolist(),
                       'removal_poles':(values[0]-rv).tolist(),'removal_residues':rwgt.tolist(),
                       'addition_first_active':float(cv[active_add[0]]-values[0]) if len(active_add) else None,
                       'removal_first_active':float(values[0]-rv[active_remove[0]]) if len(active_remove) else None},
             'operator_order':['sin_flux_xy','sin_flux_xz','electric_xy','electric_xz','current_y(qx)','current_z(qx)','rho_cos(qx)','rho_sin(qx)'],
             'source_q':q.tolist(),'spectral':spectral,'response':response,'current_contact2x2':complex_json(contacts),
             'motion':{'kind':'normalized sin_flux_xy kick, full-H exponential action','samples':exchange,
                       'energy_max_drift':float(max(energy)-min(energy)),
                       'electric_energy_range':float(max(x['electric_energy'] for x in exchange)-min(x['electric_energy'] for x in exchange))},
             'controls':{'no_hopping_loop_cosines':control_cosine,'no_hopping_ground_energy':float(tv[0]),
                         'frozen_A0_matter_gap':float(frozen[1]-frozen[0]),
                         'frozen_A0_current_response2x2':frozen_response,
                         'full_vs_frozen_current_relative_changes':nofeedback_current_change,
                         'frozen_A0_scope':'Matter-only diagnostic, not a Gauss physical full-H state.',
                         'wrongsource_projects_out':bool(bare.nnz==0)},
             'approximations':['Exact finite Gauss-reduced projected compact Hamiltonian at specified chord cutoff.',
                               'Low eigenstate poles/residues are partial spectra; full8 response uses actual full-H sparse resolvents.',
                               'Open finite graph: q labels actual external current/density profile, NOT a translation-invariant photon momentum or dispersion.',
                               'Two physical loop-flux/current channels on orthogonal faces, NOT established two infinite-volume massless photon poles.',
                               'Odd charge has explicit static opposite charge at anchor; energy is not a free isolated electron mass.',
                               'Microscopic CAR/spin assumed; correlated holes are not relativistic positrons.'],
             'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'frozen_helper_sha256':hashlib.sha256((ROOT/'rich_photon_20261014_loop.py').read_bytes()).hexdigest(),
             'runtime':{'seconds':time.time()-start,'python':sys.version,'platform':platform.platform(),'numpy':np.__version__}}
    (ROOT/(name+'_result.json')).write_text(json.dumps(receipt,indent=2)+'\n')
    np.savez_compressed(ROOT/(name+'_states.npz'),neutral_ground=ground,neutral_low_values=values,
                        charged_low_values=cv,charged_low_states=cw,removal_low_values=rv,removal_low_states=rw,
                        neutral_fock=np.array(neutral.fock),neutral_flux=neutral.flux)
    summary={k:receipt[k] for k in ('law','parameters','dimensions','ground_energy','field_active_low_gaps','loop_cosine_expectations','field_state','Gauss_diagonal_max','neutral_spin_squared','charged_ground_spin_squared','charge_pair_half_gap_with_explicit_static_references','controls')}
    summary['motion']={k:v for k,v in receipt['motion'].items() if k!='samples'}
    summary['source']={k:v for k,v in receipt['source'].items() if k not in ('addition_poles','addition_residues','removal_poles','removal_residues')}
    print(json.dumps(summary,indent=2));return 0

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--geometry',default='double_face');parser.add_argument('--cutoff',type=int,default=3)
    parser.add_argument('--background',default='integer_ions')
    parser.add_argument('--name',default='rich_photon_fullq_20261014_doubleface_L3');args=parser.parse_args()
    raise SystemExit(run(Parameters(geometry=args.geometry,cutoff=args.cutoff,background=args.background),args.name))
