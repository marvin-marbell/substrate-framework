"""NEW current/continuity and field-locality audit on full mobile frame carrier."""
import json
from pathlib import Path
import numpy as np
from scipy.linalg import eigh, expm
from rich_frame_gauss_20261014_model import carrier, frame_gauss
from rich_review_wave6_20261014_sea import scalar
ROOT=Path(__file__).resolve().parent

def main():
    r=json.loads((ROOT/'rich_review_wave6_20261014_transport_input.json').read_text())
    p=r['parameters'];local=carrier(p,1,1);internal=local['H'];d=len(internal);q=local['field']['Q']
    frame=frame_gauss();odd=np.flatnonzero(abs(frame['N']-1)<1e-8);neutral=np.flatnonzero(abs(frame['N'])<1e-8)
    inter=sum(a[np.ix_(odd,neutral)]@a[np.ix_(odd,neutral)].conj().T for a in frame['source'])
    coords=list(np.ndindex(2,2,2));index={x:i for i,x in enumerate(coords)}
    h=np.kron(np.eye(8),internal);physical=[np.zeros_like(h) for _ in range(3)];reported=[np.zeros_like(h) for _ in range(3)]
    coordinate=[np.zeros_like(h) for _ in range(3)];bonds=[]
    for x in coords:
        for axis in range(3):
            y=list(x);y[axis]+=1;y=tuple(y)
            if y not in index:continue
            a,b=index[x],index[y];aa=slice(a*d,(a+1)*d);bb=slice(b*d,(b+1)*d)
            phase=expm(1j*r['coupling']*q[axis]);block=-r['hopping']*np.kron(np.kron(np.eye(3),inter),phase)
            h[aa,bb]+=block;h[bb,aa]+=block.conj().T
            # J=-dH/dA for an external uniform link phase exp(i charge A.delta).
            physical[axis][aa,bb]=-1j*r['charge']*block;physical[axis][bb,aa]=(-1j*r['charge']*block).conj().T
            wrong=-1j*r['charge']*r['hopping']*r['coupling']*np.kron(np.kron(np.eye(3),inter),q[axis]@phase)
            reported[axis][aa,bb]+=wrong;reported[axis][bb,aa]+=wrong.conj().T
            coord=-1j*r['hopping']*r['coupling']*np.kron(np.kron(np.eye(3),inter),phase)
            coordinate[axis][aa,bb]+=coord;coordinate[axis][bb,aa]+=coord.conj().T
            bonds.append((a,b,axis))
    # Orientation above hops x+e -> x, so current continuity is J=-charge dX/dt.
    xs=[np.kron(np.diag([x[a] for x in coords]),np.eye(d)) for a in range(3)]
    velocity=[-1j*r['charge']*(h@x-x@h) for x in xs]
    continuity=[np.linalg.norm(a-b) for a,b in zip(physical,velocity)]
    wrong_cont=[np.linalg.norm(a-b) for a,b in zip(reported,velocity)]
    e,v=eigh(h);_,uv=eigh(internal);seed=np.zeros(8*d,complex);seed[:d]=uv[:,0]
    # Unlike author's conditional particle-position pulse this acts on the actual shared field factor.
    axes=np.array(r['pulse_axes']);fieldop=sum(a*b for a,b in zip(axes,q))
    pulse=np.kron(np.eye(8*6),expm(-1j*r['strength']*fieldop))
    driven=pulse@seed
    spin=np.kron(np.eye(8),np.kron(local['S'][2],np.eye(local['nb'])))
    far=np.kron(np.diag([0]*7+[1]),np.eye(d));records=[]
    for t in r['times']:
        evolve=lambda a:v@(np.exp(-1j*e*t)*(v.conj().T@a))
        a,b=evolve(driven),evolve(seed)
        delta=lambda op:float((np.vdot(a,op@a)-np.vdot(b,op@b)).real)
        records.append({'t':t,'far_position_difference':delta(far),'frame_Sz_difference':delta(spin),
                        'true_uniform_link_current_difference':[delta(j) for j in physical],
                        'reported_Qweighted_current_difference':[delta(j) for j in reported]})
    localpulse=np.kron(np.diag([1]+[0]*7),np.kron(np.eye(6),expm(-1j*r['strength']*q[2])))+np.kron(np.diag([0]+[1]*7),np.eye(d))
    site_coherence=np.kron(np.eye(8,k=1)+np.eye(8,k=-1),np.eye(d))
    zero_core=carrier(p,0,0)['H'];one_core=carrier(p,0,1)['H'];two_core=carrier(p,0,2)['H']
    out={'request':r,'dimension':len(h),'true_current_continuity':continuity,'reported_current_continuity_error':wrong_cont,
         'reported_current_differs_from_coordinate_derivative':[np.linalg.norm(a-b) for a,b in zip(reported,coordinate)],
         'source_raw_Gauss':max(np.linalg.norm(g@a@frame['P']) for g in frame['G'] for a in frame['raw_source']),
         'source_intertwiner_error':np.linalg.norm(inter-np.eye(2)),
         'shared_field_pulse_commutator_with_site_coherence':np.linalg.norm(pulse@site_coherence-site_coherence@pulse),
         'author_conditional_particle_pulse_commutator':np.linalg.norm(localpulse@site_coherence-site_coherence@localpulse),
         'records':records,'sector_boundary':{'mobile_event_l':1,'mobile_event_dim':len(h),
         'author_G_representative_l':0,'core_N0_N1_N2_dimensions':[len(zero_core),len(one_core),len(two_core)],
         'meaning':'Author G is a separate immobile l0 core restriction, NOT the192-state l1 mobile propagator or two independent position/CAR charge sector. Same parameter family is weaker than same complete Hamiltonian.'},
         'field_locality':'Only three shared Q factors; no independent field tensor at each of8 sites. Site-conditioned pulse acts jointly on particle position and shared field, not autonomous localized EM pulse. New shared-field pulse is global mode preparation, not a far-field causality proof.'}
    (ROOT/'rich_review_wave6_20261014_transport_result.json').write_text(json.dumps(out,default=scalar,indent=2)+'\n')
    print(json.dumps(out,default=scalar,indent=2))

if __name__=='__main__':main()
