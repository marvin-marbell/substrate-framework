"""New exact static sea discriminator; not a replay or a dynamical photon claim."""
from itertools import product
import json
from pathlib import Path
import resource
import time
import numpy as np
from scipy.linalg import eigh
from rich_mobile_packet_20261015_model import geometry, shell_states

ROOT=Path(__file__).resolve().parent
PREFIX='rich_photon_collective_20261015_'

def run():
    started=time.monotonic()
    c=json.loads((ROOT/'rich_mobile_packet_20261015_actual1_input.json').read_text())
    geo=geometry(c);cliff=geo['cliff'];results=[]
    # Independent exact 20-mode transverse fibres, all ten occupied and ten empty.
    for transverse in (3,5,9):
        rows=[]
        for band in (0,1,2):
            contact=0.;paramagnetic=0.;minimum_gap=100.
            for kx,ky in product(range(transverse),repeat=2):
                h=np.zeros((20,20),complex);first=np.zeros_like(h);second=np.zeros_like(h)
                for z in range(5):
                    here=slice(4*z,4*z+4);there=slice(4*((z+1)%5),4*((z+1)%5)+4)
                    angle=2*np.pi*kx/transverse;ay=2*np.pi*ky/transverse
                    h[here,here]=(c['coupling']*c['phi_center']+c['wilson']*(3-np.cos(angle)-np.cos(ay)))*cliff[0]+c['velocity']*(np.sin(angle)*cliff[1]+np.sin(ay)*cliff[2])
                    hop=-.5j*c['velocity']*cliff[3]-.5*c['wilson']*cliff[0]
                    h[here,there]+=hop;h[there,here]+=hop.conj().T
                    profile=1. if band==0 else np.cos(2*np.pi*band*z/5)
                    first[here,here]=c['charge']*profile*(c['velocity']*np.cos(angle)*cliff[1]+c['wilson']*np.sin(angle)*cliff[0])
                    second[here,here]=c['charge']**2*profile**2*(-c['velocity']*np.sin(angle)*cliff[1]+c['wilson']*np.cos(angle)*cliff[0])
                e,u=eigh(h);j=u.conj().T@first@u;d=u.conj().T@second@u
                contact+=float(np.trace(d[:10,:10]).real)
                paramagnetic+=float(2*np.sum(abs(j[10:,:10])**2/(e[:10][None,:]-e[10:][:,None])))
                minimum_gap=min(minimum_gap,float(e[10]-e[9]))
            volume=transverse**2*5
            rows.append(dict(band=band,contact=contact,paramagnetic=paramagnetic,total=contact+paramagnetic,per_cell=(contact+paramagnetic)/volume,minimum_PH_gap=minimum_gap,meaning='constantA physical torus twist' if band==0 else 'transverse magnetic profile'))
        results.append(dict(transverse=transverse,volume=transverse**2*5,full_virtual_pairs=100*transverse**2,rows=rows))
    h=geo['reference'];e,u=eigh(h);chi=np.repeat(np.cos(2*np.pi*geo['cells'][:,2]/5),4);generator=np.diag(c['charge']*chi)
    first=1j*(generator@h-h@generator);second=-(generator@generator@h-2*generator@h@generator+h@generator@generator)
    j=u.conj().T@first@u;d=u.conj().T@second@u
    contact=float(np.trace(d[:90,:90]).real);para=float(2*np.sum(abs(j[90:,:90])**2/(e[:90][None,:]-e[90:][:,None])))
    bare=[]
    for cut in (2,3,4):
        states=shell_states(cut);lookup={s:i for i,s in enumerate(states)};n=len(states);h=np.zeros((n,n),complex);q0=None
        for axis in range(8):
            a=np.zeros((n,n),complex)
            for k,s in enumerate(states):
                if s[axis]:
                    t=list(s);t[axis]-=1;a[lookup[tuple(t)],k]=np.sqrt(s[axis])
            h+=(np.diag([2*s[axis]+1 for s in states])-a@a-a.conj().T@a.conj().T)/(2*c['shell_width']**2*c['volume'])
            if axis==0:q0=(a+a.conj().T)*c['shell_width']/np.sqrt(2)
        ee,uu=eigh(h);source=q0@uu[:,0];weights=abs(uu.conj().T@source)**2
        bare.append(dict(cut=cut,dimension=n,ground=float(ee[0]),bright=[dict(gap=float(ee[k]-ee[0]),weight=float(w)) for k,w in enumerate(weights) if w>1e-10],source_norm=float(np.vdot(source,source).real)))
    out=dict(kind='NEW_STATIC_ALL_VIRTUAL_SEA_AND_OSCILLATOR_DISCRIMINATOR',static=results,periodic_gradient=dict(contact=contact,paramagnetic=para,total=contact+para,full_virtual_pairs=8100,meaning='exact periodic onsite gauge transformation, not constantA'),bare_cut=bare,decision='Use exact C4-invariant all900 single-pair plus ALL2025 double-pairs in the self-C4 (0,0) transverse fibre, then increase field shell2->3. All other fibres retain every virtual single-pair channel. This is a changed nonlinear variational family, not full multipair closure; no mass/coupling/stiffness retuning.',wall_seconds=time.monotonic()-started,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    (ROOT/(PREFIX+'probe_data.json')).write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out),flush=True)

if __name__=='__main__':run()
