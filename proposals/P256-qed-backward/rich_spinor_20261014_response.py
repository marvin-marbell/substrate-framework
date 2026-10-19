"""NEW physical discriminator: honest pole regulator, full virtual sectors, magnetic/current transfer."""
import json
from pathlib import Path
import numpy as np
from rich_spinor_20261014_model import Parameters,build_sector,solve,current,contact,scattering
from rich_spinor_20261014_calculate import encode,multiplets
ROOT=Path(__file__).resolve().parent
PREFIX='rich_spinor_20261014_actual2'


def main():
    result={'law':'octahedron-neutral-orientor-spin-transfer-v1','experiments':{}}; arrays={}
    for tag,params in [('dynamic',Parameters()),('no_spin_transfer',Parameters(rotor_spin=0)),('no_orientor_motion',Parameters(orientation=0))]:
        s=solve(build_sector(params,1)); e=s['energies']; v=s['vectors']; gs=multiplets(s)
        even=next(g for g in gs if np.mean(g['parity'])>.9)['indexes']
        odd=next(g for g in gs if np.mean(g['parity'])<-.9)['indexes']; low=even+odd
        u=v[:,even]; p=v[:,low]; qin=np.array([0,0,.35]); rows=[]
        for angle in [np.pi/3,np.pi/2,2*np.pi/3]:
            qout=.35*np.array([np.sin(angle),0,np.cos(angle)])
            epsouts=[np.array([0,1,0]),np.array([np.cos(angle),0,-np.sin(angle)])]
            for omega in [.15,.35,.6]:
                for eta in [0.025,1e-6]:
                    for helicity in [-1,1]:
                        epsin=np.array([1,helicity*1j,0])/np.sqrt(2)
                        mats=np.array([[[scattering(s,a,b,qin,epsin,qout,ep,omega,eta=eta) for a in even] for b in even] for ep in epsouts])
                        rates=np.sum(abs(mats)**2,axis=(0,1))
                        rows.append({'angle':angle,'omega':omega,'eta':eta,'helicity':helicity,'rates':rates,
                                     'asymmetry':float((rates[1]-rates[0])/sum(rates)),
                                     'spin_flip_rate':float(np.sum(abs(mats[:,0,1])**2)+np.sum(abs(mats[:,1,0])**2))})
        xyz=np.array([current(s,np.zeros(3),np.eye(3)[i]) for i in range(3)])
        static=np.zeros((3,3,2,2),complex)
        excited=[a for a in range(len(e)) if a not in even]
        w=v[:,excited]
        for i in range(3):
            for j in range(3):
                c=contact(s,np.zeros(3),np.eye(3)[i],np.zeros(3),np.eye(3)[j])
                static[i,j]=u.conj().T@c@u
                static[i,j]+=(u.conj().T@xyz[i]@w/(e[even[0]]-e[excited]))@(w.conj().T@xyz[j]@u)
                static[i,j]+=(u.conj().T@xyz[j]@w/(e[even[0]]-e[excited]))@(w.conj().T@xyz[i]@u)
        time_rows=[]
        for t in [0,.5,1,2,4]:
            evolution=(v*np.exp(-1j*e*t))@v.conj().T
            jxt=evolution.conj().T@xyz[0]@evolution
            response=1j*u.conj().T@(jxt@xyz[1]-xyz[1]@jxt)@u
            time_rows.append({'time':t,'spin_resolved_cross_current_retarded_commutator':response})
        pp=np.kron(np.kron(s['geometry']['Pp'],np.eye(2)),np.eye(4))
        ps=np.kron(np.kron(s['geometry']['Ps'],np.eye(2)),np.eye(4))
        pd=np.kron(np.kron(s['geometry']['Pd'],np.eye(2)),np.eye(4))
        weights={'s':np.real(np.diag(p.conj().T@ps@p)), 'p':np.real(np.diag(p.conj().T@pp@p)), 'd':np.real(np.diag(p.conj().T@pd@p))}
        spin=np.array([p.conj().T@a@p for a in s['spin']]); rotation=np.array([p.conj().T@a@p for a in s['angular']])
        low_rotation_residual=np.linalg.norm(rotation[0]@rotation[1]-rotation[1]@rotation[0]-1j*rotation[2])
        result['experiments'][tag]={'parameters':s['parameters'],'frequency_angle_regulator_helicity_scan':rows,
                                   'selected_indexes':low,'electronic_orbital_weights':weights,'primitive_spin_projected':spin,
                                   'total_rotation_projected':rotation,'projected_rotation_commutator_residual':float(low_rotation_residual),
                                   'static_full_current_contact_pure_gauge_norm':float(np.linalg.norm(static)),
                                   'actual_full_H_current_time_response':time_rows}
        arrays[tag+'_H']=s['H']; arrays[tag+'_V']=v; arrays[tag+'_E']=e; arrays[tag+'_static_current_contact']=static
    (ROOT/(PREFIX+'_result.json')).write_text(json.dumps(result,default=encode,indent=2))
    np.savez_compressed(ROOT/(PREFIX+'_states.npz'),**arrays)
    print(json.dumps({k:{'static_gauge_norm':a['static_full_current_contact_pure_gauge_norm'],
                        'rotation_residual':a['projected_rotation_commutator_residual'],
                        'eta1e6_asymmetry_90deg_omega035_hplus':next(r['asymmetry'] for r in a['frequency_angle_regulator_helicity_scan'] if r['angle']==np.pi/2 and r['omega']==.35 and r['eta']==1e-6 and r['helicity']==1)} for k,a in result['experiments'].items()},indent=2))

if __name__=='__main__':main()
