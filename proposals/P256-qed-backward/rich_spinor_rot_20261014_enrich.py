"""NEW material discrimination: larger angular geometry in same quantum field law."""
import json
from pathlib import Path
import numpy as np
from rich_spinor_rot_20261014_model import Parameters,build,solve,current,contact,multiplication
from rich_spinor_rot_20261014_calculate import encode,observe
from rich_spinor_rot_20261014_probe import magnetic_sources
ROOT=Path(__file__).resolve().parent;PREFIX='rich_spinor_rot_20261014_actual3'


def gauge_probe(p,arrays,tag):
    m=build(p)
    for key in ('H','vectors','energies'):m[key]=arrays[tag+'_'+key]
    c=m['charged'];nn=len(m['neutral']['basis']);nf=m['field_dimension']
    h=m['H'];v=m['vectors'];e=m['energies'];u=v[:,:2];q=np.array([0,0,.35]);zero=np.zeros(3)
    rho=p.charge*np.kron(np.kron(np.kron(multiplication(c,np.exp(1j*c['r']@q)),np.eye(2)),np.eye(nn)),np.eye(nf))
    jz=current(m,q,np.array([0,0,1]));ward=h@rho-rho@h-.35*jz
    j=current(m,zero,np.array([1,0,0]));co=contact(m,zero,np.array([1,0,0]),zero,np.array([1,0,0]));w=v[:,2:]
    static=u.conj().T@co@u+2*(u.conj().T@j@w/(e[0]-e[2:]))@(w.conj().T@j@u)
    moment,_=magnetic_sources(m)
    return {'finite_q_Galerkin_Ward_on_ground_norm':float(np.linalg.norm(ward@u)),
            'finite_q_current_on_ground_norm':float(np.linalg.norm(jz@u)),
            'static_uniform_full_receiver_norm':float(np.linalg.norm(static)),
            'actual_physical_magnetic_moment_z':np.real(np.diag(u.conj().T@moment@u))}


def main():
    arrays={};result={'law':'rotational-sphere-neutral-orientor-covariant-cavity-v1','branches':{}}
    for tag,p in [('feedback_L3_cut2',Parameters(charged_lmax=3,coupling=.45,field_cutoff=2)),
                  ('feedback_L4_cut1',Parameters(charged_lmax=4,coupling=.45))]:
        a=observe(p,tag,arrays);a['gauge_cutoff_discriminator']=gauge_probe(p,arrays,tag);result['branches'][tag]=a
    (ROOT/(PREFIX+'_result.json')).write_text(json.dumps(result,default=encode,indent=2))
    np.savez_compressed(ROOT/(PREFIX+'_states.npz'),**arrays)
    print(json.dumps({k:{'dimension':a['dimension'],'parity_gap':a['parity_gap'],
                        'spin_asymmetry':a['full_spin_receiver']['rows'][1]['spin_rate_asymmetry'],
                        'rates':a['full_spin_receiver']['rows'][1]['rates_by_initial_spin'],
                        'gauge_cutoff_discriminator':a['gauge_cutoff_discriminator']} for k,a in result['branches'].items()},default=encode,indent=2))

if __name__=='__main__':main()
