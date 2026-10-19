"""NEW angular-cutoff/source discriminator; not project validation."""
import json
from pathlib import Path
import numpy as np
from scipy.linalg import eigvalsh
from rich_spinor_rot_20261014_model import Parameters,build,solve,current,contact,multiplication
from rich_spinor_rot_20261014_calculate import encode
ROOT=Path(__file__).resolve().parent;PREFIX='rich_spinor_rot_20261014_actual2'


def magnetic_sources(m):
    p=m['params'];c=m['charged'];r=c['r'];nn=len(m['neutral']['basis']);nf=m['field_dimension']
    avec=np.cross(np.array([0,0,1]),r)/2;f=np.cross(r,avec)
    one=np.zeros_like(c['L2'],dtype=complex)
    for i in range(3):
        a=multiplication(c,f[:,i]);l=c['L'][i]
        one+=p.charge*p.inverse_inertia*(l@a+a@l)/2
    lift=lambda a:np.kron(np.kron(a,np.eye(2)),np.eye(nn))
    moment=np.kron(lift(one),np.eye(nf))
    second=np.kron(lift(p.charge**2*p.inverse_inertia*multiplication(c,np.sum(f*f,axis=1))),np.eye(nf))
    if m['field'] is not None:
        for i in range(3):
            ci=p.charge**2*p.inverse_inertia*multiplication(c,np.sum(f*np.cross(r,np.eye(3)[i]),axis=1))
            moment-=p.coupling*np.kron(lift(ci),m['field']['Q'][i])
    return moment,second


def main():
    result={'law':'rotational-sphere-neutral-orientor-covariant-cavity-v1','branches':{}};arrays={}
    for tag,p in [('L2',Parameters()),('L3',Parameters(charged_lmax=3)),('L4',Parameters(charged_lmax=4)),
                  ('field_cut1',Parameters(coupling=.45)),('field_cut2',Parameters(coupling=.45,field_cutoff=2))]:
        m=solve(build(p));c=m['charged'];v=m['vectors'];h=m['H'];e=m['energies'];u=v[:,:2]
        nn=len(m['neutral']['basis']);nf=m['field_dimension'];zero=np.zeros(3);q=np.array([0,0,.35])
        charge_density=p.charge*np.kron(np.kron(np.kron(multiplication(c,np.exp(1j*c['r']@q)),np.eye(2)),np.eye(nn)),np.eye(nf))
        ward=h@charge_density-charge_density@h-.35*current(m,q,np.array([0,0,1]))
        data={'parameters':m['parameters'],'dimension':len(h),'finite_q_Galerkin_Ward_full_norm':float(np.linalg.norm(ward)),
              'finite_q_Galerkin_Ward_on_ground_norm':float(np.linalg.norm(ward@u)),
              'finite_q_current_on_ground_norm':float(np.linalg.norm(current(m,q,np.array([0,0,1]))@u)),
              'warning':'Before-Galerkin local continuum coupling is covariant. Finite angular/field cutoff is NOT an exactly gauge-closed or Gauss-constrained model; discarded multiplication sectors generate literal Ward residual.'}
        # Static uniform pure-gauge current/contact receiver from the actual FULL spectrum.
        j=current(m,zero,np.array([1,0,0]));co=contact(m,zero,np.array([1,0,0]),zero,np.array([1,0,0]));w=v[:,2:]
        static=u.conj().T@co@u+2*(u.conj().T@j@w/(e[0]-e[2:]))@(w.conj().T@j@u)
        data['static_uniform_full_current_contact_ground_norm']=float(np.linalg.norm(static))
        moment,second=magnetic_sources(m);projected=u.conj().T@moment@u
        data['actual_physical_magnetic_moment_z']=np.real(np.diag(projected))
        b=1e-4
        minus=eigvalsh(h+b*moment+b*b*second/2,subset_by_index=[0,1])
        plus=eigvalsh(h-b*moment+b*b*second/2,subset_by_index=[0,1])
        data['actual_B_plus_minus_doublet_energies']={'B':b,'plus':plus,'minus':minus,'splitting_plus':float(plus[1]-plus[0])}
        arrays[tag+'_ward_operator']=ward;arrays[tag+'_magnetic_moment']=moment;arrays[tag+'_B_contact']=second
        arrays[tag+'_static_current_contact']=static
        result['branches'][tag]=data
    (ROOT/(PREFIX+'_result.json')).write_text(json.dumps(result,default=encode,indent=2))
    np.savez_compressed(ROOT/(PREFIX+'_states.npz'),**arrays)
    print(json.dumps({k:{a:v[a] for a in ('dimension','finite_q_Galerkin_Ward_on_ground_norm','static_uniform_full_current_contact_ground_norm','actual_physical_magnetic_moment_z')} for k,v in result['branches'].items()},default=encode,indent=2))

if __name__=='__main__':main()
