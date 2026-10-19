"""Independent NEW oblique-orbit filled Slater pulse, higher-state and wrong-current controls."""
import json
from pathlib import Path
import numpy as np
from scipy.linalg import eigh
import rich_chiral_sea_20261014_model as law

ROOT = Path(__file__).resolve().parent

def scalar(x):
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, complex): return {'real': x.real, 'imag': x.imag}
    if isinstance(x, np.generic): return scalar(x.item())
    raise TypeError(type(x).__name__)

def solve(model, request, wrong=False):
    n = len(model['H0']); cells = request['cells']; size = cells*n
    h = np.kron(np.eye(cells), model['H0'])
    hp = h.copy()
    currents = [np.zeros_like(h) for _ in range(3)]
    for x in range(cells):
        a = slice(x*n,(x+1)*n); y=(x+1)%cells; b=slice(y*n,(y+1)*n)
        h[a,b] += model['T'][0]; h[b,a] += model['T'][0].conj().T
        hp[a,b] += model['T'][0]; hp[b,a] += model['T'][0].conj().T
        currents[0][a,b] += 1j*model['params'].charge*model['T'][0]
        currents[0][b,a] -= 1j*model['params'].charge*model['T'][0].conj().T
        for i in (1,2):
            t=model['T'][i]; phase=np.exp(1j*request['transverse_k'][i-1])
            bare=phase*t+phase.conjugate()*t.conj().T
            current=1j*model['params'].charge*(phase*t-phase.conjugate()*t.conj().T)
            amplitude=request['pulse'][i-1]*np.cos(2*np.pi*x/cells+request['phase'][i-1])
            peierls=phase*np.exp(-1j*model['params'].charge*amplitude)
            h[a,a]+=bare
            hp[a,a]+=bare-amplitude*model['J'][i] if wrong else peierls*t+peierls.conjugate()*t.conj().T
            currents[i][a,a]+=current
    e,v=eigh(h); ep,vp=eigh(hp); occupied=v[:,e<0]
    source=np.zeros(size,complex)
    # A distinct complex superposition of two actual dressed local sources.
    source[:n]=(model['source'][:,0]+1j*model['source'][:,1])/np.sqrt(2)
    prepared=source-occupied@(occupied.conj().T@source)
    probability=np.vdot(prepared,prepared).real; prepared/=np.sqrt(probability)
    def evolved(values,basis,initial,t):
        coeff=basis.conj().T@initial
        phase=np.exp(-1j*values*t)
        return basis@(phase[:,None]*coeff if coeff.ndim==2 else phase*coeff)
    pulse=evolved(ep,vp,prepared,request['duration'])
    base=evolved(e,v,prepared,request['duration'])
    sea=evolved(ep,vp,occupied,request['duration'])
    empty=v[:,e>0]
    pairnumber=np.linalg.norm(empty.conj().T@sea)**2
    records=[]
    jz=np.kron(np.eye(cells),model['J'][2]); sz=np.kron(np.eye(cells),model['spin'][2])
    for t in request['times']:
        a=evolved(e,v,pulse,t); b=evolved(e,v,base,t)
        delta=lambda op:float((np.vdot(a,op@a)-np.vdot(b,op@b)).real)
        records.append({'t':t,'excess_current_difference':[delta(op) for op in currents],
                        'total_internal_Jz_difference':delta(jz),'frame_Sz_difference':delta(sz),
                        'cell_probability_difference':[(np.vdot(a[x*n:(x+1)*n],a[x*n:(x+1)*n])-np.vdot(b[x*n:(x+1)*n],b[x*n:(x+1)*n])).real for x in range(cells)],
                        'norm':np.vdot(a,a).real})
    return {'dimension':size,'occupied':occupied.shape[1],'preparation_probability':probability,
            'source_gram':np.vdot(source,source).real,'driven_sea_pair_number':pairnumber,
            'excess_occupied_overlap':np.linalg.norm(sea.conj().T@pulse),
            'driven_sea_orthonormality':np.linalg.norm(sea.conj().T@sea-np.eye(sea.shape[1])),
            'records':records,'actual_J_not_current_norm':[np.linalg.norm(currents[i]-np.kron(np.eye(cells),model['J'][i])) for i in range(3)]}

def main():
    request=json.loads((ROOT/'rich_review_wave6_20261014_sea_input.json').read_text())
    model=law.build(law.Parameters(**request['parameters']))
    e,v=eigh(model['H0']); mid=len(e)//2
    ids=np.arange(mid-2,mid+2); low=v[:,ids]
    velocities=[law.vertex(model,np.zeros(3),np.zeros(3),axis)/model['params'].charge for axis in np.eye(3)]
    alpha=[low.conj().T@a@low for a in velocities]
    c2=sum(np.trace(a@a).real/4 for a in alpha)/3
    clifford=max(np.linalg.norm(a@b+b@a-(2*c2*np.eye(4) if i==j else 0)) for i,a in enumerate(alpha) for j,b in enumerate(alpha))
    state=v[:,mid+1]
    terms=[]
    for vel in velocities:
        amplitudes=v.conj().T@vel@state
        mask=abs(e-e[mid+1])>1e-8
        contribution=np.zeros(len(e));contribution[mask]=2*abs(amplitudes[mask])**2/(e[mid+1]-e[mask])
        terms.append({'full':float(sum(contribution)),'low4_only':float(sum(contribution[ids])),
                      'discarded_source_norm':float(np.linalg.norm((np.eye(len(e))-low@low.conj().T)@vel@state))})
    q=np.array(request['probe_q']); k=np.array(request['probe_k']); h=law.hamiltonian(model,k)
    correct=sum(q[i]*law.vertex(model,k+q,k,np.eye(3)[i]) for i in range(3))
    wrong=sum(q[i]*model['J'][i] for i in range(3))
    rhs=model['params'].charge*(law.hamiltonian(model,k+q)-h)
    cn=model['C']; kin=model['conventional_kinetic']
    out={'law':model['law'],'request':request,'rest_low_interband_norms':[np.linalg.norm(v[:,mid:mid+2].conj().T@a@v[:,mid-2:mid]) for a in velocities],
         'low4_clifford':{'speed_squared':c2,'residual':clifford,'scope':'Projected rest module only, not full geometry Clifford or universal velocity.'},
         'curvature':terms,'Ward_correct':np.linalg.norm(correct-rhs),'Ward_wrong_Noether_as_current':np.linalg.norm(wrong-rhs),
         'generic_k_internal_J_commutator':[np.linalg.norm(h@j-j@h) for j in model['J']],
         'C_full':np.linalg.norm(cn@law.hamiltonian(model,-k).conjugate()@cn.conj().T+h),
         'C_conventional_positive_kinetic':np.linalg.norm(cn@(law.hamiltonian(model,-k)+kin).conjugate()@cn.conj().T+h+kin),
         'actual':solve(model,request),'wrong_Noether_source':solve(model,request,True)}
    off={**request,'pulse':[0.,0.]};out['source_off']=solve(model,off)
    (ROOT/'rich_review_wave6_20261014_sea_result.json').write_text(json.dumps(out,default=scalar,indent=2)+'\n')
    print(json.dumps(out,default=scalar,indent=2))

if __name__=='__main__':main()
