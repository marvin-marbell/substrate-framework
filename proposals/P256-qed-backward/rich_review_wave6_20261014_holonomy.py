"""NEW full-Fock oblique edge-source event with positive-frame C and wrong-current controls."""
import json
from pathlib import Path
import numpy as np
from scipy.linalg import eigh,expm
import rich_frame_holonomy_20261014_model as law
from rich_review_wave6_20261014_sea import scalar
ROOT=Path(__file__).resolve().parent

def main():
    r=json.loads((ROOT/'rich_review_wave6_20261014_holonomy_input.json').read_text());p=r['parameters'];frame=law.raw_frame()
    model=law.two_site(frame,p,r['angle']);h=model['H'];current=model['current'];nf=model['nf'];dim=len(h)
    number=model['N'];nvalues,nvectors=eigh(number);filling=4
    projected=nvectors[:,abs(nvalues-filling)<1e-8];e,v=eigh(projected.conj().T@h@projected);ground=projected@v[:,0]
    ev,uv=eigh(h)
    # Derive canonical full256-Fock C using the actual source basis, not guessed bit order.
    parity=np.diag([(-1)**i.bit_count() for i in range(16)])
    creation=[np.kron(a,np.eye(16)) for a in frame['source']]+[np.kron(parity,a) for a in frame['source']]
    _,u4=law.canonical_c(frame['source']);u=np.kron(np.eye(2),u4)
    targets=[sum(u[a,b]*creation[b].conj().T for b in range(8)) for a in range(8)]
    vacuum=np.eye(256,dtype=complex)[:,0];full=np.eye(256,dtype=complex)[:,-1];cf=np.zeros((256,256),complex)
    for state in range(256):
        original=vacuum.copy();transformed=full.copy()
        for a in reversed(range(8)):
            if state&(1<<a):original=creation[a]@original;transformed=targets[a]@transformed
        cf+=np.outer(transformed,original.conj())
    neutralparity=np.diag([(-1)**sum(n) for n in model['field']['basis']]);c=np.kron(cf,neutralparity)
    leftnumber=np.kron(np.kron(frame['N'],np.eye(16)),np.eye(nf))
    continuity=1j*(h@leftnumber-leftnumber@h)+current
    delta=r['delta'];hp=law.two_site(frame,p,r['angle']+delta)['H'];hm=law.two_site(frame,p,r['angle']-delta)['H']
    bare=law.two_site(frame,{**p,'feedback':0.},r['angle'])['current']
    sourceadd=np.column_stack([a@ground for a in model['source']]);sourcerem=np.column_stack([a.conj().T@ground for a in model['source']])
    z=complex(*r['z']);add=np.linalg.solve((e[0]+z)*np.eye(dim)-h,sourceadd);rem=np.linalg.solve((e[0]-z)*np.eye(dim)-h,sourcerem)
    gf=sourceadd.conj().T@add-(sourcerem.conj().T@rem).T
    s2=sum(s@s for s in model['S']);qop=np.kron(np.eye(256),model['field']['Q'][1]);qnum=np.kron(np.eye(256),model['field']['H']-1.5*np.eye(nf))
    prep=expm(-1j*r['neutral_pulse']*qop)@ground
    driven=expm(-1j*r['current_pulse']*current)@prep
    incorrect=expm(-1j*r['current_pulse']*model['S'][0])@prep
    records=[]
    for t in r['times']:
        evolve=lambda a:uv@(np.exp(-1j*ev*t)*(uv.conj().T@a))
        a,b,w=evolve(driven),evolve(prep),evolve(incorrect)
        deltaobs=lambda x,op:float((np.vdot(x,op@x)-np.vdot(b,op@b)).real)
        records.append({'t':t,'current_difference':deltaobs(a,current),'Nleft_difference':deltaobs(a,leftnumber),
                        'frame_S2_difference':deltaobs(a,s2),'neutral_number_difference':deltaobs(a,qnum),
                        'wrong_S_source_current_difference':deltaobs(w,current),'wrong_S_source_S2_difference':deltaobs(w,s2),
                        'norm':np.vdot(a,a).real})
    zero=law.two_site(frame,{**p,'theta':0.},r['angle'])
    spin2=sum(s@s for s in frame['S']);odd=np.flatnonzero(abs(np.diag(frame['N'])-1)<1e-8)
    out={'request':r,'dimension':dim,'filled_sector_dimension':len(e),'filled_energy':e[0],
         'C_unitarity':np.linalg.norm(c.conj().T@c-np.eye(dim)),'C_same_H':np.linalg.norm(c@h@c.conj().T-h),
         'C_opposite_source_H':np.linalg.norm(c@h@c.conj().T-law.two_site(frame,p,-r['angle'])['H']),
         'C_frame_S_preservation':max(np.linalg.norm(c@s@c.conj().T-s) for s in model['S']),
         'positive_frame_Casimir_minimum':np.min(eigh(spin2,eigvals_only=True))*p['frame_cost'],
         'raw_Gauss':max(np.linalg.norm(g@frame['P']) for g in frame['G']),
         'raw_dressed_source_leakage':max(np.linalg.norm((np.eye(224)-frame['P']@frame['P'].conj().T)@a@frame['P']) for a in frame['raw_source']),
         'current_continuity':np.linalg.norm(continuity),'wrong_omit_feedback_contact_continuity':np.linalg.norm(1j*(h@leftnumber-leftnumber@h)+bare),
         'current_first_derivative':np.linalg.norm((hp-hm)/(2*delta)-current),
         'contact_second_derivative':np.linalg.norm((hp+hm-2*h)/delta**2-model['contact']),
         'full_source_CAR':np.linalg.norm(sourceadd.conj().T@sourceadd+(sourcerem.conj().T@sourcerem).T-np.eye(8)),
         'full_G':gf,'records':records,'angle0_current':np.linalg.norm(zero['current']),
         'angle0_odd_frame_halfspin':np.linalg.norm(spin2[np.ix_(odd,odd)]-.75*np.eye(4)),
         'scope':'Actual full two-site Fock256 x supplied neutral4, interacting positive shared frame Casimir. Edge angle is a genuine source, not COM momentum. Neutral vector is not EM; source/frame/current are not interchangeable.'}
    (ROOT/'rich_review_wave6_20261014_holonomy_result.json').write_text(json.dumps(out,default=scalar,indent=2)+'\n')
    print(json.dumps(out,default=scalar,indent=2))

if __name__=='__main__':main()
