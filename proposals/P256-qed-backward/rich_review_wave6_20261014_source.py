"""NEW full192-source threepoint, parity/Clifford and exact Peierls derivative consumer."""
import json
from pathlib import Path
import numpy as np
from scipy.linalg import eigh
import rich_chiral_sea_20261014_model as law
from rich_review_wave6_20261014_sea import scalar
ROOT=Path(__file__).resolve().parent

def main():
    request=json.loads((ROOT/'rich_review_wave6_20261014_source_input.json').read_text())
    model=law.build(law.Parameters(**request['parameters']))
    k=np.array(request['k']);q=np.array(request['q']);z=complex(*request['z']);eps=np.eye(3)
    hi=law.hamiltonian(model,k);ho=law.hamiltonian(model,k+q);n=len(hi);identity=np.eye(n)
    ri=np.linalg.solve(z*identity-hi,identity);ro=np.linalg.solve(z*identity-ho,identity)
    source=model['source'];gamma=[law.vertex(model,k+q,k,a) for a in eps]
    three=[source.conj().T@ro@a@ri@source for a in gamma]
    residual=sum(q[i]*three[i] for i in range(3))-model['params'].charge*source.conj().T@(ro-ri)@source
    frame=model['frame'];p=frame['P']
    raw={'gauss_source':max(np.linalg.norm(g@a@p) for g in frame['G'] for a in frame['raw_source']),
         'bare_source_physical_norm':max(np.linalg.norm(p.conj().T@a@p) for a in frame['bare']),
         'left_casimir_on_odd':np.linalg.norm(sum(a@a for a in frame['S'])[np.ix_(np.flatnonzero(abs(frame['N']-1)<1e-8),np.flatnonzero(abs(frame['N']-1)<1e-8))]-.75*np.eye(2))}
    e,v=eigh(model['H0']);mid=n//2;low=v[:,mid-2:mid+2]
    lowhi=low.conj().T@hi@low;lowho=low.conj().T@ho@low
    lowri=np.linalg.solve(z*np.eye(4)-lowhi,np.eye(4));lowro=np.linalg.solve(z*np.eye(4)-lowho,np.eye(4));lowsource=low.conj().T@source
    lowthree=[lowsource.conj().T@lowro@(low.conj().T@a@low)@lowri@lowsource for a in gamma]
    discarded=np.eye(n)-low@low.conj().T
    delta=request['delta'];direction=np.array(request['direction'])
    def peierls(amplitude):
        h=model['H0'].copy()
        for d,t in zip(model['directions'],model['T']):
            phase=np.exp(1j*np.dot(k-model['params'].charge*amplitude*direction,d))
            h+=phase*t+phase.conjugate()*t.conj().T
        return h
    hplus=peierls(delta);hminus=peierls(-delta)
    vertex=law.vertex(model,k,k,direction)
    contact=law.contact(model,k,k,np.zeros(3),direction,np.zeros(3),direction)
    cp=model['C'];parity=model['parity']
    out={'law':model['law'],'request':request,'raw_frame':raw,'source_Gram':np.linalg.norm(source.conj().T@source-np.eye(source.shape[1])),
         'threepoint_Ward':np.linalg.norm(residual),'threepoint_norms':[np.linalg.norm(a) for a in three],
         'low4_G_error':np.linalg.norm(source.conj().T@ri@source-lowsource.conj().T@lowri@lowsource),
         'low4_threepoint_errors':[np.linalg.norm(a-b) for a,b in zip(three,lowthree)],
         'discarded_source_Gram_norm':np.linalg.norm(source.conj().T@discarded@source),
         'Peierls_first_derivative_error':np.linalg.norm((hplus-hminus)/(2*delta)+vertex),
         'Peierls_second_derivative_error':np.linalg.norm((hplus+hminus-2*hi)/delta**2-contact),
         'C_opposite_parity':np.linalg.norm(cp@parity.conjugate()@cp.conj().T+parity),
         'rest_low_parities':np.diag(low.conj().T@parity@low).real,
         'C_square_plus_one':np.linalg.norm(cp@cp.conjugate()-identity),
         'C_frame_spin_not_totalJ':[np.linalg.norm(cp@a.conjugate()@cp.conj().T+a) for a in model['spin']],
         'C_total_internal_J':[np.linalg.norm(cp@a.conjugate()@cp.conj().T+a) for a in model['J']]}
    (ROOT/'rich_review_wave6_20261014_source_result.json').write_text(json.dumps(out,default=scalar,indent=2)+'\n')
    print(json.dumps(out,default=scalar,indent=2))

if __name__=='__main__':main()
