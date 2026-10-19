"""NEW independent charged fullq join: physical sum vs unsplit full-H consumer."""
import os
for n in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[n]='1'
import json
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
import rich_photon_fullq_20261014_model as law
import rich_charged_gauss_20261014_fullq as join

def main():
    p=law.Parameters();s=law.Sector(p,2,1);ev,v=join.orthogonal_selected(s,6);g=v[:,0];e=ev[0]
    qin=np.array([.39,-.57,.81]);qout=np.array([-.63,.44,.28]);w=2.91;eta=.067
    result=join.interaction(s,g,e,ev,v,qin,qout,w,eta,1)
    split=result['physical_elastic'];split=np.array(split['real'])+1j*np.array(split['imag'])
    a,b=join.profiles(s,qout),join.profiles(s,qin)
    ji=[law.current(s,b[:,i]) for i in range(2)];jo=[law.current(s,a[:,i]).conj().T for i in range(2)]
    ident=sparse.eye(s.dimension)
    first=splu(((e+w+1j*eta)*ident-s.H).tocsc()).solve(np.column_stack([j@g for j in ji]))
    second=splu(((e-w+1j*eta)*ident-s.H).tocsc()).solve(np.column_stack([j@g for j in jo]))
    total=np.zeros((2,2),complex)
    for i in range(2):
        for j in range(2):total[i,j]=np.vdot(g,law.contact(s,a[:,i].conj(),b[:,j])@g)+np.vdot(g,jo[i]@first[:,j])+np.vdot(g,ji[j]@second[:,i])
    currents,contacts=join.spin_links(s)
    sumerrors=[np.linalg.norm((currents[k][0]+currents[k][1]-s.currents[k]).data) for k in range(len(s.edges))]
    plus=law.Sector(p,3,1);minus=law.Sector(p,1,1)
    aa=np.column_stack([law.charged_source(s,plus,i,0)[0]@g for i in range(6)])
    rr=np.column_stack([law.charged_source(s,minus,i,0,False)[0]@g for i in range(6)])
    equal=aa.conj().T@aa+(rr.conj().T@rr).T
    bare=law.charged_source(s,plus,4,0,dressed=False)[0]@g
    proper=law.charged_source(s,plus,4,0)[0]@g
    flux=[(s.flux_operator(f)-s.flux_operator(f).T)/(2j) for f in s.faces]
    ds=law.retarded_matrix(s,g,e,flux,.81+.067j)
    neutral=law.Sector(p,1,1);ne,nv=neutral.solve(2);ng=nv[:,0]
    nd=law.retarded_matrix(neutral,ng,ne[0],[(neutral.flux_operator(f)-neutral.flux_operator(f).T)/(2j) for f in neutral.faces],.81+.067j)
    out=dict(ground_energy=float(e),spin_squared=law.spin_squared(s,g),gauss=s.gauss_error,integer_electric=s.integer_electric_error,spin_split_sum_errors=sumerrors,split_vs_unsplit_physical_tensor_error=float(np.linalg.norm(split-total)),charged_source_equal_time_error=float(np.linalg.norm(equal-np.eye(6))),proper_source_norm=float(np.linalg.norm(proper)**2),bare_source_norm=float(np.linalg.norm(bare)**2),charged_vs_neutral_flux_response_distance=float(np.linalg.norm(ds-nd)),ground_only_distance=result['ground_only_difference_norm'],inelastic=result['inelastic'],scope='Same charged integer-ion H/G/D/J/C and receiver. Neutral-state D comparator is a WRONG reference, not consumed by charged tensor. G only spin-up channel; no global chargegap, propagating photon or crosssection claim.')
    Path(__file__).with_name('rich_review_wave2_20261014_join_result.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':main()
