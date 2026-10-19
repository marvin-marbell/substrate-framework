"""NEW full64 supplier/field-frame H, spatial source Ward and oblique quantum receiver."""
import json
from pathlib import Path
import numpy as np
from scipy import sparse as sp
from scipy.linalg import eigh
from scipy.sparse.linalg import expm_multiply
from rich_frame_photon_20261014_model import Parameters, Joined, spectrum, full_resolvent
from rich_review_wave6_20261014_sea import scalar
ROOT=Path(__file__).resolve().parent

def events(model,source,initial,r,wrong=False):
    pulse=np.exp(1j*sum(a*x for a,x in zip(r['field_pulse'],model.coordinates)))
    field=np.repeat(pulse,model.internal)*initial
    insertion=sum(a*j for a,j in zip(r['current_direction'],model.J))
    if wrong:insertion=sum(a*s for a,s in zip(r['wrong_spin_direction'],model.S))
    driven=expm_multiply(-1j*r['current_strength']*insertion,field)
    seeds=np.column_stack([initial,field,driven])
    n=np.array(r['spin_direction']);n/=np.linalg.norm(n)
    spin=sum(a*s for a,s in zip(n,model.S));up=model.I/2+spin
    mask=np.repeat(model.coordinates[0]+r['field_detector_mix']*model.coordinates[1]>=0,model.internal)
    records=[]
    for t in r['times']:
        states=expm_multiply(-1j*t*model.H,seeds,traceA=-1j*t*model.H.diagonal().sum()) if t else seeds
        values=[]
        for state in states.T:
            a=up@state;b=state-a
            values.append({'norm':np.vdot(state,state).real,'spin':[np.vdot(state,s@state).real for s in model.S],
                           'current':[np.vdot(state,j@state).real for j in model.J],
                           'inclusive_PVM':[np.vdot(x[m],x[m]).real for x in (a,b) for m in (mask,~mask)],
                           'source_return':np.linalg.norm(source.conj().T@state)**2,'boundary':model.boundary_probability(state),
                           'energy':np.vdot(state,model.H@state).real})
        records.append({'t':t,'states':values})
    changes=[np.linalg.norm(np.array(row['states'][1]['spin'])-row['states'][0]['spin']) for row in records]
    insertionchanges=[np.linalg.norm(np.array(row['states'][2]['spin'])-row['states'][1]['spin']) for row in records]
    return {'state_order':['own_baseline','oblique_field_unitary','same_current_unitary_after_field'],
            'maximum_field_spin_response':max(changes),'maximum_inserted_current_spin_response':max(insertionchanges),'records':records}

def main():
    r=json.loads((ROOT/'rich_review_wave6_20261014_photon_input.json').read_text());p=Parameters(**r['parameters'])
    neutral=Joined(p,0);odd=Joined(p,1);pair=Joined(p,2)
    e0,v0,quality0=spectrum(neutral.H,4);e1,v1,quality1=spectrum(odd.H,4)
    source=np.column_stack([neutral.source_to(odd,a)@v0[:,0] for a in range(2)])
    sn=sum(a*s for a,s in zip(r['spin_direction'],odd.S));_,u=eigh(source.conj().T@(sn@source));prepared=source@u[:,-1];prepared/=np.linalg.norm(prepared)
    # Select actual ground doublet by physical oblique spin, never source index.
    groundspace=v1[:,abs(e1-e1[0])<1e-8];_,u=eigh(groundspace.conj().T@(sn@groundspace));ground=groundspace@u[:,-1]
    q=np.array(r['q']);rho=odd.density_fourier(q);cart=odd.current_fourier(q);qcurrent=sum(a*j for a,j in zip(q,cart))
    continuity=odd.H@rho-rho@odd.H+qcurrent
    zi=e0[0]+complex(*r['z']);zo=zi+complex(*r['frequency_transfer'])
    right,rr=full_resolvent(odd.H,zi,source);left,rl=full_resolvent(odd.H,zo.conjugate(),source)
    rhs=-(zo-zi)*(left.conj().T@(rho@right))+source.conj().T@(rho@right)-left.conj().T@(rho@source)
    spatialthree=[left.conj().T@(j@right) for j in cart]
    correct=sum(a*j for a,j in zip(q,spatialthree))
    wrong=left.conj().T@(sum(a*s for a,s in zip(q,odd.S))@right)
    fieldcols=np.column_stack([a@ground-np.vdot(ground,a@ground)*ground for a in odd.A])
    electriccols=np.column_stack([a@ground/np.sqrt(odd.mass) for a in odd.P])
    currentcols=np.column_stack([j@ground for j in odd.J])
    z=complex(*r['z']);plus,rp=full_resolvent(odd.H,e1[0]+z,fieldcols);minus,rm=full_resolvent(odd.H,e1[0]-z,fieldcols)
    dfull=fieldcols.conj().T@plus+(fieldcols.conj().T@minus).T
    delta=r['delta'];direction=np.array(r['current_direction']);hp=odd.shifted(tuple(delta*direction));hm=odd.shifted(tuple(-delta*direction))
    jmode=sum(a*j for a,j in zip(direction,odd.J));kmode=sum(direction[i]*direction[j]*odd.K[i][j] for i in range(2) for j in range(2))
    add=np.column_stack([odd.source_to(pair,i)@ground for i in range(2)]);remove=np.column_stack([neutral.source_to(odd,i).getH()@ground for i in range(2)])
    neutraloff=Joined(p,0,induction=False);metricoff=Joined(p,1,metric=False)
    gauss=1j*sum(q[a]*(-1j*q[a]*rho/(p.epsilon*np.dot(q,q))) for a in range(3))-rho/p.epsilon
    out={'request':r,'dimensions':[neutral.size,odd.size,pair.size],'ground_quality':[quality0,quality1],
         'source_gram':np.linalg.norm(source.conj().T@source-np.eye(2)),
         'complete_addition_removal_CAR':np.linalg.norm(add.conj().T@add+(remove.conj().T@remove).T-np.eye(2)),
         'source_G':source.conj().T@right,'source_threepoint_Cartesian':spatialthree,
         'threepoint_spatial_Ward':np.linalg.norm(correct-rhs),'threepoint_wrong_S_current_Ward':np.linalg.norm(wrong-rhs),
         'resolvent_residuals':rr+rl+rp+rm,'continuity_matrix':sp.linalg.norm(continuity),'Gauss_matrix':sp.linalg.norm(gauss),
         'total_charge_N0_N1_N2':[np.unique(m.charge_data['charges'].sum(axis=1)) for m in (neutral,odd,pair)],
         'longitudinal_core_supplier_cross_range':np.ptp(odd.HL_values-neutral.HL_values),
         'field_coordinate_gram_eigenvalues':np.linalg.eigvalsh(fieldcols.conj().T@fieldcols),
         'electric_gram_eigenvalues':np.linalg.eigvalsh(electriccols.conj().T@electriccols),
         'current_gram_eigenvalues':np.linalg.eigvalsh(currentcols.conj().T@currentcols),'D_retarded':dfull,
         'Peierls_current_derivative_on_ground':np.linalg.norm(((hp-hm)/(2*delta)+jmode)@ground),
         'Peierls_contact_derivative_on_ground':np.linalg.norm(((hp+hm-2*odd.H)/delta**2-kmode)@ground),
         'neutral_induction_off_J_norm':[sp.linalg.norm(j) for j in neutraloff.J],
         'neutral_induction_off_body_norm':sp.linalg.norm(neutraloff.Hbody),
         'actual':events(odd,source,prepared,r),'metric_off':events(metricoff,source,prepared,r),
         'wrong_spin_as_current':events(odd,source,prepared,r,True)}
    (ROOT/'rich_review_wave6_20261014_photon_result.json').write_text(json.dumps(out,default=scalar,indent=2)+'\n')
    print(json.dumps(out,default=scalar,indent=2))

if __name__=='__main__':main()
