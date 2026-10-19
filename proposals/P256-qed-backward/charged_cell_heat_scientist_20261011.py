"""New source-weighted native charged heat consumer, global matrix calculus."""
from __future__ import annotations
import argparse,hashlib,json,time
from fractions import Fraction as F
from pathlib import Path
from typing import cast
import numpy as np
import sympy as sy
import harmonic_car_bridge_scientist_20261011 as native
import charged_stieltjes_scientist_20261011 as st
import charged_output_lower_certificate_20261011 as iv
import charged_cell_heat_helper_20261011 as heat
ROOT=Path(__file__).resolve().parent
PREFIX='charged_cell_heat_'
INPUTS=[*st.INPUTS,'charged_stieltjes_scientist_20261011.py','charged_kernel_review_delivery_20261011.json','three_cell_density_formation_typed3_20261011_interface.json','flux_laplace_time_request_20261011.json']

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,value): p.write_text(json.dumps(value,indent=2)+'\n')

def gram(role,n,X,N,epoch):
    paths=[]
    for k,E in enumerate(native.extraction(n)):
        y=E@X if role==0 else (E@X.T).T
        path=ROOT/f'{PREFIX}{epoch}_scratch_{role}_{k}.npy'
        np.save(path,y.reshape(-1)); paths.append(path)
    views=[np.load(p,mmap_mode='r') for p in paths]
    K=np.zeros((5,5)); blocks=0
    for start in range(0,views[0].size,32768):
        block=np.stack([v[start:start+32768] for v in views])
        K+=block@block.T; blocks+=1
    assert blocks<=300 and np.isfinite(K).all()
    del views
    for p in paths: p.unlink()
    pad=[[st.Z for _ in range(10)] for _ in range(10)]
    for i in range(5):
        for j in range(5):
            value=F(float(K[i,j]))/N
            pad[i][j]=(iv.iv(value-F(1,1<<24))[0],iv.iv(value+F(1,1<<24))[1])
    return pad,{'raw_field0_gram':K.tolist(),'blocks':blocks,'all_spectator_dimension':len(y.reshape(-1)),'entry_error_exact':'1/16777216'}

def controls(maps,hbasis,ops,ann,V,J,profiles):
    rows=[]
    R=ops[3];I=sy.eye(16)
    hole=sy.zeros(16,4)
    for p in range(4): hole[15^(1<<p),p]=(-1)**p
    for prof in profiles:
        h=cast(sy.MutableDenseMatrix,sum((sy.Rational(prof[key])*A for key,A in zip(('m_exact','c0_exact','d_exact'),hbasis[:3])),sy.zeros(4)))
        D=sum((ann[p].T*ann[q]*h[p,q] for p in range(4) for q in range(4)),sy.zeros(16))
        assert sy.simplify(hole.T*D*hole-(sy.trace(h)*sy.eye(4)-h.T))==sy.zeros(4)
        assert hole.T*R*hole==-hbasis[3].T
        for creation,u in ((True,-sy.Rational(1,2)),(False,sy.Rational(1,2))):
            for p in range(4):
                cp=ann[p].T if creation else ann[p]
                delta=(1 if creation else -1)*hbasis[3][p,p]+u
                change=sum(((cast(sy.Expr,h[q,p]) if creation else -cast(sy.Expr,h[p,q]))*(ann[q].T if creation else ann[q]) for q in range(4)),sy.zeros(16))
                left=(D+64*(R+u*I)**2)*cp-cp*(D+64*R**2)
                right=change+64*cp*(2*delta*R+delta**2*I)
                assert sy.simplify(left-right)==sy.zeros(16)
                if p<2:
                    assert delta==0
                    cropped=sum(((cast(sy.Expr,h[q,p]) if creation else -cast(sy.Expr,h[p,q]))*(ann[q].T if creation else ann[q]) for q in range(2)),sy.zeros(16))
                    assert sy.simplify((change-cropped)*V)!=sy.zeros(16,5)
                    wrongu=-u
                    wrong=(D+64*(R+wrongu*I)**2)*cp-cp*(D+64*R**2)
                    assert sy.simplify((wrong-change)*V)!=sy.zeros(16,5)
        rows.append({'role':prof['role'],'full_16CAR_affine_commutator_all4':True,'site0_delta_zero_not_global_dynamics_deletion':True,'offsite_crop_rejected_on_actual_source_slot_space':True,'wrong_affine_sign_rejected':True,'number3_signed_hole_D_R_verified':True})
    unsigned=[A.applyfunc(abs) for A in ann]
    assert unsigned[0]*unsigned[1].T+unsigned[1].T*unsigned[0]!=sy.zeros(16)
    # Exact rational supported spectra reject omitted external field and source prices.
    trueheat=heat.expminus(F(74,64)); omitted=heat.expminus(F(10,64))
    assert trueheat[1]<omitted[0]
    r=F(1,131072)
    assert F(5679,100000000000)<r<F(1,65536)
    assert F(2,5)>2*F(9,25)**2 and F(2,5)<=2*F(9,25)
    # Tangent is negative in high-energy tail; floor-Hermite remains above actual heat.
    t=F(1,16); b=F(2); a=F(5); x=F(1000000)
    ea=heat.expminus(t*a);eb=heat.expminus(t*b);ex=heat.expminus(t*x)
    tangent=iv.mul(ea,iv.iv(1-t*(x-a)))
    coeff=iv.mul(iv.sub(eb,iv.mul(ea,iv.iv(1+t*(a-b)))),iv.iv(1/(a-b)**2))
    majorant=iv.add(tangent,iv.mul(coeff,iv.iv((x-a)**2)))
    assert tangent[1]<ex[0] and majorant[0]>ex[1]
    return {'actual_profiles':rows,'unsigned_CAR_rejected':True,'external64_omission_rejected_by_heat':True,'observed_residual_is_not_certified_cap_rejected':True,'squared_infidelity_as_vector_error_rejected':True,'tail_deleted_tangent_rejected_and_global_majorant_survives_x1000000':True}

def run(epoch):
    started=time.monotonic();pins={name:sha(ROOT/name) for name in INPUTS}
    assert all(pins[k]==v for k,v in native.PINS.items())
    source=json.loads((ROOT/'three_cell_density_formation_typed3_20261011_interface.json').read_text())
    N=F(source['source_norm_squared_exact'])
    upper=json.loads((ROOT/'local_so5_ground_upper_complete_state_consumer_20261011_receipt.json').read_text())
    profiles=json.loads((ROOT/'local_so5_rep_review_recovered_corrected_freeframe_20261011_proof.json').read_text())['profiles']
    J,maps,hbasis,ops,ann,V=native.car_exact()
    control=controls(maps,hbasis,ops,ann,V,J,profiles)
    Eupper=F(upper['ground_energy_interval_exact'][1]);floor=-5120-Eupper
    assert floor>0
    eta=iv.sqrt(iv.iv(2*F(upper['selected_ground_infidelity_upper_exact'])))
    residual=F(1,65536);shift=F(46955575073,1538631694057472);delta=iv.iv(residual+shift)
    assert upper['normalized_exact_source_residual_upper_exact']==str(residual)
    assert upper['coefficient_and_action_error_upper_exact']=='1/262144'
    with np.load(ROOT/'local_so5_resum_actual04_20261011_state.npz',allow_pickle=False) as data:
        X=np.array(data['state'],copy=True).reshape(14,690880)
    rows=[];raw=[]
    for role,(n,prof) in enumerate(zip((2,126),profiles)):
        K,record=gram(role,n,X,N,epoch);raw.append({'role':prof['role'],**record})
        h=cast(sy.MutableDenseMatrix,sum((sy.Rational(prof[key])*A for key,A in zip(('m_exact','c0_exact','d_exact'),hbasis[:3])),sy.zeros(4)))
        for creation in (True,False):
            reduced=st.reduced(maps[int(creation)],h,-F(1,2) if creation else F(1,2),creation,K)
            # Norm of actual four-mode commutator c_h is Euclidean norm of its column/row.
            cnorm=[st.algebraic(sy.sqrt(sum(sy.conjugate(h[q,p] if creation else h[p,q])*(h[q,p] if creation else h[p,q]) for q in range(4)))) for p in range(2)]
            M0=[[reduced['G'][i][j] for j in range(2)] for i in range(2)]
            M1=[];M2=[]
            for i in range(2):
                r1=[];r2=[]
                for j in range(2):
                    T=st.ca(reduced['T'][i][j],st.cc(reduced['T'][j][i]))
                    T=heat.cmulreal(T,iv.iv(F(1,2)))
                    p2=iv.add(iv.mul(iv.add(cnorm[i],cnorm[j]),delta),iv.mul(delta,delta))
                    r1.append(tuple(st.widen(v,delta) for v in T))
                    r2.append(tuple(st.widen(v,p2) for v in reduced['U'][i][j]))
                M1.append(r1);M2.append(r2)
            # Exact g: K c# g=Cg, so ||K Y_g||_F²=tr U_g.
            gsecond=st.Z
            for i in range(2):
                transfer=iv.scale(iv.mul(iv.mul(cnorm[i],cnorm[i]),eta),2)
                gsecond=iv.add(gsecond,iv.add(reduced['U'][i][i][0],transfer))
            rows.append({'role':'active' if role==0 else 'aux','physical_copy_count':1 if role==0 else 63,'orientation':'creation' if creation else 'annihilation','particle_number':257 if creation else 255,'affine_u_exact':'-1/2' if creation else '1/2','one_labelled_spin_replica':True,'initial_modes':[0,1],'commutator_all_modes':[0,1,2,3],'moments':{name:[[heat.encode(z) for z in rr] for rr in matrix] for name,matrix in zip(('M0','M1','M2'),(M0,M1,M2))},'ground_second_moment_trace_upper_exact':str(F(gsecond[1],iv.Q)),'commutator_column_norm_bounds':[iv.exact_endpoints(v) for v in cnorm]})
    interface={'kind':'ACTUAL_SOURCE_WEIGHTED_WHOLE_CHARGED_SPECTRUM_MATRIX_HEAT_INTERVAL','source':'s=immutable dd06400 binary64 exact real vector/sqrt(exact N); g=accepted canonical fullground, NOT s','source_norm_squared_exact':str(N),'charged_floor_exact':str(floor),'ground_energy_upper_exact':str(Eupper),'ground_distance_exact':iv.exact_endpoints(eta),'squared_infidelity_exact':upper['selected_ground_infidelity_upper_exact'],'rigorous_residual_exact':str(residual),'formation_included_in_residual_exact':'1/262144','Rayleigh_ground_shift_exact':str(shift),'source_action_error_exact':str(residual+shift),'roles':rows,'helper_module':'charged_cell_heat_helper_20261011','helper_sha256':sha(ROOT/'charged_cell_heat_helper_20261011.py'),'own_source_sha256':sha(Path(__file__)),'input_sha256':pins,'kernel_convention':'creation rows q cols p: <c_qdagger g,exp(-tK)c_pdagger g>; annihilation returned TRANSPOSE of annihilator-source Gram: row q col p=<c_p g,exp(-tK)c_q g>','native_frame_provenance':{'actual_native_slot_embedding_J':[[str(v) for v in rr] for rr in J.tolist()],'slot_phase':['1','-I','1','1','I'],'occupied_neutral_masks':[5,3,12,10,9,6],'number3_hole_columns':'(-1)^p |15 xor 2^p>, p=0..3','spin_frame':'actual corrected harmonic CAR native modes, site0 p0,1; not imported physical Phi spin frame'},'bounds':'global floor-Hermite Loewner/polarization; no spectral nodes or finite support crop','nonclaims':['physical1 ground or phase','gap-closing pole/IR atom','commoncone','allorders linked stability','originalSOURCE']}
    times=json.loads((ROOT/'flux_laplace_time_request_20261011.json').read_text())['required_t_exact']
    evaluations=[];improvement=[]
    for row in rows:
        for t in times+['1/8','1','4']:
            value=heat.evaluate(interface,row['role'],row['orientation'],F(t))
            baseline=heat.evaluate(interface,row['role'],row['orientation'],F(t),'baseline_quadratic')
            width=max(float(F(z[k][1])-F(z[k][0])) for rr in value for z in rr for k in ('re','im'))
            oldwidth=max(float(F(z[k][1])-F(z[k][0])) for rr in baseline for z in rr for k in ('re','im'))
            evaluations.append({'role':row['role'],'orientation':row['orientation'],'t_exact':t,'matrix':value,'derivative_norm_bounds':{str(j):heat.derivative_bound(interface,row['role'],row['orientation'],F(t),j) for j in (0,1,2)},'max_entry_width_display':width})
            improvement.append({'role':row['role'],'orientation':row['orientation'],'t_exact':t,'baseline_width':oldwidth,'floor_Hermite_width':width,'ratio':width/oldwidth if oldwidth else 1})
    interface['evaluations']=evaluations
    assert pins=={name:sha(ROOT/name) for name in INPUTS}
    interface_path=ROOT/f'{PREFIX}{epoch}_interface.json'
    dump(interface_path,interface)
    receipt={'kind':'NEW_SCIENTIFIC_CHARGED_HEAT_CONSUMER_NATURAL_EXIT_RECEIPT','epoch':epoch,'own_source_sha256':sha(Path(__file__)),'input_sha256':pins,'interface_sha256':sha(interface_path),'raw_new_grams':raw,'controls':control,'material_improvement':improvement,'tail_at1_exact':heat.tail_bound(interface,F(1)),'input_source_unchanged':True,'elapsed_seconds':time.monotonic()-started}
    dump(ROOT/f'{PREFIX}{epoch}_receipt.json',receipt)
    print(json.dumps({'receipt':f'{PREFIX}{epoch}_receipt.json','interface_sha256':receipt['interface_sha256'],'evaluations':len(evaluations),'max_required_width':max(r['max_entry_width_display'] for r in evaluations if r['t_exact'] in times),'elapsed_seconds':receipt['elapsed_seconds']}),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--epoch',required=True)
    run(parser.parse_args().epoch)
