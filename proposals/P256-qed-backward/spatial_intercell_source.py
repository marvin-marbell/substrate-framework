"""Whole canonical Slater intercell response, conditional frozen-Phi body.

All old producers are read-only imports. A=1 is the actual cheaper seed.
The exact rational uniform curvature proof prices HE+HC, not (PHP)^2.
"""
from __future__ import annotations
import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import sys
import time
import zipfile
import numpy as np
from scipy.linalg import eigh, expm, expm_frechet
from threadpoolctl import threadpool_limits
import joined_phi_u1_spatial_linked_source as seed

ROOT=Path(__file__).resolve().parent
COPIES=64
PROOF={
 'source':'A=1 cheaper actual seed. B_tau=exp(-tau[h_IC,P_seed]) U_seed on WHOLE4V space; wedge of its2V occupied columns in each of64 copies over ONE common Haar field. Exact unitary and canonical norm1 pointwise imply exact joint norm1. Smooth periodic matrix exponential has every PH order, no PH truncation or finite Fourier cut. Tau varies the wave only.',
 'first_variation':'S=[h_IC,P]; dot P=-[S,P]. h_IC has only distinct-cell blocks while P,U_seed and every field derivative of the seed are cell diagonal. Q h_IC P is exactly in sectors with one cell losing and another gaining one particle. Hence onsite/intracell matter, HE/HC acting on the seed and their inner products with dotPsi vanish by actual fixed cell-number orthogonality. Tr h_IC dotP=-2||Q h_IC P||HS^2; active1+aux63 share identical h_IC but different onsite masses. Electric tangent connection r1=i U_seed^dag S_e_prime U_seed is off-cell; r0 and P0 are cell diagonal, so Tr P0 r1=Tr P0 r0 Q0 r1=0. No C/parity argument is needed.',
 'uniform_lower':'Select adjacent z-plane cell pairs. Both directed cell transitions are distinct for every evenN>=4, and no length2 z path has the same endpoints. For A0=Q_onsite h_ab P_onsite, ||A0||HS=sqrt(8 ci_z^2 gs). Seed ||P-Ponsite||<=8t gives ||A-A0||HS<=64t sqrt(c0_z^2 g0+ci_z^2 gs). Hence 64||Q h_IC P||HS^2 >=32V b^2, with b the positive bracket. Uniform Haar derivative <=-64V b^2.',
 'curvature_matter':'Let B=2sum_paths(|c0|sqrt(g0)+|ci|sqrt(gs)). The row/column Schur bound on native forward/reverse blocks gives ||h_IC||<=B, including AP/two-link collisions. Off-diagonal P/Q form of S implies ||S||<=B. P_tau double commutator has trace norm <=4B^2 rankP=8VB^2. Thus each role Taylor remainder <=4V H_role B^2 tau^2; H_role=|onsite_role|sqrt(g0)+B. No role averaging.',
 'local_trace_derivative':'Each edge occurs in one length1 and two length2 paths, with reverse included. ||h_IC,e prime||1<=8(a1+2a2). Seed ||P_e prime||1<=16 sin(t), so ||S_e prime||1<=D=16(a1+2a2)+32B sin(t), uniformly N and field; use max axis. Integral Duhamel, unitarity and trace-norm ideal property give ||R_tau,e-R_0,e||1<=|tau|D and ||R_tau,e-R_0,e-tau r1,e||1<=tau^2 B D. Derivative rank may spread; NO finite-rank assumption on exponential derivative.',
 'curvature_electric':'A=1 seed r0 has operator norm <=sin(t) on itsV active edges and0 on others. Orthogonality kills the linear Wick variation, including Berry crosscopy variation even without assuming seed mean0. Exact seed mean0 is separately inherited from the actual gate Clifford trace proof in the frozen input; linear Berry mean0 follows distinct-cell sectors. Per edge Var delta <=tau^2(2 sin(t) B D+D^2), and |mu_tau|<=tau^2 BD. ALL64 same-field Wick terms are 64 Var+64^2 mu^2. On Gauss sector COMPLETE HE64+HC64=64sum_3V Efull^2, so delta/V <=192[64(2sBD+D^2)tau^2+4096B^2D^2tau^4]. Full Ref4 stays fixed, rho=sum64(n-2), neutral exact AP sea unchanged. This prices longitudinal HC, not transverse-only HE.',
 'rational_enclosure':'Pinned exact saved g rationals are checked in [1/10,11/100] for gs and [0,3/10] for g0. sqrt(gs)>=3/10, <=1/3; sqrt(g0)<=11/20; sqrt(8)>=14/5; sin(1/128)<=1/128. b>= (21/25)|ci_z|-(11/40)|c0_z|-|ci_z|/6. All bound arithmetic is exact Fraction, with no finite-float positivity certificate.',
 'complete_action':'The curvature proof is an N-uniform complete-H energy decrease proof, not an actual variance or PHP squared. Additionally per edge second derivative of B_tau has trace norm <=8s+16s|tau|D+|tau|D2+tau^2D^2. Here D2=16(a1+2a2)+64s(a1+2a2)+32B s+64B s^2 bounds S_e second derivative by differentiating both commutators. Exterior second derivative on64 copies bounded by64 b2+64^2 b1^2 with b1=8s+|tau|D. Hence ||HE+HC Psi||<=192V(64b2+4096b1^2), using full covariant edge derivatives. Matter action <=2Vsum load H_role. Neutral eigenenergy is subtracted exactly. The triangle bounds the COMPLETE body action; actual full-H variance remains uncomputed.',
 'scope':'Conditional Clifford Phi and identity SU4, native compact beta0, A=1. No physicalPhi/SU4 lift, minimizing ground, stationarity, nonzero-hopping phase, IRatom, commoncone or originalSOURCE closure. Actual native values are finite-float observations, separate from rational uniform proof.'}

def sha(raw:bytes)->str:return hashlib.sha256(raw).hexdigest()
def archive(path:Path,table:dict[str,bytes])->dict:
    with zipfile.ZipFile(path,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for n,b in sorted(table.items()):z.writestr(n,b)
    return dict(path=str(path),sha256=sha(path.read_bytes()),members={n:sha(b) for n,b in table.items()})
def encode_f(x:F)->dict:return dict(exact=str(x),display=float(x))

def uniform(data:dict,centres:list[F])->dict:
    g0,gs=centres
    assert 0<g0<=F(3,10) and F(1,10)<=gs<=F(11,100)
    p=data['profiles'][1];s=F(1,128)
    aa=[abs(F(q['c0']))*F(11,20)+abs(F(q['ci']))/3 for q in p['paths']]
    B=2*sum(aa,F(0))
    z=next(q for q in p['paths'] if q['axis']==2 and q['length']==1)
    b=F(21,25)*abs(F(z['ci']))-F(11,40)*abs(F(z['c0']))-abs(F(z['ci']))/6
    assert b>0
    edge=max(sum((a*(1 if q['length']==1 else 2) for a,q in zip(aa,p['paths']) if q['axis']==axis),F(0)) for axis in range(3))
    D=16*edge+32*B*s
    D2=16*edge+64*s*edge+32*B*s+64*B*s*s
    H=[abs(F(r['onsite']))*F(11,20)+B for r in data['profiles'][1:]]
    role_sum=sum((int(r['load'])*h for r,h in zip(data['profiles'][1:],H)),F(0))
    negative=64*b*b
    curvature=4*role_sum*B*B+192*64*(2*s*B*D+D*D)
    quartic=192*4096*B*B*D*D
    tau=F(1,2)
    while -negative*tau+curvature*tau*tau+quartic*tau**4>=-negative*tau/2:tau/=2
    delta=-negative*tau+curvature*tau*tau+quartic*tau**4
    b1=8*s+tau*D;b2=8*s+16*s*tau*D+tau*D2+tau*tau*D*D
    action=2*role_sum+192*(64*b2+4096*b1*b1)
    assert delta<0
    return dict(all_even_N_ge4=True,all_native_fields=True,A_equals_1=True,exact_g0=str(g0),exact_gs=str(gs),b_lower=encode_f(b),S_operator_upper=encode_f(B),local_S_prime_trace_upper=encode_f(D),local_S_second_trace_upper=encode_f(D2),negative_slope_per_V=encode_f(-negative),quadratic_upper_per_V=encode_f(curvature),quartic_upper_per_V=encode_f(quartic),tau=encode_f(tau),strict_delta_H_upper_per_V=encode_f(delta),complete_body_action_norm_upper_per_V=encode_f(action),roles=[dict(role=r['role'],load=int(r['load']),onsite=str(r['onsite']),H_upper=encode_f(h)) for r,h in zip(data['profiles'][1:],H)])

def construction(native:dict,theta:np.ndarray)->dict:
    V=native['V'];d=4*V;frame=native['frame'];gam=frame/np.sqrt(np.trace(frame@frame,axis1=1,axis2=2).real/4)[:,None,None]
    U0=np.zeros((d,d),complex);P=np.zeros_like(U0);ids=np.zeros(V,int);derivatives={}
    for c,sites in enumerate(seed.tiles(native)):
        inds=np.array([4*x+j for x in sites for j in range(4)]);edges=[3*sites[0],3*sites[1]+1,3*sites[2],3*sites[0]+1]
        pc,u,du=seed.cell(theta[edges],gam)
        U0[np.ix_(inds,inds)]=u;P[np.ix_(inds,inds)]=pc;ids[sites]=c
        for e,uu in zip(edges,du):derivatives[e]=(inds,uu,uu@((np.eye(16)-np.kron(np.eye(4),gam[0]))/2)@u.conj().T+u@((np.eye(16)-np.kron(np.eye(4),gam[0]))/2)@uu.conj().T)
    p0=np.kron(np.eye(V),(np.eye(4)-gam[0])/2)
    mask=np.repeat(ids,4)[:,None]==np.repeat(ids,4)[None,:]
    return dict(P=P,U0=U0,p0=p0,mask=mask,derivatives=derivatives,cellids=ids)

def path_matrix(native:dict,profile:dict,theta:np.ndarray,edge:int|None=None,wrong:str|None=None)->np.ndarray:
    d=4*native['V'];h=np.zeros((d,d),complex)
    if edge is None:
        h=np.kron(np.eye(native['V']),float(profile['onsite'])*native['frame'][0]).astype(complex)
    for displacement,c in profile['raw_paths']:
        for p in native['Pprofiles'][0]['paths']:
            if not np.array_equal(displacement,p['displacement']):continue
            if wrong=='omit_two' and len(p['edges'])==2:continue
            if edge is not None and edge not in p['edges']:continue
            a,b=slice(4*p['source'],4*p['source']+4),slice(4*p['target'],4*p['target']+4)
            seam=1 if wrong=='no_AP' else p['seam']
            block=seam*np.exp(1j*sum(theta[e] for e in p['edges']))*np.einsum('m,mij->ij',c,native['frame'])
            if edge is not None:block*=1j
            h[a,b]+=block;h[b,a]+=block.conj().T
    return h

def tangent_case(data:dict,N:int,tau:float,full_electric:bool)->dict:
    native=seed.field.full.supplier.quantum.build_native(data['mean'],N);V=native['V'];d=4*V
    rng=np.random.default_rng(222101020+N);theta=.23*rng.normal(size=3*V)
    c=construction(native,theta);P=c['P'];U0=c['U0'];p0=c['p0'];mask=c['mask'];I=np.eye(d);Q=I-P
    roles=data['profiles'][1:];h=path_matrix(native,roles[0],theta);hic=np.where(mask,0,h);S=hic@P-P@hic;dot=-S@P+P@S
    qhp=Q@hic@P;norm=float(np.linalg.norm(qhp)**2)
    lam=.3*rng.normal(size=V);shift=native['D'].T@lam;gg=np.exp(1j*np.repeat(lam,4));lg=np.diag(np.repeat(lam,4))
    cg=construction(native,theta+shift);hg=path_matrix(native,roles[0],theta+shift);sg=np.where(mask,0,hg)@cg['P']-cg['P']@np.where(mask,0,hg)
    U=expm(-tau*S);Pt=U@P@U.conj().T;Ug=expm(-tau*sg);Pgt=Ug@cg['P']@Ug.conj().T
    gauge=float(np.max(abs(Pgt-gg[:,None]*Pt*gg.conj()[None,:])))
    dP=np.zeros_like(P);dH=np.zeros_like(h)
    # Gauge-direction derivatives are analytic commutators, independently
    # checked below against summed actual sector-local seed derivatives.
    for e,(inds,du,dp) in c['derivatives'].items():dP[np.ix_(inds,inds)]+=shift[e]*dp
    for displacement,coef in roles[0]['raw_paths']:
        for path in native['Pprofiles'][0]['paths']:
            if not np.array_equal(displacement,path['displacement']):continue
            a,b=slice(4*path['source'],4*path['source']+4),slice(4*path['target'],4*path['target']+4)
            block=1j*sum(shift[e] for e in path['edges'])*path['seam']*np.exp(1j*sum(theta[e] for e in path['edges']))*np.einsum('m,mij->ij',coef,native['frame'])
            dH[a,b]+=block;dH[b,a]+=block.conj().T
    dhic=np.where(mask,0,dH);ds=dhic@P-P@dhic+hic@dP-dP@hic
    ud=np.asarray(expm_frechet(-tau*S,-tau*ds,compute_expm=False));pder=ud@P@U.conj().T+U@dP@U.conj().T+U@P@ud.conj().T
    wrongud=np.asarray(expm_frechet(-tau*S,-tau*(dhic@P-P@dhic),compute_expm=False))
    wrongpd=wrongud@P@U.conj().T+U@dP@U.conj().T+U@P@wrongud.conj().T
    target=1j*(lg@Pt-Pt@lg)
    rows=[]
    for role in roles:
        hr=path_matrix(native,role,theta);within=np.where(mask,hr,0);energy0=float(np.trace(P@hr).real);energytau=float(np.trace(Pt@hr).real)
        rows.append(dict(role=role['role'],load=int(role['load']),onsite=str(role['onsite']),baseline_pointwise_energy=energy0,tau_pointwise_energy=energytau,delta=energytau-energy0,first_variation=float(np.trace(hr@dot).real),intracell_onsite_first_variation=float(np.trace(within@dot).real)))
    result=dict(N=N,V=V,tau=tau,rank_exact=2*V,norm_exact=1,projector_defect=float(np.max(abs(Pt@Pt-Pt))),unitary_defect=float(np.max(abs(U.conj().T@U-I))),antiHermitian_defect=float(np.max(abs(S+S.conj().T))),seed_offcell_norm=float(np.linalg.norm(np.where(mask,0,P))),tau_offcell_norm=float(np.linalg.norm(np.where(mask,0,Pt))),q_hIC_p_HS_squared=norm,all64_first_variation=-128*norm,roles=rows,cell_sector_tangent_diagonal_defect=float(np.max(abs(np.where(mask,dot,0)))),gauge_source_defect=gauge,Gauss_full_incidence_derivative_defect=float(np.max(abs(pder-target))),Gauss_seed_sector_derivative_defect=float(np.max(abs(dP-1j*(lg@P-P@lg)))),wrong_omitted_seed_derivative_defect=float(np.max(abs(wrongpd-target))),wrong_transverse_only_Gauss_defect=float(np.max(abs(target))),wrong_AP_matrix_defect=float(np.linalg.norm(h-path_matrix(native,roles[0],theta,wrong='no_AP'))),wrong_two_link_matrix_defect=float(np.linalg.norm(h-path_matrix(native,roles[0],theta,wrong='omit_two'))),reverse_Hermiticity_defect=float(np.max(abs(h-h.conj().T))),actual_pointwise_matter_delta=sum(r['load']*r['delta'] for r in rows),finite_float_semantics='Actual fixed native field, NOT Haar quadrature or interval proof.')
    assert norm>1 and result['tau_offcell_norm']>0 and gauge<1e-8 and result['Gauss_full_incidence_derivative_defect']<1e-8
    assert result['wrong_AP_matrix_defect']>1 and result['wrong_two_link_matrix_defect']>1
    assert max(abs(r['first_variation']+2*norm) for r in rows)<1e-5
    assert max(abs(r['intracell_onsite_first_variation']) for r in rows)==0
    if full_electric:
        vals,v=eigh(-1j*S);a=-1j*tau*vals;ph=np.exp(a);dif=a[:,None]-a[None,:]
        factors=np.empty_like(dif);nz=abs(dif)>1e-14;factors[nz]=(ph[:,None]-ph[None,:])[nz]/dif[nz];factors[~nz]=np.broadcast_to(ph[:,None],dif.shape)[~nz]
        _,ww=eigh(p0);W=U0@ww[:,d//2:];Y=U0@ww[:,:d//2]
        var0=var1=mean0=mean1=cross=0.;sectorcross=0.;r1mean=0.;edge_rows=[]
        for e in range(3*V):
            dp=np.zeros_like(P);r0=np.zeros_like(P)
            if e in c['derivatives']:
                inds,du,localdp=c['derivatives'][e];dp[np.ix_(inds,inds)]=localdp
                localu=U0[np.ix_(inds,inds)];r0[np.ix_(inds,inds)]=-1j*du@localu.conj().T
            hp=path_matrix(native,roles[0],theta,edge=e);hp=np.where(mask,0,hp)
            sp=hp@P-P@hp+hic@dp-dp@hic
            udp=v@(factors*(v.conj().T@(-tau*sp)@v))@v.conj().T
            rextra=-1j*U.conj().T@udp;r=r0+rextra
            old=Y.conj().T@r0@W;new=Y.conj().T@r@W
            mu0=complex(np.trace(W.conj().T@r0@W));mu1=complex(np.trace(W.conj().T@r@W))
            ev0=float(np.linalg.norm(old)**2);ev1=float(np.linalg.norm(new)**2)
            tangent=Y.conj().T@(1j*sp)@W
            sectorcross=max(sectorcross,abs(np.vdot(old,tangent)));r1mean=max(r1mean,abs(np.trace(W.conj().T@(1j*sp)@W)))
            var0+=ev0;var1+=ev1;mean0+=abs(mu0)**2;mean1+=abs(mu1)**2
            edge_rows.append(dict(edge=e,variance_seed=ev0,variance_tau=ev1,mean_tau_real=mu1.real,mean_tau_imag=mu1.imag))
        full0=64*(64*var0+4096*mean0);full1=64*(64*var1+4096*mean1)
        blocks=Pt.reshape(V,4,V,4).transpose(0,2,1,3);density=np.trace(blocks[np.arange(V),np.arange(V)],axis1=1,axis2=2).real
        cov=np.diag(density)-np.einsum('xyij,yxji->xy',blocks,blocks).real;rho=density-2
        hc=64*(64*float(np.sum(native['Lplus']*cov))+4096*float(rho@native['Lplus']@rho))
        result['actual_all_edge_fullE']=dict(edge_count=3*V,seed_HEHC=full0,tau_HEHC=full1,delta=full1-full0,tau_HC=hc,tau_HE=full1-hc,onecopy_variance_sum=var1,all64_Berry_crosscopy_price=64*4096*mean1,wrong_independent_copies_missing_cross=64*64*63*mean1,first_variation_sector_inner_defect=float(sectorcross),first_variation_mean_defect=float(r1mean),edge_observations=edge_rows)
        result['actual_pointwise_complete_H_delta']=result['actual_pointwise_matter_delta']+full1-full0
        # A periodic face-phase changes only the trial, giving a discriminating
        # exact64^2 mean law even if this tiny-tau source Berry is float-blind.
        result['shared_copy_phase_control']=dict(phase_per_copy='exp(i native contractible plaquette flux)',face_integer_norm_squared=4,exact_allcopy_HEHC_phase_price=64*64**2*4,wrong_independent_price=64*64*4,missing_crosscopy_price=64*64*63*4,semantics='Actual smooth same-field wave phase; no Hamiltonian modification.')
        assert sectorcross<1e-7 and r1mean<1e-7
    return result

def run(output:Path)->dict:
    started=time.monotonic()
    names=['induced_phi_uniform_selfconsistent_full_normal_quantum_final_root_actual_receipt.json','induced_phi_uniform_selfconsistent_full_normal_quantum_final_actual_receipt.json','induced_phi_nonuniform_fermionic_source_unitary_resumed_actual_receipt.json','joined_phi_u1_spatial_linked_corrected_weighted_20261010_receipt.json','joined_phi_u1_bulk_flux_spatial_sharp_parent_packet_20261010.json','spatial_intercell_language_actual_20261010.json']
    inputs={n:(ROOT/n).read_bytes() for n in names};saved=json.loads(inputs[names[2]])['input_descriptor']['exact_NEW_allq_BO_mean'];centres=[F(x) for x in saved['centre_exact_rational']]
    sources={**seed.SOURCE_BYTES,**seed.field.CAPTURED,Path(__file__).name:Path(__file__).read_bytes()};table={**sources,**inputs}
    assert sha(inputs[names[4]])=='50157e90c0a93c972a579990af3b7dcb270310e76646b7d27cd02b712f0f4dc9'
    pre=archive(output.with_name(output.stem+'_preexecution.zip'),table)
    data=seed.field.load_mean(g_override=[float(centres[0]),*[float(centres[1])]*3],mean_descriptor=saved)
    assert [int(r['load']) for r in data['profiles'][1:]]==[1,63]
    assert data['profiles'][1]['paths']==data['profiles'][2]['paths'] and data['profiles'][1]['onsite']!=data['profiles'][2]['onsite']
    bound=uniform(data,centres);tau=float(F(bound['tau']['exact']))
    with threadpool_limits(limits=1):cases=[tangent_case(data,4,tau,True),tangent_case(data,6,tau,False)]
    baseline=json.loads(inputs[names[3]])
    comparisons=[]
    for c in cases:
        old=next(x for x in baseline['cases'] if x['N']==c['N'])
        base=old['same_H_A_equals_1_control']
        comparisons.append(dict(N=c['N'],actual_baseline_A1_Haar_energy=base['complete_body_energy_relative_neutral_eigenenergy'],new_Haar_energy_rigorous_upper_expression='baseline_A1 + V * exact_delta_upper_per_V',strict_delta_upper=encode_f(F(bound['strict_delta_H_upper_per_V']['exact'])*c['V']),previous_stationarity=base['stationarity'],new_stationarity='UNPROVED; strict descent is not a ground/eigenstate claim',previous_weighted_energy=old['complete_body_energy_relative_neutral_eigenenergy']))
    result=dict(kind='ACTUAL_WHOLE_INTERCELL_CANONICAL_SOURCE_AND_EXACT_RATIONAL_N_UNIFORM_COMPLETE_H_DECREASE',proof=PROOF,uniform_bound=bound,cases=cases,actual_baseline_comparisons=comparisons,source_hashes={n:sha(b) for n,b in sources.items()},input_hashes={n:sha(b) for n,b in inputs.items()},preexecution=pre,runtime=dict(python=sys.executable,version=sys.version,numpy=np.__version__),unproved=['fullPhi/SU4 lift','actual complete-H variance','stationary/minimizing ground','physical-hopping phase','IRatom','commoncone','originalSOURCE'],elapsed_seconds=time.monotonic()-started)
    assert all((ROOT/n).read_bytes()==b for n,b in table.items())
    result['source_inputs_unchanged']=True
    output.write_text(json.dumps(result,indent=2)+'\n')
    executed=archive(output.with_name(output.stem+'_executed.zip'),{**table,output.name:output.read_bytes()})
    exit_manifest=output.with_name(output.stem+'_archive_manifest.json');exit_manifest.write_text(json.dumps(dict(receipt_sha256=sha(output.read_bytes()),preexecution=pre,executed=executed),indent=2)+'\n')
    return result

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();r=run(args.output)
    print(json.dumps(dict(receipt=str(args.output),sha256=sha(args.output.read_bytes()),tau=r['uniform_bound']['tau'],strict_delta_H_upper_per_V=r['uniform_bound']['strict_delta_H_upper_per_V'],cases=[{k:c[k] for k in ('N','tau_offcell_norm','all64_first_variation','actual_pointwise_matter_delta')} for c in r['cases']],elapsed_seconds=r['elapsed_seconds']),indent=2))
