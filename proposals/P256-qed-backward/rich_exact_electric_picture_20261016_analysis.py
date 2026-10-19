"""Full before-cut field covariances, free quadrature and own-state ray comparison."""
import json
from pathlib import Path
import numpy as np
import rich_exact_electric_picture_20261016_fields as f
import rich_spatial_frame_photon_20261016_fields as primitive
from rich_exact_electric_picture_20261016_bare import full_product
ROOT=Path(__file__).resolve().parent
P='rich_exact_electric_picture_20261016_'


def covariance(gram,basis,time,impulse):
    ext=primitive.basis(max(map(sum,basis))+1);d=f.primitive_images(ext,basis,time,impulse=impulse)
    B=d['b'];Pi=d['p']+np.asarray(impulse)[:,None,None]*d['embedding']
    meanB=np.array([np.einsum('ij,ij->',gram,(primitive.primitive(basis,basis,a,'q').toarray()+time/2*(primitive.primitive(basis,basis,a,'p').toarray()+impulse[a]*np.eye(len(basis))))).real for a in range(8)])
    meanP=np.array([np.einsum('ij,ij->',gram,primitive.primitive(basis,basis,a,'p').toarray()+impulse[a]*np.eye(len(basis))).real for a in range(8)])
    QQ=np.einsum('lei,mej->lmij',B.conj(),B,optimize=True);PP=np.einsum('lei,mej->lmij',Pi.conj(),Pi,optimize=True);QP=np.einsum('lei,mej->lmij',B.conj(),Pi,optimize=True)
    q2=np.einsum('ij,lmij->lm',gram,QQ,optimize=True);p2=np.einsum('ij,lmij->lm',gram,PP,optimize=True);qp=np.einsum('ij,lmij->lm',gram,QP,optimize=True)
    return dict(a_mean=meanB.tolist(),Pi_mean=meanP.tolist(),aa_covariance=(np.real(q2)-np.outer(meanB,meanB)).tolist(),PiPi_covariance=(np.real(p2)-np.outer(meanP,meanP)).tolist(),sym_aPi_covariance=(np.real(qp)-np.outer(meanB,meanP)).tolist(),primitive_CCR_error=float(np.linalg.norm(2j*np.imag(qp)-1j*np.eye(8)*np.trace(gram).real)))


def free_covariance_check():
    z,w=np.polynomial.legendre.leggauss(320);x=16*z;weight=16*w;records={}
    for cut in (1,2):
        basis=primitive.basis(cut);v=np.zeros(len(basis),dtype=np.complex128);v[0]=1
        a=np.zeros(8,dtype=int);a[6]=1;v[basis.index(tuple(a))]=1j;a[:]=0;a[0]=1;v[basis.index(tuple(a))]=.4;v/=np.linalg.norm(v)
        eps=np.zeros(8);eps[0]=.23;time=.4;Gs=[];Qs=[];Ps=[];Q2=[];P2=[];QPs=[]
        for axis in range(8):
            psi,der=f.wavefunctions(x,cut,time,impulse=eps[axis]);Gs.append(psi.conj().T@(weight[:,None]*psi));Qs.append(psi.conj().T@((weight*x)[:,None]*psi));Ps.append(psi.conj().T@(weight[:,None]*(-1j*der)));Q2.append(psi.conj().T@((weight*x*x)[:,None]*psi));P2.append(der.conj().T@(weight[:,None]*der));QPs.append(psi.conj().T@((weight*x)[:,None]*(-1j*der)))
        meansq=[];meansp=[];qq=np.zeros((8,8));pp=qq.copy();qp=qq.copy()
        for a in range(8):
            qops=list(Gs);qops[a]=Qs[a];pops=list(Gs);pops[a]=Ps[a]
            meansq.append(float(np.vdot(v,full_product(basis,qops)@v).real));meansp.append(float(np.vdot(v,full_product(basis,pops)@v).real))
            for b in range(8):
                qo=list(Gs);po=list(Gs);qpo=list(Gs)
                if a==b:qo[a]=Q2[a];po[a]=P2[a];qpo[a]=QPs[a]
                else:qo[a]=Qs[a];qo[b]=Qs[b];po[a]=Ps[a];po[b]=Ps[b];qpo[a]=Qs[a];qpo[b]=Ps[b]
                qq[a,b]=np.vdot(v,full_product(basis,qo)@v).real;pp[a,b]=np.vdot(v,full_product(basis,po)@v).real;qp[a,b]=np.vdot(v,full_product(basis,qpo)@v).real
        predicted=covariance(np.outer(v.conj(),v),basis,time,eps)
        records[str(cut)]=dict(full_aa_covariance_quadrature_error=float(np.linalg.norm(qq-np.outer(meansq,meansq)-predicted['aa_covariance'])),full_PiPi_covariance_quadrature_error=float(np.linalg.norm(pp-np.outer(meansp,meansp)-predicted['PiPi_covariance'])),full_sym_aPi_covariance_quadrature_error=float(np.linalg.norm(qp-np.outer(meansq,meansp)-predicted['sym_aPi_covariance'])),primitive_CCR_error=predicted['primitive_CCR_error'])
    return records


def main():
    result=dict(free_full_covariance=free_covariance_check(),cases={},initial_cut_ray_comparison={})
    for cut in (1,2):
        basis=primitive.basis(cut)
        for name in ['neutral','addition','removal']:
            data=np.load(ROOT/(P+f'field{cut}_{name}_raw.npz'));entry={}
            for label,key,axis,time in [('initial','input',None,0.),('baseline','baseline',None,.04),('kick0','kick0_end',0,.04),('kick2','kick2_end',2,.04)]:
                v=data[key];X=v.reshape(-1,len(basis));eps=np.zeros(8)
                if axis is not None:eps[axis]=.23
                entry[label]=covariance(X.conj().T@X,basis,time,eps)
            result['cases'][f'field{cut}_{name}']=entry
    b1=primitive.basis(1);b2=primitive.basis(2);ix=[b2.index(v) for v in b1]
    for name in ['neutral','addition','removal']:
        a=np.load(ROOT/(P+'field1_'+name+'_raw.npz'));b=np.load(ROOT/(P+'field2_'+name+'_raw.npz'));record={}
        for key in ['input','baseline','kick0_end','kick2_end']:
            x=a[key].reshape(-1,len(b1));y=b[key].reshape(-1,len(b2));overlap=np.vdot(x,y[:,ix])
            record[key]=dict(physical_same_protocol_overlap=[overlap.real.item(),overlap.imag.item()],squared_overlap=float(abs(overlap)**2))
        result['initial_cut_ray_comparison'][name]=record
    (ROOT/(P+'analysis_result.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(free_full_covariance=result['free_full_covariance'],ray_comparison=result['initial_cut_ray_comparison'])),flush=True)


if __name__=='__main__':main()
