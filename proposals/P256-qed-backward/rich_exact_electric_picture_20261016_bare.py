"""Actual free HG wavefunctions, Weyl kernels, canonical moments and wrong cuts."""
import json
from pathlib import Path
from typing import Any
import numpy as np
from scipy.linalg import expm
import rich_exact_electric_picture_20261016_fields as f
import rich_spatial_frame_photon_20261016_fields as old
ROOT=Path(__file__).resolve().parent
P='rich_exact_electric_picture_20261016_'


def full_product(basis,matrices):
    b=np.asarray(basis);out=np.ones((len(b),len(b)),dtype=np.complex128)
    for axis,m in enumerate(matrices):out*=m[b[:,axis,None],b[None,:,axis]]
    return out


def main():
    result={};raw={};time=.4;I=2.;epsilon=.23
    z,w=np.polynomial.legendre.leggauss(320);x=z*16;weights=w*16
    for cut in (1,2):
        b: list[tuple[int,...]]=[tuple(v) for v in old.basis(cut)]
        ext=old.basis(cut+1);n=len(b);v=np.zeros(n,dtype=np.complex128)
        vacuum=(0,)*8;one0=tuple([1]+[0]*7);one6=tuple([0]*6+[1,0])
        v[b.index(vacuum)]=1;v[b.index(one0)]=.4;v[b.index(one6)]=1j;v/=np.linalg.norm(v)
        impulse=np.zeros(8);impulse[0]=epsilon
        waves=[];p_m=[];q_m=[];q2_m=[];p2_m=[];grams=[];cross=[];c=np.array([.21,-.13,.17,.08,-.19,.11,.14,-.16])
        unbounded=[]
        for axis in range(8):
            psi,derivative=f.wavefunctions(x,cut,time,I,impulse[axis]);other,_=f.wavefunctions(x,cut,time,I,0.)
            gram=psi.conj().T@(weights[:,None]*psi);grams.append(gram)
            p_m.append(psi.conj().T@(weights[:,None]*(-1j*derivative)))
            q_m.append(psi.conj().T@((weights*x)[:,None]*psi))
            q2_m.append(psi.conj().T@((weights*x*x)[:,None]*psi))
            p2_m.append(derivative.conj().T@(weights[:,None]*derivative))
            cross.append(psi.conj().T@(weights[:,None]*other))
            unbounded.append(psi.conj().T@((weights*np.exp(1j*c[axis]*x))[:,None]*psi))
            waves.append(psi)
        gram=full_product(b,grams);exactW=f.displacement(b,b,c,time,I,impulse)
        numericalW=full_product(b,unbounded)
        primitive=f.primitive_images(ext,b,time,I,impulse);embedding=primitive['embedding']
        record: dict[str,Any]
        record=dict(Gram_error=float(np.linalg.norm(gram-np.eye(n))),Weyl_quadrature_error=float(np.linalg.norm(exactW-numericalW)))
        p0=np.array([old.primitive(b,b,a,'p').toarray() for a in range(8)])
        q0=np.array([old.primitive(b,b,a,'q').toarray() for a in range(8)])
        pmean=[];qmean=[];variances=[];p2total=0.
        for axis in range(8):
            ops=list(grams);ops[axis]=p_m[axis];pm=full_product(b,ops)
            ops=list(grams);ops[axis]=q_m[axis];qm=full_product(b,ops)
            ops=list(grams);ops[axis]=q2_m[axis];q2=full_product(b,ops)
            ops=list(grams);ops[axis]=p2_m[axis];pp=full_product(b,ops);p2total+=np.vdot(v,pp@v).real.item()
            pexpected=p0[axis]+impulse[axis]*np.eye(n)
            be=primitive['b'][axis];qexpected=q0[axis]+time/I*pexpected
            record[f'p{axis}_error']=float(np.linalg.norm(pm-pexpected))
            record[f'q{axis}_error']=float(np.linalg.norm(qm-qexpected))
            record[f'q2_{axis}_error']=float(np.linalg.norm(q2-be.conj().T@be))
            pmean.append(float(np.vdot(v,pm@v).real));qmean.append(float(np.vdot(v,qm@v).real));variances.append(float(np.vdot(v,q2@v).real-np.vdot(v,qm@v).real**2))
        piimage=primitive['p'];p2=sum(a.conj().T@a for a in piimage)
        energy0=float(np.vdot(v,p2@v).real/(2*I));work=p2total/(2*I)-energy0
        expectedwork=epsilon*np.vdot(v,p0[0]@v).real/I+epsilon**2/(2*I)
        overlap=f.protocol_overlap(b,b,impulse,np.zeros(8));numerical_overlap=full_product(b,cross)
        wrongkick=expm(1j*epsilon*q0[0])@v
        wrongprop=expm(-1j*time*p2/(2*I))@v
        record.update(field_dimension=n,initial_cut=cut,protocol_basis_norm=float(np.vdot(v,gram@v).real),kick_work=float(work),expected_work=float(expectedwork),kick_work_error=float(abs(work-expectedwork)),other_mode_Pi_initial_invariance=float(abs(pmean[6]-np.vdot(v,p0[6]@v).real)),Pi_means=pmean,a_means=qmean,a_variances=variances,protocol_overlap_error=float(np.linalg.norm(overlap-numerical_overlap)),wrong_cut_kick_other_mode_change=float((np.vdot(wrongkick,p0[6]@wrongkick)-np.vdot(v,p0[6]@v)).real),wrong_cut_free_Pi6_drift=float((np.vdot(wrongprop,p0[6]@wrongprop)-np.vdot(v,p0[6]@v)).real),wrong_untransformed_q6_error=float(abs(qmean[6]-np.vdot(v,q0[6]@v).real)),wrong_cut_Weyl_error=float(np.linalg.norm(expm(1j*sum(c[a]*(q0[a]+time/I*p0[a]) for a in range(8)))-exactW)))
        raw.update({f'cut{cut}_input':v,f'cut{cut}_Gram':gram,f'cut{cut}_Weyl_exact':exactW,f'cut{cut}_Weyl_quad':numericalW,f'cut{cut}_overlap':overlap,f'cut{cut}_wrongkick':wrongkick,f'cut{cut}_wrongprop':wrongprop})
        result[str(cut)]=record;print(cut,json.dumps(record),flush=True)
    (ROOT/(P+'bare_result.json')).write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(ROOT/(P+'bare_raw.npz'),**raw)


if __name__=='__main__':main()
