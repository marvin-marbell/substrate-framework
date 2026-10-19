#!/usr/bin/env python3
"""Native C/Gauss and noncompact physical neutral supplier discriminants.

Actual endpoint/full256CAR symmetry and coupled quantum creator-mode checks.
No infinite-volume photon or continuum claim.
"""
from __future__ import annotations
import argparse,hashlib,itertools,json,math,os
from pathlib import Path
from typing import Any
if __name__=='__main__':
    for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='1'
import numpy as np
from scipy.linalg import expm
from bulk_quantum_selection import BulkQuantumLaw
from original_charged_principal import original_frame
from rotation_band import RotationBand
from sea_selection import SELECTED
from noncompact_native_current import Coupled,accept,adverse,exterior


def particle_hole(rows,modes):
    lookup={row:i for i,row in enumerate(rows)}
    C=np.zeros((len(rows),len(rows)),dtype=np.complex128)
    for col,row in enumerate(rows):
        occupied=set(range(modes));sign=1
        for i in reversed(row):
            sign*=(-1)**sum(j<i for j in occupied);occupied.remove(i)
        C[lookup[tuple(sorted(occupied))],col]=sign
    return C


def wedge(U,rows):
    return np.array([[np.linalg.det(U[np.ix_(a,b)]) for b in rows] for a in rows])


def native_symmetry(size,spacing):
    band=RotationBand(a=spacing,m=.2,r=.12,v=.3,improved=False)
    # ONLY the actual native endpoint/Gram/link-defect methods are exercised.
    # The added27-flavour selector/minimum in this constructor is NEVER used.
    law=BulkQuantumLaw(size,band=band)
    rng=np.random.default_rng(2221009)
    fields=law.fields(rng.normal(size=(law.volume,4,15)))
    links=np.array([[expm(1j*np.einsum('a,aij->ij',rng.normal(size=15)*.2,law.generators)) for _ in range(3)] for _ in range(law.volume)])
    A=rng.normal(size=(law.volume,3))*3
    changed=fields.conj().copy();changed[:,0]*=-1
    conjugate_links=links.conj()
    hh=law.onebody(fields,links)
    hl=law.onebody(fields,links,A)
    hc=law.onebody(changed,conjugate_links)
    lc=law.onebody(changed,conjugate_links,-A)
    checks=dict(heavy=accept(np.linalg.norm(hc+hh.T),'actual full native heavy h(Cfields)=-h(fields)^T'),
                light=accept(np.linalg.norm(lc+hl.T),'actual full native light h(Cfields,-A)=-h(fields,A)^T'))
    signs=np.array([-1.,1.,1.,1.])
    gram=law.grams(fields);gramc=law.grams(changed)
    checks['true_Gram']=accept(np.linalg.norm(gramc-gram*signs[None,:,None]*signs[None,None,:]),'native all-ten Gram C transform')
    quartic=lambda G:float(np.sum(G*G))/spacing
    checks['quartic']=accept(abs(quartic(gramc)-quartic(gram)),'actual kappa/2 Gram quartic C symmetry')
    w,p=law.link_defects(fields,links);wc,pc=law.link_defects(changed,conjugate_links)
    checks['alignment']=accept(abs(w-wc),'native complete alignment C symmetry')
    checks['colour_plaquettes']=accept(abs(p-pc),'native complete colour plaquette C symmetry')
    generator=law.generators[6]
    gauge_fields=1j*(generator@fields-fields@generator)
    derivative_c=gauge_fields.conj();derivative_c[:,0]*=-1
    transformed_generator=-generator.conj()
    checks['all_field_Gauss_tangent']=accept(np.linalg.norm(derivative_c-1j*(transformed_generator@changed-changed@transformed_generator)), 'full colour Gauss covariant field tangent under C')
    # Independent complete256CAR lift; no occupation filter or determinant minor.
    words=list(range(256));ann=[]
    for mode in range(8):
        a=np.zeros((256,256),dtype=np.complex128)
        for word in words:
            if word>>mode&1:a[word^(1<<mode),word]=(-1)**(word&((1<<mode)-1)).bit_count()
        ann.append(a)
    h=hl[:8,:8];hprime=lc[:8,:8]
    H=sum((h[i,j]*(ann[i].conj().T@ann[j]) for i in range(8) for j in range(8)),np.zeros((256,256),dtype=np.complex128))
    Hp=sum((hprime[i,j]*(ann[i].conj().T@ann[j]) for i in range(8) for j in range(8)),np.zeros((256,256),dtype=np.complex128))
    allrows=[tuple(i for i in range(8) if word>>i&1) for word in words]
    PH=particle_hole(allrows,8)
    checks['complete256CAR']=accept(np.linalg.norm(PH@H@PH.conj().T-Hp),'actual entire256CAR particle-hole transfer')
    number=sum((a.conj().T@a for a in ann),np.zeros((256,256),dtype=np.complex128))
    checks['half_background']=accept(np.linalg.norm(PH@number@PH.conj().T-(8*np.eye(256)-number)),'entire occupation/background transformation')
    wrongfields=fields.conj()
    wrong=law.onebody(wrongfields,conjugate_links,-A)
    checks['wrong_C_time_field_sign']=adverse(np.linalg.norm(wrong+hl.T),'actual full native light h(Cfields,-A)=-h(fields,A)^T')
    return dict(size=size,volume=law.volume,onebody_dimension=4*law.volume,complete_CAR_dimension=256,arbitrary_full_fields=True,arbitrary_SU4_links=True,real_unwrapped_A=True,onsite27flavour_selector_used=False,checks=checks)


def uniform_row(spacing,size):
    band=RotationBand(a=spacing,m=.2,r=.12,v=.3,improved=False)
    kappa=2/spacing
    Cx=math.sqrt((.2+6*.12/spacing)**2+3*(.3/spacing)**2)
    gram_upper=(4*math.sqrt(2)*Cx/kappa)**(2/3)
    B=Cx*math.sqrt(2*gram_upper)
    rows=[]
    for theta in itertools.product(range(4),repeat=3):
        selected,frame=original_frame(band,size,np.array(theta)*math.pi/2,True)
        G=selected['G']
        actual=(.2+6*.12/spacing)*math.sqrt(G[0,0])+.3/spacing*sum(math.sqrt(G[j,j]) for j in range(1,4))
        if actual>B+1e-10:raise AssertionError('Native true-centre light row exceeded analytic uniform coercive Gram bound')
        rows.append(dict(centre=list(theta),true10Gram=G.tolist(),row_bound=actual,Gram_Frobenius=float(np.linalg.norm(G))))
    return dict(spacing=spacing,kappa=kappa,Cx=Cx,all_N_Gram_Frobenius_upper=gram_upper,all_N_all64_native_light_row_upper=B,actual_size=size,actual_all64=rows,max_measured_native_row=max(r['row_bound'] for r in rows),no_N0_admission_from_finite_diagnostics=True)


def coupled_mode(count,radius,UE,K,frame):
    neutral=Coupled(count,radius,0.,4,UE,K,frame)
    charged=Coupled(count,radius,0.,5,UE,K,frame)
    ev,vec,res=neutral.solve(3);E0=float(ev[0]);psi=vec[:,0];wave=neutral.reshape(psi)
    if ev[1]-ev[0]<1e-7:raise AssertionError('Use full neutral projector when ground is degenerate')
    qc=SELECTED[0]@SELECTED[2];qt=SELECTED[1]@SELECTED[3]
    Fc=wedge(np.kron(np.eye(2),qc),neutral.rows)@particle_hole(neutral.rows,8)
    Ft=wedge(np.kron(np.eye(2),qt),neutral.rows)
    parity=(-1.)**np.arange(count)
    def C(w):return (w.reshape(count,neutral.l,neutral.d)*parity[:,None,None])[:,::-1].reshape(neutral.phdim,neutral.d)@Fc.T
    def T(w):return (w.conj().reshape(count,neutral.l,neutral.d)*parity[:,None,None]).reshape(neutral.phdim,neutral.d)@Ft.T
    rng=np.random.default_rng(2221010)
    test=rng.normal(size=neutral.size)+1j*rng.normal(size=neutral.size);test/=np.linalg.norm(test)
    checks=dict(C=accept(np.linalg.norm(neutral.action(C(neutral.reshape(test)).ravel())-C(neutral.reshape(neutral.action(test))).ravel()),'actual coupled full quantum C symmetry'),T=accept(np.linalg.norm(neutral.action(T(neutral.reshape(test)).ravel())-T(neutral.reshape(neutral.action(test))).ravel()),'actual coupled full quantum time reversal'))
    trace_density=np.zeros((2,2),dtype=np.complex128)
    for x in range(2):
        for y in range(2):
            h=np.zeros((8,8),dtype=np.complex128)
            h[4*x:4*x+4,4*y:4*y+4]=np.eye(4)
            unit=exterior(h,neutral.rows)
            trace_density[x,y]=np.vdot(wave,wave@unit.T)
    checks['C_T_mode_density']=accept(np.linalg.norm(trace_density-2*np.eye(2)),'actual C/T complete-colour creator-mode density kernel')
    lookup={row:i for i,row in enumerate(charged.rows)}
    creators=[]
    for x in range(2):
        bundle=np.zeros((4,charged.d,neutral.d),dtype=np.complex128)
        for col,row in enumerate(neutral.rows):
            for a in range(4):
                i=4*x+a
                if i not in row:bundle[a,lookup[tuple(sorted((*row,i)))],col]=(-1)**sum(j<i for j in row)
        creators.append(bundle)
    eta=np.array([1.,.4+.7j]);eta/=np.linalg.norm(eta)
    B=sum((eta[x]*creators[x] for x in range(2)),np.zeros_like(creators[0]))
    source=np.array([wave@b.T for b in B]);weight=float(np.vdot(source,source).real)
    checks['complex_finite_mode_norm']=accept(abs(weight-2),'actual unfiltered complex finite-mode source norm2')
    cost=sum(np.vdot(s,charged.reshape(charged.action(s.ravel()))-E0*s) for s in source).real/weight
    packet_width=.08;nodes,weights=np.polynomial.legendre.leggauss(8)
    packet_actual=0.;packet_linear=0.
    for node,w in zip(nodes,weights):
        delta=packet_width*node
        shifted=Coupled(count,radius,delta,4,UE,K,frame)
        change=float(np.vdot(psi,shifted.action(psi)-neutral.action(psi)).real)
        packet_actual+=w*change/2
        momentum=np.tile(neutral.ell,count)
        packet_linear+=w*(UE*float(np.sum(abs(wave)**2*momentum[:,None]))*delta/3)/2
    prediction=UE*packet_width**2/18
    checks['normalizable_Bloch_packet']=accept(abs(packet_actual-prediction),'actual full coupled normalized harmonic packet error',2e-10)
    wrong_packet=adverse(abs(packet_actual),'actual full coupled normalized harmonic packet error',2e-10)
    return dict(quantum_basis=dict(oscillator=count,harmonic_radius=radius,neutral=neutral.size,charged_with_full_reference=4*charged.size),E0=E0,neutral_gap=float(ev[1]-ev[0]),eigen_residual=res,full_comparison_residual=neutral.full_residual_bound(psi,E0),trace_density_real=trace_density.real.tolist(),trace_density_imag=trace_density.imag.tolist(),eta=[dict(real=float(e.real),imag=float(e.imag)) for e in eta],finite_mode_norm=weight,finite_mode_full_Rayleigh_excess=float(cost),checks=checks,harmonic_packet=dict(Bloch_support=[-packet_width,packet_width],normalization=1.,measured_excess=packet_actual,predicted_excess=prediction,retained_linear_current=packet_linear,erased_packet_error=wrong_packet),same_R=1,packet_includes_R_in_this_two_site_discriminator=True,reference_separation_claim=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('--native-size',type=int,default=3);p.add_argument('--frame-size',type=int,default=5);p.add_argument('--spacing',type=float,default=1.);p.add_argument('--oscillators',type=int,default=7);p.add_argument('--harmonic-radius',type=int,default=4);p.add_argument('--UE',type=float,default=.37);p.add_argument('--K',type=float,default=.9);args=p.parse_args()
    band=RotationBand(a=args.spacing,m=.2,r=.12,v=.3,improved=False)
    selected,frame=original_frame(band,args.frame_size,np.zeros(3),True)
    result=dict(law='Explicit noncompact gauge-quotient quantum photon root, not native compact kernel in unreduced L2',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),inputs=vars(args),microscopic_native_C=native_symmetry(args.native_size,args.spacing),uniform_native_row=uniform_row(args.spacing,args.frame_size),actual_coupled_C_T_and_creator_mode=coupled_mode(args.oscillators,args.harmonic_radius,args.UE,args.K,frame),limits=['Sampled nativeN3/N5 symmetry/Gram controls are not N0 admission or infinite heavy/field spectrum.','Two-site quantum source includesR; finitepacket separatedR consumer is a distinct obligation.','Finite exactGalerkin comparison is not a full-space infimum/phase/mobileband/Compton proof.'],campaign_complete=False)
    print(json.dumps(result,indent=2,allow_nan=False))

if __name__=='__main__':main()
