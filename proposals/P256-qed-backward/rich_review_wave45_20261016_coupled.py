"""Fresh own44 charged-frame, actual noncompact field kicks and mechanical controls."""
import json
from pathlib import Path
import numpy as np
from scipy import sparse as sp
from scipy.sparse.linalg import expm_multiply
import rich_gauge_band_frame_20261016_model as band
import rich_coupled_frame_photon_20261016_model as law

ROOT=Path(__file__).resolve().parent;P='rich_review_wave45_20261016_'


def expectation(x,a):return float(np.vdot(x,a@x).real)


def angular_exterior(m,x):
    hm=m.matter;nf,ng,nm,nr=hm.shape;npf=m.nfield
    ga=band.angular(tuple(sorted(set(z[0] for z in hm.g['labels'])|{max(z[0] for z in hm.g['labels'])+1})))
    ra=band.angular(tuple(range(m.parameters['angular_level']+2)))
    ig=[ga['labels'].index(z) for z in hm.g['labels']];ir=[ra['labels'].index(z) for z in hm.material['labels']]
    eg=np.eye(len(ga['labels']))[:,ig];er=np.eye(len(ra['labels']))[:,ir]
    v=x.reshape(nf,ng,nm,nr,npf)
    out=np.zeros((nf,len(eg),len(er),nr,npf),complex)
    for i,a,l in __import__('itertools').product(range(3),repeat=3):
        f=hm.sector.bilinear(hm.geo['gamma'][i]);z=(f@v.reshape(nf,-1)).reshape(v.shape)
        z=np.einsum('ag,fgmrp->famrp',ga['O'][l][i].toarray()@eg,z,optimize=True)
        z=np.einsum('bm,famrp->fabrp',ra['O'][l][a].toarray()@er,z,optimize=True)
        out+=-2*np.cos(m.parameters['k'][a])*z
    for a in range(3):
        z=np.einsum('ag,fgmrp->famrp',eg,v,optimize=True)
        for b in range(3):
            zz=np.einsum('bm,famrp->fabrp',ra['O'][a][b].toarray()@er,z,optimize=True)
            zz=np.einsum('uv,fabrv->fabru',m.p[a].toarray(),zz,optimize=True)
            out+=m.body_dipole[:,b,None,None,None,None]*zz/m.parameters['cavity_inertia']
    # Other H terms are diagonal in angular labels and cannot contribute outside P.
    retained=out[:,ig,:,:,:][:,:,ir,:,:]
    outside2=max(0.,np.linalg.norm(out)**2-np.linalg.norm(retained)**2)
    return float(np.sqrt(outside2))


def one_branch(ref,neutral,number,create):
    m=law.Model(number);hm=m.matter;B=m.source_from(ref,0,create)
    x=B@neutral;strength=np.linalg.norm(x)**2
    if strength<1e-12:raise ValueError('zero actual dressed source')
    x/=np.sqrt(strength)
    v,c=m.source_vertices();J=[-m.lift(z) for z in v]
    mechanical=[1j*(d@hm.HR-hm.HR@d) for d in m.D]
    hop=[1j*(d@hm.Hrelative-hm.Hrelative@d) for d in m.D]
    parts={name:float(np.linalg.norm(m.lift(part)@x-B@(ref.lift(refpart)@neutral)/np.sqrt(strength))) for name,part,refpart in [
        ('electric',hm.HE,ref.matter.HE),('g_kinetic',hm.Hg,ref.matter.Hg),('R_kinetic',hm.HR,ref.matter.HR),('radial',hm.Hradial,ref.matter.Hradial)]}
    omitted_mechanics=np.linalg.norm(m.lift(mechanical[0])@x)
    decomp=np.linalg.norm((v[0]-mechanical[0]-hop[0])@x.reshape(m.shape))
    contact=c[0,0];compressed=-(m.D[0]@m.D[0]@hm.H-2*m.D[0]@hm.H@m.D[0]+hm.H@m.D[0]@m.D[0])
    contact_debt=np.linalg.norm(m.lift(contact-compressed)@x)
    D2wrong=sum(z@z for z in m.D)
    full_D2_debt=np.linalg.norm(m.lift(m.D2-D2wrong)@x)
    E=[m.electric(a) for a in range(3)];q=[sp.kron(m.identity,z,format='csr') for z in m.q]
    time=.17;eps=.045;free=expm_multiply(-1j*time*m.H,x)
    wrongH=m.H-m.lift(hm.HR);hoppingoff=m.H-m.lift(hm.Hrelative)
    wrongfree=expm_multiply(-1j*time*wrongH,x);offfree=expm_multiply(-1j*time*hoppingoff,x)
    rows=[];raw=dict(initial=x,free=free)
    for a in (0,1):
        plus=expm_multiply(1j*eps*q[a],x);minus=expm_multiply(-1j*eps*q[a],x)
        yp=expm_multiply(-1j*time*m.H,plus);ym=expm_multiply(-1j*time*m.H,minus)
        wrong=expm_multiply(-1j*time*wrongH,plus);off=expm_multiply(-1j*time*hoppingoff,plus)
        deltaE=np.array([expectation(yp,z)-expectation(free,z) for z in E])
        deltaJ=np.array([expectation(yp,z)-expectation(free,z) for z in J])
        # Independent retarded first variation from exact propagated q|source>.
        tangent=expm_multiply(-1j*time*m.H,1j*(q[a]@x))
        ret=np.array([2*np.vdot(free,z@tangent).real for z in J])
        fd=np.array([(expectation(yp,z)-expectation(ym,z))/(2*eps) for z in J])
        rows.append(dict(polarization=a,kick_work=expectation(plus,m.H)-expectation(x,m.H),
            full_CCR_work=eps*expectation(x,E[a])/m.parameters['cavity_inertia']+eps**2/(2*m.parameters['cavity_inertia']),
            initial_density_change=max(abs(expectation(plus,m.lift(n))-expectation(x,m.lift(n))) for n in hm.n),
            initial_current_change=[expectation(plus,z)-expectation(x,z) for z in J],
            final_delta_electric=deltaE.tolist(),final_delta_current=deltaJ.tolist(),retarded_timeorder_error=float(np.linalg.norm(fd-ret)),
            actual_omitted_HR_law_difference=float(np.linalg.norm((wrong-wrongfree)-(yp-free))),
            actual_hopping_off_law_difference=float(np.linalg.norm((off-offfree)-(yp-free))),
            final_r_feedback=expectation(yp,m.lift(hm.r))-expectation(free,m.lift(hm.r)),
            exterior=m.exterior_image(yp)))
        raw.update({f'kick{a}':plus,f'final{a}':yp,f'wrongHR{a}':wrong,f'hoppingoff{a}':off,f'tangent{a}':tangent})
    totalJ=[float(np.linalg.norm((m.H@z-z@m.H)@x)) for z in m.J]
    result=dict(number=number,create=create,net_charge=int(hm.sector.q[0].sum()),dimension=m.dimension,source_strength=float(strength),Gauss_error=hm.sector.gauss_error,
        source_component_commutators=parts,mechanical_source_vertex_norm=float(omitted_mechanics),source_mechanical_hop_decomposition_error=float(decomp),
        contact_full_vs_compressed_error=float(contact_debt),D2_full_vs_compressed_error=float(full_D2_debt),
        initial_full_angular_exterior_H_norm=angular_exterior(m,x),projected_total_J_commutator=totalJ,protocols=rows,
        source_convention='Fixed canonical variables; vertices=d_AHv=i[D,Hm], J=-vertices; length gauge uses dressed B40. Full contact before angular P.',
        limits='NEW noncompact uniform three-vector cavity, no bulk dispersion/propagation or compact-loop claim. x/y transverse to declared z. Finite field/angular Galerkin with explicit exterior; no continuum/mass/g2 claim.')
    np.savez_compressed(ROOT/(P+f'coupled_N{number}_raw.npz'),**raw)
    return result


def main():
    ref=law.Model(4);neutral=expm_multiply(-.21j*ref.H,ref.preparation())
    result=dict(kind='OWN_NEW44_SINGLE_LAW_PHYSICAL_REVIEW',neutral_dimension=ref.dimension,neutral_preparation_time=.21,branches=[])
    np.savez_compressed(ROOT/(P+'coupled_neutral_raw.npz'),neutral=neutral)
    for number,create in [(5,True),(3,False)]:
        entry=one_branch(ref,neutral,number,create);result['branches'].append(entry)
        (ROOT/(P+'coupled_result.json')).write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(entry),flush=True)


if __name__=='__main__':main()
