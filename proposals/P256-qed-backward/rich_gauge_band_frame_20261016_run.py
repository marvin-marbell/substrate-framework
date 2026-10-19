"""Actual finite quantum-field/full-Fock dynamics and adverse measurements."""
import json
from pathlib import Path
import numpy as np
from scipy import sparse as sp
from scipy.sparse.linalg import expm_multiply
import rich_gauge_band_frame_20261016_model as m

ROOT=Path(__file__).resolve().parent
P='rich_gauge_band_frame_20261016_'


def expect(v,a):return float(np.vdot(v,a@v).real)


def measure(model,v):
    h=model.H@v
    return dict(norm=float(np.vdot(v,v).real),energy=float(np.vdot(v,h).real),
                variance=float(np.vdot(h,h).real-abs(np.vdot(v,h))**2),
                r=expect(v,model.r),force=expect(v,model.force),beta=expect(v,model.beta),
                electric_energy=expect(v,model.HE),frame_energy=expect(v,model.Hg),
                material_energy=expect(v,model.HR),mass_supplier_energy=expect(v,model.Hmass),
                relative_energy=expect(v,model.Hrelative),
                J2=sum(float(np.linalg.norm(a@v)**2) for a in model.J),
                E=[expect(v,a) for a in model.E],density=[expect(v,a) for a in model.n],
                current=[expect(v,a) for a in model.currents],
                torque_g=[expect(v,a) for a in model.torque_g],torque_R=[expect(v,a) for a in model.torque_R])


def peierls_probe():
    geo=m.geometry();gamma=geo['gamma'];k=np.array(m.DEFAULTS['k']);q=np.array([.11,-.08,.06])
    h=lambda p:sum(-2*np.cos(p[a])*gamma[a] for a in range(3))
    vertex=[2*np.sin(k[a]+q[a]/2)*gamma[a] for a in range(3)]
    wrong=[2*np.sin(k[a])*gamma[a] for a in range(3)]
    delta=h(k+q)-h(k)
    return dict(midpoint_ward_error=float(np.linalg.norm(delta-sum(2*np.sin(q[a]/2)*vertex[a] for a in range(3)))),
                wrong_q0_ward_error=float(np.linalg.norm(delta-sum(2*np.sin(q[a]/2)*wrong[a] for a in range(3)))),
                meaning='unprojected infinite embedded-position parent band vertex; separate from finite tree-path currents')


def main():
    raw={};result={'kind':'ACTUAL_NEW_CENTRAL_BAND_FRAME_TREE_SOURCE_DYNAMICS','parameters':m.DEFAULTS,'peierls':peierls_probe()}
    reference=m.Model(4);initial=reference.preparation();raw['reference_input']=initial
    result['neutral']={'shape':reference.shape,'initial':measure(reference,initial)}
    times=np.array([0.,.12,.24]);raw['times']=times
    neutral=expm_multiply(-1j*reference.H,initial,start=0,stop=times[-1],num=len(times),traceA=-1j*reference.H.diagonal().sum())
    raw['neutral_states']=neutral
    result['neutral']['dynamics']=[measure(reference,v) for v in neutral]
    result['branches']={}
    for number,create,name,component in [(5,True,'addition',0),(3,False,'removal',1)]:
        model=m.Model(number);sources=[model.source_from(reference,mu,create) for mu in range(8)]
        packets=[b@initial for b in sources];weights=[float(np.linalg.norm(v)**2) for v in packets]
        v=packets[component]/np.sqrt(weights[component]);raw[name+'_input']=v
        evolved=expm_multiply(-1j*model.H,v,start=0,stop=times[-1],num=len(times),traceA=-1j*model.H.diagonal().sum())
        raw[name+'_states']=evolved
        derivatives=[]
        for axis in range(3):
            # Literal physical LEFT derivative of Dband(g); primitive CAR scalars.
            rhs=sum((-model.geo['spin'][axis][component,nu] if create else model.geo['spin'][axis][component,nu].conjugate())*packets[nu] for nu in range(8))
            lhs=model.J[axis]@packets[component]-sources[component]@(reference.J[axis]@initial)
            derivatives.append(float(np.linalg.norm(lhs-rhs)))
        noether=[];continuity=[];electric_derivative=[]
        for state in evolved:
            noether.append(max(float(np.linalg.norm(model.H@(a@state)-a@(model.H@state))) for a in model.J))
            for site in range(8):
                charge_rate=1j*(model.H@(model.n[site]@state)-model.n[site]@(model.H@state))
                divergence=sum(model.geo['incidence'][site,e]*(model.currents[e]@state) for e in range(8))
                continuity.append(float(np.linalg.norm(charge_rate-divergence)))
        # True source-electric connection is [E,B]=sum_s path_s D_mu,s cdag_s.
        b=sources[component];electric_connection=[];missing_connection=[]
        for e in range(8):
            delta=model.E[e]@(b@initial)-b@(reference.E[e]@initial)
            electric_connection.append(float(np.linalg.norm(delta)))
            missing_connection.append(float(np.linalg.norm(model.E[e]@v)))
        # Bare bandS rotates microsites with no Wilson E shift. G action on its
        # output is exactly charge change at fixed original electric labels.
        bare_gauss=[];correct_gauss=[]
        for axis in range(3):
            bare=model.sector.bilinear(model.geo['spin'][axis]);defect=0.
            state0=evolved[1].reshape(model.shape)
            for row,col in zip(*bare.nonzero()):
                amplitude=bare[row,col]*state0[col]
                dq=model.sector.q[row]-model.sector.q[col]
                defect+=float(np.linalg.norm(amplitude)**2)*float(dq@dq)
            bare_gauss.append(defect);correct_gauss.append(model.sector.gauss_error)
        # A phase on the tree path generator changes H by the full connection;
        # electric derivative/contact retain all g,R,r,Fock sectors.
        e=0;eps=2e-6;estate=model.sector.electric[:,e]
        phase=np.repeat(estate,model.shape[1]*model.shape[2]*model.shape[3])
        state=evolved[1];u=np.exp(1j*eps*phase)
        hp=u.conj()*(model.H@(u*state));hm=u*(model.H@(u.conj()*state))
        path_derivative=(hp-hm)/(2*eps)
        electric_derivative=float(np.linalg.norm(path_derivative-model.currents[e]@state))
        # Positive finite-cone force is an actual derivative, including frame
        # inverse-r² and beta*r, not mass held fixed during evolution.
        r=m.DEFAULTS['radial_nodes'];raw[name+'_electric_basis']=model.sector.electric
        raw[name+'_masks']=np.array(model.sector.masks)
        result['branches'][name]=dict(number=number,charge=number-4,boundary_flux=number-4,shape=model.shape,
            source_component=component,source_weights=weights,total_source_weight=sum(weights),
            source_Noether_derivative_errors=derivatives,Noether_H_commutator_state_errors=noether,
            Gauss_continuity_errors=continuity,Gauss_basis_error=model.sector.gauss_error,
            source_electric_connection_norms=electric_connection,
            missing_source_electric_connection_norms=missing_connection,
            bare_bandS_Gauss_violation_norm2=bare_gauss,Wilson_bandS_Gauss_error=correct_gauss,
            path_Peierls_derivative_error=electric_derivative,
            dynamics=[measure(model,w) for w in evolved],
            source_energy_cost=expect(v,model.H)-expect(initial,reference.H),
            angular_Gram_errors=[model.g['gram_error'],model.material['gram_error']],
            angular_cuts=dict(g=sorted(set(x[0] for x in model.g['labels'])),R=sorted(set(x[0] for x in model.material['labels']))))
        print(name,json.dumps(result['branches'][name]['dynamics'][-1]),flush=True)
        del model,sources,packets,evolved
    result['limits']=['NEW tree-Wilson law, not full periodic cube gauge model; no independent loop photons',
        'central-only sewing is declared; extra frame/taste species retained, no full RIGHT quotient',
        'fullFock256 across N; observed N3/4/5 only, no intermediate band/sea projection',
        'complete angular multiplets but finite g/R cut and three interior radial nodes; no convergence claim',
        'r is quantum positive matrix norm in declared cone metric; beta*r explicit NEW staggered mass supplier',
        'source charged energy cost includes electric/frame/internal costs; not Lorentz/Dirac inertial mass or g2',
        'addition/removal are opposite boundary charges, no proved C symmetry and no positron inference',
        'unprojected CAR exact before angular regulator; projected source CAR boundary leakage not silently assumed zero']
    (ROOT/(P+'result.json')).write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(ROOT/(P+'raw.npz'),**raw)
    print(json.dumps({'branches':list(result['branches']),'result':P+'result.json'}),flush=True)


if __name__=='__main__':main()
