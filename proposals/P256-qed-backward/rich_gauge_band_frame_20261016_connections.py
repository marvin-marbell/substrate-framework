"""Fresh-state quantum field/source derivatives, angular payment and band basis."""
import json
from pathlib import Path
import numpy as np
from scipy import sparse as sp
from scipy.sparse.linalg import expm_multiply
import rich_gauge_band_frame_20261016_model as m

ROOT=Path(__file__).resolve().parent
P='rich_gauge_band_frame_20261016_'


def main():
    ref=m.Model(4);v=ref.preparation();result={};raw={}
    q=np.array([.11,-.08,.06]);k=np.array(m.DEFAULTS['k']);model=m.Model(5)
    b=model.source_from(ref,0,True);packet=b@v;packet/=np.linalg.norm(packet)
    # Deliberately physical electric pulse, not a bare g/S relabelling.
    phase=np.repeat(model.sector.electric[:,0],np.prod(model.shape[1:]))
    state=np.exp(.37j*phase)*packet;raw['pulse_input']=state
    evolved=expm_multiply(-.19j*model.H,state,traceA=-.19j*model.H.diagonal().sum());raw['pulse_output']=evolved
    state=evolved
    ghigh=np.array([x[0]==1.5 for x in model.g['labels']]);rhigh=np.array([x[0]==1 for x in model.material['labels']])
    probability=abs(state.reshape(model.shape))**2
    result['angular_cut_payment']=dict(g_highest_multiplet_weight=float(probability[:,ghigh,:,:].sum()),
        R_highest_multiplet_weight=float(probability[:,:,rhigh,:].sum()),
        g_complete_blocks=[.5,1.5],R_complete_blocks=[0,1],no_convergence_claim=True)
    result['source_EOM']={}
    pieces=['HE','Hradial','Hg','HR','Hmass','Hrelative']
    for name in pieces:
        a=getattr(model,name)@(b@v)-b@(getattr(ref,name)@v)
        result['source_EOM'][name]=float(np.linalg.norm(a));raw['source_EOM_'+name]=a
    total=model.H@(b@v)-b@(ref.H@v)
    raw['source_EOM_total']=total
    result['source_EOM']['total_norm']=float(np.linalg.norm(total))
    result['source_EOM']['source_H0_without_electric_frame_defect']=float(np.linalg.norm(raw['source_EOM_HE']+raw['source_EOM_Hg']))
    # Full quantum angular field vertex; no clamped O(g^-1 R) values.
    axis=[]
    for a in range(3):
        axis.append(sum((model.relative_terms[(i*3+a)*3+ell]/(-2*np.cos(k[a])) for i in range(3) for ell in range(3)),sp.csr_matrix(model.H.shape,dtype=complex)))
    delta=sum((-2*np.cos(k[a]+q[a])+2*np.cos(k[a]))*(axis[a]@state) for a in range(3))
    mid=sum(2*np.sin(q[a]/2)*2*np.sin(k[a]+q[a]/2)*(axis[a]@state) for a in range(3))
    wrong=sum(2*np.sin(q[a]/2)*2*np.sin(k[a])*(axis[a]@state) for a in range(3))
    result['quantum_midpoint']=dict(correct_error=float(np.linalg.norm(delta-mid)),wrong_q0_error=float(np.linalg.norm(delta-wrong)))
    # Actual k-dependent embedded<->periodic orbital basis.  Density rho must
    # connect the two momenta, and EVERY H component, including HE, transforms.
    sites=model.geo['sites'];occupation=model.sector.n
    tk=np.exp(1j*(occupation@(sites@k)));tp=np.exp(1j*(occupation@(sites@(k+q))))
    tk=np.repeat(tk,np.prod(model.shape[1:]));tp=np.repeat(tp,np.prod(model.shape[1:]));rho=tp.conj()*tk
    hk=model.H
    hdelta=sum(((-2*np.cos(k[a]+q[a])+2*np.cos(k[a]))*axis[a] for a in range(3)),sp.csr_matrix(hk.shape,dtype=complex))
    hp=hk+hdelta
    lhs=tp.conj()*(hp@(tp*(rho*state)))-rho*tk.conj()*(hk@(tk*state))
    rhs=tp.conj()*sum(2*np.sin(q[a]/2)*2*np.sin(k[a]+q[a]/2)*(axis[a]@(tk*state)) for a in range(3))
    wrong_lhs=tp.conj()*(hp@(tp*state))-tk.conj()*(hk@(tk*state))
    wrong_rhs=tk.conj()*sum(2*np.sin(q[a]/2)*2*np.sin(k[a]+q[a]/2)*(axis[a]@(tk*state)) for a in range(3))
    result['band_basis_connection']=dict(correct_ward_error=float(np.linalg.norm(lhs-rhs)),wrong_same_k_identity_density_error=float(np.linalg.norm(wrong_lhs-wrong_rhs)),density_connection_nonidentity_norm=float(np.linalg.norm((rho-1)*state)))
    # Force dH/dr: differentiate local potential/centrifugal/mass pieces at
    # fixed canonical radial Hilbert measure u, including g/r connection.
    p=model.parameters;r=model.rnodes;eps=1e-6;fi,gi,mi,ri=[sp.eye(n) for n in model.shape]
    def local_h(shift):
        x=r+shift
        potential=p['radial_stiffness']*(x-p['radial_centre'])**2/2+3/(8*p['radial_mass']*x**2)
        return m.tensor(fi,gi,mi,sp.diags(potential))+m.tensor(fi,model.g['casimir'],mi,sp.diags(1/(2*p['radial_mass']*x**2)))+p['root_coupling']*m.tensor(model.sector.bilinear(model.geo['beta']),gi,mi,sp.diags(x))
    finite=-((local_h(eps)-local_h(-eps))@state)/(2*eps)
    result['mass_supplier_force_derivative_error']=float(np.linalg.norm(finite-model.force@state))
    result['pulse_observables']=dict(J2=sum(float(np.linalg.norm(a@state)**2) for a in model.J),
        r=float(np.vdot(state,model.r@state).real),current=[float(np.vdot(state,a@state).real) for a in model.currents],
        torque_g=[float(np.vdot(state,a@state).real) for a in model.torque_g],
        torque_R=[float(np.vdot(state,a@state).real) for a in model.torque_R])
    raw['quantum_midpoint_delta']=delta;raw['quantum_midpoint_wrong']=wrong
    (ROOT/(P+'connections_result.json')).write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(ROOT/(P+'connections_raw.npz'),**raw)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
