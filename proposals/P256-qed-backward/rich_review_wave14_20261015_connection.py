"""P i[H,D] P versus i[PHP,PDP]: actual continuum-product/source owner audit."""
import json
from pathlib import Path
import sys
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
from rich_rotcov_gauge_20261015_model import build,london_profiles


def norm(a):return float(sparse.linalg.norm(a) if sparse.issparse(a) else np.linalg.norm(a))
def comm(a,b):return a@b-b@a


def consume(p,reference,label):
    m=build(p);h=m['H'];body=int(np.prod(m['dims'][:3]));nf=m['dims'][-1];ident=np.eye(4);charge=p['charge']*p.get('gauge_coupling',1.)
    if reference is None:
        selected=np.flatnonzero((abs(m['N'].diagonal().real-2)<1e-8)&(abs(m['J'][2].diagonal().real)<1e-8));es,vs=eigsh(h[selected][:,selected],k=1,which='SA',tol=1e-11);reference=np.zeros(h.shape[0],complex);reference[selected]=vs[:,0]
    shape=h.shape;radial_current=[sparse.csr_matrix(shape,dtype=complex) for _ in range(6)];original_current=[x.copy() for x in radial_current];full_product=[x.copy() for x in radial_current]
    for f,q in enumerate(m['supplier_quadratures']):
        original,radial,chi,gradient=london_profiles(q['points'],p);mass=p['supplier_masses'][f]
        linear=[q['linear'](profile,mass,-charge) for profile in [radial,original,gradient]]
        cross=[charge**2/mass*q['integrate'](np.einsum('xai,xbi->xab',profile,radial)) for profile in [radial,original,gradient]]
        def embed(op,field):
            pair=np.kron(op,ident) if f==0 else np.kron(ident,op)
            return sparse.kron(sparse.eye(body),sparse.kron(sparse.csr_matrix(pair),sparse.csr_matrix(field)),format='csr')
        for a in range(6):
            for target,li,cr in zip([radial_current,original_current,full_product],linear,cross):
                target[a]-=embed(li[a],np.eye(nf))
                for b in range(6):target[a]-=embed(cr[:,:,a,b],m['field']['Q'][b])
    # P i[H,D] P has only the continuum covariant kinetic commutator:
    # -e {p+e AP, grad chi}/(2m). All coordinate potentials AND electric
    # (Pi-e chi)^2 commute chi BEFORE projection, but cut matrices need not.
    channels=[]
    for a in range(6):
        body_current=m['current'][a]-original_current[a];radial_total=body_current+radial_current[a]
        corrected=radial_total+full_product[a];cut_dipole=1j*comm(h,m['dipole'][a]);complement=full_product[a]-cut_dipole
        ccr_only=radial_total-1j*comm(h,m['P'][a]);old_eom=m['current'][a]-1j*comm(h,m['physical_P'][a])
        potential_cut=1j*comm(m['longitudinal_cross'],m['dipole'][a]);electric_cut=1j*comm(m['field_energy'],m['dipole'][a])
        supplier_cut=1j*comm(m['supplier_energy']+m['gauge_matter_coupling'],m['dipole'][a])
        channels.append(dict(family='TE' if a<3 else 'TM',axis=a%3,current_owner_vs_continuum_product_operator=norm(m['current'][a]-corrected),current_owner_vs_continuum_product_ground=norm((m['current'][a]-corrected)@reference),old_cut_commutator_source_ground=norm((m['current'][a]-m['covariant_current'][a])@reference),actual_complement_operator=norm(complement),actual_complement_ground=norm(complement@reference),complement_vs_reported_discrepancy=norm((m['current'][a]-m['covariant_current'][a])-complement),projected_potential_comm_ground=norm(potential_cut@reference),projected_electric_comm_ground=norm(electric_cut@reference),supplier_comm_missing_full_product_ground=norm((full_product[a]-supplier_cut)@reference),physical_P_cut_Heisenberg_defect_ground=norm(old_eom@reference),canonical_P_CCR_only_defect_ground=norm(ccr_only@reference),EOM_component_identity_operator=norm(old_eom-ccr_only-complement),remaining_Heisenberg_defect_after_only_dipole_product_correction=norm((m['current'][a]-(1j*comm(h,m['P'][a])+full_product[a]))@reference)))
    return dict(profile=label,dimension=h.shape[0],channels=channels,interpretation='No H/current fitted, no stiffness subtraction. Original-A continuum current is source owner. P i[H,D]P is projected from continuum gradients BEFORE cut; omitted complement in i[PHP,PDP] exactly witnesses old source discrepancy. Canonical-field CCR/product projection is a separate residual in a naively cut Heisenberg commutator. Neither residual is declared fundamental physics or gauge-complete Maxwell.' )


def main():
    inp,out=map(Path,sys.argv[1:3]);req=json.loads(inp.read_text());p=json.loads(Path(req['owner_input']).read_text());data=np.load(req['owner_npz'])
    result=dict(kind='INDEPENDENT_CONTINUUM_CONNECTION_COMPLEMENT_WITNESS_SAME_NEW_LAW',definition='Delta=P i[H,D] P-i[PHP,PDP]=i(P H(1-P)D P-P D(1-P)H P). D=-e chi. Direct source P Udag J_original U P=I_body-dH_AP/dq+P i[H,D]P. Operator products are formed BEFORE cut.',London=consume(p,data['ground'],'actual_London'),constant_gradient=consume(dict(p,gauge_profile='constant'),None,'constant_original_A_gradient'),scope='Same exact author4 cutoff2 law; NEW source/operator consumer, no author edits, no old producer replay or new coupling. Current covariance defect is an operator-product construction issue, not an irreducible physical obstruction.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
