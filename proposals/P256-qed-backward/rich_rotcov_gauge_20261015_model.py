"""NEW radial Wilson/London basis with electric Berry and Born-Huang terms.

This is a gauge-including Galerkin construction, not a subtraction/fitted
stiffness. Physical original-A current and finite covariant dipole current are
both retained so the remaining projection defect cannot be hidden.
"""
import numpy as np
from scipy.sparse import eye
from rich_rotcov_induced_20261015_model import oscillator, quadrature, profiles, comm
from rich_rotcov_gauge_20261015_electrostatics import cross_density, shell_kernel
from rich_rotcov_gauge_20261015_sparse import SparseRoundoff
from rich_rotcov_gauge_20261015_primitive import build as primitive_build


def london_profiles(points,p):
    original,metrics=profiles(points,p['field_width'],p['epsilon'])
    chi=np.zeros((len(points),6));gradient=np.zeros_like(original)
    if p.get('gauge_profile')=='constant':
        original[:]=0;original[:,:3,:]=np.eye(3)[None,:,:]
        chi[:,:3]=points;gradient[:,:3,:]=np.eye(3)[None,:,:]
    else:
        t,wt=np.polynomial.legendre.leggauss(48);t=(t+1)/2;wt=wt/2
        r2=np.sum(points*points,axis=1);f=np.exp(-r2[:,None]*t[None,:]**2/(2*p['field_width']**2))
        scalar=f@wt;derivative=-(f*t[None,:]**2)@wt/p['field_width']**2
        chi[:,3:]=scalar[:,None]*points/np.sqrt(metrics[3])
        gradient[:,3:,:]=(scalar[:,None,None]*np.eye(3)[None,:,:]+derivative[:,None,None]*points[:,:,None]*points[:,None,:])/np.sqrt(metrics[3])
    return original,original-gradient,chi,gradient


def build(p):
    # Pure constructor only: fresh whole shell, raw frame, neutral supplier
    # Coulomb and primitive E. No predecessor state, response or CLI is reused.
    sparse=SparseRoundoff();csr_matrix=sparse.csr;kron=sparse.kron
    model=primitive_build(p,sparse);dims=model['dims'];ns=dims[3];nf=dims[4];body=np.prod(dims[:3]);e=p['charge']*p.get('gauge_coupling',1.)
    body_currents=[a.copy() for a in model['current']]
    sup=model['supplier'];osc=oscillator(p['supplier_cut']);n=len(osc['basis']);identity=np.eye(n)
    qs=[quadrature(osc,length,p['radial_quadrature']) for length in p['supplier_lengths']]
    def emat(op):return kron(eye(body),kron(csr_matrix(op),eye(nf)),format='csr')
    def mix(op,field):return kron(eye(body),kron(csr_matrix(op),csr_matrix(field)),format='csr')
    linear=[];quadratic=[];direct_linear=[];direct_cross=[];direct_contact=[];chis=[];chi_squares=[]
    for f,q in enumerate(qs):
        original,radial,chi,grad=london_profiles(q['points'],p)
        linear.append(q['linear'](radial,p['supplier_masses'][f],-e))
        direct_linear.append(q['linear'](original,p['supplier_masses'][f],-e))
        quadratic.append(e**2*q['integrate'](np.einsum('xai,xbi->xab',radial,radial))/(2*p['supplier_masses'][f]))
        direct_cross.append(e**2*q['integrate'](np.einsum('xai,xbi->xab',original,radial))/p['supplier_masses'][f])
        direct_contact.append(e**2*q['integrate'](np.einsum('xai,xbi->xab',original,original))/p['supplier_masses'][f])
        chis.append(q['integrate'](chi));chi_squares.append(q['integrate'](np.einsum('xa,xa->x',chi,chi)))
    def pair(a,b):return np.kron(a,identity)+np.kron(identity,b)
    chi_total=[pair(chis[0][:,:,a],chis[1][:,:,a]) for a in range(6)]
    square=pair(chi_squares[0],chi_squares[1])+2*sum(np.kron(chis[0][:,:,a],chis[1][:,:,a]) for a in range(6))
    born_huang=square-sum(a@a for a in chi_total)
    hm=csr_matrix(model['H'].shape,dtype=complex);electric=model['field_energy'].copy();h=model['H'].copy()
    current=[a.copy() for a in model['current']];contact=[[a.copy() for a in row] for row in model['contact']]
    for a in range(6):
        ba=pair(linear[0][a],linear[1][a]);hm+=mix(ba,model['field']['Q'][a])
        electric-=e*mix(chi_total[a],model['field']['P'][a])
        current[a]-=emat(pair(direct_linear[0][a],direct_linear[1][a]))
        for b in range(6):
            coefficient=pair(quadratic[0][:,:,a,b],quadratic[1][:,:,a,b]);hm+=mix(coefficient,model['field']['QQ'][a][b])
            current[a]-=mix(pair(direct_cross[0][:,:,a,b],direct_cross[1][:,:,a,b]),model['field']['Q'][b])
            contact[a][b]+=emat(pair(direct_contact[0][:,:,a,b],direct_contact[1][:,:,a,b]))
    electric+=e**2*emat(square)/2
    h+=hm+electric-model['field_energy']
    # Complete same-kernel longitudinal body/supplier/central-ion density.
    g=quadrature(oscillator(p['geometry_cut']),p['length'],p['radial_quadrature']);cross=cross_density(p,oscillator(p['geometry_cut']),g,qs)
    core_number=np.real(model['N'].diagonal()).reshape(dims)[:,0,0,0,0]-2
    electrostatic=csr_matrix(h.shape,dtype=complex)
    for bi,radius in enumerate(p['radii']):
        projector=np.zeros((dims[1],dims[1]));projector[bi,bi]=1
        full=kron(csr_matrix(np.diag(core_number)),kron(csr_matrix(projector),kron(csr_matrix(cross[bi]),eye(nf))),format='csr')
        # Replace old unsoftened collective self term by the same softened
        # density kernel used in ALL new longitudinal cross terms.
        correction=p['charge']**2*(shell_kernel(np.array([radius]),radius,p['coulomb_softening'])[0]-1/radius)/(8*np.pi*p['epsilon'])
        full+=kron(csr_matrix(np.diag(core_number**2)),kron(csr_matrix(projector),eye(dims[2]*ns*nf)),format='csr')*correction
        electrostatic+=full
    h+=electrostatic
    model['H']=h.tocsr();model['current']=current;model['contact']=contact;model['field_energy']=electric.tocsr();model['longitudinal_cross']=electrostatic
    model['physical_P']=[model['P'][a]-e*emat(chi_total[a]) for a in range(6)]
    model['dipole']=[-e*emat(a) for a in chi_total];model['chi_total']=chi_total;model['born_huang']=born_huang
    model['direct_original_current']=current
    # Co-moving-basis functional source includes the time-dependent connection.
    # It is NOT set to zero just because a static gradient is absorbed in basis.
    covariant=[]
    for a in range(6):
        derivative=emat(pair(linear[0][a],linear[1][a]))
        for b in range(6):derivative+=mix(pair(quadratic[0][:,:,a,b]+quadratic[0][:,:,b,a],quadratic[1][:,:,a,b]+quadratic[1][:,:,b,a]),model['field']['Q'][b])
        covariant.append(body_currents[a]-derivative+1j*comm(h,model['dipole'][a]))
    model['covariant_current']=covariant
    # Physical angular parts are U† L U and U†(Q cross Pi)U. Their summed
    # correction cancels exactly; they need not give commuting cut PVMs.
    physical_sup=[[v.copy() for v in family] for family in model['SL']]
    physical_field=[[v.copy() for v in family] for family in model['F']]
    for family in range(2):
        for axis in range(3):
            b=3*family+(axis+1)%3;c=3*family+(axis+2)%3
            physical_field[family][axis]-=e*(mix(chi_total[c],model['field']['Q'][b])-mix(chi_total[b],model['field']['Q'][c]))
            for f in range(2):
                cb=np.kron(chis[f][:,:,b],identity) if f==0 else np.kron(identity,chis[f][:,:,b])
                cc=np.kron(chis[f][:,:,c],identity) if f==0 else np.kron(identity,chis[f][:,:,c])
                physical_sup[f][axis]+=e*(mix(cc,model['field']['Q'][b])-mix(cb,model['field']['Q'][c]))
    model['physical_SL']=physical_sup;model['physical_F']=physical_field;model['gauge_matter_coupling']=hm;model['parameters']=p
    model['chi_flavours']=chis;model['supplier_quadratures']=qs
    model['sparse_roundoff']=sparse.snapshot()
    return model




def shifted_h(model,external):
    """Fixed physical-basis original-A source, including contact, independently."""
    h=model['H'].copy()
    for a in range(6):
        h-=external[a]*model['direct_original_current'][a]
        for b in range(6):h+=external[a]*external[b]*model['contact'][a][b]/2
    return h
