"""Finite actual N4 operator Taylor observations, not global interval proof."""
from __future__ import annotations
from fractions import Fraction as F
from types import SimpleNamespace
from typing import Any
import numpy as np
from scipy.linalg import expm
import induced_phi_nonuniform_field_car as field
import induced_phi_nonuniform_fermionic_source_unitary as source
import induced_phi_nonuniform_gauss_quantum as gauss
from bulk_quantum_selection import BulkQuantumLaw
from bosonic_original_law_join import OriginalBosonicLaw
from rotation_band import RotationBand


def graph(native: dict[str, Any]) -> np.ndarray:
    sites = np.asarray(native['sites'],dtype=int)
    return np.stack([np.ravel_multi_index(((sites+np.eye(3,dtype=int)[axis])%4).T,(4,4,4)) for axis in range(3)],axis=1)


def components(fields: np.ndarray, links: np.ndarray, target: np.ndarray, a: float) -> tuple[float,float,float]:
    G = np.einsum('xmij,xnji->xmn',fields,fields).real/4
    quartic = float(np.sum(G*G)/a)
    W = 0.
    magnetic = 0.
    for x in range(64):
        for i in range(3):
            U = links[3*x+i]
            W += float(np.linalg.norm(fields[x]-U@fields[target[x,i]]@U.conj().T)**2)
        for i,j in ((0,1),(0,2),(1,2)):
            U = links[3*x+i]@links[3*target[x,i]+j]@links[3*target[x,j]+i].conj().T@links[3*x+j].conj().T
            magnetic += .7*float(np.linalg.norm(U-np.eye(4))**2)/2
    return quartic,W/40,magnetic


def h_jets(data: dict[str, Any], profile: dict[str, Any], slow: np.ndarray, fast: np.ndarray, C: np.ndarray, wrong: str = '') -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    native,geom = data['native'],data['geom']
    T = geom['M']/np.sqrt(8)
    B = native['frame']
    df = np.einsum('a,amij->mij',slow,geom['deltas'])/8
    Q,Qd = source.adapted_jet(B,df,T,geom['Q'])
    dPhi = df[None]+np.einsum('xma,aij->xmij',(fast@Q.T).reshape(64,4,15),T)
    ddPhi = 2*np.einsum('xma,aij->xmij',(fast@Qd.T).reshape(64,4,15),T)
    X = np.einsum('ea,aij->eij',C.reshape(192,15),T)
    h = np.zeros((256,256),complex)
    h1 = np.zeros_like(h)
    h2 = np.zeros_like(h)
    for x in range(64):
        sl = slice(4*x,4*x+4)
        h[sl,sl] = float(profile['onsite'])*B[0]
        h1[sl,sl] = float(profile['onsite'])*dPhi[x,0]
        h2[sl,sl] = float(profile['onsite'])*ddPhi[x,0]
    for path in field.native_paths(profile,native):
        if wrong == 'one_link' and len(path['edges']) == 1:
            continue
        if wrong == 'two_link' and len(path['edges']) == 2:
            continue
        edges = path['edges']
        U1 = sum((1j*X[e] for e in edges),np.zeros((4,4),complex))
        U2 = sum((-X[e]@X[e] for e in edges),np.zeros((4,4),complex))
        for k,e in enumerate(edges):
            for e2 in edges[k+1:]:
                U2 -= 2*X[e]@X[e2]
        if wrong == 'commuting_order':
            U2 = U1@U1
        x,y = int(path['source']),int(path['target'])
        c = path['coefficients']
        Bc = np.einsum('m,mij->ij',c,B)
        dx,dy = (np.einsum('m,mij->ij',c,dPhi[z]) for z in (x,y))
        ddx,ddy = (np.einsum('m,mij->ij',c,ddPhi[z]) for z in (x,y))
        if wrong == 'endpoint':
            dy,ddy = dx,ddx
        hop = path['seam']*Bc
        first = path['seam']*(dx+Bc@U1+U1@Bc+dy)/2
        second = path['seam']*(ddx+2*dx@U1+Bc@U2+U2@Bc+2*U1@dy+ddy)/2
        if wrong == 'AP_seam':
            hop,first,second = hop/path['seam'],first/path['seam'],second/path['seam']
        sx,sy = slice(4*x,4*x+4),slice(4*y,4*y+4)
        for matrix,value in ((h,hop),(h1,first),(h2,second)):
            matrix[sx,sy] += value
            matrix[sy,sx] += value.conj().T
    sign = -1 if profile['scale'] == 'heavy' else 1
    return sign*h,sign*h1,sign*h2


def sea_jets(h: np.ndarray, h1: np.ndarray, h2: np.ndarray) -> tuple[float,float,float]:
    energy,V = np.linalg.eigh(h)
    if not energy[127] < 0 < energy[128]:
        raise ArithmeticError('Actual occupied128 control has no gap')
    N,E = V[:,:128],V[:,128:]
    v = N.conj().T@h1@E
    second = float(np.trace(N.conj().T@h2@N).real+2*np.sum(abs(v)**2/(energy[:128,None]-energy[None,128:])))
    return float(energy[:128].sum()),float(np.trace(N.conj().T@h1@N).real),second


def run(data: dict[str, Any]) -> dict[str, Any]:
    rng = np.random.default_rng(22210105760)
    native,geom = data['native'],data['geom']
    T = geom['M']/np.sqrt(8)
    target = graph(native)
    a = float(native['a'])
    slow = rng.normal(size=45)
    slow *= .41/np.linalg.norm(slow)
    fast = rng.normal(size=(64,45))
    fast -= fast.mean(axis=0)
    fast *= .53/np.linalg.norm(fast)
    C = rng.normal(size=(64,3,15))
    C *= .47/np.linalg.norm(C)
    B = native['frame']
    df = np.einsum('a,amij->mij',slow,geom['deltas'])/8
    Q,Qd = source.adapted_jet(B,df,T,geom['Q'])
    f0 = np.einsum('xma,aij->xmij',(fast@Q.T).reshape(64,4,15),T)
    X = np.einsum('ea,aij->eij',C.reshape(192,15),T)
    O,Od = gauss.orbit(B,T),gauss.orbit(df,T)
    H = O.T@O/8
    dH = (Od.T@O+O.T@Od)/8
    L = f0[:,None]-f0[target]-1j*(X.reshape(64,3,1,4,4)@B-B@X.reshape(64,3,1,4,4))
    # Independent literal derivative and frozen ALLq real-space Hessian.
    Lnorm = float(np.sum(abs(L)**2)/40)
    diff = fast[:,None]-fast[target]
    colour = C.reshape(192,15)
    align_hessian = float(np.sum(diff**2)/80+np.einsum('ea,ab,eb->',colour,H,colour)/10)
    colour_forcing = float(np.einsum('ea,ab,eb->',colour,dH,colour)/10)
    curl = np.asarray([colour[3*x+i]+colour[3*target[x,i]+j]-colour[3*target[x,j]+i]-colour[3*x+j]
                       for x in range(64) for i,j in ((0,1),(0,2),(1,2))])
    magnetic_hessian = .175*float(np.sum(curl**2))
    fhat = (np.fft.fftn(fast.reshape(4,4,4,45),axes=(0,1,2))/8).reshape(64,45)
    chat = (np.fft.fftn(C.reshape(4,4,4,45),axes=(0,1,2))/8).reshape(64,45)
    allq_hessian = 0.
    for index,q in enumerate(gauss.QINTEGER):
        z = np.r_[fhat[index],chat[index]]
        K = gauss.build_geometry(native,geom,q)['Kfast90']
        allq_hessian += float(np.vdot(z,K@z).real/2)
    allq_hessian_defect = abs(allq_hessian-align_hessian-magnetic_hessian)
    quartic_hessian = float((slow@geom['K_literal']@slow+np.einsum('xa,ab,xb->',fast,geom['K_literal'],fast))/2)
    phi1 = df[None]+f0
    G0 = np.einsum('mij,nji->mn',B,B).real/4
    G1 = (np.einsum('mij,xnji->xmn',B,phi1)+np.einsum('xmij,nji->xmn',phi1,B)).real/4
    G2 = np.einsum('xmij,xnji->xmn',phi1,phi1).real/4
    quartic_direct = float(np.sum(G1**2+2*G0*G2)/a)
    quartic_first = float(2*np.einsum('mn,xmn->',G0,G1)/a)
    basequart = float(64*np.sum(G0*G0)/a)
    primitive_rows = []
    donor_error = 0.
    potential_law = OriginalBosonicLaw(4,RotationBand(a=a),1.,1.,1.)
    raw_graph: Any = SimpleNamespace(volume=64,target=target)
    for step in (.04,.02,.01):
        fields,_,links,_ = source.field_ray(data,step,slow,fast,C)
        v4,va,vm = components(fields,links,target,a)
        W,P = BulkQuantumLaw.link_defects(raw_graph,fields,links.reshape(64,3,4,4))
        literal = sum(potential_law.field_potential_at(fields[x],[(fields[target[x,i]],links[3*x+i]) for i in range(3)]) for x in range(64))
        donor_error = max(donor_error,abs(literal-(v4+va)),abs(.7*P-vm),abs(W/40-va))
        v4res = v4-basequart-quartic_first*step-quartic_hessian*step**2
        vares = va-align_hessian*step**2-colour_forcing*step**3
        vmres = vm-magnetic_hessian*step**2
        sn = step*np.linalg.norm(slow)
        fn = step*np.linalg.norm(fast)
        cn = step*np.linalg.norm(C)
        fastnorm = fn+cn
        caps = (2*sn**3+sn**4/64+(5+64*8192)*sn*fn**2+12*fn**3+fn**4,
                .3*fastnorm**3+.1*fastnorm**4+sn**2*cn**2/80,
                4.2*cn**3+1.4*cn**4)
        if any(abs(value)>cap for value,cap in zip((v4res,vares,vmres),caps,strict=True)):
            raise ArithmeticError('Observed actual potential violates derived bound')
        primitive_rows.append({'ray_step':step,'coordinate_radius':float(step*np.sqrt(np.linalg.norm(slow)**2+np.linalg.norm(fast)**2+np.linalg.norm(C)**2)),'quartic_residual_after_exact_degree2':v4res,'alignment_residual_after_actual_degree2_and_retained_colour_t3':vares,'plaquette_residual_after_actual_degree2':vmres,'bounds':caps,'scalar_tail_scope':'Finite observations outside tiny proof collar; global inequalities are independently derived. Physical Lambda factors kept symbolic, not multiplied into floats.'})
    # Deliberately violate zero-mean: a slow/fast quadratic then survives.
    uniform = rng.normal(size=45)
    uniform /= np.linalg.norm(uniform)
    fast_wrong = fast+uniform[None]/8
    fw = np.einsum('xma,aij->xmij',(fast_wrong@Q.T).reshape(64,4,15),T)
    js = (np.einsum('mij,nji->mn',B,df)+np.einsum('mij,nji->mn',df,B)).real/4
    jf = (np.einsum('mij,xnji->xmn',B,fw)+np.einsum('xmij,nji->xmn',fw,B)).real/4
    crossG = (np.einsum('mij,xnji->xmn',df,fw)+np.einsum('xmij,nji->xmn',fw,df)).real/4
    wrong_slowfast = float(np.sum(2*js*jf+2*G0*crossG)/a)
    correct_cross = float(np.sum(2*js*(np.einsum('mij,xnji->xmn',B,f0)+np.einsum('xmij,nji->xmn',f0,B)).real/4+2*G0*(np.einsum('mij,xnji->xmn',df,f0)+np.einsum('xmij,nji->xmn',f0,df)).real/4)/a)
    # Ordinary SU4 face order is physically detectable at a finite noncommuting point.
    point = .3
    links = np.asarray([expm(1j*point*x) for x in X])
    x,i,j = 0,0,1
    correctword = links[3*x+i]@links[3*target[x,i]+j]@links[3*target[x,j]+i].conj().T@links[3*x+j].conj().T
    wrongword = links[3*target[x,i]+j]@links[3*x+i]@links[3*target[x,j]+i].conj().T@links[3*x+j].conj().T
    wrongword_defect = float(np.linalg.norm(correctword-wrongword))
    matter_rows = []
    adverse = {key:0. for key in ('one_link','two_link','endpoint','commuting_order','AP_seam')}
    allsea_second_unit = 0.
    charged_only_second = 0.
    for profile in data['profiles']:
        h,h1,h2 = h_jets(data,profile,slow,fast,C)
        mean = np.broadcast_to(B,(64,4,4,4))
        identity = np.broadcast_to(np.eye(4),(192,4,4))
        mean_error = float(np.linalg.norm(h-field.assemble(profile,native,mean,identity)))
        E0,E1,E2 = sea_jets(h,h1,h2)
        allsea_second_unit += profile['load']*E2
        if profile['scale'] != 'heavy':
            charged_only_second += profile['load']*E2
        ray = []
        for step in (.02,.01,.005):
            fields,fd,links,ld = source.field_ray(data,step,slow,fast,C)
            actual = field.assemble(profile,native,fields,links)
            _,exactdh = source.h_jet(profile,native,fields,fd,links,ld)
            energy = float(np.linalg.eigvalsh(actual)[:128].sum())
            remainder = energy-E0-step*E1-step**2*E2/2
            htail = float(np.linalg.norm(actual-h-step*h1-step**2*h2/2))
            ray.append({'step':step,'literal_filled_sea':energy,'Taylor_remainder':remainder,'remainder_divided_step_cubed':remainder/step**3,'full_matrix_Taylor_remainder_HS':htail,'actual_ray_derivative_minus_first_and_second_order_HS':float(np.linalg.norm(exactdh-h1-step*h2))})
        for wrong in adverse:
            wh,wh1,wh2 = h_jets(data,profile,slow,fast,C,wrong)
            adverse[wrong] = max(adverse[wrong],float(np.linalg.norm(wh-h)+np.linalg.norm(wh1-h1)+np.linalg.norm(wh2-h2)))
        if mean_error>1e-10 or any(abs(row['Taylor_remainder'])>1e6*row['step']**3 for row in ray):
            raise ArithmeticError('Actual finite matter Taylor control failed')
        if not ray[-1]['full_matrix_Taylor_remainder_HS']<ray[0]['full_matrix_Taylor_remainder_HS']/40:
            raise ArithmeticError('Independent h2 does not give cubic Taylor remainder')
        matter_rows.append({'role':profile['role'],'copies':profile['load'],'signed_heavy':profile['scale']=='heavy','physical_energy_multiplier':'2^8192' if profile['scale']=='heavy' else '1','mean_literal_operator_defect':mean_error,'E0_E1_E2_unscaled':[E0,E1,E2],'ray_Taylor_observations':ray,'first_HS':float(np.linalg.norm(h1)),'second_HS':float(np.linalg.norm(h2))})
    heavysecond = F.from_float(matter_rows[0]['E0_E1_E2_unscaled'][2])
    chargedsecond = F.from_float(charged_only_second)
    wrong_heavy = abs((F(2)**8192-1)*heavysecond)
    wrong_aux = abs(F(62)*F.from_float(matter_rows[2]['E0_E1_E2_unscaled'][2]))
    if max(abs(Lnorm-align_hessian),abs(quartic_direct-quartic_hessian),abs(correct_cross),allq_hessian_defect)>1e-10 or donor_error>1e-8:
        raise ArithmeticError('Actual primitive law/Hessian binding failed')
    if min(abs(wrong_slowfast),abs(colour_forcing),wrongword_defect,*adverse.values())<=1e-8 or not wrong_heavy or not wrong_aux:
        raise ArithmeticError('Adverse primitive/matter Taylor control blind')
    allq_control = {'literal_all64q_fast_Hessian_defect':allq_hessian_defect,
                   'native_all64q_fast_quadratic_action':allq_hessian,
                   'literal_alignment_plus_ordered_plaquette_quadratic_action':align_hessian+magnetic_hessian,
                   'rootXi_potential_piece':colour_forcing,
                   'rootXi_kinetic_piece_is_separate':True}
    return {'summary':{'primitive_Hessian_binding_max_defect':max(abs(Lnorm-align_hessian),abs(quartic_direct-quartic_hessian),allq_hessian_defect),'independent_native_fullform_potential_max_defect':donor_error,'actual_zero_mean_slow_fast_quadratic':correct_cross,'wrong_nonzero_mean_slow_fast_quadratic':wrong_slowfast,'retained_linear_s_colour_fast_Hessian_t3':colour_forcing,'wrong_SU4_face_order_matrix_defect':wrongword_defect,'wrong_matter_jet_defects':adverse,'wrong_omitted_physical_heavy_scale_nonzero':bool(wrong_heavy),'wrong_auxiliary_load1_not63_nonzero':bool(wrong_aux)},'all64q_primitive_Hessian_control':allq_control,'primitive_rays':primitive_rows,'matter_profiles':matter_rows,'physical_scale_controls':{'heavy_omission_second_energy_jet_exact_from_observed_binary_coefficient':str(wrong_heavy),'auxiliary_omission_second_energy_jet_exact_from_observed_binary_coefficient':str(wrong_aux),'correct_full65_second_energy_jet':'2^8192*('+str(heavysecond)+')+('+str(chargedsecond)+')','allunit_wrong_second_jet':allsea_second_unit,'finite_coefficient_is_not_an_interval':True},'control_dimensions':{'N':4,'V':64,'normal':5760,'slow':45,'fastPhi_zero_mean':2835,'colour':2880,'physical_oriented_edges':192,'ordered_faces':192,'ambient_with_local_angles':6720,'h_modes':256,'occupied':128,'independent_copies':65},'scope':'Actual native pure potential consumers and independently differentiated h0/h1/h2; observations are ordinary floating finite controls, some outside tiny collar to discriminate. Global support+3R and outward weighted norm proof lives in the separate derivation; no finite-control agreement is promoted to global certification.'}
