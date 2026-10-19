"""NEW per-copy N4 Phi45/local-SU4/U1 finite-CAR93 experiment.

This new nonlinear law extends the dynamic-Phi ordered endpoint operator by
exp(i charge sum(theta_edges)). TEST couplings and a negative-band finite
Slater reference are inputs, not an interacting vacuum or campaign root.
No Pi, Coulomb, Maxwell kernel, clipping, toron subtraction or fitted pole.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import induced_phi_nonuniform_field_car as old
import numpy as np
from scipy.linalg import expm
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
SOURCE_BYTES = {**old.CAPTURED, Path(__file__).name: Path(__file__).read_bytes()}
Q_SELECTED = ((0, 0, 0), (1, 0, 0), (1, 2, 0), (2, 0, 0))
U1 = np.array([60, 76, 92])
OLD90 = np.r_[np.arange(45), np.arange(45,60), np.arange(61,76), np.arange(77,92)]


def charge(profile: dict[str, Any]) -> int:
    return 0 if profile['scale'] == 'heavy' else 1


def generators(profile: dict[str, Any], geom: dict[str, Any], scalar_scale: float = 1.) -> np.ndarray:
    return np.concatenate((geom['M']/np.sqrt(8), (charge(profile)*scalar_scale*np.eye(4))[None]))


def assemble(profile: dict[str, Any], native: dict[str, Any], fields: np.ndarray,
             links: np.ndarray, theta: np.ndarray, charge_override: int | None = None,
             omit_second_phase: bool = False) -> np.ndarray:
    """Literal NEW nonlinear h, arbitrary Phi endpoints and noncommuting links."""
    c = charge(profile) if charge_override is None else charge_override
    h = np.zeros((256,256), complex)
    for x in range(64):
        h[4*x:4*x+4,4*x:4*x+4] = float(profile['onsite'])*fields[x,0]
    for path in old.native_paths(profile,native):
        x,y = int(path['source']),int(path['target'])
        ordered = np.eye(4,dtype=complex)
        for j,edge in enumerate(path['edges']):
            phase = 1. if omit_second_phase and j == 1 else np.exp(1j*c*theta[edge])
            ordered = ordered @ (phase*links[edge])
        fx,fy = np.einsum('m,mij->ij',path['coefficients'],fields[x]),np.einsum('m,mij->ij',path['coefficients'],fields[y])
        hop = path['seam']*(fx@ordered+ordered@fy)/2
        h[4*x:4*x+4,4*y:4*y+4] += hop
        h[4*y:4*y+4,4*x:4*x+4] += hop.conj().T
    return (-1 if profile['scale']=='heavy' else 1)*h


def vertices(profile: dict[str, Any], native: dict[str, Any], geom: dict[str, Any],
             terms: list[dict[str, Any]], q: np.ndarray, scalar_scale: float = 1.) -> np.ndarray:
    t = generators(profile,geom,scalar_scale)
    v = np.zeros((64,93,4,4),complex)
    v[:,:45] = float(profile['onsite'])*geom['deltas'][:,0][None]/8
    for term in terms:
        d,axis,length = term['displacement'],term['axis'],term['length']
        phase,eq = np.exp(1j*old.THETA@d),np.exp(1j*q@d)
        sf = sum(np.exp(1j*q[axis]*j) for j in range(length))/8
        dh,hop = term['dPhi'],term['hop']
        v[:,:45] += phase[:,None,None,None]*(1+eq)*dh[None]/16
        v[:,:45] += (phase.conj()/eq)[:,None,None,None]*(1+eq)*dh.conj().transpose(0,2,1)[None]/16
        cv = .5j*(hop@t+t@hop)*sf
        cr = -.5j*(hop.conj().T@t+t@hop.conj().T)*sf
        sl = slice(45+16*axis,61+16*axis)
        v[:,sl] += phase[:,None,None,None]*cv[None]+(phase.conj()/eq)[:,None,None,None]*cr[None]
    return (-1 if profile['scale']=='heavy' else 1)*v


def contact(profile: dict[str, Any], geom: dict[str, Any], terms: list[dict[str, Any]],
            q: np.ndarray, projector: np.ndarray) -> np.ndarray:
    """Complete ordered D2h[-q,+q], including scalar mixed and scalar-scalar."""
    t = generators(profile,geom)
    anti = t[:,None]@t[None]+t[None]@t[:,None]
    result = np.zeros((93,93),complex)
    sign = -1 if profile['scale']=='heavy' else 1
    for term in terms:
        axis,length,d = term['axis'],term['length'],term['displacement']
        dh,hop = term['dPhi'],term['hop']
        phase = np.exp(1j*old.THETA@d)
        em = np.exp(-1j*q@d)
        sf = sum(np.exp(1j*q[axis]*j) for j in range(length))
        pc = .5j/64*(dh[:,None]@t[None]+em*t[None]@dh[:,None])*sf
        dr = dh.conj().transpose(0,2,1)
        pcr = -.5j/64*(t[None]@dr[:,None]+em*dr[:,None]@t[None])*sf
        sl = slice(45+16*axis,61+16*axis)
        result[:45,sl] += sign*(np.einsum('kij,k,abji->ab',projector,phase,pc)+np.einsum('kij,k,abji->ab',projector,phase.conj(),pcr))
        d2 = -length*anti/(2*64)
        d2r = d2.copy()
        for j in range(length):
            for ell in range(j+1,length):
                e = np.exp(1j*q[axis]*(ell-j))
                d2 -= (e*t[:,None]@t[None]+e.conjugate()*t[None]@t[:,None])/64
                d2r -= (e*t[None]@t[:,None]+e.conjugate()*t[:,None]@t[None])/64
        cc = .5*(hop@d2+d2@hop)
        ccr = .5*(d2r@hop.conj().T+hop.conj().T@d2r)
        result[sl,sl] += sign*(np.einsum('kij,k,abji->ab',projector,phase,cc)+np.einsum('kij,k,abji->ab',projector,phase.conj(),ccr))
    result[45:,:45] = result[:45,45:].conj().T
    return result


def build(profile: dict[str, Any], data: dict[str, Any], qi: tuple[int,int,int]) -> dict[str, Any]:
    native,geom = data['native'],data['geom']
    terms = old.path_terms(profile,native,geom)
    q = 2*np.pi*np.asarray(qi)/4
    h = old.bloch_h(profile,native,terms)
    bands,u = np.linalg.eigh(h)
    if not np.all(bands[:,1]<0) or not np.all(bands[:,2]>0):
        raise ArithmeticError('Selected negative-band Slater reference requires BOTH occupied/empty bands')
    kout = np.asarray([np.ravel_multi_index(tuple((k+qi)%4),(4,4,4)) for k in old.QINTEGERS])
    v = vertices(profile,native,geom,terms,q)
    w,z = u[:,:,:2],u[:,:,2:]
    row = dict(profile=profile,q=q,q_integer=np.asarray(qi),h=h,bands=bands,U=u,vertices=v,kout=kout,
               PH_plus=z[kout].conj().transpose(0,2,1)[:,None]@v@w[:,None],
               PH_minus=w[kout].conj().transpose(0,2,1)[:,None]@v@z[:,None],
               gap_plus=bands[kout,2:,None]-bands[:,None,:2],
               gap_minus=bands[:,None,2:]-bands[kout,:2,None],terms=terms)
    row['bubble'] = old.response(row,0.)
    row['contact'] = contact(profile,geom,terms,q,w@w.conj().transpose(0,2,1))
    row['static'] = row['contact']-row['bubble']
    return row


def maxabs(x: np.ndarray) -> float:
    return float(np.max(np.abs(x)))


def require(errors: dict[str,float], tolerance: float = 5e-9) -> None:
    if max(errors.values(),default=0.) > tolerance:
        raise ArithmeticError(json.dumps(errors))


def finite_covariance(data: dict[str, Any]) -> dict[str, Any]:
    """Arbitrary nonuniform fields/links, independent local SU4 and U1 gauges."""
    native,geom = data['native'],data['geom']
    rng = np.random.default_rng(20261009)
    t = geom['M']/np.sqrt(8)
    fields = native['frame'][None]+.017*np.einsum('xa,amij->xmij',rng.normal(size=(64,45)),geom['deltas'])
    links = np.asarray([expm(1j*np.einsum('a,aij->ij',c,t)) for c in .19*rng.normal(size=(192,15))])
    gauges = np.asarray([expm(1j*np.einsum('a,aij->ij',c,t)) for c in .31*rng.normal(size=(64,15))])
    theta,alpha = .23*rng.normal(size=192),.37*rng.normal(size=64)
    targets = np.asarray([np.ravel_multi_index(tuple((native['sites'][e//3]+np.eye(3,dtype=int)[e%3])%4),(4,4,4)) for e in range(192)])
    fp = gauges[:,None]@fields@gauges.conj().transpose(0,2,1)[:,None]
    lp = gauges[np.arange(192)//3]@links@gauges[targets].conj().transpose(0,2,1)
    tp = theta+alpha[np.arange(192)//3]-alpha[targets]
    results = []
    for profile in data['profiles']:
        h = assemble(profile,native,fields,links,theta)
        hp = assemble(profile,native,fp,lp,tp)
        local = np.exp(1j*charge(profile)*alpha)[:,None,None]*gauges
        expected = (local[:,None]@h.reshape(64,4,64,4).transpose(0,2,1,3)@local.conj().transpose(0,2,1)[None]).transpose(0,2,1,3).reshape(256,256)
        errors = dict(local_SU4_and_U1=maxabs(hp-expected),hermitian=maxabs(h-h.conj().T))
        if charge(profile)==0:
            errors['neutral_theta_independence'] = maxabs(h-assemble(profile,native,fields,links,theta+.61))
            wrong = maxabs(h-assemble(profile,native,fields,links,theta,charge_override=1))
            if wrong < 1e-5:
                raise ArithmeticError('Wrong neutral charge control did not reach theta dependence')
        else:
            wrong = maxabs(hp-assemble(profile,native,fp,lp,tp,omit_second_phase=True))
            if wrong < 1e-5:
                raise ArithmeticError('Omitted second phase control failed to discriminate')
        require(errors)
        results.append(dict(role=profile['role'],errors=errors,wrong_law_defect=wrong))
    return dict(seed=20261009,arbitrary_nonuniform_Phi=True,noncommuting_links=True,species=results)


def abelian_bridge(row: dict[str, Any], data: dict[str, Any]) -> dict[str,float]:
    """Independent primitive Abelian paths/current/contact at this NEW test point."""
    native,profile,q = data['native'],row['profile'],row['q']
    p = next(p for p in native['Pprofiles'] if p['label']==profile['role'])
    v = np.zeros((3,256,256),complex)
    cc = np.zeros((3,3),complex)
    fs = np.exp(1j*native['sites']@q)/8
    for path in p['paths']:
        x,y,ax = path['source'],path['target'],path['axis']
        phase_sum = sum(fs[e//3] for e in path['edges'])
        hop = path['hop']
        v[ax,4*x:4*x+4,4*y:4*y+4] += 1j*phase_sum*hop
        v[ax,4*y:4*y+4,4*x:4*x+4] += -1j*phase_sum*hop.conj().T
        cc[ax,ax] += -abs(phase_sum)**2*(np.trace(p['P'][4*y:4*y+4,4*x:4*x+4]@hop)+np.trace(p['P'][4*x:4*x+4,4*y:4*y+4]@hop.conj().T))
    fourier = np.kron(np.exp(1j*native['sites']@old.THETA.T)/8,np.eye(4))
    ev = np.zeros_like(v)
    eh = np.zeros((256,256),complex)
    for k,out in enumerate(row['kout']):
        ev[:,4*out:4*out+4,4*k:4*k+4] = row['vertices'][k,U1]
        eh[4*k:4*k+4,4*k:4*k+4] = row['h'][k]
    errors = dict(primitive_Abelian_operator=maxabs(fourier.conj().T@p['hcentre']@fourier-eh),
                  primitive_Abelian_current=maxabs(fourier.conj().T@v@fourier-ev),
                  primitive_Abelian_contact=maxabs(cc-row['contact'][np.ix_(U1,U1)]))
    require(errors)
    return errors


def literal_contact(row: dict[str, Any], data: dict[str, Any]) -> np.ndarray:
    """Independent real-space ordered contact, all93 pairs and AP paths."""
    profile,native,geom = row['profile'],data['native'],data['geom']
    fs = np.exp(1j*native['sites']@row['q'])/8
    t = generators(profile,geom)
    anti = t[:,None]@t[None]+t[None]@t[:,None]
    w = row['U'][:,:,:2]
    pm = w@w.conj().transpose(0,2,1)
    fourier = np.exp(1j*native['sites']@old.THETA.T)/8
    p = np.einsum('xp,pab,yp->xayb',fourier,pm,fourier.conj(),optimize=True).reshape(256,256)
    result = np.zeros((93,93),complex)
    for path in old.native_paths(profile,native):
        x,y,axis = path['source'],path['target'],path['axis']
        hop = np.einsum('m,mij->ij',path['coefficients'],native['frame'])
        dh = np.einsum('m,amij->aij',path['coefficients'],geom['deltas'])
        dr = dh.conj().transpose(0,2,1)
        du = sum((1j*t*fs[e//3] for e in path['edges']),np.zeros((16,4,4),complex))
        dur = -du
        pc = (dh[:,None]*fs[x].conjugate()@du[None]+du[None]@dh[:,None]*fs[y].conjugate())/2
        pcr = (dur[None]@dr[:,None]*fs[x].conjugate()+dr[:,None]*fs[y].conjugate()@dur[None])/2
        d2,d2r = np.zeros((16,16,4,4),complex),np.zeros((16,16,4,4),complex)
        for j,e in enumerate(path['edges']):
            d2 -= anti*abs(fs[e//3])**2/2
            d2r -= anti*abs(fs[e//3])**2/2
            for f in path['edges'][j+1:]:
                a,b = fs[e//3].conjugate()*fs[f//3],fs[e//3]*fs[f//3].conjugate()
                d2 -= t[:,None]@t[None]*a+t[None]@t[:,None]*b
                d2r -= t[None]@t[:,None]*a+t[:,None]@t[None]*b
        cc,ccr = (hop@d2+d2@hop)/2,(d2r@hop.conj().T+hop.conj().T@d2r)/2
        block,back = p[4*y:4*y+4,4*x:4*x+4],p[4*x:4*x+4,4*y:4*y+4]
        factor = path['seam']*(-1 if profile['scale']=='heavy' else 1)
        sl = slice(45+16*axis,61+16*axis)
        result[:45,sl] += factor*(np.einsum('ij,abji->ab',block,pc)+np.einsum('ij,abji->ab',back,pcr))
        result[sl,sl] += factor*(np.einsum('ij,abji->ab',block,cc)+np.einsum('ij,abji->ab',back,ccr))
    result[45:,:45] = result[:45,45:].conj().T
    return result


def row_controls(row: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    profile,native,geom = row['profile'],data['native'],data['geom']
    prior = old.build_species(profile,native,geom,row['q_integer'])
    errors = {name+'_old90_subblock':maxabs(row[name][np.ix_(OLD90,OLD90)]-prior[oldname])
              for name,oldname in [('contact','contact'),('bubble','B'),('static','sea')]}
    errors['old90_currents'] = maxabs(row['vertices'][:,OLD90]-prior['vertices'])
    errors['literal_all93_ordered_contact'] = maxabs(literal_contact(row,data)-row['contact'])
    d = 1-np.exp(1j*row['q'])
    direction = np.zeros(93,complex); direction[U1] = d
    contracted = np.einsum('a,kaij->kij',d,row['vertices'][:,U1])
    density = charge(profile)*np.eye(4)/8
    target = 1j*(density@row['h']-row['h'][row['kout']]@density)
    errors['U1_density_operator_Ward'] = maxabs(contracted-target)
    errors['full93_longitudinal_contact_minus_bubble'] = maxabs(row['static']@direction)
    errors['mixed_Phi_U1_Ward'] = maxabs((row['static']@direction)[:45])
    errors['mixed_SU4_U1_Ward'] = maxabs((row['static']@direction)[np.setdiff1d(np.arange(45,93),U1)])
    for name in ('contact','bubble','static'):
        errors[name+'_hermiticity'] = maxabs(row[name]-row[name].conj().T)
    wrong = {}
    if charge(profile):
        errors.update(abelian_bridge(row,data))
        bad = vertices(profile,native,geom,row['terms'],row['q'],1/np.sqrt(8))
        wrong['identity_over_sqrt8_current_defect'] = maxabs(bad[:,U1]-row['vertices'][:,U1])
        if np.linalg.norm(d)>0:
            wrong['identity_over_sqrt8_density_Ward_defect'] = maxabs(np.einsum('a,kaij->kij',d,bad[:,U1])-target)
            omitted = row['static'].copy()
            omitted[:45,U1] -= row['contact'][:45,U1]
            omitted[U1,:45] -= row['contact'][U1,:45]
            wrong['omitted_Phi_U1_contact_longitudinal_defect'] = maxabs((omitted@direction)[:45])
        scalar_defects = [value for key,value in wrong.items() if key.startswith('identity_over_sqrt8')]
        if min(scalar_defects)<1e-6:
            raise ArithmeticError('Wrong scalar normalization does not discriminate: '+str(wrong))
    else:
        errors['neutral_U1_vertices_zero'] = maxabs(row['vertices'][:,U1])
        errors['neutral_U1_contact_zero'] = maxabs(row['contact'][:,U1])
    require(errors)
    return dict(errors=errors,wrong_cases=wrong,
                mixed_contact_control_scope='Mean-point longitudinal projection can be symmetry blind; rejection of omitted Phi/U1 contact uses the separate nonlinear noncommuting literal mixed-derivative probe, not this diagnostic.')


def literal_mixed_probe(data: dict[str, Any]) -> dict[str, Any]:
    """Finite mixed derivatives of an actual noncommuting length-two endpoint."""
    native,geom,profile = data['native'],data['geom'],data['profiles'][1]
    path = next(p for p in old.native_paths(profile,native) if p['length']==2 and p['source']==3)
    x,y = path['source'],path['target']; coeff = path['coefficients']
    t = geom['M']/np.sqrt(8)
    links = [expm(.23j*t[1]),expm(-.19j*t[4])]
    ordered = links[0]@links[1]
    rng = np.random.default_rng(22293)
    dx = np.einsum('a,amij->mij',rng.normal(size=45),geom['deltas'])
    dy = np.einsum('a,amij->mij',rng.normal(size=45),geom['deltas'])
    fx = native['frame']+.03*dx; fy = native['frame']-.02*dy
    theta = [.17,-.29]
    def endpoint(a: float,b: float, omit: bool = False) -> np.ndarray:
        phase = np.exp(1j*(theta[0]+b+(0 if omit else theta[1]+.7*b)))
        xx,yy = np.einsum('m,mij->ij',coeff,fx+a*dx),np.einsum('m,mij->ij',coeff,fy+a*dy)
        return path['seam']*phase*(xx@ordered+ordered@yy)/2
    analytic = 1.7j*path['seam']*np.exp(1j*sum(theta))*(np.einsum('m,mij->ij',coeff,dx)@ordered+ordered@np.einsum('m,mij->ij',coeff,dy))/2
    steps = []
    for eps in (1e-3,3e-4):
        derivative = (endpoint(eps,eps)-endpoint(eps,-eps)-endpoint(-eps,eps)+endpoint(-eps,-eps))/(4*eps**2)
        bad = (endpoint(eps,eps,True)-endpoint(eps,-eps,True)-endpoint(-eps,eps,True)+endpoint(-eps,-eps,True))/(4*eps**2)
        error,defect = maxabs(derivative-analytic),maxabs(bad-analytic)
        if error > 1e-4 or defect < 1e-3 or maxabs(analytic)<1e-3:
            raise ArithmeticError('Literal mixed derivative probe failed')
        steps.append(dict(epsilon=eps,correct_error=error,omitted_contact_defect=maxabs(analytic),omitted_two_link_phase_defect=defect,wrong_scalar_normalization_defect=maxabs(analytic*(1-1/np.sqrt(8)))))
    return dict(source=x,target=y,AP_seam=path['seam'],length=2,ordered_link_commutator=maxabs(links[0]@links[1]-links[1]@links[0]),steps=steps)


def matrix_report(matrix: np.ndarray) -> dict[str, Any]:
    eig = np.linalg.eigvalsh(matrix)
    return dict(matrix=old.encode(matrix),signed_eigenvalues=eig.tolist(),minimum=float(eig[0]),maximum=float(eig[-1]),negative_below_minus_1e_8=int(np.sum(eig < -1e-8)))


def weighted(rows: list[dict[str, Any]], name: str) -> np.ndarray:
    result = np.zeros((93,93),complex)
    for row in rows:
        load = row['profile']['load']
        result += row[name] if load == 1 else load*row[name]
    return result


def run() -> dict[str, Any]:
    started = time.monotonic()
    data = old.load_mean(g_override=[Fraction(1,4),Fraction(1,16),Fraction(1,16),Fraction(1,16)],
                         mean_descriptor=dict(law='NEW explicit TEST finite Slater point',exact_g=['1/4','1/16','1/16','1/16']))
    if [p['load'] for p in data['profiles']] != [1,1,63]:
        raise ArithmeticError('Original primitive copy loads changed')
    covariance = finite_covariance(data)
    mixed = literal_mixed_probe(data)
    selected = []
    for qi in Q_SELECTED:
        rows = [build(p,data,qi) for p in data['profiles']]
        species = []
        for row in rows:
            species.append(dict(role=row['profile']['role'],load=row['profile']['load'],charge=charge(row['profile']),
                                reference='two negative occupied and two positive empty bands at every AP momentum',
                                bands_all64=row['bands'].tolist(),minimum_PH_gap=float(min(row['gap_plus'].min(),row['gap_minus'].min())),
                                dual_PH_shapes=[list(row['PH_plus'].shape),list(row['PH_minus'].shape)],
                                controls=row_controls(row,data),
                                matrices={name:matrix_report(row[name]) for name in ('contact','bubble','static')}))
        total = {name:weighted(rows,name) for name in ('contact','bubble','static')}
        charged = {name:weighted(rows[1:],name) for name in ('contact','bubble','static')}
        selected.append(dict(q_integer=list(qi),species=species,all_species_load_weighted={k:matrix_report(v) for k,v in total.items()},all_charged_load_weighted={k:matrix_report(v) for k,v in charged.items()}))
    sources = {name:dict(sha256=hashlib.sha256(raw).hexdigest(),unchanged=raw==(ROOT/name).read_bytes()) for name,raw in sorted(SOURCE_BYTES.items())}
    if not all(s['unchanged'] for s in sources.values()):
        raise ArithmeticError('Source changed during actual experiment')
    return dict(law='NEW_JOINED_DYNAMIC_PHI_LOCAL_SU4_U1_FINITE_CAR93',
                finite_operator=dict(N=4,V=64,AP_momenta=64,coordinates=93,Phi=45,SU4=45,U1=3,
                    order='Phi45; axis0 SU4T0..14,U1; axis1 SU4T0..14,U1; axis2 SU4T0..14,U1',
                    scalar_generator='charge I4; NOT I4/sqrt8',Fourier='exp(i q x)/sqrt64',loads=[1,1,63],charges=[0,1,1],
                    selected_q=[list(q) for q in Q_SELECTED],couplings=['1/4','1/16','1/16','1/16']),
                descriptor=data['descriptor'],covariance=covariance,literal_mixed_derivative=mixed,selected=selected,
                interpretation=dict(signed_instabilities='Negative static eigenvalues are exposed finite Slater sea curvatures, including toron curvature. No clipping/subtraction, no global ground claim, no dressed pole or cone inference.',
                    physical_scale='Per-copy Lambda=1 operator displays. Declared heavy/charged scale separation retained in profiles; no addition promoted to full Lambda=2^8192 quantum source.',
                    remaining_debts=['Pi/electric dynamics not implemented','scalar-density/Coulomb response not implemented; allcharged rho would be required','full interacting dynamic-Phi/SU4/U1 physical source absent','dressed massless photon dispersion/residue/common matter cone unproved','macro/continuum/Compton transfer unproved'],
                    old_sources='Dynamic Phi/SU4 source and fixed-Phi Abelian/Pi/Coulomb source are different laws/states; only finite operator bridge at NEW test point observed.'),
                source_at_execution=sources,runtime_seconds=time.monotonic()-started)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        result = run()
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(output=str(args.output),sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),runtime_seconds=result['runtime_seconds'],source_stable=True)))


if __name__ == '__main__':
    main()
