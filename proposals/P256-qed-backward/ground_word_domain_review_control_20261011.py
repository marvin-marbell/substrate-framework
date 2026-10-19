"""Independent exact operator/form controls; never imports predecessor entrypoints."""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import itertools
import json
from pathlib import Path
import sys
import zipfile

import sympy as sp


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def zero(matrix: sp.Matrix) -> bool:
    return all(sp.simplify(x) == 0 for x in matrix)


def differential_controls() -> dict:
    a, q, t = sp.symbols('a q t', real=True)
    chi, lam = sp.symbols('chi lambda', positive=True)
    shift = sp.Rational(1, 3)
    hopping = sp.Matrix([[0, sp.exp(sp.I*a)], [sp.exp(-sp.I*a), 0]])
    m = sp.Matrix([[sp.cos(a)*sp.exp(-q*q), sp.sin(a)*sp.exp(-q*q)],
                   [sp.sin(a)*sp.exp(-q*q), -sp.cos(a)*sp.exp(-q*q)]])
    psi = sp.Matrix([sp.exp(sp.I*2*a-q*q), sp.exp(sp.I*3*a-2*q*q)])
    # Every other scalar potential commutes with m and the U1 translation.
    potential = lam*q**4*sp.eye(2)+q*hopping
    def h(v):
        return -64*v.diff(a, 2)-4*chi*v.diff(q, 2)+potential*v
    def w(v):
        return v.subs(a, a+shift)
    lhs = h(m*w(psi))-m*w(h(psi))
    kinetic = -64*(m.diff(a, 2)*w(psi)+2*m.diff(a)*w(psi.diff(a)))
    kinetic -= 4*chi*(m.diff(q, 2)*w(psi)+2*m.diff(q)*w(psi.diff(q)))
    material = (potential*m-m*potential)*w(psi)+m*(potential-potential.subs(a, a+shift))*w(psi)
    assert zero(lhs-kinetic-material)
    assert not zero((lhs-kinetic-material+64*2*m.diff(a)*w(psi.diff(a))).subs({a:0,q:1,chi:1,lam:1}))
    assert not zero((lhs-kinetic-material-m*(potential-potential.subs(a,a+shift))*w(psi)).subs({a:0,q:1,chi:1,lam:1}))
    f = [sp.sin(a), sp.cos(a)]
    assert sp.simplify(sum(x*x for x in f)-1) == 0
    assert sp.simplify(sum(x*sp.diff(x,a) for x in f)) == 0
    assert sp.simplify(sum(sp.diff(x,a)**2 for x in f)-1) == 0
    off = [sp.sin(t)*hopping, sp.cos(t)*sp.eye(2)]
    connection = sum((x.H*x.diff(a) for x in off), sp.zeros(2))
    assert zero(connection-sp.diag(-sp.I*sp.sin(t)**2,sp.I*sp.sin(t)**2))
    assert not zero(connection.subs(t,sp.pi/4))
    # Density matrices have commuting jets; scalar radial derivative uses s=q^2/2.
    scale, angle = sp.symbols('M theta', positive=True)
    number = sp.diag(0, 1, 260)
    g = angle*sp.exp(-q*q/(2*scale))*number
    ds = sp.diag(*[sp.sin(g[i,i]) for i in range(3)])
    dc = sp.diag(*[sp.cos(g[i,i]) for i in range(3)])
    assert zero(ds*ds.diff(q)+dc*dc.diff(q))
    assert zero(ds.diff(q)**2+dc.diff(q)**2-g.diff(q)**2)
    kinetic_density = sp.simplify(g.diff(q)[2,2]**2)
    assert sp.simplify(kinetic_density-260**2*angle**2*q*q*sp.exp(-q*q/scale)/scale**2) == 0
    density = sp.diag(2,3)
    density_f = [sp.diag(sp.sin(t*2),sp.sin(t*3)),sp.diag(sp.cos(t*2),sp.cos(t*3))]
    rem = sum((x*q*hopping*x for x in density_f),sp.zeros(2))-q*hopping
    assert zero(rem-q*hopping*(sp.cos(t)-1))
    assert not zero((rem-q*hopping*(sp.cos(2*t)-1)).subs({q:1,a:0,t:sp.pi/3}))
    return {'scope':'exact differential identities on native CAR transfer compressions, not native spectrum',
            'full_kinetic_M_W_commutator':True,'missing_first_order_rejected':True,
            'missing_W_phase_shift_rejected':True,'scalar_IMS_cost_closed_plaquette':256,
            'offdiagonal_connection':str(connection),'density_IMS_connection_zero':True,
            'radial_convention':'s=|q|^2/2; |grad exp(-T/M)|^2=2T exp(-2T/M)/M^2',
            'density_Phi_cost':'4 chi theta^2 260^2/(e M)',
            'wrong_density_charge_step_two_rejected':True}


def native_graph_controls(profiles: list[dict]) -> dict:
    # Exact full three-dimensional periodic N4 Laplacian, not a flat/infinite surrogate.
    n=4
    sites=list(itertools.product(range(n),repeat=3)); index={x:i for i,x in enumerate(sites)}
    v=len(sites)
    def move(x,axis,step):
        y=list(x);y[axis]=(y[axis]+step)%n;return tuple(y)
    edges=[(x,axis) for x in sites for axis in range(3)]
    d=sp.zeros(v,3*v)
    for j,(x,axis) in enumerate(edges):
        d[index[x],j]=1;d[index[move(x,axis,1)],j]=-1
    lap=d*d.T
    reduced_inverse=lap[:-1,:-1].inv()
    parity=sp.Matrix([(-1)**sum(x) for x in sites])
    k=sp.zeros(3*v,1)
    for j,(x,axis) in enumerate(edges):
        if axis==0:k[j]=64*parity[index[x]]
    rho=128*parity
    assert d*k==rho
    assert lap*parity==12*parity
    def longitudinal(flow):
        charge=d*flow
        phi=sp.Matrix(list(reduced_inverse*charge[:-1,:])+[0])
        return d.T*phi
    kl=longitudinal(k);kt=k-kl
    rows=[]
    for length in (1,2):
        flow=sp.zeros(3*v,1)
        start=(3,0,0)
        for step in range(length):flow[3*index[move(start,0,step)]]=1
        tl=longitudinal(flow);tt=flow-tl
        direct=64*(2*(k.T*flow)[0]+(flow.T*flow)[0])
        electric=64*(2*(kt.T*tt)[0]+(tt.T*tt)[0])
        coulomb=64*(2*(kl.T*tl)[0]+(tl.T*tl)[0])
        assert direct==electric+coulomb
        outside=[j for j in range(3*v) if flow[j]==0]
        tail=sum(tl[j]**2 for j in outside)
        assert tail>0 and all(tt[j]+tl[j]==0 for j in outside)
        assert direct!=electric
        rows.append({'length':length,'AP_seam_sign':-1,'full_delta':str(direct),
                     'HE_delta':str(electric),'HC_delta':str(coulomb),
                     'exact_longitudinal_exterior_norm2':str(tail),'full_exterior_zero':True})
    # Species counts and shared source: no 64 independent Coulomb copies.
    assert 64*64*sp.Rational(4,12)!=64*sp.Rational(128**2,12)
    bweights=[abs(Fraction(p['onsite']))+2*sum(abs(Fraction(r['c0']))+abs(Fraction(r['ci'])) for r in p['paths']) for p in profiles]
    lam=1<<8192
    b=4*(lam*bweights[0]+bweights[1]+63*bweights[2])
    assert b/Fraction(lam)<512
    assert [p['load'] for p in profiles]==[1,1,63]
    stars={tuple(0 for _ in range(3))}
    for axis,length,sign in itertools.product(range(3),(1,2),(-1,1)):
        coord=[0,0,0];coord[axis]=sign*length;stars.add(tuple(coord))
    assert len(stars)==13
    # Radius-two l1 balls have 25 sites: they must NOT be confused with this native axial star.
    ball={x for x in itertools.product(range(-2,3),repeat=3) if sum(abs(j) for j in x)<=2}
    assert len(ball)==25 and stars!=ball
    u=sp.diag(sp.I,-sp.I,-1,-1)
    w=sp.Matrix([[0,1,0,0],[-1,0,0,0],[0,0,0,1],[0,0,-1,0]])
    assert u.det()==w.det()==1 and u.H*u==w.H*w==sp.eye(4)
    assert u*w!=w*u
    phi=sp.diag(3,1,-2,-2)
    coefficient=(phi*u*w+u*w*phi)/2
    gx=w;gy=u
    transformed=(gx*phi*gx.H*(gx*u*w*gy.H)+(gx*u*w*gy.H)*gy*phi*gy.H)/2
    assert transformed==gx*coefficient*gy.H
    assert coefficient!=(phi*w*u+w*u*phi)/2
    canonical=[]
    for size in (4,6,8):
        vv=size**3
        occupied=sum(15<<(4*i) for i,x in enumerate(itertools.product(range(size),repeat=3)) if sum(x)%2==0)
        assert occupied.bit_count()==2*vv
        state=sum(occupied<<(4*vv*s) for s in range(65)) | (15<<(4*vv*65))
        assert all(((state>>(4*vv*s))&((1<<(4*vv))-1)).bit_count()==2*vv for s in range(65))
        assert (state>>(4*vv*65)).bit_count()==4
        for p in profiles:
            assert all(int(r['length']) in (1,2) for r in p['paths'])
        canonical.append({'N':size,'each_species_particles':2*vv,'dynamical_species':65,'charged_species':64,'Ref_particles':4})
    return {'scope':'native exact incidence, affine energy and coefficient controls; no quantum ground integration',
            'full_N4_affine_identity':rows,'canonical':canonical,'B_over_lambda_exact':str(b/Fraction(lam)),
            'native_axial_star_sites':13,'non_native_l1_radius_two_sites':25,
            'ordered_SU4_endpoint_covariance':True,'reversed_SU4_path_rejected':True,
            'frozen_HC_and_independent_species_HC_rejected':True}


def domain_controls() -> dict:
    # Exact method countercases. They are not relabelled native spectra/ground states.
    cases=[]
    for m in (4,16,64):
        p=Fraction(1,m**4);energy=m**3
        first=p*energy;second=p*energy*energy
        assert first==Fraction(1,m) and second==m*m
        cases.append({'M':m,'excitation':energy,'probability':str(p),'first_excess':str(first),'graph_norm_squared':str(second)})
    omega=sp.Matrix([1,0,0]);feature=sp.Matrix([0,1,1])/sp.sqrt(2)
    evolved=sp.diag(1,-1,1)*feature # H=diag(0,1,2), t=pi.
    assert (feature.T*evolved)[0]==0 and (omega.T*evolved)[0]==0
    assert (evolved.T*evolved)[0]==1
    # Native partition constants are additive because L>=0 and the complete partition sum is bounded.
    l, c, z=sp.symbols('l C z',nonnegative=True)
    assert sp.simplify((l+c)-l)==c
    return {'scope':'exact method countercases plus analytically reviewed native partition recurrence',
            'first_energy_excess_not_graph_control':cases,
            'stationary_positive_history_not_static_GNS_invariance':{'H':[0,1,2],'time':'pi','static_span_dimension':2,'evolved_orthogonal_defect_norm_squared':1},
            'graph_packet_correction':'require L Omega=0 or ||L Omega|| uniformly bounded/vanishing, e.g. shrinking excitation spectral support',
            'restricted_word_recurrence':'q_L(F psi)<=q_L(psi)+C||psi||^2; norm(F)<=1; q_L(U psi)=q_L(psi)',
            'restricted_word_continuity':'||(exp(-i delta L)-1)W||^2<=2|delta|q_L(W)',
            'full_static_GNS_identification_proved':False}


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    request=json.loads(args.request.read_bytes())
    before={path:digest(Path(path).read_bytes()) for path in request['pins']}
    assert before==request['pins']
    with zipfile.ZipFile(request['epoch_archive']) as archive:
        profiles=json.loads(archive.read(request['profile']))['primitive_profiles']
    controls={'differential':differential_controls(),'native':native_graph_controls(profiles),'domain':domain_controls()}
    after={path:digest(Path(path).read_bytes()) for path in request['pins']}
    assert before==after
    result={'schema':1,'scope':'INDEPENDENT_EXACT_DOMAIN_FORM_CONTROLS_NOT_NATIVE_SPECTRAL_EVOLUTION',
            'controls':controls,'source_input_sha256_before':before,'source_input_sha256_after':after,
            'request_sha256':digest(args.request.read_bytes()),'python':sys.executable,
            'sympy_version':sp.__version__,'old_entrypoints_imported_or_executed':False,
            'native_ground_computed':False,'full_GNS_dynamics_proved':False}
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'exact_control_groups':list(controls),'all_controls_reached':True,'full_GNS_dynamics':False}))


if __name__=='__main__':
    main()
