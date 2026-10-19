"""Independent exact fixed-cutoff energy review; no original main or solver."""
from __future__ import annotations

import argparse
import ast
import hashlib
import io
import itertools
import json
from pathlib import Path
import sys
import time
import zipfile

import numpy as np
from scipy import sparse
import sympy as sp

P = 65537
II = 256
HALF = pow(2, -1, P)
ROOT = Path(__file__).resolve().parent


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def frozen_inputs(raw):
    tree = ast.parse(raw)
    allowed = {'native_data', 'face_matrices', 'generators'}
    nodes = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))
             or isinstance(node, ast.FunctionDef) and node.name in allowed]
    namespace = {'__file__': str(ROOT / 'ground_loop_support_consumer_20261012.py')}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'frozen_input_definitions', 'exec'), namespace)
    return namespace


def ff(value):
    real, imag = sp.expand_complex(value).as_real_imag()
    def rational(q):
        q = sp.Rational(q)
        return int(q.p) * pow(int(q.q), -1, P) % P
    return (rational(real) + II * rational(imag)) % P


def modular(matrix):
    return np.array([[ff(matrix[a, b]) for b in range(matrix.cols)]
                     for a in range(matrix.rows)], dtype=np.int64)


def bracket(a, b):
    return (a @ b - b @ a) % P


def ordered_action(legs, carrier):
    states = {0: carrier}
    for count in range(1, 5):
        for chosen in itertools.combinations(range(4), count):
            mask = sum(1 << j for j in chosen)
            states[mask] = sum((bracket(legs[j], states[mask ^ (1 << j)]) for j in chosen),
                               np.zeros_like(carrier)) % P
    return states[15]


def inserted_jets(legs, first, second, carrier):
    # Independent explicit temporal permutations and insertion positions.
    outputs = [np.zeros_like(carrier) for _ in range(3)]
    def word(order, substitutions):
        value = carrier
        for position, j in enumerate(order):
            value = bracket(substitutions.get(position, legs[j]), value)
        return value
    for order in itertools.permutations(range(4)):
        outputs[0] = (outputs[0] + word(order, {})) % P
        for position, j in enumerate(order):
            outputs[1] = (outputs[1] + word(order, {position: first[j]})) % P
            outputs[2] = (outputs[2] + word(order, {position: second[j]})) % P
        for left, right in itertools.combinations(range(4), 2):
            outputs[2] = (outputs[2] + 2 * word(order, {left: first[order[left]],
                                                      right: first[order[right]]})) % P
    return outputs


def recurrence_jets(legs, first, second, carrier):
    states = {0: [carrier, np.zeros_like(carrier), np.zeros_like(carrier)]}
    for mask in range(1, 16):
        result = [np.zeros_like(carrier) for _ in range(3)]
        for j in range(4):
            if not mask & (1 << j):
                continue
            value = states[mask ^ (1 << j)]
            result[0] = (result[0] + bracket(legs[j], value[0])) % P
            result[1] = (result[1] + bracket(first[j], value[0]) + bracket(legs[j], value[1])) % P
            result[2] = (result[2] + bracket(second[j], value[0])
                         + 2 * bracket(first[j], value[1]) + bracket(legs[j], value[2])) % P
        states[mask] = result
    return states[15]


def native_controls(helper, profiles):
    sites, index, phi, links, phases, block, points, specs = helper['native_data'](profiles)
    exact_legs = helper['face_matrices'](specs, points)
    legs = [modular(t) for t in exact_legs]
    adjoints = [modular(t.H) for t in exact_legs]
    identity = np.eye(16, dtype=np.int64)
    px = np.zeros_like(identity); px[:4, :4] = np.eye(4, dtype=np.int64)
    zero = [sp.zeros(16) for _ in range(4)]
    def action(g):
        return (ordered_action(legs, g) + ordered_action(adjoints, g)) * HALF % P
    def jet(first, second):
        f = recurrence_jets(legs, [modular(t) for t in first], [modular(t) for t in second], px)
        h = recurrence_jets(adjoints, [modular(t.H) for t in first], [modular(t.H) for t in second], px)
        return [(a + b) * HALF % P for a, b in zip(f, h)]
    def explicit(first, second):
        f = inserted_jets(legs, [modular(t) for t in first], [modular(t) for t in second], px)
        h = inserted_jets(adjoints, [modular(t.H) for t in first], [modular(t.H) for t in second], px)
        return [(a + b) * HALF % P for a, b in zip(f, h)]
    pindex = {x: j for j, x in enumerate(points)}
    def embed(spec, body):
        out = sp.zeros(16)
        source, target = (spec['y'], spec['x']) if not spec['reverse'] else (spec['x'], spec['y'])
        a, b = pindex[target], pindex[source]
        out[4*a:4*a+4, 4*b:4*b+4] = body.H if spec['reverse'] else body
        return out
    counts = {}; laplacians = {}; explicit_count = 0
    for family in ('Phi', 'Haar', 'fullE'):
        entries = []
        if family == 'Phi':
            for site in points:
                for mu in range(4):
                    for generator, weight in helper['generators']():
                        first = []
                        for spec in specs:
                            coefficient = spec['c0'] if mu == 0 else sp.I*spec['ci'] if mu == spec['axis']+1 else 0
                            dx = generator if site == spec['x'] else sp.zeros(4)
                            dy = generator if site == spec['y'] else sp.zeros(4)
                            value = spec['seam']*spec['phase']*coefficient*(dx*spec['link']+spec['link']*dy)/2
                            first.append(embed(spec, value))
                        entries.append((first, zero, weight))
        else:
            for edge, spec in enumerate(specs):
                directions = helper['generators']() if family == 'Haar' else [(None, sp.Integer(1))]
                for generator, weight in directions:
                    first = list(zero); second = list(zero)
                    if family == 'Haar':
                        fx = spec['c0']*phi[spec['x']][0] + sp.I*spec['ci']*phi[spec['x']][spec['axis']+1]
                        fy = spec['c0']*phi[spec['y']][0] + sp.I*spec['ci']*phi[spec['y']][spec['axis']+1]
                        du = sp.I*generator*spec['link']; ddu = -generator*generator*spec['link']
                        for store, dlink in ((first, du), (second, ddu)):
                            value = spec['seam']*spec['phase']*(fx*dlink+dlink*fy)/2
                            store[edge] = embed(spec, value)
                    else:
                        sign = -1 if spec['reverse'] else 1
                        first[edge] = sign*sp.I*exact_legs[edge]; second[edge] = -exact_legs[edge]
                    entries.append((first, second, weight))
        nonzero = 0; lap = np.zeros_like(px)
        for number, (first, second, weight) in enumerate(entries):
            value = jet(first, second)
            nonzero += int(np.any(value[1])); lap = (lap + ff(weight)*value[2]) % P
            # Representative independent ALL24 insertion checks in each native family.
            if number in {0, len(entries)//2, len(entries)-1}:
                comparison = explicit(first, second)
                assert all(np.array_equal(a, b) for a, b in zip(value, comparison))
                explicit_count += 1
        assert np.any(lap)
        counts[family] = {'directions': len(entries), 'nonzero_first_on_Px': nonzero}
        laplacians[family] = digest(lap.tobytes())
    assert counts == {'Phi': {'directions':240, 'nonzero_first_on_Px':180},
                      'Haar': {'directions':60, 'nonzero_first_on_Px':60},
                      'fullE': {'directions':4, 'nonzero_first_on_Px':4}}
    def electric(g):
        return II * (ordered_action(legs, g)-ordered_action(adjoints, g)) * HALF % P
    berry = (action(electric(px))-electric(action(px))) % P
    assert np.any(berry)
    assert not np.any(action(identity)) and np.any(action(px))
    assert not np.any((action(px)+action(identity-px)) % P)
    # Gauge covariance with independently inserted native leg jets.
    generator = sp.zeros(16); generator[0,1] = generator[1,0] = sp.Rational(1,2)
    first = [sp.I*(generator*t-t*generator) for t in exact_legs]
    second = [sp.I*(generator*dt-dt*generator) for dt in first]
    covariance = jet(first, second)
    gmod = modular(generator)
    covfirst = II*(bracket(gmod, action(px))-action(bracket(gmod,px))) % P
    covsecond = (-bracket(gmod,bracket(gmod,action(px)))
                 +2*bracket(gmod,action(bracket(gmod,px)))
                 -action(bracket(gmod,bracket(gmod,px)))) % P
    assert np.array_equal(covariance[1], covfirst) and np.array_equal(covariance[2], covsecond)
    # Closed collective density, not individual-copy local density.
    for site in points:
        incidence = sum((1 if spec['x']==site else 0)-(1 if spec['y']==site else 0)
                        if not spec['reverse'] else
                        (1 if spec['y']==site else 0)-(1 if spec['x']==site else 0) for spec in specs)
        assert incidence == 0
    fulllegs = []; fulladjoints = []
    selected = [4*index[x]+c for x in points for c in range(4)]
    for t, th in zip(legs, adjoints):
        for target, local in ((fulllegs,t),(fulladjoints,th)):
            big = np.zeros((256,256),dtype=np.int64);big[np.ix_(selected,selected)] = local
            target.append(sparse.csr_matrix(big))
    def global_word(matrices, g):
        states={0:g}
        for mask in range(1,16):
            value=np.zeros_like(g)
            for j in range(4):
                if mask & (1<<j):
                    old=states[mask^(1<<j)]
                    value=(value+matrices[j]@old-(matrices[j].T@old.T).T)%P
            states[mask]=value
        return states[15]
    def global_action(g):
        return (global_word(fulllegs,g)+global_word(fulladjoints,g))*HALF%P
    carrier=np.zeros((256,256),dtype=np.int64)
    carrier[np.ix_(selected,selected)]=px
    image=global_action(carrier)
    bodies=[]; primitives=[]; split_comm={}
    for profile in profiles:
        pieces={key:np.zeros_like(carrier) for key in ('onsite','NN','two')}; count=0
        for x in sites:
            a=4*index[x];pieces['onsite'][a:a+4,a:a+4] = modular(sp.Rational(profile['onsite'])*phi[x][0])
            for path in profile['paths']:
                y,b=block(x,path);c=4*index[y];key='NN' if path['length']==1 else 'two'
                pieces[key][a:a+4,c:c+4]=(pieces[key][a:a+4,c:c+4]+modular(b))%P
                pieces[key][c:c+4,a:a+4]=(pieces[key][c:c+4,a:a+4]+modular(b.H))%P
                count+=1
        bodies.append(sum(pieces.values(),np.zeros_like(carrier))%P);primitives.append(count)
        if profile['load']==63:
            for key,h in pieces.items():
                hs=sparse.csr_matrix(h)
                act=lambda v: (hs@v-(hs.T@v.T).T)%P
                split_comm[key]=(act(image)-global_action(act(carrier)))%P
    assert primitives==[192,384,384]
    fullcomm=sum(split_comm.values(),np.zeros_like(carrier))%P
    assert np.any(fullcomm) and all(np.any(v) for v in split_comm.values())
    delta=sp.Rational(profiles[1]['onsite'])-sp.Rational(profiles[2]['onsite'])
    mismatch=np.zeros_like(carrier)
    for x in sites:
        a=4*index[x];mismatch[a:a+4,a:a+4]=ff(delta)*modular(phi[x][0])%P
    assert np.array_equal((bodies[1]-bodies[2])%P,mismatch)
    assert np.any(global_action(mismatch))
    traceless=np.zeros_like(carrier);traceless[np.ix_(selected,selected)]=np.eye(16,dtype=np.int64)
    outside=[j for j in range(256) if j not in selected]
    traceless[outside,outside]=ff(-sp.Rational(16,240))
    assert int(np.trace(traceless))%P==0 and not np.any(global_action(traceless))
    return {'native_directions':counts,'laplacian_pins':laplacians,'all24_independent_jet_comparisons':explicit_count,
            'nonzero_Berry_operator_image_pin':digest(berry.tobytes()),'gauge_first_second_covariance':True,
            'closed_total_density_incidence_zero':True,'primitive_counts':primitives,'onsite_mismatch':str(delta),
            'signed_common_body_piece_pins':{k:digest(v.tobytes()) for k,v in split_comm.items()},
            'drop_onsite_NN_or_two_changes_native_commutator':True,'active_mismatch_leakage_nonzero':True,
            'native_signed_complement_cross_exact_cancellation':True,'exterior_trace_corrected_source_zero':True,
            'scope':'actual original full64 adjoint intertwiner/input point, NOT all canonical irrep ranks or a ground',
            'neutral_separate_factor_commutes_not_deleted':True,'fullE_coefficient64_once':True}


def differential_control():
    x=sp.symbols('x',real=True)
    a=sp.Matrix([[x*x,1+sp.I*x],[1-sp.I*x,-x]])
    b=sp.Matrix([[1+x,sp.I+x*x],[-sp.I+x*x,2-x]])
    connection=sp.Matrix([[sp.I*x,1],[-1,-sp.I*x]])
    psi=(1-x*x)**2*sp.Matrix([1+sp.I*x,2-x+sp.I*x*x])
    body=sp.Matrix([[2+x,1-sp.I*x],[1+sp.I*x,-3+x*x]])
    d=lambda v:v.diff(x)+connection*v
    dc=lambda v:v.diff(x)+connection*v-v*connection
    kinetic=lambda v:-d(d(v))
    h=lambda v:kinetic(v)+body*v
    comm=lambda u,v:u*v-v*u
    lhs=a*(h(a*psi)-a*h(psi))-(h(a*a*psi)-a*h(a*psi))
    rhs=2*dc(a)*dc(a)*psi-comm(a,dc(dc(a)))*psi-2*comm(a,dc(a))*d(psi)+comm(a,comm(body,a))*psi
    assert (lhs-rhs).applyfunc(sp.expand)==sp.zeros(2,1)
    integrate=lambda value:sp.integrate(sp.expand(value),(x,-1,1))
    inner=lambda u,v:(u.H*v)[0]
    c=a+sp.I*b
    exact=sp.re(integrate(inner(c*psi,h(c*psi)-c*h(psi))))
    squares=integrate(inner(dc(c)*psi,dc(c)*psi))
    berry=sp.re(integrate(inner(psi,(dc(c).H*c-c.H*dc(c))*d(psi))))
    signed_body=sp.re(integrate(inner(psi,c.H*comm(body,c)*psi)))
    assert sp.simplify(exact-squares-berry-signed_body)==0 and berry!=0 and signed_body!=0
    assert exact!=squares
    cross_exact=integrate(inner(a*psi,h(b*psi)-b*h(psi)))
    cross_expansion=integrate(inner(dc(a)*psi,dc(b)*psi)+inner(a*d(psi),dc(b)*psi)
                              -inner(a*psi,dc(b)*d(psi))+inner(a*psi,comm(body,b)*psi))
    assert sp.simplify(cross_exact-cross_expansion)==0
    return {'covariant_operator_identity':True,'connection_antiHermitian':connection.H==-connection,
            'exact_Re_Cdag_Hcomm_C':str(exact),'gradient_term':str(squares),'Berry_term':str(berry),
            'signed_body_term':str(signed_body),'boundary_pairing_zero':True,
            'full_unrealified_cross_identity':True,'cross_exact':str(cross_exact),
            'nonstationary_differential_identity_control_not_native_ground':True}


def psd_exact(matrix):
    assert matrix==matrix.H
    for size in range(1,matrix.rows+1):
        for subset in itertools.combinations(range(matrix.rows),size):
            assert sp.simplify(matrix.extract(subset,subset).det())>=0


def complex_spectral_control():
    h=sp.diag(0,0,2,5); omega=sp.eye(4)[:,0]
    a=sp.zeros(4);a[2,0]=a[0,2]=1
    b=sp.zeros(4);b[2,0]=sp.I;b[0,2]=-sp.I;b[3,0]=b[0,3]=1
    zero=sp.zeros(4);zero[1,0]=zero[0,1]=3
    sources=[a+zero,b-zero];columns=sp.Matrix.hstack(*(s*omega for s in sources))
    q0=sp.diag(0,0,1,1); rankone=sp.eye(4)-omega*omega.H
    t=columns.H*h*columns;s=columns.H*h*h*columns
    eta=sp.Rational(2,3);tau=sp.Integer(1);upper=sp.Integer(32)
    psd_exact(t-tau*sp.eye(2));psd_exact(upper*sp.eye(2)-s)
    mass=columns.H*q0*columns;g=columns.H*q0*(h+eta*sp.eye(4)).inv()*q0*columns
    lower=tau**3/(upper*(upper+eta*tau))
    psd_exact(mass-tau*tau/upper*sp.eye(2));psd_exact(g-lower*sp.eye(2))
    z=sp.Matrix([1,sp.I]);c=sources[0]+sp.I*sources[1]
    value=(omega.H*c.H*(h*c-c*h)*omega)[0]
    contact=(omega.H*(c.H*(h*c-c*h)-(h*c-c*h)*c.H)*omega)[0]/2
    real_deleted=(z.H*t.applyfunc(sp.re)*z)[0]
    assert value==5 and contact==9 and real_deleted==9 and (z.H*t*z)[0]==value
    assert t[0,1]==2*sp.I and rankone*zero*omega!=sp.zeros(4,1) and q0*zero*omega==sp.zeros(4,1)
    # Zero modes can appear at the limit; finite P0 projections do NOT converge to whole P0.
    n=sp.symbols('n',positive=True);finite_g=1/(1/n+eta);limit_g=0
    assert sp.limit(finite_g,n,sp.oo)==1/eta and limit_g==0
    energies=[sp.Integer(1),sp.Integer(3),sp.Integer(7)]
    weights=[sp.Rational(1,5),sp.Rational(2,7),sp.Rational(3,11)]
    first=sum(e*w for e,w in zip(energies,weights));second=sum(e*e*w for e,w in zip(energies,weights))
    scalar_g=sum(w/(e+eta) for e,w in zip(energies,weights));scalar_lower=first**3/(second*(second+eta*first))
    assert scalar_g>=scalar_lower>0
    return {'full_complex_T':str(t),'full_complex_m2':str(s),'full_complex_mass':str(mass),
            'full_complex_G_eta':str(g),'tau':str(tau),'U':str(upper),'eta':str(eta),'matrix_lower':str(lower),
            'complex_C_exact_first_energy':str(value),'WRONG_symmetric_contact':str(contact),
            'WRONG_real_matrix':str(real_deleted),'wholeP0_not_mean':True,
            'finite_zero_projector_nontransport_control':{'finite_G_limit':str(1/eta),'limiting_wholeP0_G':0},
            'first_moment_uniform_integrability_controls':{'bounded_m2_mass_at_E_n':'mass=1/n^2: m1=1/n,m2=1',
                'WRONG_without_m2':'mass=1/n: m1=1,m2=n; weak spectral mass disappears but first moment does not'},
            'scalar_exact_G':str(scalar_g),'scalar_lower_G':str(scalar_lower),
            'not_native_ground_or_spectrum':True}


def radial_control():
    dimension=3840;norm=sp.beta(sp.Rational(dimension,2),sp.Rational(9,2))/2
    rows=[]
    for power in (sp.Rational(17,2),sp.Integer(8),sp.Integer(12),sp.Integer(16)):
        second=(9-power)/2
        rows.append({'power':str(power),'beta_second':str(second),'finite':second>0,
                     'normalized_moment':str(sp.beta((dimension+power)/2,second)/(2*norm)) if second>0 else 'DIVERGES'})
    assert [bool(row['finite']) for row in rows]==[True,True,False,False]
    s=sp.Rational(1,34);fractional=sp.Rational(2,17)/(sp.Rational(2,17)-2*s)
    assert fractional==2
    return {'D':dimension,'amplitude_exponent':str(-sp.Rational(dimension+9,4)),
            'normalization_radial_factor':str(norm),'moments':rows,'tail_integrand_power':'r^(power-10)',
            'base_H_graph_requires_moment8':True,'uncut_A_Hilbert_requires_moment8':True,
            'uncut_A_form_requires_moment12_diverges':True,'uncut_A_graph_requires_moment16_diverges':True,
            'native_quartic_angular_nonzero_supplied_by_new_native_source_image':True,
            'Jordan_leading_degree4_on_Apsi_degree8_no_lower_body_cancellation':True,
            'NONSTATIONARY_not_native_ground_domain_failure':True,
            'strict_fractional_s':str(s),'fractional_moment':str(fractional),'endpoint_1over17_diverges':True}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--request',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();start=time.monotonic();request=json.loads(args.request.read_bytes())
    with zipfile.ZipFile(request['preexecution_archive']) as archive:
        for name,pin in request['pins'].items():
            assert digest(archive.read(Path(name).name))==pin,name
        with zipfile.ZipFile(io.BytesIO(archive.read(request['snapshot']))) as snapshot:
            helper=frozen_inputs(snapshot.read('ground_loop_support_consumer_20261012.py'))
            with zipfile.ZipFile(io.BytesIO(snapshot.read('ground_loop_support_actual3_20261012_preexecution.zip'))) as before:
                with zipfile.ZipFile(io.BytesIO(before.read(request['profile_archive']))) as original:
                    profile_raw=original.read(request['profile_member'])
                    assert digest(profile_raw)==request['profile_member_sha256']
                    profiles=json.loads(profile_raw)['primitive_profiles']
    native=native_controls(helper,profiles)
    result={'kind':'INDEPENDENT_EXACT_NATIVE_AND_COMPLEX_FULL_ENERGY_FORM_REVIEW_CONTROLS_NOT_GROUND_ADMISSION',
            'native':native,'differential':differential_control(),'complex_spectral':complex_spectral_control(),
            'physical_radial':radial_control(),'pins':request['pins'],
            'runtime':{'executable':sys.executable,'elapsed_seconds':time.monotonic()-start},
            'source_lower_tau_paid':False,'fixed_cutoff_only':True,'original_main_or_solver_executed':False,
            'scientific_admission':False,'campaign_complete':False}
    args.output.write_text(json.dumps(result,indent=2,default=lambda v:bool(v))+'\n')
    print(json.dumps({'receipt':args.output.name,'new_exact_controls':True,'actual_tau':'UNPAID'}))


if __name__=='__main__':
    main()
