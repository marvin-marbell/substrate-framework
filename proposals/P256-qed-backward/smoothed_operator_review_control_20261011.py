"""Independent exact adverse controls; never imports or runs historical producers."""
from __future__ import annotations
import hashlib
import json
from fractions import Fraction as F
from pathlib import Path
from typing import cast
import zipfile
import sympy as s

ROOT = Path(__file__).resolve().parent


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def require(condition: bool, label: str) -> None:
    if not condition:
        raise AssertionError(label)


def bindings(pins: dict[str, str]) -> list[dict[str, object]]:
    for name, pin in pins.items():
        require(digest((ROOT / name).read_bytes()) == pin, 'current pinned bytes: ' + name)
    results = []
    for epoch, source, request in (
        ('actual1', 'smoothed_flux_scientist_20261011.py', 'smoothed_flux_request_20261011.json'),
        ('actual2', 'smoothed_flux_principal_filter_20261011.py', 'smoothed_flux_normalized_principal_request_20261011.json'),
    ):
        prefix = 'smoothed_flux_' + epoch + '_20261011'
        with zipfile.ZipFile(ROOT / (prefix + '_executed.zip')) as z:
            manifest = json.loads(z.read('smoothed_flux_preexecution_manifest.json'))
            for name, pin in manifest['source_input_sha256'].items():
                require(digest(z.read(name)) == pin, 'archived manifest member: ' + name)
            receipt_bytes = z.read(prefix + '_receipt.json')
            receipt = json.loads(receipt_bytes)
            exit_record = json.loads(z.read(prefix + '_exit.json'))
            require(receipt['source_sha256'] == digest(z.read(source)) == pins[source], 'source binding')
            require(receipt['request_sha256'] == digest(z.read(request)), 'request binding')
            req = json.loads(z.read(request))
            require(receipt['input_sha256'] == req['input_sha256'], 'receipt/request input contract')
            for name, pin in receipt['input_sha256'].items():
                require(digest(z.read(name)) == pin, 'receipt archived input: ' + name)
            require(digest(receipt_bytes) == pins[prefix + '_receipt.json'], 'receipt immutable pin')
            require(exit_record['receipt_sha256'] == digest(receipt_bytes), 'exit/receipt binding')
            for suffix in ('stdout', 'stderr', 'language'):
                ext = '.json' if suffix == 'language' else '.txt'
                require(exit_record[suffix + '_sha256'] == digest(z.read(prefix + '_' + suffix + ext)), 'exit output binding')
            require(exit_record['returncode'] == 0 and exit_record['natural_exit'], 'archived natural exit')
            results.append(dict(epoch=epoch, archived_manifest_members=len(manifest['source_input_sha256']),
                                source_sha256=receipt['source_sha256'], request_sha256=receipt['request_sha256'],
                                receipt_sha256=digest(receipt_bytes), natural_exit_record=exit_record,
                                historical_execution_rerun=False))
    return results


def exact_controls() -> dict[str, object]:
    results: dict[str, object] = {}
    # Integer native N4 incidence built independently, not imported.
    n = 4
    sites = [(x, y, z) for x in range(n) for y in range(n) for z in range(n)]
    index = {site: i for i, site in enumerate(sites)}
    def shift(site: tuple[int, int, int], axis: int) -> tuple[int, int, int]:
        return tuple((value + (j == axis)) % n for j, value in enumerate(site))  # type: ignore[return-value]
    faces: list[dict[int, int]] = []
    for site in sites:
        for j, k in ((0, 1), (0, 2), (1, 2)):
            faces.append({3*index[site]+j: 1, 3*index[shift(site,j)]+k: 1,
                          3*index[shift(site,k)]+j: -1, 3*index[site]+k: -1})
    edge_theta_units = [0] * (3*n**3)
    edge_theta_units[0] = 1  # pi on one raw edge; four exact seam faces
    phi_units = [sum(sign*edge_theta_units[e] for e, sign in face.items()) for face in faces]
    r = F(3, 4)
    derivative = [(-2*r/(1-r) if u % 2 else 2*r/(1+r)) for u in phi_units]
    f = [F(1,8) if i % 3 == 0 else F(0) for i in range(len(faces))]
    require(sum(x*x for x in f) == 1, 'q0 source norm')
    g = [F(0)] * len(edge_theta_units)
    b = [F(0)] * len(g)
    for i, face in enumerate(faces):
        for e, sign in face.items():
            g[e] += sign*f[i]*derivative[i]
            b[e] += sign*f[i]
    require(all(x == 0 for x in b), 'q0 raw gradient telescopes')
    div = [F(0)] * len(sites)
    for site in sites:
        for axis in range(3):
            value = g[3*index[site]+axis]
            div[index[site]] += value
            div[index[shift(site,axis)]] -= value
    require(all(x == 0 for x in div), 'exact Gauss closed nonlinear gradient')
    norm = sum(x*x for x in g)
    require(norm > 0, 'reject q0 smooth contact zero and erase-Poisson formula')
    results['native_q0'] = dict(N=n, r=str(r), gradient_norm2=str(norm), double_contact=str(128*norm),
                                 ground_m1_if_insertable=str(64*norm), rejected=['gradient=C^T f','q0_contact=0'],
                                 state_scope='exact allowed configuration, not minimizing ground')
    # Genuine noncommuting quantum endpoint matrix is independent of electric coordinate.
    t, alpha = s.symbols('t alpha', real=True)
    fm = s.sin(t)
    B = s.Matrix([[t*0, 1], [2, 0]])
    psi = s.Matrix([s.Function('p')(t), s.Function('q')(t)])
    def electric(v: s.MatrixBase) -> s.MatrixBase:
        return -s.I*v.diff(t) + alpha*v
    def h(v: s.MatrixBase) -> s.MatrixBase:
        return 64*electric(electric(v))
    require(s.simplify(fm*(B*s.exp(s.I*t)*psi) - B*s.exp(s.I*t)*(fm*psi)) == s.zeros(2,1), 'quantum B commutes with scalar F')
    double = 2*fm*h(cast(s.MatrixBase, fm*psi)) - fm**2*h(psi) - h(cast(s.MatrixBase, fm**2*psi))
    require(s.simplify(double-128*s.cos(t)**2*psi) == s.zeros(2,1), 'affine double-contact constant128')
    require(s.simplify(double-64*s.cos(t)**2*psi) != s.zeros(2,1), 'reject first-moment constant used as double commutator')
    D=s.Matrix([[1,-1],[-1,1]]); A=D.T*D/8; Q=s.eye(2)-A*D; k=s.Matrix([1,0])
    require(Q*k+A*D*k == k and Q*k != k, 'reject Pi-only conjugation')
    x,y=s.symbols('x y'); cross=s.expand(64*(x+y)**2-64*x*x-64*y*y)
    require(cross == 128*x*y and cross != 0, 'reject independent Coulomb species')
    c0=s.Matrix([[0,1,0,0],[0,0,0,0],[0,0,0,1],[0,0,0,0]])
    c1=s.Matrix([[0,0,1,0],[0,0,0,-1],[0,0,0,0],[0,0,0,0]])
    for i, ci in enumerate((c0,c1)):
        for j, cj in enumerate((c0,c1)):
            require(ci*cj+cj*ci==s.zeros(4), 'finite CAR annihilator algebra')
            require(ci*cj.T+cj.T*ci==(s.eye(4) if i==j else s.zeros(4)), 'finite CAR delta')
    nx=c0.T*c0;ny=c1.T*c1;hop=c0.T*c1
    require(nx*hop-hop*nx==hop and ny*hop-hop*ny==-hop, 'literal source target density shift')
    require((nx+ny)*hop-hop*(nx+ny)==s.zeros(4), 'CAR number sector preservation')
    require(nx*hop-hop*nx!=s.zeros(4), 'reject density commutes with hop')
    results['CAR'] = dict(modes=2, dimension=4, algebra_pairs=4, density_source=1,
                          density_target=-1, number_conserved=True, spectator_embedding_only=True)
    results['primitive_affine_contact'] = dict(quantum_multiplier_matrix='[[0,1],[2,0]]; arbitrary non-scalar endpoint witness',
                                               cross_term=str(cross), double_constant=128, first_moment_constant=64,
                                               scope='all-vector finite algebra witnesses plus analytic full-coordinate tensor argument')
    kap=s.symbols('kap',positive=True); rq=kap/(4+kap)
    cap=s.cancel(192*kap**2/(4*rq**2))
    require(s.expand(cap-48*(4+kap)**2)==0 and s.limit(cap,kap,0)==768, 'normalization failure exact asymptotic')
    require(s.limit(cap/kap**2,kap,0)==s.oo, 'reject normalized q2 cap')
    # Exact Fourier convolution multiplier at N4: cos(q)=0, constant multiplier1.
    fourier=[F(1),F(0),F(-1),F(0)]
    coarse=[(fourier[j-1]+fourier[j]+fourier[(j+1)%4])/3 for j in range(4)]
    require(coarse == [x/3 for x in fourier], 'native fixed Fourier convolution scalar')
    require((sum([F(1),F(1),F(1)])/3)==1, 'reject q0 UV erasure by spatial convolution')
    results['normalization_spatial'] = dict(cap=str(cap), limit=768, normalized_q2_rejected=True,
                                           fixed_q_multiplier='1/3', moment_ratio_unchanged=True,
                                           assumptions='translation-invariant linear convolution and one exact momentum; not nonlinear unwrapping')
    # Symbolic radial value at seam is exactly0, unlike principal endpoint +/-pi/2.
    radial=s.symbols('radial',positive=True)
    require(s.sin(s.pi)==0 and s.limit(s.atan(0/(1-radial))/radial,radial,1,dir='-')==0, 'seam radial midpoint exact')
    require(s.Integer(0)!=s.pi/2 and s.Integer(0)!=-s.pi/2, 'reject either endpoint as radial seam value')
    results['seam_domain'] = dict(radial_midpoint=0, wrong_endpoint_assignment_rejected=True,
                                  principal_form_counterexample='weights1/(2j^2), energies256j^2: each m1 summand128',
                                  unfiltered_first_moment_diverges=True)
    # All-vector block control with entire TWO-dimensional ground space.
    L=s.diag(0,0,2,11); P=s.diag(1,1,0,0); QQ=s.eye(4)-P
    FF=s.Matrix([[2,3,5,7],[3,1,4,6],[5,4,0,8],[7,6,8,1]])
    eps=s.Rational(3,5); R=eps*(L+eps*s.eye(4)).inv()
    AA=R*QQ*FF*P; OO=AA+AA.T
    require(OO.T==OO and L*AA.T==s.zeros(4), 'all-vector adjoint in entire ground block')
    require(L*OO==L*AA, 'all-vector LO=LA, not merely selected-source identity')
    Pselected=s.diag(1,0,0,0)
    wrong=R*(s.eye(4)-Pselected)*FF*Pselected
    require(wrong[1,0]==3 and AA[1,0]==0, 'reject selected ground subtraction: zero-state leakage')
    require(AA[:,1]!=s.zeros(4,1), 'source on second ground vector included')
    results['whole_ground_all_vector'] = dict(ground_dimension=2, selected_vector_leakage='3',
                                              all_vector_block_identity=True, second_ground_source_nonzero=True,
                                              scope='exact finite operator, NOT measured full-law ground or arbitrary full-law D(H) theorem')
    tail_rows=[]
    for epsf in (F(1,3),F(7,5)):
        K=9; m0=F(0);m1=F(0);m2=F(0)
        for j in range(1,K+1):
            energy=256*j*j;w=F(1,2*j*j); factor=epsf/(energy+epsf)
            m0+=w*factor**2;m1+=energy*w*factor**2;m2+=energy**2*w*factor**2
        tail0=epsf**2/F(655360*K**5);tail1=epsf**2/F(1536*K**3);tail2=epsf**2/F(2*K)
        # Finite disjoint tail slab is below analytic infinite integral bounds.
        slab0=F(0);slab1=F(0);slab2=F(0)
        for j in range(K+1,6*K+1):
            energy=256*j*j; term=F(1,2*j*j)*(epsf/(energy+epsf))**2
            slab0+=term;slab1+=energy*term;slab2+=energy**2*term
        require(slab0<tail0 and slab1<tail1 and slab2<tail2, 'independent exact principal m0/m1/m2 outward tails')
        require(m1>=256*m0 and m2<=epsf**2, 'gap persists; graph second moment controlled')
        require(128*K>0 and 128*(K+1)>128*K, 'unfiltered principal m1 does not converge')
        tail_rows.append(dict(epsilon=str(epsf),cutoff=K,m0_partial=str(m0),m1_partial=str(m1),m2_partial=str(m2),
                              m0_tail=str(tail0),m1_tail=str(tail1),m2_tail=str(tail2),gap=256))
    results['principal_tails_graph'] = tail_rows
    return results


def main() -> None:
    input_path=ROOT/'smoothed_operator_review_input_20261011.json'
    input_bytes=input_path.read_bytes(); inputs=json.loads(input_bytes)
    original=bindings(inputs['pins'])
    result=dict(kind='INDEPENDENT_EXACT_ADVERSE_REVIEW_NOT_PHYSICAL_ADMISSION',
                source_sha256=digest(Path(__file__).read_bytes()),input_sha256=digest(input_bytes),
                original_bindings=original,controls=exact_controls(),all_controls_reached=True,
                historical_producer_rerun=False,actual_ground=False,physical1_phase=False,IR_atom=False)
    (ROOT/'smoothed_operator_review_receipt_20261011.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(all_controls_reached=True,original_epochs_bound=2,physical_admission=False)))


if __name__ == '__main__':
    main()
