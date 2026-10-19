"""Independent small native CAR/harmonic review; no producer or solver entrypoint."""
import ast
import hashlib
import itertools
import json
import math
from fractions import Fraction
from pathlib import Path
import numpy as np
import scipy
from scipy import sparse
from scipy.linalg import null_space
import sympy as s

ROOT = Path(__file__).resolve().parent
PREFIX = 'local_so5_rep_review_recovered_'
source = ROOT/'local_so5_resum_source_20261011.py'
raw = source.read_bytes()
tree = ast.parse(raw)
# Only bounded algebra/action functions, never imports/run/old control functions.
names = {'representation', 'full_action', 'density_moments'}
nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
for node in nodes:
    node.decorator_list = []
scope = dict(np=np, sparse=sparse)
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), scope)
results = {'source_sha256': hashlib.sha256(raw).hexdigest(), 'versions': {'numpy': np.__version__, 'scipy': scipy.__version__, 'sympy': s.__version__}, 'extracted_bounded_functions': sorted(names), 'no_producer_checker_or_solver_entrypoint': True}

# CAR from independent Jordan-Wigner tensor factors, then exact dGamma restriction.
I2 = s.eye(2); Z = s.diag(1,-1); destroy = s.Matrix([[0,1],[0,0]])
a = [s.kronecker_product(*[I2 if k>p else (destroy if k==p else Z) for k in reversed(range(4))]) for p in range(4)]
assert all(a[p]*a[q].H+a[q].H*a[p] == (s.eye(16) if p==q else s.zeros(16)) for p in range(4) for q in range(4))
x = s.Matrix([[0,1],[1,0]]); z = s.diag(-1,1)
one = [s.diag(-1,1,-1,1), s.kronecker_product(x,-z), s.BlockMatrix([[s.zeros(2),s.I*x],[-s.I*x,s.zeros(2)]]).as_explicit(), s.diag(s.Rational(1,2),s.Rational(1,2),-s.Rational(1,2),-s.Rational(1,2))]
V = s.zeros(16,5)
for j, mask in enumerate([5,3,12,10]): V[mask,j]=1
V[9,4]=V[6,4]=1/s.sqrt(2)
singlet = (s.eye(16)[:,9]-s.eye(16)[:,6])/s.sqrt(2)
ops=[]
for h in one:
    full = sum((h[p,q]*a[p].H*a[q] for p in range(4) for q in range(4)),s.zeros(16))
    small = s.simplify(V.H*full*V)
    assert s.simplify(full*V-V*small)==s.zeros(16,5)
    assert full*singlet==s.zeros(16,1)
    ops.append(small)
G=s.zeros(5); G[0,3]=G[3,0]=-1; G[1,2]=G[2,1]=1; G[4,4]=1
assert all(o.T*G+G*o==s.zeros(5) for o in ops)
assert G[0,0]==0
algebra=ops.copy()
changed=True
while changed:
    changed=False
    for u,v in itertools.combinations(algebra.copy(),2):
        w=u*v-v*u
        before=s.Matrix.hstack(*[s.Matrix(o).reshape(25,1) for o in algebra])
        if before.row_join(w.reshape(25,1)).rank()>before.rank():
            algebra.append(w); changed=True
assert len(algebra)==10
results['exact_native']={'full16_CAR':True,'wedge2_vector5_and_inert1':True,'invariant_G_and_isotropic_ac':True,'generated_so5_dimension':len(algebra)}
NOPS=[np.array(o,dtype=complex) for o in ops]

def collective(o,n):
    return sum((np.kron(np.kron(np.eye(5**k),o),np.eye(5**(n-k-1))) for k in range(n)),np.zeros((5**n,5**n),complex))

def harmonic_embedding(n):
    # Explicit symmetric tensors, then trace contraction: no CG or Casimir input.
    occupations=[k for k in itertools.product(range(n+1),repeat=5) if sum(k)==n]
    sym=np.zeros((5**n,len(occupations)))
    for j,occ in enumerate(occupations):
        word=sum(([i]*k for i,k in enumerate(occ)),[])
        perms=set(itertools.permutations(word))
        for perm in perms:
            idx=0
            for i in perm: idx=5*idx+i
            sym[idx,j]=1/math.sqrt(len(perms))
    if n==1: return sym
    trace=np.einsum('ij,ijk->k',np.array(G,float),sym[:,0].reshape(5,5,-1))[:,None]
    for j in range(1,len(occupations)):
        trace=np.column_stack((trace,np.einsum('ij,ijk->k',np.array(G,float),sym[:,j].reshape(5,5,-1))))
    return sym@null_space(trace)

results['small_harmonics']=[]
for n in [1,2,3]:
    E=harmonic_embedding(n)
    dim=(n+1)*(n+2)*(2*n+3)//6
    assert E.shape[1]==dim
    native=[collective(o,n) for o in NOPS]
    restricted=[E.conj().T@o@E for o in native]
    compressed=[scope['representation'](n,1,0,0,False)[0].toarray(),scope['representation'](n,0,1,0,False)[0].toarray(),scope['representation'](n,0,0,1,False)[0].toarray(),np.diag(scope['representation'](n,0,0,0,False)[1])]
    invariant=max(np.linalg.norm(o@E-E@r) for o,r in zip(native,restricted))
    probes=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[1.3,-.7,.4,.2],[-.2,.9,1.1,-.3]]
    errors=[]
    for coefficients in probes:
        left=sum(c*o for c,o in zip(coefficients,restricted))
        right=sum(c*o for c,o in zip(coefficients,compressed))
        left+=.37*(restricted[3]@restricted[3]); right+=.37*(compressed[3]@compressed[3])
        errors.append(float(np.max(np.abs(np.linalg.eigvalsh(left)-np.linalg.eigvalsh(right)))))
    # ac tensor source, moments challenge both source assignment and generator phases.
    hn=sum(c*o for c,o in zip(probes[-1],native)); hc=sum(c*o for c,o in zip(probes[-1],compressed))
    v=np.eye(5**n,dtype=complex)[:,0]
    labels=scope['representation'](n,0,0,0,False)[2]
    w=np.eye(dim,dtype=complex)[:,labels.index((n,0,0))]
    moment_errors=[]
    for power in range(5):
        moment_errors.append(float(abs(v[0]-w[labels.index((n,0,0))])))
        v=hn@v; w=hc@w
    wrong=sum(c*o for c,o in zip(probes[-1],[compressed[0],math.sqrt(2)*compressed[1],compressed[2],compressed[3]]))
    wrong_error=float(np.max(np.abs(np.linalg.eigvalsh(E.conj().T@hn@E)-np.linalg.eigvalsh(wrong))))
    assert invariant<1e-10 and max(errors)<1e-10 and max(moment_errors)<1e-9 and wrong_error>.1
    results['small_harmonics'].append({'n':n,'dimension':dim,'trace_kernel_invariance_error':float(invariant),'mixed_density_square_spectrum_error':max(errors),'source_moments_through_power4_error':max(moment_errors),'wrong_CG_factor_spectral_error':wrong_error})

# Exact recurrence is independently summed from forward CG weights.
for n in [1,2,3,126]:
    prior=Fraction(0)
    for l in range(n+1):
        summed=Fraction(0)
        for p,q in itertools.product(range(l+1),repeat=2):
            summed+=2*Fraction((p+1)*(l-q+1)+(l-p+1)*(q+1),(l+1)**2)
        average=summed/(l+1)**2
        assert average==Fraction((l+2)**2,(l+1)**2)
        f=Fraction((n-l)*(n+l+3)*(l+1),2*(l+2))
        assert average*f+prior==n*(n+3)-l*(l+2)
        prior=f
    assert prior==0
results['exact_CG_recurrence']={'n':[1,2,3,126],'CG_trace_computed_from_all_weights':True,'terminal_zero':True,'dimensions':[14,690880,14*690880]}

# Actual action on small complete joint space against explicit Kronecker matrix.
A,ra,_=scope['representation'](2,1.3,-.7,.4)
B,rb,_=scope['representation'](1,2.1,-.7,.4)
rho=ra[:,None]+rb[None,:]; u=.17
diagonal=A.diagonal()[:,None]+B.diagonal()[None,:]+64*(rho+u)**2
AA=A.copy(); BB=B.copy(); AA.setdiag(0); BB.setdiag(0); AA.eliminate_zeros(); BB.eliminate_zeros()
H=np.kron(A.toarray(),np.eye(5))+np.kron(np.eye(14),B.toarray())+np.diag(64*(rho.reshape(-1)+u)**2)
rng=np.random.default_rng(2221011); v=rng.normal(size=70); v/=np.linalg.norm(v)
y=scope['full_action'](v,diagonal,AA.indptr,AA.indices,AA.data,BB.indptr,BB.indices,BB.data)
assert np.linalg.norm(y-H@v)<1e-10
zero_projector=np.diag((rho.reshape(-1)==0).astype(float))
commutator=float(np.linalg.norm(H@zero_projector-zero_projector@H))
assert commutator>1
mean,second,zero=scope['density_moments'](v,rho)
assert abs(zero-v@zero_projector@v)<1e-14
# Bare constant once: replacing u=0 changes H by128u rho+64u² I.
assert np.max(abs(64*(rho+u)**2-64*rho**2-128*u*rho-64*u*u))<1e-12
# Connected subtraction need not remove a degenerate ground block.
rho_wrong=np.array([[0.,1.,0.],[1.,0.,0.],[0.,0.,0.]])
psi=np.array([1.,0.,0.]); Qexc=np.diag([0.,0.,1.])
assert np.linalg.norm(rho_wrong@psi)**2-(psi@rho_wrong@psi)**2==1 and np.linalg.norm(Qexc@rho_wrong@psi)**2==0
results['complete_action_semantics']={'small_dimension':70,'explicit_Kronecker_error':float(np.linalg.norm(y-H@v)),'rho_zero_projector_not_spectral_commutator_norm':commutator,'bare64u2_once':True,'connected_mean_not_ground_block_counterexample':{'mean_subtracted_weight':1,'excited_spectral_weight':0},'PHP_square_counterexample':{'H':[[0,1],[1,0]],'P_dimension':1,'full_residual_squared':1,'projected_square':0}}

proof=json.loads((ROOT/'local_so5_resum_freeframe_proof_20261011.json').read_text())
m,d,e,c=s.symbols('m d e c',real=True,nonzero=True)
locals_=dict(m=m,d=d,e=e,c=c,I=s.I)
U=s.Matrix([[s.sympify(z,locals=locals_) for z in row] for row in proof['U_columns_exact_symbolic']])
R=s.Matrix([[s.sympify(z,locals=locals_) for z in row] for row in proof['rho_free_exact_symbolic']])
def remainder_is_zero(matrix):
    for expression in matrix:
        numerator=s.cancel(expression).as_numer_denom()[0]
        if s.simplify(s.Poly(numerator,e).rem(s.Poly(e**2-m**2-d**2,e)).as_expr())!=0:return False
    return True
assert remainder_is_zero(U.H*U-s.eye(5))
native_H=m*ops[0]+c*ops[1]+d*ops[2]
labelled_energies=s.diag(-2*e,2*c,0,-2*c,2*e)
original_H_identity=remainder_is_zero(native_H*U-U*labelled_energies)
assert not original_H_identity
swapped_energies=s.diag(-2*e,-2*c,0,2*c,2*e)
assert remainder_is_zero(native_H*U-U*swapped_energies)
corrected_U=s.diag(1,1,1,1,-1)*U
assert remainder_is_zero(native_H*corrected_U-corrected_U*labelled_energies)
assert remainder_is_zero(corrected_U.H*ops[3]*corrected_U-R)
literal_Bplus_action=[str(z) for z in ops[1]*U[:,1]]
literal_Bplus_expected=[str(z) for z in 2*U[:,1]]
assert remainder_is_zero(U.H*ops[3]*U-R)
assert (U.H*U-s.eye(5)).subs({m:3,d:4,e:6})!=s.zeros(5)
for p in proof['profiles']:
    assert s.Rational(p['e_squared_exact'])==s.Rational(p['m_exact'])**2+s.Rational(p['d_exact'])**2
results['exact_free5']={'frozen_native_H_U_identity':original_H_identity,'frozen_U_diagonal_for_actual_native_B':['-2e','-2c','0','2c','2e'],'literal_Bplus_actual_B_action':literal_Bplus_action,'literal_Bplus_claimed_B_action':literal_Bplus_expected,'flip_fifth_native_row_restores_H_and_preserves_rho':True,'exact_profile_e2_equals_m2_plus_d2':True,'wrong_e_rejected':True,'roles_distinct':proof['profiles'][0]['m_exact']!=proof['profiles'][1]['m_exact']}

# Full length-two support includes the intermediate even where endpoints miss target.
allocation=[]
for n in [4,6]:
    totals={1:Fraction(0),2:Fraction(0)}; counts={1:0,2:0}; endpoint_missed=0
    def block(vertex):return(vertex[0]//2,vertex[1],vertex[2])
    for length in [1,2]:
        for vertex in itertools.product(range(n),repeat=3):
            for axis in range(3):
                path=[]
                for j in range(length+1):
                    row=list(vertex);row[axis]=(row[axis]+j)%n;path.append(tuple(row))
                touched={block(row) for row in path}
                assert sum((Fraction(1,len(touched)) for _ in touched),Fraction(0))==1
                if (0,0,0) in touched:
                    counts[length]+=1
                    if not(length==1 and len(touched)==1):totals[length]+=Fraction(1,len(touched))
                    if length==2 and (0,0,0) not in {block(path[0]),block(path[-1])}:endpoint_missed+=1
    assert counts=={1:11,2:16} and totals=={1:Fraction(5),2:Fraction(6)} and endpoint_missed==4
    allocation.append({'N':n,'touching_counts':counts,'allocated_exact':{k:str(v) for k,v in totals.items()},'endpoint_only_wrong_control_missed':endpoint_missed})
results['reciprocal_allocation']=allocation
assert hashlib.sha256(source.read_bytes()).hexdigest()==results['source_sha256']
results['frozen_source_unchanged']=True
results['own_source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
(ROOT/(PREFIX+'control_20261011_receipt.json')).write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results,indent=2),flush=True)
