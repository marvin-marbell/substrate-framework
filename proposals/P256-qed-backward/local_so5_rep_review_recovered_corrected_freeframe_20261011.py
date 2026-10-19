"""New exact binary-profile/native-phase/free5 binding; no old producer entrypoint."""
from pathlib import Path
import hashlib
import json
import numpy as np
import sympy as s
ROOT=Path(__file__).resolve().parent
PREFIX='local_so5_rep_review_recovered_'
inputs=['joined_phi_u1_ground_photon_collective_delivered_actual_20261010.npz','induced_phi_uniform_selfconsistent_full_normal_quantum_final_actual_receipt.json','local_so5_resum_source_20261011.py','local_so5_resum_freeframe_proof_20261011.json']
hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in inputs}
proof=json.loads((ROOT/inputs[3]).read_text())
cfg=json.loads((ROOT/inputs[1]).read_text())['input_descriptor']['original_micro_config_UNCHANGED']
with np.load(ROOT/inputs[0],allow_pickle=False) as data:phi=data['Phi_binary_section']
rat=lambda z:s.Rational(float(z))
PHI=[s.Matrix([[rat(z.real)+s.I*rat(z.imag) for z in row] for row in matrix]) for matrix in phi]
h=s.Matrix([[1,0,1,0],[0,1,0,1],[-1,0,1,0],[0,-1,0,1]])
f=[h.T*x*h/2 for x in PHI]
scale=rat(phi[0,0,2].real)
assert f[0]==scale*s.diag(-1,-1,1,1)
q=s.trace(f[1]*f[1])/4
assert f[1]*f[1]==q*s.eye(4)
gamma=f[1]/s.sqrt(q)
spin=s.Matrix.hstack(s.eye(4)[:,0],gamma[:,0],s.eye(4)[:,1],gamma[:,1])
assert spin.H*spin==s.eye(4)
# Literal RotationBand/scalar_paths binary coefficients: SELECTED trace has exact
# unit normalization, nearest coefficient -r/a and -i*v/(2a), onsite m+4.5r/a.
nearest_mass=rat(-cfg['light_r']/cfg['a'])
nearest_derivative=rat(-cfg['light_v']/(2*cfg['a']))
T=nearest_mass*f[0]+s.I*nearest_derivative*f[1]
t=s.simplify(spin.H*T*spin)
c0=s.simplify(-t[0,0]);d=s.simplify(t[0,1]/s.I)
block=s.Matrix([[-c0,s.I*d],[s.I*d,c0]])
assert t==s.diag(block,block)
assert c0<0 and d<0
mvar,dvar,evar,cvar=s.symbols('m d e c',real=True,nonzero=True)
locals_=dict(m=mvar,d=dvar,e=evar,c=cvar,I=s.I)
U=s.Matrix([[s.sympify(z,locals=locals_) for z in row] for row in proof['U_columns_exact_symbolic']])
R=s.Matrix([[s.sympify(z,locals=locals_) for z in row] for row in proof['rho_free_exact_symbolic']])
# Exact full Fock exterior action, separately for binary physical spin t and source.
Z=s.diag(1,-1);I=s.eye(2);lower=s.Matrix([[0,1],[0,0]])
a=[s.kronecker_product(*[I if k>p else(lower if k==p else Z) for k in reversed(range(4))]) for p in range(4)]
V=s.zeros(16,5)
for j,mask in enumerate([5,3,12,10]):V[mask,j]=1
V[9,4]=V[6,4]=1/s.sqrt(2)
def wedge(mat):return s.simplify(V.H*sum((mat[p,q]*a[p].H*a[q] for p in range(4) for q in range(4)),s.zeros(16))*V)
A1=s.diag(-1,1,-1,1)
B1=s.kronecker_product(s.Matrix([[0,1],[1,0]]),s.diag(1,-1))
C1=s.BlockMatrix([[s.zeros(2),s.I*s.Matrix([[0,1],[1,0]])],[-s.I*s.Matrix([[0,1],[1,0]]),s.zeros(2)]]).as_explicit()
rho1=s.diag(s.Rational(1,2),s.Rational(1,2),-s.Rational(1,2),-s.Rational(1,2))
A,B,C,rho=map(wedge,[A1,B1,C1,rho1])
S=s.diag(1,-1,-1,1)
assert S.H*A1*S==A1 and S.H*B1*S==-B1 and S.H*C1*S==C1 and S.H*rho1*S==rho1
F=s.diag(1,1,1,1,-1)
assert F*A*F==A and F*B*F==-B and F*C*F==C and F*rho*F==rho
D=s.diag(-2*evar,2*cvar,0,-2*cvar,2*evar)
def zero_mod(matrix):
    for expr in matrix:
        num=s.cancel(expr).as_numer_denom()[0]
        if s.simplify(s.Poly(num,evar).rem(s.Poly(evar**2-mvar**2-dvar**2,evar)).as_expr())!=0:return False
    return True
Ucorr=F*U
Hsource=mvar*A+cvar*B+dvar*C
Hphysical=mvar*A-cvar*B+dvar*C
assert zero_mod(Ucorr.H*Ucorr-s.eye(5))
assert zero_mod(Hsource*Ucorr-Ucorr*D)
assert zero_mod(Ucorr.H*rho*Ucorr-R)
assert zero_mod(Hphysical*U-U*D)
assert zero_mod(U.H*rho*U-R)
assert not zero_mod(Hsource*U-U*D)
G=s.zeros(5);G[0,3]=G[3,0]=-1;G[1,2]=G[2,1]=1;G[4,4]=1
assert F.T*G*F==G
records=[]
for role,count in [('active',1),('auxiliary',63)]:
    mass=rat(cfg[role+'_m']+4.5*cfg['light_r']/cfg['a'])*scale
    expected=next(p for p in proof['profiles'] if p['role']==role)
    assert mass==s.Rational(expected['m_exact']) and c0==s.Rational(expected['c0_exact']) and d==s.Rational(expected['d_exact'])
    assert mass**2+d**2==s.Rational(expected['e_squared_exact'])
    physical4=s.BlockMatrix([[mass*s.diag(-1,1),block],[block.H,mass*s.diag(-1,1)]]).as_explicit()
    assert physical4==mass*A1-c0*B1+d*C1
    assert S.H*physical4*S==mass*A1+c0*B1+d*C1
    records.append(dict(role=role,count=count,spin_replicas=2*count,m_exact=str(mass),c0_exact=str(c0),d_exact=str(d),e_squared_exact=str(mass**2+d**2),c0_float=float(c0),d_float=float(d),native_binary_Cl2_split_exact=True,source_phase_binding_exact=True))
result=dict(kind='NEW_EXACT_BINARY_NATIVE_PHASE_BOUND_FREE5_INTERTWINER_NOT_GROUND_ADMISSION',source_and_input_sha256=hashes,profiles=records,native_basis_order=['ac','ab','cd','bd','(ad+bc)/sqrt2'],natural_binary_hopping_2x2=[[str(z) for z in row] for row in block.tolist()],source_oneparticle_phase=[[str(z) for z in row] for row in S.tolist()],frozen_U_same_source_basis_identity=False,frozen_U_natural_physical_basis_identity=True,corrected_source_U=[[str(z) for z in row] for row in Ucorr.tolist()],free_energy_diagonal=['-2e','2c0','0','-2c0','2e'],free_density_matrix=[[str(z) for z in row] for row in R.tolist()],algebraic_relation='e^2=m^2+d^2,e>0,m>0,c0<0',corrected_source_unitarity_H_density_identities=True,phase_preserves_rho_G_and_ac_up_to_global_sign=True,arbitrary_replica_extension='F^tensor n preserves G-traceless Sym^n(5), sends ac^tensor n to itself up to phase, and intertwines collective native A/B/C/rho. Independent trace-kernel controls and exact CG recurrence in recovered control support fixed branching normalization. Active2 and auxiliary126 have distinct m; tensor dimension14*690880=9672320.',parent_pair_reusable_after_new_input_binding=True,parent_inertia_or_Ritz_certificate_reviewed=False,scope='Selected frozen binary two-site one-edge tree, canonical Cl2 per spin, selected top harmonic active2/aux126. No ambient fullground or lattice/field/pole inference.',own_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
assert all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha for name,sha in hashes.items())
result['source_and_inputs_unchanged']=True
(ROOT/(PREFIX+'corrected_freeframe_20261011_proof.json')).write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2),flush=True)
