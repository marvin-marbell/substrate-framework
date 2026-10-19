import json,itertools,hashlib
from pathlib import Path
import numpy as np
import sympy as s
import bulk_electric_phase_consumer as bound
root=Path.cwd()
frame,scale=bound.frozen.exact_frame((root/bound.INPUTS[0]).read_bytes())
cfg=json.loads((root/bound.INPUTS[1]).read_bytes())['input_descriptor']['original_micro_config_UNCHANGED']
profiles=bound.frozen.profiles(cfg,frame,scale)
A=s.diag(-1,1,-1,1)
Z=s.diag(-1,1);X=s.Matrix([[0,1],[1,0]])
B=s.kronecker_product(s.Matrix([[0,1],[1,0]]),Z)
C=s.Matrix.vstack(s.Matrix.hstack(s.zeros(2),s.I*X),s.Matrix.hstack(-s.I*X,s.zeros(2)))
R=s.diag(s.Rational(1,2),s.Rational(1,2),-s.Rational(1,2),-s.Rational(1,2))
masks=[sum(1<<i for i in p) for p in itertools.combinations(range(4),2)]
def wedge(h):
 o=s.zeros(6)
 for j,mask in enumerate(masks):
  for p in range(4):
   for q in range(4):
    z=bound.car(mask,p,q)
    if z is not None:o[masks.index(z[0]),j]+=z[1]*h[p,q]
 return o
ops=[wedge(h) for h in [A,B,C,R]]
null=s.Matrix.vstack(*ops).nullspace();assert len(null)==1
v=s.zeros(6,5)
for j,mask in enumerate([5,3,12,10]):v[masks.index(mask),j]=1
for mask in [9,6]:v[masks.index(mask),4]=1/s.sqrt(2)
assert v.H*v==s.eye(5)
assert all(o*v==v*(v.H*o*v) for o in ops)
a,b,c,r=[s.simplify(v.H*o*v) for o in ops]
zl=s.diag(-s.Rational(1,2),s.Rational(1,2),-s.Rational(1,2),s.Rational(1,2),0)
zr=s.diag(-s.Rational(1,2),-s.Rational(1,2),s.Rational(1,2),s.Rational(1,2),0)
pl=s.zeros(5);pr=s.zeros(5)
pl[1,0]=pl[3,2]=1;pr[2,0]=pr[3,1]=1
assert a==2*(zl+zr) and r==zl-zr
assert c==-2*((pl-pl.H)/(2*s.I)+(pr-pr.H)/(2*s.I))
assert list(b[:,4])==[0,s.sqrt(2),-s.sqrt(2),0,0]
# Exact Lie-span rank, using independent flattened matrix rows.
basis=[]
def add(o):
 col=s.Matrix(list(o));old=s.Matrix.hstack(*basis) if basis else s.zeros(25,0)
 if s.Matrix.hstack(old,col).rank()>len(basis):basis.append(col);return True
 return False
for o in [a,b,c,r]:add(o)
i=0
while i<len(basis):
 x=s.Matrix(5,5,list(basis[i]))
 for y in list(basis):add(x*s.Matrix(5,5,list(y))-s.Matrix(5,5,list(y))*x)
 i+=1
assert len(basis)==10
# Actual native matrix: two equivalent Cl2 spin sectors, without changing couplings.
def cx(pair):return complex(float(pair[0]),float(pair[1]))
f=[np.array([[cx(z) for z in row] for row in m]) for m in frame]
gamma=f[1]/np.sqrt(np.trace(f[1]@f[1]).real/4)
spin=np.column_stack([np.eye(4)[:,0],gamma@np.eye(4)[:,0],np.eye(4)[:,1],gamma@np.eye(4)[:,1]])
assert np.max(abs(spin.conj().T@spin-np.eye(4)))<1e-12
native=[]
for p in profiles:
 path=next(z for z in p['paths'] if z['axis']==0 and z['length']==1)
 hop=np.array([[cx(z) for z in row] for row in path['matrix']]);t=spin.conj().T@hop@spin
 assert np.max(abs(t[:2,2:]))<1e-12 and np.max(abs(t[:2,:2]-t[2:,2:]))<1e-12
 cc=-t[0,0].real;dd=(t[0,1]/1j).real
 assert np.max(abs(t[:2,:2]-np.array([[-cc,1j*dd],[1j*dd,cc]])))<1e-12
 native.append(dict(role=p['role'],count=p['count'],onsite_exact=str(p['onsite']),c0_native=cc,ci_native=dd,spin_replicas=2*p['count']))
result=dict(kind='EXACT_NATIVE_DIMER_SO5_HARMONIC_REPRESENTATION_DESIGN_CONTROL_NOT_BLOCK_GROUND',lie_dimension=10,singlet_dimension=1,canonical_two_spin_particles_per_replica=2,Cl4_spin_replicas=native,source_initial_vector='all ac, null-trace highest weight; all126 auxiliary replicas identical, active2 distinctonsite',harmonic_dimension_formula='(n+1)(n+2)(2n+3)/6',auxiliary_n126_dimension=127*128*255//6,active_n2_dimension=14,complete_active_auxiliary_tensor_dimension=14*(127*128*255//6),SO4_basis='l0..n,jL=jR=l/2,mL,mR',h='2m(JLz+JRz)-2d(JLy+JRy)+cB_n +64(rho_active+rho_aux+u)^2',rho='JLz-JRz',B_forward='sqrt((n-l)(n+l+3)(l+1)/(2(l+2))) * product CG(j,m;1/2,q | j+1/2,m+q), weights w_(+,-)=sqrt2, w_(-,+)=-sqrt2; backward adjoint',scope='Exactly invariant selected canonical spin-sector local one-link/tree body. NOT full lattice/Floquet/ground/photon; other lattice bonds, two-links, Phi/SU4 and boundary action unpaid. No SO5 N126 matrix or stationary state computed here.',input_sha256={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in [bound.INPUTS[0],bound.INPUTS[1],'bulk_electric_phase_typed_20261010.json']},old_main_or_root_run=False,campaign_complete=False)
Path('joined_phi_u1_local_so5_design_control_20261010_receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
