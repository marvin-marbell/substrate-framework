from __future__ import annotations
import hashlib, itertools, json, time
from pathlib import Path
import sympy as s
ROOT=Path(__file__).resolve().parent
INPUT='local_so5_rep_review_recovered_corrected_freeframe_20261011_proof.json'
PIN=hashlib.sha256((ROOT/INPUT).read_bytes()).hexdigest();OWN=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();START=time.monotonic()
a,b,e,c=s.symbols('a b e c',real=True)
m=e*(a*a-b*b);d=2*e*a*b
A=s.diag(-1,1,-1,1)
B=s.kronecker_product(s.Matrix([[0,1],[1,0]]),s.diag(1,-1))
X=s.Matrix([[0,1],[1,0]]);Z=s.zeros(2)
C=Z.row_join(s.I*X).col_join((-s.I*X).row_join(Z))
R=s.diag(s.Rational(1,2),s.Rational(1,2),-s.Rational(1,2),-s.Rational(1,2))
H=m*A+c*B+d*C
Wplus=s.Matrix([[1,0],[0,1],[1,0],[0,-1]])/s.sqrt(2)
Wminus=s.Matrix([[1,0],[0,1],[-1,0],[0,1]])/s.sqrt(2)
U=s.Matrix.hstack(Wplus*s.Matrix([a,-s.I*b]),Wminus*s.Matrix([a,s.I*b]),s.I*Wplus*s.Matrix([-s.I*b,a]),s.I*Wminus*s.Matrix([s.I*b,a]))
D4=s.diag(-e+c,-e-c,e+c,e-c)
rf=(a*a-b*b)/2;rg=a*b
R4=s.Matrix([[0,rf,0,-rg],[rf,0,rg,0],[0,rg,0,rf],[-rg,0,rf,0]])
def reduce(z):return s.rem(s.expand(z),b*b+a*a-1,b)
def zero(M):return all(reduce(z)==0 for z in M)
assert zero(U.conjugate().T*U-s.eye(4))
assert zero(U.conjugate().T*H*U-D4) and zero(U.conjugate().T*R*U-R4)

def second(M,n):
    masks=[sum(1<<q for q in comb) for comb in itertools.combinations(range(4),n)]
    out=s.zeros(len(masks))
    for j,mask in enumerate(masks):
        for p in range(4):
            for q in range(4):
                if mask>>q&1:
                    sign=(-1)**((mask&((1<<q)-1)).bit_count());tail=mask^(1<<q)
                    if not tail>>p&1:
                        sign*=(-1)**((tail&((1<<p)-1)).bit_count())
                        out[masks.index(tail|(1<<p)),j]+=sign*M[p,q]
    return out,masks
h2,_=second(D4,2);r2,masks2=second(R4,2)
# Consistent free5 joint phase: highest pair has a minus sign, not an
# independently chosen density phase. This matches corrected 102966f3.
V=s.zeros(6,5)
for j,(mask,coef) in enumerate([(3,1),(5,1),(9,1/s.sqrt(2)),(10,1), (12,-1)]):V[masks2.index(mask),j]=coef
V[masks2.index(6),2]=1/s.sqrt(2)
D5=s.diag(-2*e,2*c,0,-2*c,2*e)
R5=s.zeros(5)
for j in [1,3]:
    for i,z in [(0,rg),(2,s.sqrt(2)*rf),(4,-rg)]:R5[i,j]=R5[j,i]=z
assert zero(h2*V-V*D5) and zero(r2*V-V*R5)
assert zero(V.T*V-s.eye(5))
wrongV=V.copy();wrongV[:,4]=-wrongV[:,4]
assert not zero(r2*wrongV-wrongV*R5)
# Full number3 CAR, not an inferred scalar hole or a neutral gap.
h3,masks3=second(H,3);r3,_=second(R,3)
J=s.zeros(4)
for p in range(4):J[masks3.index(15^(1<<p)),p]=(-1)**p
assert zero(J.T*h3*J+H.T) and zero(J.T*r3*J+R.T)
Q=s.kronecker_product(s.eye(2),X)
assert zero(Q*(-H.T)*Q-H) and zero(Q*(-R.T)*Q+R)
S4=s.diag(1,-1,1,-1);S5=s.diag(1,-1,1,-1,1)
assert zero(S4*D4*S4-D4) and zero(S4*R4*S4+R4)
assert zero(S5*D5*S5-D5) and zero(S5*R5*S5+R5)
result={'kind':'EXACT_JOINT_FREE_SPINOR_VECTOR_DENSITY_AND_NUMBER3_CAR_PHASE_CONTROL','oneparticle_free_energies':['-e+c0','-e-c0','e+c0','e-c0'],'spinor_density_entries':'R01=R23=m/(2e), R03=-d/(2e), R12=d/(2e); symmetric partners, other entries0','relations':'e^2=m^2+d^2, m>0; a^2+b^2=1, m=e(a^2-b^2), d=2eab','exact_joint_spinor_D_R':True,'matches_existing_corrected_free5_joint_pair':True,'wrong_highest_wedge_phase_rejected':True,'full_number3_hole_D_R_verified':True,'number3_pair_equivalent_to_number1':True,'whole_free_R_sign_flip_commutes_D':True,'affine_half_offsets_have_same_spectrum_by_whole_R_flip':True,'selected_charged_spectrum_certified':False,'charged_gap_certified':False,'physicalPhi_ground':False,'phase':False,'pole':False,'next_input':'Containing charged branch: active vector count1 plus auxiliary126 plus active spinor4, OR active2 plus auxiliary125 plus auxiliary spinor4. Actual one-hop annihilation/creation image containment remains to bind; no gap from this algebra alone. Hcell=D+64(Rtotal+u)^2 with u=+/-1/2, plus external bridge64 perunit flow in the two-cell source chart.','input_sha256':{INPUT:PIN},'own_source_sha256':OWN,'source_and_input_unchanged':hashlib.sha256((ROOT/INPUT).read_bytes()).hexdigest()==PIN and hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==OWN,'elapsed_seconds':time.monotonic()-START}
(ROOT/'charged_spinor_joint_free_phase_control_20261011_receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
