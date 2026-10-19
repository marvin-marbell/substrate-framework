"""Independent bounded controls; imports no frozen producer or checker."""
import itertools
import json
from collections import deque
import numpy as np
from scipy.linalg import expm
from scipy.sparse import csr_matrix, diags, eye, kron, bmat
from scipy.sparse.linalg import expm_multiply


def lift(h, masks):
    lookup = {m: i for i, m in enumerate(masks)}
    rows, cols, vals = [], [], []
    for j, m in enumerate(masks):
        for p, q in zip(*np.nonzero(h)):
            if not (m >> q) & 1:
                continue
            s = (-1)**((m & ((1 << q)-1)).bit_count())
            a = m ^ (1 << q)
            if (a >> p) & 1:
                continue
            s *= (-1)**((a & ((1 << p)-1)).bit_count())
            rows.append(lookup[a | (1 << p)])
            cols.append(j)
            vals.append(s*h[p, q])
    return csr_matrix((vals, (rows, cols)), shape=(len(masks), len(masks)))


def sign(a):
    return (-1)**sum(a[i] > a[j] for i in range(len(a)) for j in range(i+1, len(a)))

rng = np.random.default_rng(910111)
x = rng.normal(size=(4, 4))+1j*rng.normal(size=(4, 4))
x = (x+x.conj().T)/2
x -= np.trace(x)*np.eye(4)/4
pairs = list(itertools.combinations(range(4), 2))
m4 = [sum(1 << i for i in p) for p in pairs]
sing = np.zeros((6, 6))
for i, p in enumerate(pairs):
    comp = tuple(k for k in range(4) if k not in p)
    sing[i, pairs.index(comp)] = sign(p+comp)/np.sqrt(6)
g = lift(x, m4).toarray()
singlet_error = np.linalg.norm(g@sing+sing@g.T)
rdm_error = np.linalg.norm(sing@sing.T-np.eye(6)/6)

m8 = [m for m in range(256) if m.bit_count() == 4]
idx = {m: i for i, m in enumerate(m8)}
v = np.zeros(4900, complex)
for p in pairs:
    cp = tuple(i for i in range(4) if i not in p)
    for q in pairs:
        cq = tuple(i for i in range(4) if i not in q)
        a = sum(1 << i for i in p)+sum(1 << (4+i) for i in q)
        b = sum(1 << i for i in cp)+sum(1 << (4+i) for i in cq)
        v[idx[a]*70+idx[b]] = sign(p+cp)*sign(q+cq)/6
T = rng.normal(size=(4, 4))+1j*rng.normal(size=(4, 4))
h = np.block([[np.zeros((4, 4)), T], [T.conj().T, np.zeros((4, 4))]])
L = lift(h, m8)
I = eye(70)
H = kron(L, I)+kron(I, L)
r = np.array([(m & 15).bit_count()-2 for m in m8])
rho = (r[:, None]+r[None, :]).ravel()
D = diags(2*rho*rho)
C = D@H-H@D
wrongD = kron(diags(2*r*r), I)+kron(I, diags(2*r*r))
wrongC = wrongD@H-H@wrongD
step = .017
state = expm_multiply(-step*C, v)
wrongstate = expm_multiply(-step*wrongC, v)
# Differential covariance: derivative generator is i[rho,C].
R = diags(rho)
Cp = 1j*(R@C-C@R)
Cpp = -(R@(R@C-C@R)-(R@C-C@R)@R)
z = csr_matrix(C.shape)
block = bmat([[-step*C,z,z],[-step*Cp,-step*C,z],[-step*Cpp,-2*step*Cp,-step*C]], format='csr')
action = expm_multiply(block, np.concatenate([v, np.zeros(9800)]))
j1, j2 = action[4900:9800], action[9800:]
gauss1 = np.linalg.norm(j1-1j*rho*state)
gauss2 = np.linalg.norm(j2+rho*rho*state)
wrong_drop_longitudinal = np.linalg.norm(j1)

allm = list(range(256))
P = np.diag([1]*4+[0]*4)
s = h@P-P@h
S = lift(s, allm).toarray()
comm_error = np.linalg.norm(S-(lift(h,allm)@lift(P,allm)-lift(P,allm)@lift(h,allm)).toarray())
sectors = {str(k): float(np.linalg.norm(S[np.ix_([m for m in allm if m.bit_count()==k], [m for m in allm if m.bit_count()==k])])) for k in range(9)}
f = expm(-.021*S)[:,15]
one = expm(-.021*s)@P@expm(.021*s)
neutral_error = abs(np.vdot(f,lift(h,allm)@f)-np.trace(h@one))
# Shared-field product has both mixed terms in its second derivative.
A = np.array([[0,1],[-1,0]],complex)
B = 1j*np.array([[1,0],[0,-1]],complex)
q = .31
U, V = expm(q*A), expm(q*B)
fullsecond = A@A@U@V+2*A@U@B@V+U@B@B@V
omitted = A@A@U@V+U@B@B@V
shared_cross_gap = np.linalg.norm(fullsecond-omitted)

# Independent spanning-tree native N4 geometry.
n=4
sites=list(itertools.product(range(n),repeat=3)); lookup={x:i for i,x in enumerate(sites)}
edges=[]
for site in sites:
    for axis in range(3):
        t=list(site);t[axis]=(t[axis]+1)%n
        edges.append((lookup[site],lookup[tuple(t)]))
inc=np.zeros((64,192),int)
adj=[[] for _ in sites]
for e,(a,b) in enumerate(edges):
    inc[a,e]=1;inc[b,e]=-1
    adj[a].append((b,e,1));adj[b].append((a,e,-1))
seen={0}; queue=deque([0]); paths=np.zeros((64,192),int); tree=set()
while queue:
    a=queue.popleft()
    for b,e,s in adj[a]:
        if b not in seen:
            seen.add(b);queue.append(b);tree.add(e)
            paths[b]=paths[a];paths[b,e]+=s
chords=[e for e in range(192) if e not in tree]
Bmat=np.zeros((192,len(chords)),int)
for j,e in enumerate(chords):
    a,b=edges[e]; Bmat[:,j]=paths[a]-paths[b];Bmat[e,j]+=1
Gram=Bmat.T@Bmat
beta=np.linspace(-.04,.06,len(chords))
metric_error=abs(np.linalg.norm(Bmat@beta)**2-beta@Gram@beta)
wrong_identity_gap=abs(beta@Gram@beta-beta@beta)
# Chain rule for holonomy z=B^T theta gives E=-i B grad_z.
k=np.arange(len(chords))%3-1
raw_E=Bmat@k
wrong_config_E=Bmat@np.linalg.solve(Gram.astype(float), k.astype(float))
wrong_config_gap=np.linalg.norm(raw_E-wrong_config_E)
trace=int(np.trace(Gram)); d=len(chords)
from fractions import Fraction
w=Fraction(1,64*d*trace)
packet_price=64*trace*d*w*w
out=dict(kind='INDEPENDENT_CAR_GAUSS_DIFFERENTIAL_CYCLE_CONTRARY_CONTROLS', seed=910111, singlet_error=float(singlet_error), wedge_rdm_error=float(rdm_error), charged_pointnorm_error=float(abs(np.vdot(state,state)-1)), fullrho_tangent_error=float(np.linalg.norm(C@v-2*H@v)), wrong_percopy_state_gap=float(np.linalg.norm(state-wrongstate)), gauss_first_error=float(gauss1), gauss_second_error=float(gauss2), wrong_drop_HC_gap=float(wrong_drop_longitudinal), neutral_allsector_commutator_error=float(comm_error), neutral_number_sector_norms=sectors, neutral_onebody_fullCAR_error=float(neutral_error), omitted_shared_second_jet_cross_gap=float(shared_cross_gap), N=4, cycle_dimension=d, cycle_divergence_zero=bool(not np.any(inc@Bmat)), chord_identity=bool(np.array_equal(Bmat[chords],np.eye(d))), gram_trace=trace, exact_packet_upper=str(packet_price), metric_error=float(metric_error), wrong_identity_metric_gap=float(wrong_identity_gap), wrong_configuration_cycle_E_gap=float(wrong_config_gap), controls_not_global_proof=True)
assert singlet_error<1e-13 and rdm_error<1e-13
assert out['charged_pointnorm_error']<1e-13 and gauss1<1e-13 and gauss2<1e-13
assert out['wrong_percopy_state_gap']>1e-4 and wrong_drop_longitudinal>.01
assert comm_error<1e-12 and neutral_error<1e-12
assert all(sectors[str(k)]>0 for k in range(1,8))
assert shared_cross_gap>1 and not np.any(inc@Bmat)
assert metric_error<1e-12 and wrong_identity_gap>.01 and wrong_config_gap>1
assert d==129 and trace==738 and packet_price==w
print(json.dumps(out, sort_keys=True))
