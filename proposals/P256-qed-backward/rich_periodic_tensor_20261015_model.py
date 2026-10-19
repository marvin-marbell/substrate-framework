"""Gauge-exact finite TeNPy MPS/MPO; explicit spinful Jordan-Wigner factors.

No inherited evolved state. 32 physical sites (8 Fock4,24 electric3), int64
U(1) encoding of ALL local Gauss laws, N and Sz. All tensors are graded only
by the conserved integer; fermionic CAR is explicit, never inferred from q.
"""
from __future__ import annotations
import numpy as np
from tenpy.linalg import np_conserved as npc
from tenpy.models.lattice import Lattice
from tenpy.models.model import CouplingModel
from tenpy.networks.site import Site
from tenpy.networks.mps import MPS
from tenpy.networks.mpo import MPOEnvironment
from rich_periodic_photon_20261015_model import geometry, source_geometry

POSITIONS, EDGES, INCIDENCE = geometry()
PREFIX = 'rich_periodic_tensor_20261015_'


def local_matrices():
    up = np.zeros((4, 4)); down = up.copy()
    for n in range(4):
        if n & 1: up[n ^ 1, n] = 1
        if n & 2: down[n ^ 2, n] = (-1)**(n & 1)
    return up, down


class GaugeTensor:
    def __init__(self):
        self.sites = []
        self.matter = {}; self.field = {}
        self.ops = []
        self.chinfo = npc.ChargeInfo([1], ['encoded_N_G_Sz'])
        up, down = local_matrices()
        self.c = [up, down]
        self.parity = np.diag([1, -1, -1, 1])
        n = np.array([0, 1, 1, 2]); sz = np.array([0, 1, -1, 0])
        for v in range(8):
            self.matter[v] = len(self.sites)
            q = n + 64*((n-1)*32**v + sz*32**8)
            self.add_site(q, ['0', 'u', 'd', 'ud'], {'N':np.diag(n), 'Rho':np.diag(n-1), 'Rho2':np.diag((n-1)**2)})
            for e in range(3*v, 3*v+3):
                a, b, _ = EDGES[e]
                self.field[e] = len(self.sites)
                E = np.array([-1, 0, 1])
                self.add_site(64*E*(32**a-32**b), ['-', '0', '+'], {'E':np.diag(E), 'E2':np.diag(E**2)})
        self.lat = Lattice([1], self.sites, bc='open', bc_MPS='finite')
        self.hop_terms = []
        for e, (a, b, _) in enumerate(EDGES):
            for direction in (1, -1):
                src, dst = (a,b) if direction == 1 else (b,a)
                for spin in (0,1):
                    factors = {}
                    # Ordered product c_dst^dagger c_src of explicit GLOBAL JW factors.
                    for v, dagger in ((dst,True),(src,False)):
                        for w in range(v):
                            i = self.matter[w]
                            factors[i] = factors.get(i,np.eye(4)) @ self.parity
                        i = self.matter[v]
                        factors[i] = factors.get(i,np.eye(4)) @ (self.c[spin].T if dagger else self.c[spin])
                    factors[self.field[e]] = np.diag(np.ones(2), -direction)
                    term = [(self.register(i,A), i) for i,A in sorted(factors.items()) if not np.array_equal(A,np.eye(A.shape[0]))]
                    self.hop_terms.append((-1.,term,e,direction))
        self.diag_terms = [(2.,[('Rho2',i)]) for i in self.matter.values()] + [(1/32,[('E2',i)]) for i in self.field.values()]
        self.H = self.mpo(self.diag_terms + [(x,t) for x,t,_,_ in self.hop_terms])

    def add_site(self, q, labels, ops):
        leg = npc.LegCharge.from_qflat(self.chinfo, np.asarray(q,dtype=np.int64)[:,None])
        self.sites.append(Site(leg, labels, sort_charge=True, **ops))
        self.ops.append({})

    def register(self, i, A):
        key = A.tobytes()
        if key not in self.ops[i]:
            name = 'X'+str(len(self.ops[i]))
            self.sites[i].add_op(name, A, need_JW=False, permute_dense=True)
            self.ops[i][key] = name
        return self.ops[i][key]

    def mpo(self, terms):
        model = CouplingModel(self.lat)
        for strength, term in terms:
            if abs(strength) > 1e-15:
                model.add_local_term(strength, [(op,[0,i]) for op,i in term])
        return model.calc_H_MPO()

    def seed(self):
        labels = ['0']*32
        for v,i in self.matter.items(): labels[i] = 'u' if v%2 == 0 else 'd'
        return MPS.from_product_state(self.sites, labels, dtype=complex, unit_cell_width=1)

    def step_mpo(self, dt):
        return self.mpo([(1.,[('Id',0)])] + [(-dt*x,t) for x,t in self.diag_terms] + [(-dt*x,t) for x,t,_,_ in self.hop_terms])

    def sources(self, q):
        geo = source_geometry(q)
        result = []
        for p in range(2):
            coeff = geo['coefficients'][p]
            result.append(self.mpo([(coeff[e]/16,[('E',i)]) for e,i in self.field.items()]))
        for p in range(2):
            coeff = geo['coefficients'][p]
            result.append(self.mpo([(1j*d*x*coeff[e],t) for x,t,e,d in self.hop_terms]))
        contact_mpos = []
        for p in range(2):
            row = []
            for r in range(2):
                terms = [(-x*geo['coefficients'][p,e]*geo['coefficients'][r,e],t) for x,t,e,_ in self.hop_terms if abs(geo['coefficients'][p,e]*geo['coefficients'][r,e])>1e-15]
                # Orthogonal Cartesian link directions have EXACTLY zero mixed
                # twist derivative; no empty-charge MPO is needed for scalar0.
                row.append(self.mpo(terms) if terms else None)
            contact_mpos.append(row)
        return geo, result, contact_mpos

    def loop_sources(self, q):
        from rich_periodic_photon_20261015_model import loops
        result = []
        for length in (2,4):
            terms = []
            count = 0
            for flux in loops():
                support = [e for e,f in enumerate(flux) if f]
                if len(support) != length: continue
                # For winding loops choose only axes transverse to q. This
                # retains differences of parallel loops at nonzero q on L2.
                if length == 2 and q[EDGES[support[0]][2]]: continue
                origin = EDGES[support[0]][0]
                phase = (-1.)**int(POSITIONS[origin]@np.asarray(q))
                count += 1
                for direction in (1,-1):
                    term = []
                    for e in support:
                        T = np.diag([1.,1.],-direction*flux[e])
                        i = self.field[e]
                        term.append((self.register(i,T),i))
                    terms.append((phase*direction/(2j*np.sqrt(8)),term))
            result.append((self.mpo(terms),count))
        return result


def matrix_element(bra, op, ket):
    # TeNPy MPOEnvironment explicitly excludes both MPS.norm prefactors;
    # MPS.overlap includes them. Keep mixed K and source Gram in one metric.
    return complex(np.conj(bra.norm)*ket.norm*MPOEnvironment(bra, op, ket).full_contraction(0))


def physical_tensor_norm_benchmark(model):
    ket = model.seed()
    image = ket.copy()
    model.H.apply_naively(image)
    image.canonical_form(renormalize=False)
    exact = float(np.real(image.overlap(image)))
    mixed = matrix_element(image,model.H,ket)
    image.norm *= 3.25
    scaled = matrix_element(image,model.H,ket)
    result = {'physical_H_image_norm2':exact,'mixed_H_image_real':mixed.real,
              'mixed_error':float(abs(mixed-exact)),
              'scaled_bra_factor':3.25,'scaled_H_image_real':scaled.real,
              'scaling_error':float(abs(scaled-3.25*mixed)),
              'consumer':'actual32node spinfulCAR/gaugeMPO H acting on fresh Gauss0 Neel state; finite-MPS norm factors, not copy/wiring'}
    if max(result['mixed_error'],result['scaling_error'])>1e-9:
        raise ArithmeticError(result)
    return result


def encoded_charge_proof():
    return {'encoding':'N+64[sum_v G_v 32^v+(2Sz)32^8],target8',
            'bounds':{'N':[0,16],'G_v':[-7,7],'2Sz':[-8,8]},
            'proof':'Modulo64 forces N=8 as |N-8|<=8<64. Then lowest signed digit G0 is divisible32 and |G0|<=7<32, hence zero; divide32 and repeat through G7; remaining2Sz=0. Integer no-carry decoding, not unsigned-digit assumption.',
            'int64_absolute_upper_bound':int(16+64*(7*sum(32**v for v in range(8))+8*32**8)),
            'fermion_parity':'q mod2=n mod2; nevertheless explicit JW used. Field contributions even. No intrinsic grading assumed.'}


def small_benchmark():
    up, down = local_matrices(); P = np.diag([1,-1,-1,1])
    c = [np.kron(up,np.eye(4)),np.kron(down,np.eye(4)),np.kron(P,up),np.kron(P,down)]
    car = max(np.linalg.norm(a@b.T+b.T@a-(np.eye(16) if i==j else 0)) for i,a in enumerate(c) for j,b in enumerate(c))
    E = np.diag([-1,0,1]); U = np.diag([1.,1.],-1)
    N0 = c[0].T@c[0]+c[1].T@c[1]; N1 = c[2].T@c[2]+c[3].T@c[3]
    G0 = np.kron(N0-np.eye(16),np.eye(3))+np.kron(np.eye(16),E)
    G1 = np.kron(N1-np.eye(16),np.eye(3))-np.kron(np.eye(16),E)
    H = np.kron(2*((N0-np.eye(16))@(N0-np.eye(16))+(N1-np.eye(16))@(N1-np.eye(16))),np.eye(3))+np.kron(np.eye(16),E@E/32)
    for s in range(2):
        T = np.kron(c[2+s].T@c[s],U); H -= T+T.T
    keep = np.flatnonzero((np.diag(G0)==0)&(np.diag(G1)==0))
    eigen = np.linalg.eigvalsh(H[np.ix_(keep,keep)])
    expected=(4+1/32-np.sqrt((4+1/32)**2+16))/2
    return {'CAR_error':car,'Gauss_commutators':[float(np.linalg.norm(H@G-G@H)) for G in (G0,G1)],'physical_dimension':len(keep),'ground':float(eigen[0]),'analytic_error':float(abs(eigen[0]-expected)),'full_dimension':48}
