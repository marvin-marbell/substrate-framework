"""Exact disjoint-pair CAR kernels; full intermediate pair complements retained."""
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy import sparse
import rich_composite_geometry_20261015_model as model
from rich_composite_geometry_20261015_science import load
from rich_composite_geometry_20261015_runtime import primitive_for
from rich_composite_geometry_20261015_cloud_probe import neutral_state,inner
from rich_composite_geometry_20261015_source_eom import bare_state


def sources(root):
    data=load(root/'rich_composite_geometry_20261015_correlated_onehop_neutral1_neutral.pkl');primitive=primitive_for(data);coupling=load(root/'rich_composite_geometry_20261015_correlated_onehop_reference1_J0_coupling.pkl');states=np.load(root/'rich_composite_geometry_20261015_correlated_onehop_reference1_J0_states.npz')
    omega=neutral_state(data,coupling,states['states'][:,0]);baryon=model.source_terms([0,0,0],'half_triplet',1);lepton=[([(model.mode(2,1,0,3),True)],1.+0j)]
    white=bare_state(primitive,omega,lepton);packets={'0':omega,'B':bare_state(primitive,omega,baryon),'L':white,'BL':bare_state(primitive,white,baryon)}
    return primitive,packets,{name:float(inner(packet,packet).real) for name,packet in packets.items()}


def raw_fock_action(primitive,pair,component,flavour,white,dagger):
    # Independent32-bit Fock expansion, NOT tensor primitive.hop/string reuse.
    output=defaultdict(complex);spin=np.eye(2) if component==0 else model.PAULI[component-1]
    for first,a in primitive.local.vectors[pair[0]].items():
        for second,b in primitive.local.vectors[pair[1]].items():
            mask=first|(second<<16)
            for s,t in np.ndindex(2,2):
                if spin[s,t]==0:continue
                for member in ([3] if white else range(3)):
                    creation=16+8*flavour+4*s+member;annihilation=8*flavour+4*t+member;coefficient=spin[s,t]
                    if dagger:creation,annihilation=annihilation,creation;coefficient=coefficient.conjugate()
                    if not(mask>>annihilation)&1:continue
                    sign=(-1)**((mask&((1<<annihilation)-1)).bit_count());changed=mask^(1<<annihilation)
                    if(changed>>creation)&1:continue
                    sign*=(-1)**((changed&((1<<creation)-1)).bit_count());changed|=1<<creation
                    output[changed]+=a*b*sign*coefficient
    return output


class PairAlgebra:
    def __init__(self,primitive,packets,species):
        self.primitive=primitive;self.species=species;self.white=species=='white';self.edge=2 if self.white else 0;self.offset=2 if self.white else 0
        self.pairs=sorted({key[self.offset:self.offset+2] for packet in packets.values() for key in packet});self.lookup={pair:i for i,pair in enumerate(self.pairs)};self.output_pairs=list(self.pairs);self.output_lookup=dict(self.lookup);self.raw={};self.checks=[]
        self.reference=next(iter(packets['0']));flavours=(1,) if self.white else (0,1)
        for component in range(4):
            for flavour in flavours:
                for dagger in (False,True):
                    rows=[];columns=[];values=[];sample=None
                    for column,pair in enumerate(self.pairs):
                        key=list(self.reference);key[self.offset:self.offset+2]=pair;key[-1]=0;image=primitive.hop({tuple(key):1.+0j},self.edge,flavour,self.white,component,dagger)
                        pair_image=defaultdict(complex)
                        for target,value in image.items():
                            out=target[self.offset:self.offset+2]
                            if target[-1]!=0:raise AssertionError('Chosen tree/white hop changed PW: factorization invalid')
                            if out not in self.output_lookup:self.output_lookup[out]=len(self.output_pairs);self.output_pairs.append(out)
                            rows.append(self.output_lookup[out]);columns.append(column);values.append(value);pair_image[out]+=value
                        if sample is None and pair_image:sample=(pair,pair_image)
                    self.raw[component,flavour,dagger]=(rows,columns,values)
                    # Four physical raw-norm checks cover spin, flavour and reverse CAR signs.
                    selected=(component,flavour,dagger) in ([(0,1,False),(2,1,True)] if self.white else [(0,0,False),(1,1,True)])
                    if selected and sample is not None:
                        expanded=raw_fock_action(primitive,sample[0],component,flavour,self.white,dagger);fock_norm=sum(abs(v)**2 for v in expanded.values());coordinate_norm=sum(abs(v)**2 for v in sample[1].values());error=abs(fock_norm-coordinate_norm)
                        self.checks.append(dict(component=component,flavour=flavour,dagger=dagger,independent_Fock_norm2=fock_norm,pair_coordinate_norm2=coordinate_norm,error=error))
                        if error>2e-10:raise AssertionError(('Independent CAR pair norm mismatch',self.checks[-1]))
        shape=(len(self.output_pairs),len(self.pairs));self.raw={key:sparse.coo_matrix((value[2],(value[0],value[1])),shape=shape).tocsr() for key,value in self.raw.items()};self.cache={}
        # Spectator/Jordan-Wigner factorization on actual four source families.
        for packet in packets.values():
            key=next(iter(packet));pair=key[self.offset:self.offset+2];column=self.lookup[pair]
            for component in (0,2):
                image=primitive.hop({key:1.+0j},self.edge,1,self.white,component,False);direct={target:coefficient for target,coefficient in image.items()}
                col=self.raw[component,1,False].getcol(column).tocoo();factored={}
                for row,value in zip(col.row,col.data):
                    target=list(key);target[self.offset:self.offset+2]=self.output_pairs[row];factored[tuple(target)]=value
                if max((abs(direct.get(k,0)-factored.get(k,0)) for k in direct.keys()|factored.keys()),default=0)>2e-12:raise AssertionError('Global spectator CAR parity factorization failed')
    def operator(self,kind,component,theta):
        key=(kind,component,float(theta))
        if key not in self.cache:
            matrix=sparse.csr_matrix((len(self.output_pairs),len(self.pairs)),dtype=complex)
            for flavour in ((1,) if self.white else (0,1)):
                charge=-1. if self.white else (2/3 if flavour==0 else -1/3)
                for dagger in (False,True):
                    direction=-1 if dagger else 1;phase=np.exp(1j*direction*charge*theta);base=self.primitive.graph['weights'][self.edge]*(self.primitive.p['scalar_coupling'] if component==0 else 1j*self.primitive.p['vector_velocity']*direction)
                    coefficient=base*(phase-1 if kind=='delta' else -1j*direction*charge*phase if kind=='current' else -charge*charge*phase)
                    matrix+=coefficient*self.raw[component,flavour,dagger]
            self.cache[key]=matrix.tocsr()
        return self.cache[key]
    def single(self,kind,component,theta):return self.operator(kind,component,theta)[:len(self.pairs)].tocsr()
    def product(self,left,right):return (left.conj().T@right).tocsr()
    def receipt(self):
        full=self.operator('current',0,0.);projected=full[:len(self.pairs)];missing=full.conj().T@full-projected.conj().T@projected
        return dict(species=self.species,input_pair_dimension=len(self.pairs),full_onehop_output_pair_dimension=len(self.output_pairs),independent_Fock_checks=self.checks,unprojected_pair_product_complement_Frobenius=float(sparse.linalg.norm(missing)),unprojected_pair_product_complement_trace=float(missing.diagonal().sum().real),scope='KdagK uses ALL generated two-site CAR intermediates; only final source expectation is contracted. No source-PAP replacement, no evolved/output Gram input.')


class SourceContraction:
    def __init__(self,packet,q,w):
        self.keys=list(packet);self.values=np.array([packet[key] for key in self.keys]);self.bra=self.values.conj();self.q=q;self.w=w
        self.indices=[(q.lookup[key[:2]],w.lookup[key[2:4]],key[-1]) for key in self.keys];self.lookup={key:i for i,key in enumerate(self.indices)}
        self.qlocal=sorted({key[0] for key in self.indices});self.wlocal=sorted({key[1] for key in self.indices})
        self.by_w_pw={}
        for row,(qi,wi,pw) in enumerate(self.indices):self.by_w_pw.setdefault((wi,pw),{})[qi]=row
    def material(self,a,b):
        a=a.tocsc();b=b.tocsc();rows=[];cols=[];values=[]
        for column,(qi,wi,pw) in enumerate(self.indices):
            qa,qb=a.indptr[qi:qi+2];wa,wb=b.indptr[wi:wi+2]
            for wr,wv in zip(b.indices[wa:wb],b.data[wa:wb]):
                bra=self.by_w_pw.get((wr,pw))
                if bra is None:continue
                for qr,qv in zip(a.indices[qa:qb],a.data[qa:qb]):
                    row=bra.get(qr)
                    if row is not None:rows.append(row);cols.append(column);values.append(qv*wv)
        return sparse.coo_matrix((values,(rows,cols)),shape=(len(self.keys),)*2).tocsr()
    def expectation(self,a,b,r,phi):
        matrix=self.material(a,b);changed=(matrix@self.values.reshape(len(self.keys),70)).reshape(-1,10,7)
        return np.einsum('nrf,rs,fg,nsg->',self.bra,r,phi,changed,optimize=True)
