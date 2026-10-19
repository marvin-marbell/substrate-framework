"""Physical-site-charge projected packet space; no compressed-charge exponentiation.
A native vector is sum_h P_h V W_h x_h in FULL CAR32 x PW19.
P_h are exact Laurent coefficients of exp(i chi rho_site0), h=3 rho_site0.
Sparse band determinants describe seeds/images only, NOT the resulting orbit states.
"""
from __future__ import annotations
from itertools import product
from functools import lru_cache
import numpy as np
from scipy import sparse
from scipy.linalg import eigh,qr
from scipy.sparse.linalg import LinearOperator
from rich_spatial_colour_root_20261015_model import (
    mode,string,frame,loop_field,scalar_field,so3,graph,pair_pattern,new_C_mask,
    spatial_source_terms,site_colour_terms,generators,common_root,clean,
    connection_data,law,route_current,observable,source_operator,TensorOperator)

CHARGES3=np.array([2 if f==0 and c<3 else 0 if f==0 else -1 if c<3 else -3
    for f,s,c in product(range(2),range(2),range(4))],int)
IDENTITY=np.eye(19,dtype=complex)


def tidy(packet):
    return {m:v for m,v in packet.items() if np.linalg.norm(v)>1e-13}


def combine(packets,weights):
    out={}
    for packet,weight in zip(packets,weights):
        if weight:
            for mask,value in packet.items():out[mask]=out.get(mask,np.zeros(19,complex))+weight*value
    return tidy(out)


def bilinear(packet,terms,field=IDENTITY):
    out={}
    for mask,vector in packet.items():
        fv=field@vector
        for a,b,coefficient in terms:
            step=string(mask,[(a,True),(b,False)])
            if step is not None:out[step[0]]=out.get(step[0],np.zeros(19,complex))+step[1]*coefficient*fv
    return tidy(out)


def forward(packet,loop,edge,f,colourless,spin,adjoint=False):
    bridge=np.array([[.5,.5],[-.5,-.5]])
    result=[];members=[3] if colourless else range(3)
    for a,b in product(members,repeat=2):
        if (edge==0 or colourless) and a!=b:continue
        terms=[(mode(f,m,s,a),mode(f,n,t,b),bridge[m,n]*spin[s,t])
            for m,n,s,t in product(range(2),repeat=4) if spin[s,t]!=0]
        field=IDENTITY if edge==0 or colourless else loop['U'][b,a].conj().T
        if adjoint:terms=[(b,a,np.conj(c)) for a,b,c in terms];field=field.conj().T
        result.append(bilinear(packet,terms,field))
    return combine(result,[1]*len(result))


def colour_charge(packet,t):return bilinear(packet,site_colour_terms(t))


def electric_image(packet,loop,p,g):
    pieces=[{m:loop['C']@v for m,v in packet.items()}];weights=[p['electric']*g['lengths'][1]/2]
    for t,r in zip(generators(),loop['R']):
        first=combine([colour_charge(packet,t),{m:r@v for m,v in packet.items()}],[1,1])
        second=combine([colour_charge(first,t),{m:r@v for m,v in first.items()}],[1,1])
        pieces.append(second);weights.append(p['electric']*g['lengths'][0]/2)
    return combine(pieces,weights)


def spin_image(packet,axis):
    s=frame()['sigma'][axis]/2
    terms=[(mode(f,m,a,c),mode(f,m,b,c),s[a,b]) for f,m,a,b,c in product(range(2),range(2),range(2),range(2),range(4)) if s[a,b]!=0]
    return bilinear(packet,terms)


def conjugate(packet):
    swap=np.zeros((19,19));swap[0,0]=1;swap[1:10,10:]=np.eye(9);swap[10:,1:10]=np.eye(9)
    out={}
    for mask,value in packet.items():
        target,sign=new_C_mask(mask);out[target]=sign*(swap@value)
    return tidy(out)


def packet_matrix(packets):
    masks=sorted(set().union(*(x.keys() for x in packets)));lookup={m:i for i,m in enumerate(masks)};rr=[];cc=[];vv=[]
    for j,packet in enumerate(packets):
        for mask,vector in packet.items():
            for a in np.flatnonzero(abs(vector)>1e-14):rr.append(19*lookup[mask]+int(a));cc.append(j);vv.append(vector[a])
    return masks,sparse.csr_matrix((vv,(rr,cc)),shape=(19*len(masks),len(packets)))


@lru_cache(maxsize=None)
def charge_polynomial(single,dist,double_charge):
    # z=exp(i chi/3); physical period6pi. All coefficients are dyadic rationals.
    values={double_charge:1.}
    for r,n,d in zip((-3,-1,0,2),single,dist):
        for flipped in [False]*(n-d)+[True]*d:
            nxt={}
            for k,v in values.items():
                nxt[k]=nxt.get(k,0.)+v*(-.5 if flipped else .5)
                nxt[k+r]=nxt.get(k+r,0.)+v*.5
            values=nxt
    return tuple((k,v) for k,v in values.items() if abs(v)>1e-15)


def charge_grams(left,right):
    """<left_i|P_h|right_j>, exact full-site Laurent projection BEFORE cuts."""
    lm,la=packet_matrix(left);rm,ra=packet_matrix(right)
    lp=[pair_pattern(m) for m in lm];rp=[pair_pattern(m) for m in rm];groups={}
    for j,(s,d,c,sign) in enumerate(rp):groups.setdefault((s,d),[]).append(j)
    entries={h:([],[],[]) for h in range(-12,13)}
    for i,(single,double,choice,sign) in enumerate(lp):
        ns=tuple(sum(bool(single&(1<<a)) and CHARGES3[a]==r for a in range(16)) for r in (-3,-1,0,2))
        offset=sum(int(CHARGES3[a]) for a in range(16) if double&(1<<a))
        for j in groups.get((single,double),[]):
            diff=choice^rp[j][2]
            ds=tuple(sum(bool(diff&(1<<a)) and CHARGES3[a]==r for a in range(16)) for r in (-3,-1,0,2))
            if ds[2]:continue
            for h,v in charge_polynomial(ns,ds,offset):
                if h not in entries:raise ValueError('physical site-charge range violated')
                rr,cc,vv=entries[h];rr.append(i);cc.append(j);vv.append(v*sign*rp[j][3])
    out={}
    for h,(rr,cc,vv) in entries.items():
        if vv:
            kernel=sparse.csr_matrix((vv,(rr,cc)),shape=(len(lm),len(rm)))
            out[h]=clean(la.conj().T@sparse.kron(kernel,sparse.eye(19),format='csr')@ra,1e-14)
    return out


def split_spin(packet):
    groups={}
    for mask,v in packet.items():
        m=sum((1 if (i%8)//4==0 else -1) for i in range(32) if mask&(1<<i))
        groups.setdefault(m,{})[mask]=v
    return groups


def normalize_seeds(packets):
    seeds=[];labels=[]
    for packet in packets:
        for m,part in split_spin(packet).items():
            norm=np.sqrt(sum(float(np.vdot(v,v).real) for v in part.values()))
            if norm>1e-12:seeds.append({k:v/norm for k,v in part.items()});labels.append(m)
    _,matrix=packet_matrix(seeds);gram=(matrix.conj().T@matrix).toarray();labels=np.array(labels,int)
    kept=[];records=[]
    for m in sorted(set(labels)):
        ids=np.flatnonzero(labels==m);g=gram[np.ix_(ids,ids)];e=eigh(g,eigvals_only=True)
        threshold=max(1e-12,1e-11*max(0.,e[-1]));rank=int(sum(e>threshold))
        _,_,pivots=qr(g,pivoting=True);kept.extend(ids[pivots[:rank]])
        records.append(dict(twice_Sz=int(m),input_count=len(ids),rank=rank,threshold=float(threshold),smallest_kept=float(e[e>threshold][0]) if rank else None))
    return [seeds[i] for i in kept],labels[kept],records


class OrbitBasis:
    """Native full-Fock isometry, represented by exact charge projectors and seed Gram."""
    def __init__(self,packets,p):
        self.seeds,self.seed_m,self.seed_rank_records=normalize_seeds(packets);self.grams=charge_grams(self.seeds,self.seeds)
        self.blocks={};self.slices={};self.h=[];self.m=[];self.rank_records=[];start=0
        for h,g in sorted(self.grams.items()):
            cols=[];mlabels=[]
            for m in sorted(set(self.seed_m)):
                selected=np.flatnonzero(self.seed_m==m);block=g[selected][:,selected].toarray()
                asym=np.linalg.norm(block-block.conj().T)
                if asym>1e-9:raise ValueError('non-Hermitian native charge Gram')
                values,vectors=eigh((block+block.conj().T)/2)
                threshold=max(p.get('rank_absolute',1e-12),p.get('rank_relative',1e-10)*max(0.,values[-1]))
                if values[0]<-1e-9:raise ValueError('indefinite full charge Gram')
                keep=values>threshold;w=np.zeros((len(self.seeds),int(sum(keep))),complex)
                w[selected]=vectors[:,keep]/np.sqrt(values[keep])[None,:]
                cols.append(w);mlabels.extend([int(m)]*w.shape[1])
                self.rank_records.append(dict(h=int(h),twice_Sz=int(m),seed_count=len(selected),rank=int(sum(keep)),threshold=float(threshold),min_eigenvalue=float(values[0]),max_eigenvalue=float(values[-1]),smallest_kept=float(values[keep][0]) if any(keep) else None,hermiticity_error=float(asym)))
            w=np.column_stack(cols)
            if w.shape[1]:
                self.blocks[h]=w;self.slices[h]=slice(start,start+w.shape[1]);start+=w.shape[1]
                self.h.extend([h]*w.shape[1]);self.m.extend(mlabels)
        self.h=np.array(self.h,int);self.m=np.array(self.m,int);self.dimension=start
        self.isometry_error=max(float(np.linalg.norm(w.conj().T@self.grams[h]@w-np.eye(w.shape[1]))) for h,w in self.blocks.items())

    def projected(self,images,target,shift):
        grams=charge_grams(target.seeds,images);out=sparse.lil_matrix((target.dimension,self.dimension),dtype=complex)
        for hr,wr in self.blocks.items():
            hl=hr+int(shift)
            if hl in target.blocks and hl in grams:
                value=target.blocks[hl].conj().T@grams[hl]@wr
                out[target.slices[hl],self.slices[hr]]=value
        return clean(out,1e-11)

    def operator(self,function,shift=0):return self.projected([function(x) for x in self.seeds],self,shift)

    def image_metric(self,images,shift):
        grams=charge_grams(images,images);out=sparse.lil_matrix((self.dimension,self.dimension),dtype=complex)
        for h,w in self.blocks.items():
            if h+shift in grams:out[self.slices[h],self.slices[h]]=w.conj().T@grams[h+shift]@w
        return clean(out,1e-11)


def neutral_seeds(loop,p,g):
    sea=sum(1<<mode(f,1,s,c) for f,s,c in product(range(2),range(2),range(4)))
    vacuum=np.zeros(19,complex);vacuum[0]=1;seed={sea:vacuum};packets=[seed];sigma=frame()['sigma']
    for edge in range(2):
        for spin_index,spin in enumerate([np.eye(2)]+sigma):
            fs=[];bs=[];qs=[]
            for f,white in product(range(2),range(2)):
                fs.append(forward(seed,loop,edge,f,white,spin));bs.append(forward(seed,loop,edge,f,white,spin,True))
                qs.append(([2/3,0.] if f==0 else [-1/3,-1.])[white])
            a=combine(fs,[1]*4);b=combine(bs,[1]*4)
            aq=combine(fs,qs);bq=combine(bs,qs)
            packets.append(combine([a,b],[1,1] if spin_index==0 else [1j,-1j]))
            packets.append(combine([aq,bq],[-1j,1j] if spin_index==0 else [1,1]))
    packets.append(electric_image(seed,loop,p,g));packets.append({m:loop['B']@v for m,v in seed.items()})
    return packets+[conjugate(x) for x in packets]


def lower_terms(terms,j,m):
    result=[];denom=np.sqrt((j+m)*(j-m+1))
    for operations,value in terms:
        for a,(index,creation) in enumerate(operations):
            if creation and (index%8)//4==0:
                changed=list(operations);changed[a]=(index+4,True);result.append((changed,value/denom))
    return result


def source_terms_all(site,kind,j):
    # Frozen canonical threehalf polynomial has m=+1/2, NOT highest m=+3/2.
    terms=spatial_source_terms(site,kind,j)
    if j==1.5:
        raised=[]
        for operations,value in terms:
            for a,(index,creation) in enumerate(operations):
                if creation and (index%8)//4==1:
                    changed=list(operations);changed[a]=(index-4,True)
                    raised.append((changed,value/np.sqrt(3)))
        terms=raised
    out={j:terms};m=j
    while m>-j:terms=lower_terms(terms,j,m);m-=1;out[m]=terms
    return out


def insert(packet,terms):
    result={}
    for mask,vector in packet.items():
        for operations,value in terms:
            step=string(mask,operations)
            if step is not None:result[step[0]]=result.get(step[0],np.zeros(19,complex))+step[1]*value*vector
    return tidy(result)


def source_definitions(kind):
    placements=[('0',0),('1',1)]+([('split1',(0,0,1)),('split2',(0,1,1))] if kind=='baryon' else [])
    out={}
    for label,site in placements:
        for j in ([.5,1.5] if kind=='baryon' else [.5]):
            for m,terms in source_terms_all(site,kind,j).items():
                shift=0
                for operations,value in terms[:1]:
                    sites=[site]*len(operations) if isinstance(site,int) else site
                    shift=sum(int(CHARGES3[((i//16)*2+(i%8)//4)*4+i%4])*s0 for (i,_),s0 in zip(operations,[1-x for x in sites]))
                out[f'{label}:{j}:{m}']=dict(terms=terms,shift=shift,spin=j,m=m)
    return out


def anti_terms(terms):
    return [([(index^12,False) for index,_ in operations],value*np.prod([(-1)**((i%8)//4) for i,_ in operations])) for operations,value in terms]


def make_ops(basis,loop,p,g,total_charge):
    n=basis.dimension;eye=sparse.eye(n,format='csr');q=total_charge*eye;rho=sparse.diags(basis.h/3,format='csr');delta=q-2*rho
    op=dict(Q=q,rho0=rho,rho1=q-rho,density_delta=delta,density_delta2=clean(delta@delta))
    op['electric']=basis.operator(lambda x:electric_image(x,loop,p,g))
    for name,key in [('B','B'),('C','C'),('PW_boundary','boundary')]:op[name]=basis.operator(lambda x,key=key:{m:loop[key]@v for m,v in x.items()})
    op['S']=[basis.operator(lambda x,a=a:spin_image(x,a)) for a in range(3)]
    op['S2']=clean(sum(s@s for s in op['S']));op['S_z_labels']=basis.m/2
    op['hops']=[]
    for edge in range(2):
        channels=[]
        for f,white in product(range(2),range(2)):
            charge=([2/3,0.] if f==0 else [-1/3,-1.])[white]
            matrices=[basis.operator(lambda x,sp=sp:forward(x,loop,edge,f,white,sp),-round(3*charge)) for sp in [np.eye(2)]+frame()['sigma']]
            channels.append((charge,matrices))
        op['hops'].append(channels)
    return op


class BodyLift(LinearOperator):
    """Rectangular body source tensor identity_R,phi without allocating repeated CSR."""
    def __init__(self,matrix,factor):
        self.matrix=matrix;self.factor=factor
        super().__init__(dtype=np.dtype(complex),shape=(matrix.shape[0]*factor,matrix.shape[1]*factor))
    def _matvec(self,x):return (self.matrix@np.asarray(x).reshape(self.matrix.shape[1],self.factor)).ravel()
    def _rmatvec(self,x):return (self.matrix.conj().T@np.asarray(x).reshape(self.matrix.shape[0],self.factor)).ravel()
    def _matmat(self,x):return np.column_stack([self._matvec(x[:,j]) for j in range(x.shape[1])])


def charge_site0_image(packet):
    diagonal={}
    for mask,v in packet.items():
        charge=sum(int(CHARGES3[((i//16)*2+(i%8)//4)*4+i%4])/3 for i in range(32) if mask&(1<<i))
        diagonal[mask]=charge*v/2
    terms=[(mode(f,m,s,c),mode(f,1-m,s,c),CHARGES3[(f*2+s)*4+c]/6)
        for f,m,s,c in product(range(2),range(2),range(2),range(4))]
    return combine([diagonal,bilinear(packet,terms)],[1,1])


def build(p,checkpoint=None):
    loop=loop_field();scalar=scalar_field(p);rotor=so3(p);geometry=graph(p,rotor)
    neutral=OrbitBasis(neutral_seeds(loop,p,geometry),p);bases={'neutral':neutral};definitions={};source_matter={};source_metric_full={}
    if checkpoint:checkpoint('neutral_basis',neutral)
    for name,kind in [('lepton','lepton'),('baryon','baryon')]:
        defs=source_definitions(kind);definitions[name]=defs
        packets=[insert(seed,item['terms']) for item in defs.values() for seed in neutral.seeds]
        bases[name]=OrbitBasis(packets,p)
        if checkpoint:checkpoint(name+'_basis',bases[name])
        source_matter[name]={};source_metric_full[name]={}
        for label,item in defs.items():
            images=[insert(seed,item['terms']) for seed in neutral.seeds]
            source_matter[name][label]=neutral.projected(images,bases[name],item['shift'])
            source_metric_full[name][label]=neutral.image_metric(images,item['shift'])
    ops={};sectors={}
    for name,basis in bases.items():
        total=0. if name=='neutral' else -1. if name=='lepton' else 1.
        op=make_ops(basis,loop,p,geometry,total);op['N']=(16 if name=='neutral' else 17 if name=='lepton' else 19)*sparse.eye(basis.dimension,format='csr');ops[name]=op
        if checkpoint:checkpoint(name+'_operators',op)
    cneutral=sparse.lil_matrix((neutral.dimension,)*2,dtype=complex)
    cg=charge_grams(neutral.seeds,[conjugate(x) for x in neutral.seeds])
    for h,w in neutral.blocks.items():
        if -h in neutral.blocks and -h in cg:cneutral[neutral.slices[-h],neutral.slices[h]]=neutral.blocks[-h].conj().T@cg[-h]@w
    cneutral=clean(cneutral);cmatter={'neutral':cneutral}
    for name,anti in [('lepton','antilepton'),('baryon','antibaryon')]:
        basis=bases[name];a=OrbitBasis.__new__(OrbitBasis);a.__dict__.update(basis.__dict__)
        a.seeds=[conjugate(x) for x in basis.seeds];a.seed_m=basis.seed_m.copy();a.h=-basis.h;a.blocks={-h:w for h,w in basis.blocks.items()};a.slices={-h:v for h,v in basis.slices.items()};a.grams={-h:v for h,v in basis.grams.items()};bases[anti]=a
        op={key:value for key,value in ops[name].items() if key not in ['connection','J','J2']}
        for key in ['Q','rho0','rho1','density_delta']:op[key]=-ops[name][key]
        op['N']=32*sparse.eye(basis.dimension)-ops[name]['N']
        op['hops']=[[(q,[mats[0].conj().T]+[-f.conj().T for f in mats[1:]]) for q,mats in edge] for edge in ops[name]['hops']]
        ops[anti]=op;cmatter[name]=sparse.eye(basis.dimension,format='csr');source_matter[anti]={};source_metric_full[anti]={}
        for label,item in definitions[name].items():
            images=[insert(seed,anti_terms(item['terms'])) for seed in neutral.seeds]
            source_matter[anti][label]=neutral.projected(images,a,-item['shift'])
            source_metric_full[anti][label]=neutral.image_metric(images,-item['shift'])
    for name,basis in bases.items():
        op=ops[name];op['connection']=connection_data(op,rotor,p)
        op['J']=[clean(sparse.kron(s,sparse.eye(10))+sparse.kron(sparse.eye(basis.dimension),l)) for s,l in zip(op['S'],rotor['L'])];op['J2']=clean(sum(j@j for j in op['J']))
        sectors[name]=dict(dimension=basis.dimension,native_isometry=basis,charge3_site0=basis.h,twice_Sz=basis.m,charge_PVM={h:sparse.diags((basis.h==h).astype(float),format='csr') for h in sorted(set(basis.h))},P=None,masks=None)
    sources={name:{label:BodyLift(mat,10*len(scalar['grid'])) for label,mat in values.items()} for name,values in source_matter.items()}
    c={name:source_operator(mat,rotor,scalar) for name,mat in cmatter.items()}
    return dict(loop=loop,scalar=scalar,rotor=rotor,geometry=geometry,sectors=sectors,operators=ops,sources=sources,source_matter=source_matter,source_metric_full=source_metric_full,C=c,C_matter=cmatter,native=bases,source_definitions=definitions,root=common_root())
