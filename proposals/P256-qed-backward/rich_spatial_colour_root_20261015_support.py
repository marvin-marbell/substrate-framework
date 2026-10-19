"""Actual full-CAR/current/density support from exact new spatial H/state snapshots.
No producer replay: consumes captured selected states, applies full pre-pair and
pre-SO3 multiplication operators, and bounds the unmeasured colour-electric tail.
"""
from __future__ import annotations
from itertools import product
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
import numpy as np
from scipy import sparse
from rich_spatial_colour_root_20261015_model import (
    mode,one_body,frame,masks_for,loop_field,so3,graph,clean,project,density_image,
    electric_and_density,connection_data,law,TensorOperator)


def matrix(data,name):
    return sparse.csr_matrix((data[name+'_data'],data[name+'_indices'],data[name+'_indptr']),shape=tuple(data[name+'_shape']))


def norm(value):return float(np.linalg.norm(value))

def eps(a,b,c):return 0 if len({a,b,c})<3 else 1 if (a,b,c) in [(0,1,2),(1,2,0),(2,0,1)] else -1


def raw_primitive_image(sector,targets,loop,edge,spin,kind,scale):
    masks=sector['masks'];bridge=np.array([[.5,.5],[-.5,-.5]])
    result=sparse.csr_matrix((19*len(targets),19*len(masks)),dtype=complex)
    for f in range(2):
        for colourless in [False,True]:
            q=([0.,-1.] if colourless else [2/3,-1/3])[f]
            if kind.startswith('current') and not q:continue
            if kind=='current_scalar':coefficients=(-1j*q,1j*q)
            elif kind=='current_vector':coefficients=(q,q)
            elif kind=='H_scalar':coefficients=(1.,1.)
            else:coefficients=(1j,-1j)
            members=[3] if colourless else range(3)
            for reverse,coefficient in enumerate(coefficients):
                bmatrix=bridge.conj().T if reverse else bridge
                for a,b in product(members,repeat=2):
                    if (edge==0 or colourless) and a!=b:continue
                    terms=[(mode(f,m,s,a),mode(f,n,t,b),bmatrix[m,n]*spin[s,t])
                        for m,n,s,t in product(range(2),repeat=4) if spin[s,t]!=0]
                    body=one_body(masks,terms,targets)
                    field=np.eye(19) if edge==0 or colourless else loop['U'][a,b] if reverse else loop['U'][b,a].conj().T
                    result+=coefficient*scale*sparse.kron(body,field,format='csr')
    return clean(result@sector['P'])


def orientation_functions(rotor,g,p,kind):
    basis=rotor['basis'];r=rotor['R'][:,:,0];nq=len(r);functions=[]
    components=range(3) if kind in ['current','magnetic'] else [0]
    for physical in components:
        terms=[]
        for edge in range(2):
            if kind=='H':
                terms.append(basis)
                for axis in range(3):terms.append(g['tangents'][:,edge,axis,None]*basis)
            else:
                line=g['coefficients'][:,edge,physical+(3 if kind=='magnetic' else 0)]
                terms.append(line[:,None]*basis)
                for axis in range(3):terms.append((line*g['tangents'][:,edge,axis])[:,None]*basis)
        if kind!='H':
            convection=np.zeros((nq,10),complex)
            for angular in range(3):
                lbasis=basis@rotor['L'][angular].toarray()
                if kind=='current':
                    for b in range(3):
                        e=eps(angular,b,physical)
                        if e:
                            derivative=sum(eps(angular,b,c)*r[:,c] for c in range(3))
                            convection+=p['vertex_separation']/(4*p['orientation_inertia'])*e*(2*r[:,b,None]*lbasis+1j*derivative[:,None]*basis)
                else:
                    alpha=(1. if angular==physical else 0.)-r[:,angular]*r[:,physical]
                    derivative=-1j*sum(eps(angular,physical,c)*r[:,angular]*r[:,c] for c in range(3))
                    convection+=p['vertex_separation']**2/(16*p['orientation_inertia'])*(2*alpha[:,None]*lbasis+derivative[:,None]*basis)
            terms.append(convection)
        functions.append(np.concatenate(terms,axis=1)*np.sqrt(rotor['weights'])[:,None])
    full=np.concatenate(functions,axis=0)
    _,s,vh=np.linalg.svd(full,full_matrices=False);keep=s>1e-12*s[0]
    return dict(s=s[keep],factor=vh[keep].T,rank=int(sum(keep)),discarded_function_norm=float(np.linalg.norm(s[~keep])))


def full_image_norm(images,state,sector,scalar,factor,phi_flags,block_rows):
    nf=len(scalar['grid']);x=state.reshape(sector['dimension'],10,nf);plain=x.reshape(sector['dimension'],-1)
    weighted=(x*scalar['grid'][None,None,:]).reshape(sector['dimension'],-1);total=0.
    for start in range(0,images[0].shape[0],block_rows):
        stop=min(start+block_rows,images[0].shape[0]);values=[]
        for operator,phi in zip(images,phi_flags):
            image=(operator[start:stop]@(weighted if phi else plain)).reshape(stop-start,10,nf)
            values.append(image.transpose(0,2,1).reshape((stop-start)*nf,10))
        combined=np.concatenate(values,axis=1)
        transformed=(combined@factor['factor'])*factor['s'][None,:]
        total+=float(np.sum(abs(transformed)**2))
    return total


def main():
    inp,out=map(Path,sys.argv[1:3]);request=json.loads(inp.read_text());root=inp.parent;prefix=request['producer_prefix']
    producer_input=root/(prefix+'_input.json');operator_path=root/(prefix+'_operators.npz');state_path=root/(prefix+'_states.npz');preflight_path=root/(prefix+'_preflight.json')
    p=json.loads(producer_input.read_text());preflight=json.loads(preflight_path.read_text());started=time.time()
    data=np.load(operator_path);states=np.load(state_path);scalar=dict(H=data['scalar_H'],grid=data['scalar_grid'],phi=np.diag(data['scalar_grid']))
    rotor=so3(p);g=graph(p,rotor);loop=loop_field();factors={kind:orientation_functions(rotor,g,p,kind) for kind in ['current','magnetic','H']}
    results={}
    for name,delta in [('neutral',(0,0,0,0)),('baryon',(2,1,0,0)),('lepton',(0,0,0,1))]:
        pp=matrix(data,name+'_P');masks=[int(x) for x in data[name+'_masks']]
        sector=dict(masks=masks,P=pp,dimension=pp.shape[1],gauge_error=preflight['local_Gauss'][name])
        state=states['reference'] if name=='neutral' else states[name+'_pole_0:0.5'];nf=len(scalar['grid']);x=state.reshape(sector['dimension'],-1)
        targets=masks_for(delta,p['pair_cut']+1);charge={'neutral':0.,'baryon':1.,'lepton':-1.}[name]
        # Exact local density moments and graph tree electric, without full Gauss rebuild.
        electric=electric_and_density(sector,loop,p,g);density=density_image(sector)
        d=clean(density['embedding'].conj().T@density['delta_image'])
        d2=clean(density['delta_image'].conj().T@density['delta_image'])
        full_d2=float(np.sum(x.conj()*(d2@x)).real);retained_d2=norm(d@x)**2
        dipole_retained=0.
        for axis in range(3):
            op=TensorOperator(sparse.kron(d,p['vertex_separation']/2*rotor['coordinate'][axis][0],format='csr'),sparse.csr_matrix((sector['dimension']*10,)*2),scalar)
            dipole_retained+=norm(op@state)**2
        entry=dict(state='neutral actual reference' if name=='neutral' else 'actual lowest source-bright halfspin pole',
            density_delta_full_norm_squared=full_d2,density_delta_retained_norm_squared=retained_d2,
            omitted_local_EM_charge_norm_squared=(full_d2-retained_d2)/4,
            omitted_vector_dipole_norm_squared=p['vertex_separation']**2/4*full_d2-dipole_retained,
            current_target_raw_masks=len(targets),original_pair_cut=p['pair_cut'],
            meaning='Full raw CAR density/current images BEFORE band-pair and material-SO3 cut, within SAME PW19 and DVR. Positive differences are exact omitted operator support, NOT energy/resolvent error estimates.')
        op=dict(Q=charge*sparse.eye(sector['dimension'],format='csr'),density_delta=d,density_delta2=d2,hops=[])
        for edge in range(2):
            op['hops'].append([(q,[matrix(data,f'{name}_hop_{edge}_{channel}_{spin}') for spin in range(4)])
                for channel,q in enumerate([2/3,0.,-1/3,-1.])])
        op['connection']=connection_data(op,rotor,p)
        images=[];flags=[];sigma=frame()['sigma']
        for edge in range(2):
            images.append(raw_primitive_image(sector,targets,loop,edge,np.eye(2),'current_scalar',p['scalar_coupling']*g['weights'][edge]));flags.append(True)
            for spin in sigma:
                images.append(raw_primitive_image(sector,targets,loop,edge,spin,'current_vector',p['vector_velocity']*g['weights'][edge]));flags.append(False)
        # D and totalQ images embedded in the SAME complete next-pair target space.
        lookup={m:i for i,m in enumerate(targets)};inclusion=sparse.csr_matrix((np.ones(len(masks)),([lookup[m] for m in masks],range(len(masks)))),shape=(len(targets),len(masks)))
        embedding=sparse.kron(inclusion,sparse.eye(19),format='csr')@pp
        from rich_spatial_colour_root_20261015_model import density_terms
        dimage=sparse.kron(one_body(masks,density_terms(),targets),sparse.eye(19),format='csr')@pp
        for kind,last_image,offset in [('current',dimage,0),('magnetic',charge*embedding,3)]:
            full=full_image_norm(images+[last_image],state,sector,scalar,factors[kind],flags+[False],request['block_rows'])
            retained=sum(norm(law(op,rotor,scalar,p,g,derivative=(offset+i,))@state)**2 for i in range(3))
            entry[kind]=dict(full_vector_norm_squared=full,retained_vector_norm_squared=retained,
                omitted_norm_squared=full-retained,omitted_norm=float(np.sqrt(max(0.,full-retained))),
                full_product_function_rank=factors[kind]['rank'],discarded_function_norm=factors[kind]['discarded_function_norm'])
        del images,embedding,dimage
        # Full H-hop tail is computed exactly; full electric tail gets a rigorous finite-spectrum bound.
        images=[];flags=[]
        for edge in range(2):
            images.append(raw_primitive_image(sector,targets,loop,edge,np.eye(2),'H_scalar',p['scalar_coupling']*g['weights'][edge]));flags.append(True)
            for spin in sigma:
                images.append(raw_primitive_image(sector,targets,loop,edge,spin,'H_vector',p['vector_velocity']*g['weights'][edge]));flags.append(False)
        full_hop=full_image_norm(images,state,sector,scalar,factors['H'],flags,request['block_rows'])
        h0=TensorOperator(matrix(data,name+'_H0_one'),matrix(data,name+'_H0_phi'),scalar,field_energy=True)
        c=project(sector,sparse.kron(sparse.eye(len(masks)),loop['C'],format='csr'))
        b=project(sector,sparse.kron(sparse.eye(len(masks)),loop['B'],format='csr'))
        core=clean(sparse.kron(electric['electric']-p['plaquette']*b,sparse.eye(10),format='csr')+
            sparse.kron(sparse.eye(sector['dimension']),rotor['L2'],format='csr')/(2*p['orientation_inertia']))
        hopping=TensorOperator(h0.one-core,h0.phi,scalar)
        retained_hop=norm(hopping@state)**2;eta_hop=float(np.sqrt(max(0.,full_hop-retained_hop)))
        tree=p['electric']*g['lengths'][0]*electric['tree_C']/2
        mean_tree=float(np.sum(x.conj()*(tree@x)).real);projected_tree_norm=norm(tree@x)**2
        maximum=p['electric']*g['lengths'][0]/2*(40/3)
        eta_tree_upper=float(np.sqrt(max(0.,maximum*mean_tree-projected_tree_norm)))
        entry['H_support']=dict(full_hopping_norm_squared=full_hop,retained_hopping_norm_squared=retained_hop,
            exact_omitted_hopping_norm=eta_hop,tree_energy_mean=mean_tree,retained_tree_energy_norm_squared=projected_tree_norm,
            tree_energy_operator_upper_bound=maximum,omitted_tree_energy_norm_upper_bound=eta_tree_upper,
            full_H0_omitted_norm_interval=[max(0.,eta_hop-eta_tree_upper),eta_hop+eta_tree_upper],
            bound_derivation='At one vertex four spin/flavour colour triplets imply local SU3 highest weights p+q<=4. Adding PW19 right electric3/3* gives p+q<=5, Casimir<=40/3. Thus0<=A_tree<=M; full||A psi||^2<=M<A>, omitted||Apsi||^2<=M<A>-||P Apsi||^2. Scalar/DVR, L2 and loop-only terms preserve these pair/orientation cuts. Triangle inequality combines exact hopping tail with electric bound. Bound is generally LOOSE, no energy-gap theorem or higher-PW/DVR control.')
        results[name]=entry
        del images
    provenance={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in [producer_input,operator_path,state_path,preflight_path]}
    payload=dict(kind='NEW_SPATIAL_FULL_CURRENT_DENSITY_AND_HOP_SUPPORT_PHYSICAL_CONSUMER',producer_prefix=prefix,inputs=provenance,results=results,
        geometry_quadrature='Haar j0+j1 state products times current polynomial degree<=2, squared degree<=6; azimuth8/polar6 integrates the full products. The factor SVD drops only numerical zero singular values at1e-12 relative, recorded.',
        resource=dict(peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,elapsed_seconds=time.time()-started),
        limitations='Exact omitted current/magnetic/density/hopping support refers to SAME PW19/DVR parent law; higher colour and scalar field states absent. Electric full H tail is bounded, not computed exactly. No source projection rescaling, continuum/transport admission or old producer replay.')
    out.write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(dict(results=results,resource=payload['resource']),indent=2))


if __name__=='__main__':main()
