#!/usr/bin/env python3
"""Actual canonical square/auxiliary/Haar/Gauss/normal-field range jets.

The tree-Wilson homotopy changes no integer gauge charge.  This native
invariant-spin control is not an admission of the full microscopic campaign.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
from scipy.linalg import eigh
from scipy.sparse import coo_matrix

ROOT=Path(__file__).resolve().parent
SOURCE_BYTES_AT_IMPORT={name:(ROOT/name).read_bytes() for name in (
    'covariant_native_haar_range_jets.py','covariant_induced_photon.py',
    'sea_selection.py','rotation_band.py')}
from covariant_induced_photon import Parameters,phi_frame

GRADE=5


def check(condition: bool,message: str) -> None:
    if not condition:raise ArithmeticError(message)


def multiply(a: np.ndarray,b: np.ndarray) -> np.ndarray:
    result=np.zeros_like(a)
    for n in range(GRADE+1):
        for j in range(n+1):result[n]+=a[j]*b[n-j]
    return result


def determinant_jets(entries: np.ndarray) -> np.ndarray:
    """Permutation determinant: no inverse of a potentially singular minor."""
    size=entries.shape[-1]
    result=np.zeros(entries.shape[:-2],complex)
    for permutation in itertools.permutations(range(size)):
        parity=sum(permutation[i]>permutation[j] for i in range(size) for j in range(i+1,size))
        product=np.zeros_like(result);product[0]=1
        for i,column in enumerate(permutation):product=multiply(product,entries[...,i,column])
        result+=(-1)**parity*product
    return result


def scalar_sqrt(a: np.ndarray) -> np.ndarray:
    result=np.zeros_like(a);result[0]=np.sqrt(a[0])
    for n in range(1,GRADE+1):
        result[n]=(a[n]-sum((result[j]*result[n-j] for j in range(1,n)),0))/(2*result[0])
    return result


def square_jets(gamma: np.ndarray,kappa: float,theta: float) -> tuple[np.ndarray,np.ndarray]:
    """w_t=1+t(exp(i theta)-1), not exp(i t theta)."""
    B=np.zeros((GRADE+1,4,4),complex);B[0]=np.eye(4)/(2*kappa)
    F=np.zeros_like(B)
    for edge,(x,y,mu,orientation) in enumerate(((0,1,1,1),(1,2,2,1),(2,3,1,-1),(3,0,2,-1))):
        wp=np.exp(1j*theta)-1 if edge==3 else 0j
        wm=wp.conjugate()
        B[0,x,y]-=.5;B[0,y,x]-=.5
        B[1,x,y]-=.5*wp;B[1,y,x]-=.5*wm
        z=orientation*gamma[mu,0,1]/2
        F[0,x,y]+=z;F[0,y,x]-=z
        F[1,x,y]+=z*wp;F[1,y,x]-=z*wm
    inverse=np.zeros_like(B);inverse[0]=np.linalg.inv(B[0])
    for n in range(1,GRADE+1):inverse[n]=-inverse[0]@B[1]@inverse[n-1]
    conjugate=np.array([x.conj().T for x in F])
    T=np.zeros((GRADE+1,8,8),complex)
    for n in range(GRADE+1):
        T[n,:4,:4]=inverse[n]
        T[n,4:,4:]=B[n]
        for i in range(n+1):
            T[n,:4,4:]-=inverse[i]@F[n-i]
            T[n,4:,:4]-=conjugate[i]@inverse[n-i]
            for j in range(n-i+1):T[n,4:,4:]+=conjugate[i]@inverse[j]@F[n-i-j]
    permutation=[0,4,1,5,2,6,3,7]
    T=T[:,permutation,:][:,:,permutation]
    values,vectors=eigh(T[0]);check(bool(values[0]>0),'Actual positive free onebody square')
    roots=np.sqrt(values);S=np.zeros_like(T);S[0]=(vectors*roots)@vectors.conj().T
    for n in range(1,GRADE+1):
        rhs=T[n]-sum((S[j]@S[n-j] for j in range(1,n)),np.zeros((8,8),complex))
        S[n]=vectors@((vectors.conj().T@rhs@vectors)/(roots[:,None]+roots[None,:]))@vectors.conj().T
    prefactor=scalar_sqrt(determinant_jets(B))
    return S,prefactor


def auxiliary_half(gamma: np.ndarray,kappa: float,theta: float) -> tuple[np.ndarray,float]:
    """Unchanged full canonical auxiliary on the physical edge 3 -> 0."""
    phase=np.exp(1j*theta);B=np.eye(2,dtype=complex)/(2*kappa)
    B[1,0]-=.5*phase;B[0,1]-=.5*phase.conjugate()
    F=np.zeros((2,2),complex);z=-gamma[2,0,1]/2
    F[1,0]+=z*phase;F[0,1]-=z*phase.conjugate()
    inverse=np.linalg.inv(B)
    T=np.block([[inverse,-inverse@F],[-F.conj().T@inverse,B+F.conj().T@inverse@F]])
    permutation=[0,2,1,3];T=T[np.ix_(permutation,permutation)]
    values,vectors=eigh(T);check(bool(values[0]>0),'Actual positive auxiliary onebody')
    return (vectors*np.sqrt(values))@vectors.conj().T,float(np.sqrt(np.linalg.det(B).real))


def density(light: tuple[int,...],auxiliary: tuple[int,...],charge: int) -> tuple[int,int,int,int]:
    result=[sum(mode//2==x for mode in light)-1 for x in range(4)]
    for x in (0,3):result[x]+=sum((0 if mode<2 else 3)==x for mode in auxiliary)-1
    result[0]-=charge
    return tuple(result) # type: ignore[return-value]


def flux(rho: tuple[int,int,int,int],n: int) -> tuple[int,int,int,int]:
    result=[];value=n
    for r in rho:value+=r;result.append(value)
    check(result[-1]==n,'Literal total body+auxiliary+Ref charge vanishes')
    return tuple(result) # type: ignore[return-value]


def space(charge: int,cutoff: int) -> dict[str,Any]:
    light=list(itertools.combinations(range(8),4+charge))
    auxiliary=list(itertools.combinations(range(4),2))
    states=[(l,a) for l in light for a in auxiliary]
    middle=[i for i,(l,a) in enumerate(states) if density(l,a,charge)==(0,0,0,0)]
    physical=[]
    for i,(l,a) in enumerate(states):
        rho=density(l,a,charge)
        for n in range(-cutoff,cutoff+1):
            electric=flux(rho,n)
            if max(abs(e) for e in electric)<=cutoff:physical.append((i,n,electric))
    return dict(charge=charge,light=light,auxiliary=auxiliary,states=states,middle=middle,
        physical=physical,index={(i,n):j for j,(i,n,_) in enumerate(physical)},
        state_index={state:i for i,state in enumerate(states)})


def reflected_factor_jets(gamma: np.ndarray,y: np.ndarray,normal_half: np.ndarray,
                          physical: dict[str,Any],grid: int) -> np.ndarray:
    middle_states=[physical['states'][i] for i in physical['middle']]
    light_columns=sorted(set(l for l,_ in middle_states));aux_columns=sorted(set(a for _,a in middle_states))
    li={l:i for i,l in enumerate(light_columns)};ai={a:i for i,a in enumerate(aux_columns)}
    columns_l=np.array([li[l] for l,_ in middle_states]);columns_a=np.array([ai[a] for _,a in middle_states])
    row_l=np.array(physical['light']);col_l=np.array(light_columns)
    row_a=np.array(physical['auxiliary']);col_a=np.array(aux_columns)
    naux=len(row_a);row_state=np.array([i for i,_,_ in physical['physical']])
    row_flux=np.array([n%grid for _,n,_ in physical['physical']])
    blocks=[];tail=0.
    for value in y:
        actual=gamma.copy();actual[1]*=1+float(value)
        samples=[]
        for theta in 2*math.pi*np.arange(grid)/grid:
            S,prefactor=square_jets(actual,.09,float(theta))
            entries=S[:,row_l[:,None,:,None],col_l[None,:,None,:]]
            canonical=determinant_jets(entries)
            dressed=np.zeros_like(canonical)
            for n in range(GRADE+1):
                for j in range(n+1):dressed[n]+=prefactor[j]*canonical[n-j]
            ah,ap=auxiliary_half(actual,.04,float(theta))
            am=ap*np.linalg.det(ah[row_a[:,None,:,None],col_a[None,:,None,:]])
            # Full auxiliary canonical matrix, not an occupied ground overlap.
            combined=dressed[:,:,columns_l][:,:,None,:]*am[None,None,:,columns_a]
            samples.append(combined.reshape(GRADE+1,len(physical['states']),len(middle_states)))
        coefficients=np.fft.fft(np.array(samples),axis=0)/grid
        block=coefficients[row_flux,:,row_state,:].transpose(1,0,2)
        blocks.append(block)
        for n in range(GRADE+1):
            forbidden=[j for j in range(grid) if min(j,grid-j)>n+2]
            tail=max(tail,float(np.linalg.norm(coefficients[forbidden,n])) if forbidden else 0.)
    rows=len(physical['physical']);columns=len(middle_states);nodes=len(y)
    result=np.zeros((GRADE+1,nodes*rows,nodes*columns),complex)
    for output in range(nodes):
        for input_node in range(nodes):
            result[:,output*rows:(output+1)*rows,input_node*columns:(input_node+1)*columns]=normal_half[output,input_node]*blocks[input_node]
    physical['Taylor_Fourier_forbidden_norm']=tail
    return result


def common_range_jets(F: np.ndarray,scale: float) -> dict[str,Any]:
    F=F/math.sqrt(scale);size=F.shape[2]
    # A fixed UNITARY domain change, not a nonunitary range substitution.
    # Forming F0dag F0 first squares its condition number and loses the small
    # physical singular directions.  Contract already-whitened columns instead.
    left,q,right=np.linalg.svd(F[0],full_matrices=False)
    check(bool(q[-1]>0),'Only actual positive common Haar range is inverted')
    F=np.array([coefficient@right.conj().T for coefficient in F])
    F[0]=left*q
    white=F/q[None,None,:];white[0]=left
    relative=np.zeros((GRADE+1,size,size),complex)
    G=np.zeros_like(relative)
    for n in range(GRADE+1):
        for j in range(n+1):relative[n]+=white[j].conj().T@white[n-j]
        G[n]=q[:,None]*relative[n]*q[None,:]
    G[0]=np.diag(q*q)
    Q=np.zeros_like(G);Q[0]=np.diag(q)
    inverse=np.zeros_like(G);inverse[0]=np.diag(1/q)
    for n in range(1,GRADE+1):
        rhs=G[n]-sum((Q[j]@Q[n-j] for j in range(1,n)),np.zeros((size,size),complex))
        Q[n]=rhs/(q[:,None]+q[None,:])
        inverse[n]=-inverse[0]@sum((Q[j]@inverse[n-j] for j in range(1,n+1)),np.zeros((size,size),complex))
    V=np.zeros_like(F)
    for n in range(GRADE+1):
        for j in range(n+1):V[n]+=F[j]@inverse[n-j]
    rows=[]
    for n in range(1,GRADE+1):
        orthogonal=sum((V[j].conj().T@V[n-j] for j in range(n+1)),np.zeros((size,size),complex))
        norm=float(np.linalg.norm(relative[n],2))
        rows.append(dict(grade=n,whitened_Taylor_coefficient_norm=norm,
            actual_R_derivative_norm=math.factorial(n)*norm,
            actual_polar_V_Taylor_norm=float(np.linalg.norm(V[n],2)),
            polar_isometry_coefficient_defect=float(np.linalg.norm(orthogonal,2))))
    moving=V[1]-V[0]@(V[0].conj().T@V[1])
    return dict(F=F,G=G,Q=Q,V=V,rows=rows,range_rank=size,
        minimum_nonzero_G=float(q[-1]**2),maximum_G=float(q[0]**2),
        zero_order_isometry_defect=float(np.linalg.norm(V[0].conj().T@V[0]-np.eye(size),2)),
        actual_first_moving_null_norm=float(np.linalg.norm(moving,2)))


def source_coefficients(gamma: np.ndarray,packet: np.ndarray) -> np.ndarray:
    S,_=square_jets(gamma,.09,0.)
    values,U=eigh(S[0]@S[0]);stable=U[:,values<1]
    raw=np.kron(packet[:,None],np.eye(2));psi=stable@stable.conj().T@raw
    return psi/np.linalg.norm(psi,'fro')


def source(neutral: dict[str,Any],charged: dict[str,Any],coefficients: np.ndarray,
           embedding: np.ndarray,nodes: int) -> Any:
    nn=len(neutral['physical']);nc=len(charged['physical'])
    paths=((0,0,0,0),(-1,0,0,0),(-1,-1,0,0),(0,0,0,1))
    weights=coefficients@embedding.conj().T
    rows=[];columns=[];data=[];defect=0
    for column,(state,n,electric) in enumerate(neutral['physical']):
        occupied,aux=neutral['states'][state];mask=sum(1<<mode for mode in occupied)
        for mode in range(8):
            if mode in occupied:continue
            target=charged['state_index'][(tuple(sorted((*occupied,mode))),aux)]
            target_flux=tuple(a+b for a,b in zip(electric,paths[mode//2]));target_n=target_flux[3]
            target_row=charged['index'].get((target,target_n))
            if target_row is None:continue
            check(charged['physical'][target_row][2]==target_flux,'Actual source Wilson string matches integer electric flux')
            rho=density(charged['states'][target][0],aux,1)
            defect=max(defect,max(abs(target_flux[x]-target_flux[(x-1)%4]-rho[x]) for x in range(4)))
            below=(mask&((1<<mode)-1)).bit_count()
            for ref in range(4):
                for node in range(nodes):
                    rows.append((ref*nodes+node)*nc+target_row);columns.append(node*nn+column)
                    data.append((-1)**(below+ref)*weights[mode,ref])
    check(defect==0,'Literal Ref4->Ref3 source preserves local integer Gauss')
    return coo_matrix((data,(rows,columns)),shape=(4*nodes*nc,nodes*nn)).tocsr()


def curl(physical: dict[str,Any],nodes: int) -> Any:
    size=len(physical['physical']);rows=[];columns=[];data=[]
    for column,(state,n,_) in enumerate(physical['physical']):
        for shift,amplitude in ((1,1/(2j)),(-1,-1/(2j))):
            target=physical['index'].get((state,n+shift))
            if target is None:continue
            for node in range(nodes):rows.append(node*size+target);columns.append(node*size+column);data.append(amplitude)
    return coo_matrix((data,(rows,columns)),shape=(nodes*size,nodes*size)).tocsr()


def boundary_jets(neutral: dict[str,Any],charged: dict[str,Any],S: Any,Sin: Any,Bc: Any) -> list[dict[str,Any]]:
    Vn=neutral['V'];Vc=charged['V'];Fn=neutral['F'];Fc=charged['F']
    rows=[];nc=Vc.shape[1];mc=Vc.shape[2];mn=Vn.shape[2]
    sourceV=np.zeros((GRADE+1,4*mc,mn),complex)
    half=np.zeros_like(sourceV);receiver=np.zeros((GRADE+1,mn,4*mc),complex)
    rawS=[(S@Vn[j]).reshape(4,nc,mn) for j in range(GRADE+1)]
    rawF=[(S@Fn[j]).reshape(4,nc,mn) for j in range(GRADE+1)]
    outS=[(Sin@Vn[j]).reshape(4,nc,mn) for j in range(GRADE+1)]
    BVc=[Bc@Vc[j] for j in range(GRADE+1)]
    for n in range(GRADE+1):
        for j in range(n+1):
            for ref in range(4):
                sourceV[n,ref*mc:(ref+1)*mc]+=Vc[j].conj().T@rawS[n-j][ref]
                half[n,ref*mc:(ref+1)*mc]+=Fc[j].conj().T@rawF[n-j][ref]
                receiver[n,:,ref*mc:(ref+1)*mc]+=outS[j][ref].conj().T@BVc[n-j]
        reconstructed=np.zeros_like(half[n])
        for i in range(n+1):
            for j in range(n-i+1):
                for ref in range(4):
                    reconstructed[ref*mc:(ref+1)*mc]+=charged['Q'][i]@sourceV[j,ref*mc:(ref+1)*mc]@neutral['Q'][n-i-j]
        rows.append(dict(grade=n,actual_source_V_Taylor_norm=float(np.linalg.norm(sourceV[n],2)),
            actual_positive_half_boundary_Taylor_norm=float(np.linalg.norm(half[n],2)),
            actual_receiver_V_Taylor_norm=float(np.linalg.norm(receiver[n],2)),
            half_boundary_polar_reconstruction_defect=float(np.linalg.norm(half[n]-reconstructed,2))))
    return rows


def run() -> dict[str,Any]:
    p=Parameters(kappa_power=400);gamma4,metric=phi_frame(p)
    values,U=eigh(1j*gamma4[0]@gamma4[1]@gamma4[2]);embedding=U[:,values>0]
    _,frame=eigh(embedding.conj().T@gamma4[0]@embedding);embedding=embedding@frame[:,::-1]
    gamma=np.array([embedding.conj().T@g@embedding for g in gamma4[:3]])
    check(max(np.linalg.norm(a@b+b@a-2*np.eye(2)*(i==j)) for i,a in enumerate(gamma) for j,b in enumerate(gamma))<1e-12,'Actual invariant Clifford sector')
    nodes=3;grid=32;cutoff=12;sigma=.03;radius=.12
    y=np.linspace(-.75*radius,.75*radius,nodes);dy=float(y[1]-y[0])
    logk=-y*y/(4*sigma*sigma)-sigma*sigma*y*y/(radius*radius*(radius*radius-y*y))
    k=np.exp(logk);k/=np.linalg.norm(k)
    D=np.zeros((nodes-1,nodes))
    for j in range(nodes-1):D[j,j]=math.sqrt(k[j+1]/k[j])/dy;D[j,j+1]=-math.sqrt(k[j]/k[j+1])/dy
    Hphi=2*sigma*sigma*D.T@D;eig,vec=eigh(Hphi)
    normal_half=(vec*np.exp(-eig/2))@vec.T
    neutral=space(0,cutoff);charged=space(1,cutoff)
    Fn=reflected_factor_jets(gamma,y,normal_half,neutral,grid)
    Fc=reflected_factor_jets(gamma,y,normal_half,charged,grid)
    scale=float(np.linalg.norm(Fn[0],2)**2)
    rn=common_range_jets(Fn,scale);rc=common_range_jets(Fc,scale)
    packet=np.array([.1,.8,.4,.1]);packet/=np.linalg.norm(packet)
    outpacket=np.array([.5,.4,.1,.6]);outpacket/=np.linalg.norm(outpacket)
    S=source(neutral,charged,source_coefficients(gamma,packet),embedding,nodes)
    Sout=source(neutral,charged,source_coefficients(gamma,outpacket),embedding,nodes)
    Bn=curl(neutral,nodes);Bc=curl(charged,nodes)
    boundaries=boundary_jets(rn,rc,S,Sout,Bc)
    photon=[]
    for n in range(GRADE+1):
        B=sum((rn['V'][j].conj().T@(Bn@rn['V'][n-j]) for j in range(n+1)),np.zeros((rn['range_rank'],rn['range_rank']),complex))
        photon.append(dict(grade=n,actual_photon_V_Taylor_norm=float(np.linalg.norm(B,2))))
    supplier_path=ROOT/'covariant_induced_photon_capsule_kappa400_full125.json'
    supplier_bytes=supplier_path.read_bytes();supplier=json.loads(supplier_bytes)
    inputs=dict(native_active_kappa=.09,native_auxiliary_kappa=.04,native_auxiliary_sites=[0,3],
        active_canonical_numbers=[4,5],auxiliary_canonical_number=2,original_Ref=4,prepared_Ref=3,
        integer_tagged_charge=1,integer_Ref_hole_charge=-1,normal_nodes=y.tolist(),normal_sigma=sigma,
        normal_radius=radius,normal_energy_eigenvalues=eig.tolist(),Fourier_grid=grid,electric_cutoff=cutoff,
        source_packet=packet.tolist(),receiver_packet=outpacket.tolist(),maximum_Taylor_grade=GRADE,
        family='treeWilson+t(directWilson-treeWilson), in fixed integer Gauss-reduced coordinates',
        unchanged_auxiliary='FULL six canonical occupations of the positive two-site charge1 species',
        supplier_capsule=supplier_path.name,supplier_capsule_sha256=hashlib.sha256(supplier_bytes).hexdigest(),
        supplier_input_descriptor_sha256=supplier['input_descriptor_sha256'])
    return dict(input_descriptor=inputs,input_descriptor_sha256=hashlib.sha256(json.dumps(inputs,sort_keys=True).encode()).hexdigest(),
        source_at_import={name:hashlib.sha256(data).hexdigest() for name,data in SOURCE_BYTES_AT_IMPORT.items()},
        source_bytes_at_import={name:data.decode() for name,data in SOURCE_BYTES_AT_IMPORT.items()},physical_mean_descriptor=metric,
        neutral=dict(physical_dimension=len(neutral['physical'])*nodes,bare_Haar_joint_Gauss_rank=rn['range_rank'],
            actual_minimum_nonzero_G=rn['minimum_nonzero_G'],actual_maximum_G=rn['maximum_G'],
            zero_order_isometry_defect=rn['zero_order_isometry_defect'],first_moving_null_norm=rn['actual_first_moving_null_norm'],
            Fourier_forbidden_coefficient_norm=neutral['Taylor_Fourier_forbidden_norm'],actual_R_rows=rn['rows']),
        charged=dict(physical_dimension=len(charged['physical'])*nodes,Ref_hole_channels=4,
            bare_Haar_joint_Gauss_rank=rc['range_rank'],actual_minimum_nonzero_G=rc['minimum_nonzero_G'],
            actual_maximum_G=rc['maximum_G'],zero_order_isometry_defect=rc['zero_order_isometry_defect'],
            first_moving_null_norm=rc['actual_first_moving_null_norm'],
            Fourier_forbidden_coefficient_norm=charged['Taylor_Fourier_forbidden_norm'],actual_R_rows=rc['rows']),
        actual_source_receiver_V_jet_rows=boundaries,actual_photon_V_jet_rows=photon,
        full125_RHS_descriptor=dict(supplier_capsule_sha256=hashlib.sha256(supplier_bytes).hexdigest(),
            supplier_input_descriptor_sha256=supplier['input_descriptor_sha256'],supplier_source_at_import=supplier['source_at_import'],
            supplier_runtime_seconds=supplier['runtime']['elapsed_seconds'],input_packet_modes=125,
            status='Actual all8 RHS is preserved in this bound supplier capsule; native constants are NOT macro admission.'),
        equations=[
            'P0 is bare electric Haar intersected with literal total active+auxiliary+Ref local Gauss. It is fixed throughout t; charge and statistical seam characters are not scaled.',
            'F(t)=Kphi^1/2 SF_active(t) tensor SF_aux P0, with integer-Fourier/Gauss physical output. G(t)=F(t)dag F(t). Phi heat is inside G, not commuted past active or auxiliary SF.',
            'V(t)=F(t)G(t)^-1/2 is the EXACT full canonical polar isometry, not the static auxiliary ground isometry. All native auxiliary occupations and normal-field levels remain.',
            'At t0 each active orderr Fourier degree<=r and auxiliary degree<=2. Hence degree<=r+2; grid32 and electriccut12 include every physical Taylor row throughgrade5 without an electric current tail.',
            'Reported R_derivative=r! G0^-1/2 G_r G0^-1/2; G_r denotes the ordinary Taylor coefficient. Coefficient and derivative norms are separately labelled.',
            'The source is stable-band dressed creation times literal Ref4 annihilation, including the full complex invariant-spin embedding and CAR signs. Receiver uses a distinct packet and coherent two-orientation integer-cycle curl.',
            'Source and receiver are pulled back with BOTH moving polar isometries. Positive half boundary is Fcharged dag S Fneutral = sqrtGcharged (Vcharged dag S Vneutral) sqrtGneutral.'],
        actual_gauge_charge_changed=False,microscopic_admission_proved=False,
        scope='Actual native positive canonical/Haar/U1-Gauss/one-normal-coordinate quantum control; invariant two-spin active square and two-site auxiliary. NOT full Cartesian Phi/SU4 Gauss, fullmacro NF, along-g remainder, or induced-photon source admission.')


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    started=time.monotonic();result=run();result['runtime_seconds']=time.monotonic()-started
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:result[key] for key in ('source_at_import','input_descriptor_sha256','neutral','charged',
        'actual_source_receiver_V_jet_rows','actual_photon_V_jet_rows','runtime_seconds')}))


if __name__=='__main__':
    main()
