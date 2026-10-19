"""Independent pivoted-QR degree-three SO3 words and finite Phi algebra."""
from itertools import combinations_with_replacement,product
import numpy as np
from scipy import sparse
from scipy.linalg import qr,solve_triangular
import rich_composite_geometry_20261015_model as model


def construct(p):
    # Degree<=3 times degree<=3: fixed Haar quadrature, not a cap scan.
    nodes,legendre=np.polynomial.legendre.leggauss(7);rotations=[];weights=[]
    def zrotation(a):return np.array([[np.cos(a),-np.sin(a),0.],[np.sin(a),np.cos(a),0.],[0.,0.,1.]])
    for a,b,c in product(range(9),range(7),range(9)):
        z=nodes[b];s=np.sqrt(1-z*z);middle=np.array([[z,0.,s],[0.,1.,0.],[-s,0.,z]])
        rotations.append(zrotation(a*2*np.pi/9)@middle@zrotation(c*2*np.pi/9));weights.append(legendre[b]/162)
    rotations=np.array(rotations);weights=np.array(weights);flat=rotations.reshape(-1,9)
    monomials=[()]+[word for degree in (1,2,3) for word in combinations_with_replacement(range(9),degree)]
    values=np.column_stack([np.prod(flat[:,word],axis=1) if word else np.ones(len(weights)) for word in monomials])
    _,upper,pivots=qr(np.sqrt(weights)[:,None]*values,mode='economic',pivoting=True)
    diagonal=abs(np.diag(upper));rank=int(np.count_nonzero(diagonal>1e-11))
    if rank!=84:raise AssertionError(('Degree-three SO3 dimension is not1+9+25+49',rank))
    selected=[monomials[i] for i in pivots[:rank]];transform=solve_triangular(upper[:rank,:rank],np.eye(rank));basis=values[:,pivots[:rank]]@transform
    input_basis=np.column_stack([np.ones(len(weights)),np.sqrt(3)*flat]);embedding=basis.T@(weights[:,None]*input_basis)
    coordinates=np.array([[basis.T@((weights*rotations[:,i,a])[:,None]*basis) for a in range(3)] for i in range(3)])
    angular=[]
    for axis in range(3):
        generator=np.array([[-model.epsilon(axis,i,j) for j in range(3)] for i in range(3)],float)
        differentiated_flat=(rotations@generator).reshape(-1,9);columns=[]
        for word in selected:
            derivative=np.zeros(len(weights))
            for slot,index in enumerate(word):derivative+=differentiated_flat[:,index]*np.prod(flat[:,word[:slot]+word[slot+1:]],axis=1)
            columns.append(derivative)
        derivative=np.column_stack(columns)@transform;angular.append(basis.T@(weights[:,None]*(-1j*derivative)))
    graph=model.geometry(p);inertia=np.asarray(graph['inertia_inverse']);kinetic=sum((inertia[a,b]*(angular[a]@angular[b])/2 for a,b in product(range(3),repeat=2)),np.zeros((84,84),complex));scalar,rotor=model.field_and_rotor(p)
    phi=np.diag(scalar['grid']);tphi=scalar['kinetic']/p['scalar_volume_ratio'];kphi=phi@tphi@phi-phi@phi@tphi-tphi@phi@phi+phi@tphi@phi
    root_error=max(np.linalg.norm(embedding.T@coordinates[i,a]@embedding-sparse.csr_matrix(rotor['coordinate'][i][a]).toarray()) for i,a in product(range(3),repeat=2))
    receipt=dict(word_dimension=rank,degree_certificate='Input j<=1 times two coordinate factors reaches exactly j<=3; Trot preserves j. No claim of global commuting coordinates on all84states.',polynomial_Haar_Gram_error=float(np.linalg.norm(basis.T@(weights[:,None]*basis)-np.eye(84))),input_embedding_error=float(np.linalg.norm(embedding.T@embedding-np.eye(10))),input_R10_coordinate_error=float(root_error),minimum_retained_QR_diagonal=float(diagonal[83]),maximum_discarded_QR_diagonal=float(max(diagonal[84:])),Phi_double_commutator_eigenvalues=np.linalg.eigvalsh(kphi).tolist(),Phi_double_commutator_trace=float(np.trace(kphi)),inertia=graph['inertia'].tolist())
    if max(receipt['polynomial_Haar_Gram_error'],receipt['input_embedding_error'],root_error)>2e-10:raise AssertionError(('Exact field word metric failed',receipt))
    return dict(p=p,rotations=rotations,weights=weights,basis=basis,input_basis=input_basis,E=embedding,R=coordinates,T=kinetic,scalar=scalar,Phi=phi,Kphi=kphi,graph=graph,receipt=receipt)


def direction(field,source_edge,receiver_edge):
    ds=field['graph']['axes'][source_edge];dr=field['graph']['axes'][receiver_edge];e=field['E'];t=field['T'];rs=np.einsum('a,iaxy->ixy',ds,field['R']);rr=np.einsum('a,iaxy->ixy',dr,field['R']);identity=np.eye(7)
    source_lifts=[(e,field['Phi'])]+[(rs[i]@e,identity) for i in range(3)];receiver_lifts=[(e,field['Phi'])]+[(rr[i]@e,identity) for i in range(3)]
    vertices=[(0,0,e,field['Kphi'])];errors=[];first=[]
    for i,j in product(range(3),repeat=2):
        a=rs[i];b=rr[j];word=(a@t@b-a@b@t-t@b@a+b@t@a)@e
        first.append(float(np.linalg.norm((a@b-b@a)@e)))
        derivative=np.zeros(len(field['weights']))
        for u,v in product(range(3),repeat=2):
            left=np.einsum('qia,a->qi',field['rotations'],np.cross(np.eye(3)[u],ds))[:,i];right=np.einsum('qia,a->qi',field['rotations'],np.cross(np.eye(3)[v],dr))[:,j]
            derivative+=field['graph']['inertia_inverse'][u,v]*left*right
        analytic=field['basis'].T@((field['weights']*derivative)[:,None]*field['input_basis']);errors.append(float(np.linalg.norm(word-analytic)));vertices.append((i+1,j+1,analytic,identity))
    receipt=dict(source_edge=source_edge,receiver_edge=receiver_edge,axis_dot=float(ds@dr),first_full_coordinate_commutator_max=max(first),rotor_kinetic_vs_derivative_tensor_error=max(errors),sign='Delta<J>(u)=-u²/2<[DeltaH,[Tshared,J]]>+O(u³); full first commutator vanishes. SV/VS=0 by tensor-factor algebra, not dropped channels.')
    if max(first+errors)>2e-10:raise AssertionError(('Primitive rotation identity failed',receipt))
    return dict(source=source_lifts,receiver=receiver_lifts,vertices=vertices,receipt=receipt)


def gram(left,right):return left[0].conj().T@right[0],left[1].conj().T@right[1]
