"""Primitive field-word lifts BEFORE R10/source projection.
Input j<=1 plus two coordinate factors reaches exactly j<=3 (84 functions).
This is a degree certificate, NOT a regulator/convergence campaign.
"""
from itertools import combinations_with_replacement,product
import numpy as np
from scipy import sparse
import rich_composite_geometry_20261015_model as model


def rotation_grid():
    azimuth=9;nodes,weights=np.polynomial.legendre.leggauss(7);rotations=[];haar=[]
    def rz(angle):return np.array([[np.cos(angle),-np.sin(angle),0.],[np.sin(angle),np.cos(angle),0.],[0.,0.,1.]])
    for a,(z,w),g in product(range(azimuth),zip(nodes,weights),range(azimuth)):
        s=np.sqrt(1-z*z);ry=np.array([[z,0.,s],[0.,1.,0.],[-s,0.,z]]);rotations.append(rz(2*np.pi*a/azimuth)@ry@rz(2*np.pi*g/azimuth));haar.append(w/(2*azimuth*azimuth))
    return np.asarray(rotations),np.asarray(haar)


def so3_words(p):
    rotations,weights=rotation_grid();flat=rotations.reshape(-1,9);monomials=[()]+[indices for degree in (1,2,3)for indices in combinations_with_replacement(range(9),degree)];values=np.column_stack([np.prod(flat[:,indices],axis=1)if indices else np.ones(len(weights))for indices in monomials]);weighted=np.sqrt(weights)[:,None]*values;_,singular,vh=np.linalg.svd(weighted,full_matrices=False)
    # Peter-Weyl 1+9+25+49=84: complete polynomial functions of degree<=3.
    transform=vh[:84].T/singular[:84];basis=values@transform;old=np.column_stack([np.ones(len(weights)),np.sqrt(3)*flat]);embedding=basis.T@(weights[:,None]*old);coordinate=np.empty((3,3,84,84),complex)
    for i,a in product(range(3),repeat=2):coordinate[i,a]=basis.T@((weights*rotations[:,i,a])[:,None]*basis)
    generators=[]
    for axis in range(3):
        real_generator=np.array([[-model.epsilon(axis,i,j)for j in range(3)]for i in range(3)],float);derivative=(rotations@real_generator).reshape(-1,9);columns=[]
        for indices in monomials:
            value=np.zeros(len(weights))
            for slot,index in enumerate(indices):value+=derivative[:,index]*np.prod(flat[:,indices[:slot]+indices[slot+1:]],axis=1)
            columns.append(value)
        differentiated=np.column_stack(columns)@transform;generators.append(basis.T@(weights[:,None]*(-1j*differentiated)))
    graph=model.geometry(p);inverse=graph['inertia_inverse'];kinetic=sum((.5*inverse[a,b]*(generators[a]@generators[b])for a,b in product(range(3),repeat=2)),np.zeros((84,84),complex));scalar,rotor=model.field_and_rotor(p)
    root_error=max(float(np.linalg.norm(embedding.conj().T@coordinate[i,a]@embedding-sparse.csr_matrix(rotor['coordinate'][i][a]).toarray()))for i,a in product(range(3),repeat=2))
    return {'p':p,'rotations':rotations,'weights':weights,'basis':basis,'old_basis':old,'embedding':embedding,'coordinate':coordinate,'right_generators':np.asarray(generators),'kinetic':kinetic,'scalar':scalar,'graph':graph,
        'receipt':{'input_rotor_dimension':10,'word_output_dimension':84,'exact_degree_certificate':'Input degree<=1 plus TWO R multiplication factors gives degree<=3,all l0..3 irreps. Trot preserves l. All intermediate complements for these words retained. 9x9x7 Haar quadrature integrates required polynomial/Gaunt products; NOT a biggercap/precision scan.','Gram_error':float(np.linalg.norm(basis.T@(weights[:,None]*basis)-np.eye(84))),'input_embedding_error':float(np.linalg.norm(embedding.conj().T@embedding-np.eye(10))),'root_R10_coordinate_error':root_error,'minimum_retained_singular':float(singular[83]),'maximum_discarded_singular':float(singular[84]),'right_generator_Hermiticity':max(float(np.linalg.norm(g-g.conj().T))for g in generators),'inertia':graph['inertia'].tolist(),'Phi_law':'Exact original finitePhi7 kinetic/grid/potential; no canonicalCCR assumed.'}}


def direction_fields(field,source_edge,receiver_edge):
    graph=field['graph'];source_direction=graph['axes'][source_edge];receiver_direction=graph['axes'][receiver_edge];coordinate=field['coordinate'];embedding=field['embedding'];kinetic=field['kinetic'];scalar=field['scalar'];phi=scalar['phi'];tphi=scalar['kinetic']/field['p']['scalar_volume_ratio'];kphi=phi@(tphi@phi-phi@tphi)-(tphi@phi-phi@tphi)@phi
    source=np.einsum('a,iaxy->ixy',source_direction,coordinate);receiver=np.einsum('a,iaxy->ixy',receiver_direction,coordinate);lifts=[('identity',embedding,np.eye(7))]
    for prefix,operators in (('receiver',receiver),('source_delta',source),('source_current',source),('source_contact',source)):
        lifts.append((prefix+'_0',embedding,phi))
        lifts.extend((prefix+'_'+str(axis+1),operators[axis]@embedding,np.eye(7))for axis in range(3))
    lifts.append(('vertex_00',embedding,kphi));first_errors=[];derivative_errors=[];vv=[]
    for i,j in product(range(3),repeat=2):
        sr=source[i];rr=receiver[j];double=(sr@kinetic@rr-sr@rr@kinetic-kinetic@rr@sr+rr@kinetic@sr)@embedding;vv.append(double);lifts.append((f'vertex_{i+1}{j+1}',double,np.eye(7)));first_errors.append(float(np.linalg.norm((sr@rr-rr@sr)@embedding)))
        derivative_value=np.zeros(len(field['weights']))
        for a,b in product(range(3),repeat=2):
            ea=np.eye(3)[a];eb=np.eye(3)[b];ds=np.einsum('nij,j->ni',field['rotations'],np.cross(ea,source_direction))[:,i];dr=np.einsum('nij,j->ni',field['rotations'],np.cross(eb,receiver_direction))[:,j];derivative_value+=graph['inertia_inverse'][a,b]*ds*dr
        analytic=field['basis'].T@((field['weights']*derivative_value)[:,None]*field['old_basis']);derivative_errors.append(float(np.linalg.norm(double-analytic)))
    rgram=np.array([[left[1].conj().T@right[1]for right in lifts]for left in lifts]);phigram=np.array([[right[2].T@left[2].conj()for right in lifts]for left in lifts]);input_projector=embedding@embedding.conj().T;complement=np.array([[left[1].conj().T@(np.eye(84)-input_projector)@right[1]for right in lifts]for left in lifts])
    return {'lifts':lifts,'Rgram':rgram,'Phigram':phigram,'Rcomplement':complement,'receipt':{'source_edge':source_edge,'receiver_edge':receiver_edge,'source_receiver_axis_dot':float(source_direction@receiver_direction),'full_coordinate_first_commutator_input_norm_max':max(first_errors),'rotor_double_commutator_derivative_identity_error_max':max(derivative_errors),'Phi_double_commutator':kphi.tolist(),'Phi_double_commutator_eigenvalues':np.linalg.eigvalsh(kphi).tolist(),'Phi_double_commutator_trace':float(np.trace(kphi).real),'Phi_CCR_assumption':'NONE: Kphi=[Phi,[Tphi,Phi]] computed from actual finite7 matrices; signed nonconstant algebra retained.','retarded_sign':'Delta<J>(u)=-u²/2 <[DeltaHsource,[Tphi+Trot,Jreceiver]]>+O(u³),because full firstcomm=0. SV/VS=0; coherentSS+VV retained.','rotor_tensor':'sum_ab Iinv_ab (R(e_a cross d_source))_i(R(e_b cross d_receiver))_j. Isotropic: [(d_source.dot.d_receiver)deltaij-(R d_receiver)_i(R d_source)_j]/I,with OWNING microscopicphase/spin/hop factors outside.','unshared_colour_kinetic':'HE commutes whiteCAR and Phi/R; Jacobi plus source/receivercommutation gives leadingcrossHE=0. PW19 retained in owning states; chosen tree0/white2 have identityU,not silently discarded colourflux.'}}
