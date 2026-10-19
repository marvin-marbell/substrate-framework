"""Physical source response, same-initial decoupled-field control and electric overflow."""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import expm_multiply
from rich_quantum_common_field_20261015_model import rotor, whole_law, compressed


def source_response(source,state,kicked,evolved,baseline_states):
    if source is None:return {}
    norm=lambda vector:float(np.linalg.norm(source.conj().T@vector)**2)
    initial=norm(state)
    return dict(kind='Raw reverse-CAR-source strength ||Sdag psi||^2; not an inclusive probability',baseline=[norm(x) for x in baseline_states],reaction=[norm(x) for x in evolved],excess=[norm(x)-norm(y) for x,y in zip(evolved,baseline_states)],field_only_initial_change=norm(kicked)-initial)


def frozen_control(sector,state,kicked,scalar,field,p,time):
    matter=sparse.block_diag([sparse.csr_matrix(b['V'].conj().T@b['H0']@b['V']) for b in sector['blocks']],format='csr')
    charge=compressed(sector,'Q',scalar)
    coefficient=(1/p['charge_radius']-1/p['boundary_radius'])/(8*np.pi*p['permittivity'])
    matter+=coefficient*charge@charge
    h=sparse.kron(matter,field['identity'])+sparse.kron(sparse.eye(sector['dimension']),field['H'])
    offset=float(np.vdot(state,h@state).real);h-=offset*sparse.eye(h.shape[0])
    baseline=expm_multiply(-1j*time*h,state);pulse=expm_multiply(-1j*time*h,kicked)
    response={}
    for key in ['phi','phi2','C','B','positive']:
        op=sparse.kron(compressed(sector,key,scalar),field['identity'])
        response[key]=float(np.vdot(pulse,op@pulse).real-np.vdot(baseline,op@baseline).real)
    return dict(time=float(time),protocol='Same actual full-H initial stationary state and same field-only kick, paired propagation with all EM Peierls paths fixed at A=0. Scalar/colour/matter parameters untouched. Control current is its true derivative ZERO, not full-law current relabelled.',material_excess=response,physical_current=[0.,0.])


def field_overflow(sector,state,scalar,field,p):
    extended=dict(p);extended['electric_cut']=p['electric_cut']+3
    big=rotor(extended);h,_,_=whole_law(sector,scalar,big,p)
    old=field['electric'];new=big['electric'];n=sector['dimension'];size=len(new)
    slots=np.array([np.flatnonzero(new==x)[0]*size+np.flatnonzero(new==y)[0] for x in old for y in old])
    inclusion=sparse.csr_matrix((np.ones(len(slots)),(slots,np.arange(len(slots)))),shape=(big['size'],field['size']))
    embedding=sparse.kron(sparse.eye(n),inclusion,format='csr')
    image=h@(embedding@state);outside=image-embedding@(embedding.conj().T@image)
    return dict(norm=float(np.linalg.norm(outside)),extended_electric_cut=extended['electric_cut'],meaning='Actual H image lost by the two-coordinate Fourier boundary, evaluated with all charge harmonics through3; distinct from population/energy residuals.')


def retarded_current(sector,state,h,j,k,field,p):
    """Both retarded orders/contact; sparse resolvents span the entire retained H."""
    from scipy.sparse.linalg import gmres, LinearOperator
    estate=float(np.vdot(state,h@state).real);identity=sparse.eye(h.shape[0])
    centered=h-estate*identity
    vectors=np.column_stack([op@state for op in j])
    cross=sparse.csr_matrix(h.shape,dtype=complex)
    for harmonic in [2,-1,-3]:
        shift=field['shifts'][harmonic][2];sine=(shift-shift.conj().T)/(2j)
        cross-=p['vector_velocity']*(harmonic/3)**2*sparse.kron(sector['vector'][2,harmonic],sine)
    contact=np.array([[float(np.vdot(state,k[0]@state).real),float(np.vdot(state,cross@state).real)],[float(np.vdot(state,cross@state).real),float(np.vdot(state,k[1]@state).real)]])
    result=[]
    for frequency in p['source_frequencies']:
        tensors=[];errors=[]
        for sign in [-1,1]:
            matrix=(frequency+1j*p['eta'])*identity+sign*centered
            diagonal=matrix.diagonal();preconditioner=LinearOperator(matrix.shape,matvec=lambda x:x/diagonal)
            solutions=[]
            for axis in range(2):
                solution,info=gmres(matrix,vectors[:,axis],M=preconditioner,rtol=1e-10,atol=1e-12,restart=100,maxiter=1000)
                solutions.append(solution);errors.append(dict(sign=sign,axis=axis,solver_info=int(info),residual=float(np.linalg.norm(matrix@solution-vectors[:,axis]))))
            tensors.append(vectors.conj().T@np.column_stack(solutions))
        chi=tensors[0]-tensors[1].T+contact
        result.append(dict(frequency=frequency,real=chi.real.tolist(),imag=chi.imag.tolist(),resolvent_errors=errors))
    return dict(contact=contact.tolist(),retarded=result,all_retained_virtuals=h.shape[0],convention='chi_ij=<J_i (omega-H+E+ieta)^-1 J_j>-<J_j (omega+H-E+ieta)^-1 J_i>+K_ij; K=+Hsecond. No source-normalized pair yield.')
