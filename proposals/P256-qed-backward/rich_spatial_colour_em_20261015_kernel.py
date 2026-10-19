"""Actual 3D path/profile/Gauss kernel consumer before solving the changed whole H."""
from itertools import product
import json
from pathlib import Path
import sys
import numpy as np
from scipy import sparse
from rich_spatial_colour_root_20261015_model import so3,graph
from rich_spatial_colour_em_20261015_fields import profiles,electric_gram,gaussian_green,field_shell,route_coefficients,phase_kernel,endpoint_connections,mechanical_kernels


def main():
    inp,out=map(Path,sys.argv[1:3]);p=json.loads(inp.read_text());rotor=so3(p);geometry=graph(p,rotor);field=field_shell(p['em_cut'],p['em_basis_scale']);coefficients=route_coefficients(geometry,rotor,p)
    x,w=np.polynomial.hermite.hermgauss(8);indices=np.array(list(product(range(8),repeat=3)));nodes=x[indices];weights=np.prod(w[indices],axis=1)
    points=p['em_width']*nodes;values=profiles(points,p['em_width'],p['epsilon']);normalizer=p['epsilon']*p['em_width']**3*weights*np.exp(np.sum(nodes*nodes,axis=1))
    gram=np.einsum('n,nai,nbi->ab',normalizer,values,values)
    convolution=[]
    for centre in geometry['vertices']:
        sampled=profiles(centre+np.sqrt(2)*p['charge_sigma']*nodes,p['em_width'],p['epsilon'])
        average=np.einsum('n,nai->ai',weights/np.pi**1.5,sampled)
        exact=profiles(centre[None,:],p['em_width'],p['epsilon'],p['charge_sigma'])[0]
        convolution.append(float(np.linalg.norm(average-exact)))
    divergence=[];h=1e-5
    for point in np.array([[.2,.3,.4],[-.4,.1,.6]]):
        div=np.zeros(6)
        for axis in range(3):
            displacement=np.eye(3)[axis]*h
            plus=profiles((point+displacement)[None,:],p['em_width'],p['epsilon'],p['charge_sigma'])[0]
            minus=profiles((point-displacement)[None,:],p['em_width'],p['epsilon'],p['charge_sigma'])[0]
            div+=(plus[:,axis]-minus[:,axis])/(2*h)
        divergence.append(div.tolist())
    loop=coefficients['body'][1]-coefficients['body'][0]
    full=phase_kernel(rotor,geometry,coefficients,field,1,-1.,spin_axis=0)
    phase=phase_kernel(rotor,geometry,coefficients,field,1,-1.)
    orientation=rotor['integrate'](geometry['tangents'][:,1,0])
    wrong=sparse.kron(orientation,sparse.eye(field['dimension']))@phase
    continuity=[]
    for edge,route in enumerate(geometry['quadrature']):
        for k in [np.array([.5,0.,0.]),np.array([0.,.5,0.]),np.array([0.,0.,1.])]:
            line=np.einsum('n,ni->i',np.exp(-1j*(route['points']@k)),route['dl'])
            ends=np.exp(-1j*(geometry['vertices'][0]@k))-np.exp(-1j*(geometry['vertices'][1]@k))
            gaussian=np.exp(-p['charge_sigma']**2*(k@k)/2)
            continuity.append(dict(edge=edge,wavevector=k.tolist(),same_density_kinematic_error=float(abs(gaussian*(1j*k@line-ends)))))
    connections=endpoint_connections(geometry,rotor,p);mechanical=mechanical_kernels(rotor,field,connections,p['orientation_inertia'])
    result=dict(
        kind='ACTUAL_SPATIAL_EM_GEOMETRY_KERNEL_NOT_PHOTON_PROOF',parameters=p,
        vertices=geometry['vertices'].tolist(),paths=[x.tolist() for x in geometry['paths']],
        electric_raw_Gram=electric_gram(p['em_width'],p['epsilon']).tolist(),
        numerical_normalized_Gram=gram.tolist(),normalized_Gram_error=float(np.linalg.norm(gram-np.eye(6))),
        gaussian_convolution_errors=convolution,transverse_divergence=divergence,
        body_route_coefficients=coefficients['body'].tolist(),closed_loop_TE_flux_vector=loop[:3].tolist(),
        closed_loop_TM_flux_vector=loop[3:].tolist(),closed_loop_TE_norm=float(np.linalg.norm(loop[:3])),
        closed_loop_TM_norm=float(np.linalg.norm(loop[3:])),full_R_spin_phase_vs_wrong_separate_product=float(np.linalg.norm(full-wrong)),
        same_density_continuity=continuity,
        Coulomb=dict(self=float(gaussian_green(0.,p['charge_sigma'],p['epsilon'])),cross=float(gaussian_green(p['vertex_separation'],p['charge_sigma'],p['epsilon']))),
        field_basis_dimension=field['dimension'],material_orientation_dimension=rotor['dimension'],
        primitive_electric_spectrum=np.linalg.eigvalsh(field['H_E']).tolist(),
        mechanical_linear_norms=[float(np.linalg.norm(x)) for x in mechanical['linear']],
        mechanical_quadratic_norms=[float(np.linalg.norm(x)) for x in mechanical['quadratic']],
        limits='Both loop fluxes are actual path integrals, not polarizations by counting. Field shells are basis regulators; primitive Pi² spectrum is NOT a magnetic oscillator or photon dispersion. Kinematic same-density identity precedes CAR/pair/angular projection; actual projected Ward defects must still be measured.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
