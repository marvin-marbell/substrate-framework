"""Independent finite local graph-U1 covariance, not lab rotational EM."""
import json
from pathlib import Path
import numpy as np
from scipy import sparse
import rich_composite_geometry_20261015_model as model
from rich_composite_geometry_20261015_science import load
from rich_native_body_review_20261016_math import data_from_labels,peierls
ROOT=Path(__file__).resolve().parent
P='rich_native_body_review_20261016_'
RUN='rich_composite_geometry_20261015_correlated_dressing3'


def main():
    meta=load(ROOT/(RUN+'_body_basis.pkl'));data=data_from_labels(meta['labels']);scalar,_=model.field_and_rotor(meta['p'])
    matrices={name:sparse.load_npz(ROOT/(RUN+'_op_'+name+'.npz')) for name in ('Gram','rotation','electric','plaquette','quark_vector','white_vector','quark_scalar','white_scalar')}
    g=matrices['Gram'];hop0=matrices['quark_vector']+matrices['white_vector'];cp=matrices['quark_scalar']+matrices['white_scalar'];c0=matrices['rotation']+matrices['electric']+matrices['plaquette']+hop0
    alpha=.27;phase=sparse.diags(np.exp(1j*alpha*data['Q'][:,3]));inverse=phase.conj().T
    rotated0=phase@c0@inverse;rotatedphi=phase@cp@inverse
    phased0=c0+peierls(hop0,data,2,0,alpha)+peierls(hop0,data,3,0,alpha)
    phasedphi=cp+peierls(cp,data,2,0,alpha)+peierls(cp,data,3,0,alpha)
    dg=phase@g@inverse-g;d0=rotated0-phased0;dp=rotatedphi-phasedphi
    metric_error=float(sparse.linalg.norm(dg));vector_error=float(sparse.linalg.norm(d0));scalar_error=float(sparse.linalg.norm(dp));full_error=float(np.sqrt(7*vector_error**2+float(np.dot(scalar['grid'],scalar['grid']))*scalar_error**2))
    if max(metric_error,vector_error,scalar_error)>2e-10:raise AssertionError('Full native material count gauge orbit is not closed')
    wrong0=rotated0-(c0+peierls(hop0,data,3,0,alpha));wrongphi=rotatedphi-(cp+peierls(cp,data,3,0,alpha))
    wrong_error=float(np.sqrt(7*sparse.linalg.norm(wrong0)**2+float(np.dot(scalar['grid'],scalar['grid']))*sparse.linalg.norm(wrongphi)**2))
    try:
        assert wrong_error<2e-10,'Local node3 gauge phase wrongly applied to chord3 only; incident edge2 omitted'
    except AssertionError as error:rejection=str(error)
    else:raise AssertionError('Wrong one-edge gauge control did not reach consumer')
    result=dict(alpha=alpha,node=3,phased_incident_edges=[2,3],metric_covariance_error=metric_error,C0_covariance_error=vector_error,Cphi_covariance_error=scalar_error,fullPhi_native_H_covariance_error=full_error,wrong_one_edge_gauge_error=wrong_error,wrong_case_expected_assertion=rejection,scope='Finite material graphU1 count-gauge orbit and same microscopic link derivative only. Not labrotational convection, fullspace local Maxwell gauge completion, dynamical photons or source-EOM Gdot.')
    np.savez(ROOT/(P+'gauge_raw.npz'),Qsite3=data['Q'][:,3],gauge_phase=np.exp(1j*alpha*data['Q'][:,3]),correct_defects=np.array([metric_error,vector_error,scalar_error,full_error]),wrong_one_edge_defect=wrong_error)
    (ROOT/(P+'gauge_result.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
