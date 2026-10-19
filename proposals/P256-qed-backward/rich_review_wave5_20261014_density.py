"""NEW matched-density shared-field consumer and actual discarded-source witness."""
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
from rich_soft_field_20261014_model import Parameters
from rich_soft_field_20261014_density import DensityJoint
from rich_review_wave5_20261014_soft import audit
from rich_review_wave5_20261014_consumer import encode

ROOT=Path(__file__).resolve().parent
PREFIX='rich_review_wave5_20261014_density'


def leakage(model,g):
    p=model.projection; d=model.retain; total=0.; current=0.
    for index,phi in enumerate(model.phi):
        h,j,_=model.matter.operators(phi)
        reduced=g.reshape(len(model.phi),model.md)[index]
        if model.suppliers==1:
            v=p@reduced
            hv=h@v; jv=j@v
            total+=np.linalg.norm(hv-p@(p.conj().T@hv))**2
            current+=np.linalg.norm(jv-p@(p.conj().T@jv))**2
        else:
            # Exact factorized embedding in full 64 x 64 matter, no full H allocation.
            v=p@reduced.reshape(d,d)@p.T
            hv=h@v+(h@v.T).T
            jv=j@v+(j@v.T).T
            total+=np.linalg.norm(hv-p@(p.conj().T@hv@p.conj())@p.T)**2
            current+=np.linalg.norm(jv-p@(p.conj().T@jv@p.conj())@p.T)**2
    return {'full64_per_supplier_H_discarded_ground_residual':np.sqrt(total),'full64_per_supplier_current_discarded_source_norm':np.sqrt(current)}


def main():
    request=json.loads((ROOT/'rich_review_wave5_20261014_soft_input.json').read_text())
    p=Parameters(**{**request['parameters'],'shape':tuple(request['parameters']['shape'])})
    results={}
    for count in [1,2]:
        pp=p if count==1 else replace(p,shape=(p.shape[0],p.shape[1],2*p.shape[2]))
        model=DensityJoint(pp,1,suppliers=count,retain=12)
        result,g,_,_=audit(model)
        result.update(leakage(model,g))
        result['supplier_count']=count; result['retained_per_supplier']=model.retain
        levels=np.linalg.eigvalsh(model.matter.operators(0.)[0].toarray())
        result['retained_endpoint_gap']=float(levels[model.retain]-levels[model.retain-1])
        results['suppliers'+str(count)]=result
    full=json.loads((ROOT/'rich_review_wave5_20261014_soft_result.json').read_text())['cases']['joint_n1']
    result={'new_retained_states':12,'matched_density_cases':results,'one_full64_reference_gap':full['gap'],'one_projected_vs_full64_gap_difference':results['suppliers1']['gap']-full['gap'],'two_vs_one_density_matched_gap_ratio':results['suppliers2']['gap']/results['suppliers1']['gap'],'meaning':'Actual shared phi with two microscopic operator sums; low matter projection is variational and discarded H/current source norms are not zero. Fixed finite density helps this projected branch, not proof of infinite matter/gauge photon phase.'}
    (ROOT/(PREFIX+'_result.json')).write_text(json.dumps(encode(result),indent=2))
    print(json.dumps(encode(result)))


if __name__=='__main__':
    main()
