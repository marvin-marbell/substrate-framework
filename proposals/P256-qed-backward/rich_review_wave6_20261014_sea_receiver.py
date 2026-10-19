"""Materially changed allowed transverse momenta after stationary-derivative adverse case."""
import json
from pathlib import Path
import rich_chiral_sea_20261014_model as law
from rich_review_wave6_20261014_sea import solve, scalar
ROOT=Path(__file__).resolve().parent

def main():
    request=json.loads((ROOT/'rich_review_wave6_20261014_sea_receiver_input.json').read_text())
    model=law.build(law.Parameters(**request['parameters']))
    result={'law':model['law'],'request':request,'actual':solve(model,request),
            'wrong_Noether_source':solve(model,request,True),
            'source_off':solve(model,{**request,'pulse':[0.,0.]})}
    (ROOT/'rich_review_wave6_20261014_sea_receiver_result.json').write_text(json.dumps(result,default=scalar,indent=2)+'\n')
    print(json.dumps(result,default=scalar,indent=2))

if __name__=='__main__':main()
