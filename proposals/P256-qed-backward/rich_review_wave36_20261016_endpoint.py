"""Owner-aligned creation-index sigma(mu,m), complex TrU endpoint readout.
Supersedes ONLY actual1 endpoint family; original Phi EOM remains unchanged.
"""
import json
from pathlib import Path
import resource
import time
from rich_review_wave36_20261016_consume import endpoint_consumer
ROOT=Path(__file__).resolve().parent
P='rich_review_wave36_20261016_'


def main():
    started=time.time();result=endpoint_consumer()
    result['supersession']='actual1 endpoint only used transposed creation index and ReTrU rather than owner-declared sigma(mu,m),complexTrU. actual1 source EOM/current/wrong-derivative results unaffected. Original raw/capsule retained, not relabelled.'
    result['elapsed_seconds']=time.time()-started;result['peak_RSS_KiB']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    (ROOT/(P+'endpoint_result.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
