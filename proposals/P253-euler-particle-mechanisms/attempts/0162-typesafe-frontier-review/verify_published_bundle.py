#!/usr/bin/env python3
"""Portable integrity and policy-invariant checks for the published review bundle."""
from __future__ import annotations
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
GRAPH=HERE/'trajectory-policy.json'
REPORT=HERE/'trajectory-policy-report.md'
LOG=HERE/'trajectory-policy-responses.jsonl'
EXPECTED={
 'trajectory-policy.json':'938157e3723bed6962b75080c6b7af1e09805729c36fc8b25d2a33d9748bdc45',
 'trajectory-policy-report.md':'e4e679323a5f9bbaba01ebd38ab73d4eb7133761cac4aab968c9b83d1291bb87',
 'trajectory-policy-responses.jsonl':'447a5b4c58b45498d7a1fb86bdfc45b953967cea770723c3f95341e4b1aaba96',
}

def check(label,condition):
    if not condition: raise AssertionError(label)
    print(f'[PASS] {label}')

def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def when(value): return datetime.fromisoformat(value.replace('Z','+00:00'))

def main():
    for name,expected in EXPECTED.items():
        check(f'{name} matches published SHA-256',digest(HERE/name)==expected)
    graph=json.loads(GRAPH.read_text(encoding='utf-8'))
    report=REPORT.read_text(encoding='utf-8')
    rows=[json.loads(line) for line in LOG.read_text(encoding='utf-8').splitlines() if line.strip()]
    check('schema version is 1',graph['schema_version']==1)
    check('model is jev-1.13.0',graph['model']=='jev-1.13.0')
    nodes=graph['nodes']; by_id={node['id']:node for node in nodes}
    check('168 unique historical episode nodes',len(nodes)==len(by_id)==168)
    check('every episode completes after its decision snapshot',all(when(n['decision_at'])<=when(n['available_at']) for n in nodes))
    check('168 time slices',len(graph['time_slices'])==len(nodes))
    slices={s['decision_episode']:s for s in graph['time_slices']}
    check('every node has one time slice',set(slices)==set(by_id))
    chronology_counts=[]
    chronology_hashes=[]
    for target_id,slice_ in slices.items():
        target=by_id[target_id]
        eligible=sorted(n['id'] for n in nodes if when(n['available_at'])<when(target['decision_at']))
        chronology_counts.append(slice_['eligible_completed_episodes']==len(eligible))
        state_hash=hashlib.sha256(json.dumps(eligible,sort_keys=True).encode()).hexdigest()
        chronology_hashes.append(slice_['eligible_ids_sha256']==state_hash)
    check('every time slice has the exact eligible count',all(chronology_counts))
    check('every time slice has the exact eligible-state hash',all(chronology_hashes))
    check('all graph edges join published episode nodes',all(e['source'] in by_id and e['target'] in by_id for e in graph['edges']))
    edge_types=Counter(e['type'] for e in graph['edges'])
    check('edge classes are explicit references, chronology, and repeated failures',set(edge_types)=={'explicit_reference','chronological_successor','repeated_failure_family'})
    audits=graph['historical_decision_audits']
    check('28 historical decision audits',len(audits)==graph['counts']['decision_audits']==28)
    check('159 historical counterfactuals',sum(len(a['counterfactuals']) for a in audits)==graph['counts']['historical_counterfactual_edges']==159)
    check('20 clean missed opportunities',len(graph['missed_opportunities'])==graph['counts']['missed_opportunities']==20)
    check('30 hindsight-sensitive leads',len(graph['hindsight_sensitive_leads'])==graph['counts']['hindsight_sensitive_leads']==30)
    check('historical confidence classes remain separate',all(x['historical_policy_confidence']=='cleaner_time_slice' for x in graph['missed_opportunities']) and all(x['historical_policy_confidence']=='hindsight_sensitive' for x in graph['hindsight_sensitive_leads']))
    rankings=graph['current_action_rankings']
    check('five current obligations',len(rankings)==5)
    check('65 current action candidates',sum(map(len,rankings.values()))==graph['counts']['current_action_candidates']==65)
    expected_types={'historical_episode_transfer','accepted_claim_transfer','provisional_code_transfer','failure_transfer'}
    for obligation,candidates in rankings.items():
        present={c['candidate_type'] for c in candidates}
        check(f'{obligation}: every evidence class remains visible',expected_types<=present)
        check(f'{obligation}: each evidence class has a channel anchor',expected_types<={c['candidate_type'] for c in candidates if c.get('channel_anchor')})
    charge=rankings['p253_charge_converter']
    check('unlicensed U(1) extension is not the default charge action',charge[0]['source_id']!='0068')
    check('12 learned operation priors',len(graph['operation_priors'])==12)
    check('operation priors are bounded and have positive evidence mass',all(0<=p['mean_useful_transition']<=1 and p['effective_episode_mass']>0 for p in graph['operation_priors'].values()))
    interventions=graph['stuck_interventions']
    check('five exact-failure interventions',len(interventions)==graph['counts']['stuck_interventions']==5)
    check('every intervention retains an exact failure and eight candidates',all(v['failure_statement'] and len(v['candidates'])==8 for v in interventions.values()))
    check('819 raw TypeSafe response records parse',len(rows)==819)
    check('every raw inference hash is unique',len({r['inference_sha256'] for r in rows})==len(rows))
    check('graph pins the published response ledger',graph['generated_from']['response_log_sha256']==digest(LOG))
    check('credential material is absent',all(token not in (GRAPH.read_text()+LOG.read_text()) for token in ['TYPESAFE_API_KEY=','Authorization: Bearer ','"api_key":']))
    sections=['Historical decision audits','Missed-opportunity map','Counterfactual next-action rankings','Learned research-operation priors','Exact-failure stuck interventions']
    check('report exposes all five policy products',all(section in report for section in sections))
    print()
    print(f'Verified {len(nodes)} nodes, {len(graph["edges"])} graph edges, {len(rows)} raw responses, and 5 policy products.')

if __name__=='__main__': main()
