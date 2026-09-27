"""Audit a completed paired coding run against its frozen data and server receipts."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def require(condition,message):
    if not condition:raise ValueError(message)


def audit(run,dataset,receipts):
    manifest=json.loads((dataset/'manifest.json').read_text())
    protocol=json.loads((run/'protocol.json').read_text())
    done=json.loads((run/'completion.json').read_text())
    require(protocol['manifest']==manifest,'Protocol dataset differs')
    require(protocol['steps']==80 and protocol['group_size']==4,'Unexpected training schedule')
    require(protocol['sandbox_wall_timeout_seconds']==30,'Unexpected execution budget')
    rows={}
    for split in ('train','heldout'):
        require(digest(dataset/f'{split}.json')==manifest['sha256'][split],'Dataset hash differs')
        rows[split]=json.loads((dataset/f'{split}.json').read_text())
    require(len(rows['train'])==80 and len(rows['heldout'])==20,'Incomplete dataset')
    train_ids=[r['id'] for r in rows['train']];heldout_ids=[r['id'] for r in rows['heldout']]
    require(len(set(train_ids+heldout_ids))==100,'Duplicate or overlapping task IDs')
    server={}
    for receipt in records(receipts):
        key=(receipt['phase'],receipt['id'],receipt['completion'])
        server.setdefault(key,[]).append(receipt['result'])
    manifest_hash=digest(dataset/'manifest.json')
    def check_result(row,phase):
        result=row['result'];checks=result['checks'];fields=result['field_checks']
        require(result['dataset_manifest_sha256']==manifest_hash,'Reward receipt dataset differs')
        require(result['wall_timeout_seconds']==30,'Reward receipt execution budget differs')
        require(len(checks)==24 and all(type(v) is bool for v in checks),'Invalid test vector')
        require(len(fields)==24 and all(type(v) is bool for f in fields for v in f.values()),'Invalid field vector')
        require(result['total']==24 and result['passed']==sum(checks),'Incorrect test count')
        require(type(result['success']) is bool and result['success']==all(checks),'Incorrect success flag')
        test_score=sum(checks)/24;field_score=sum(sum(f.values()) for f in fields)/sum(len(f) for f in fields)
        mode=protocol['training_reward_mode'] if phase=='train' else 'tests'
        reward=test_score if mode=='tests' else .5*(test_score+field_score)
        require(result['reward_mode']==mode,'Reward mode differs')
        for key,want in [('reward',reward),('test_score',test_score),('field_score',field_score)]:
            require(math.isclose(result[key],want,rel_tol=1e-12,abs_tol=1e-12),'Reward arithmetic differs: '+key)
        require(result in server.get((phase,row['id'],row['completion']),[]),'Missing matching Vultr receipt')
    events=records(run/'optimizer-events.jsonl');rollouts=records(run/'rollouts.jsonl')
    require([e['step'] for e in events]==list(range(1,81)),'Optimizer steps incomplete or duplicated')
    require(len(rollouts)==320,'Incomplete training rollouts')
    for i,event in enumerate(events):
        group=rollouts[i*4:(i+1)*4]
        require(event['task']==train_ids[i],'Training task sequence differs')
        require(all(r['step']==i+1 and r['id']==train_ids[i] for r in group),'Training rollout sequence differs')
        require(event['rewards']==[r['result']['reward'] for r in group],'Optimizer rewards differ from rollouts')
        for row in group:check_result(row,'train')
    phases={}
    for phase in ('before','after'):
        episodes=records(run/phase/'episodes.jsonl')
        require([r['id'] for r in episodes]==heldout_ids,'Heldout order or completeness differs')
        for row in episodes:check_result(row,phase)
        summary=json.loads((run/phase/'summary.json').read_text())
        require(summary['complete'] and summary['completed']==summary['expected']==20,'Incomplete evaluation summary')
        require(summary['successes']==sum(r['result']['success'] for r in episodes),'Summary success count differs')
        for key,result_key in [('mean_reward','reward'),('mean_field_score','field_score')]:
            require(math.isclose(summary[key],sum(r['result'][result_key] for r in episodes)/20,abs_tol=1e-12),'Summary mean differs')
        phases[phase]=episodes
    pairs=[{'id':b['id'],'before':b['result'],'after':a['result']} for b,a in zip(phases['before'],phases['after'],strict=True)]
    require(done['pairs']==pairs,'Completion pairing differs')
    b=sum(p['before']['success'] for p in pairs);a=sum(p['after']['success'] for p in pairs)
    require(done['before_success']==b and done['after_success']==a and done['heldout']==20,'Completion scores differ')
    require(done['percentage_point_gain']==(a-b)*5,'Completion gain differs')
    require(done['changed_parameters']==events[-1]['changed_parameters']>0,'Missing adapter weight changes')
    require(math.isclose(done['delta_l2'],events[-1]['delta_l2']) and done['delta_l2']>0,'Adapter delta differs')
    paths=['protocol.json','completion.json','optimizer-events.jsonl','rollouts.jsonl','before/summary.json','after/summary.json','before/episodes.jsonl','after/episodes.jsonl','adapter/adapter_model.safetensors']
    return {'verified':True,'run':run.name,'model':protocol['model'],'steps':80,'heldout':20,
        'before_successes':b,'after_successes':a,'gain_percentage_points':(a-b)*5,
        'new_successes':sum(not p['before']['success'] and p['after']['success'] for p in pairs),
        'regressions':sum(p['before']['success'] and not p['after']['success'] for p in pairs),
        'mean_test_score':{phase:sum(r['result']['test_score'] for r in values)/20 for phase,values in phases.items()},
        'mean_field_score':{phase:sum(r['result']['field_score'] for r in values)/20 for phase,values in phases.items()},
        'reward_contrast_batches':sum(len(set(e['rewards']))>1 for e in events),
        'changed_adapter_parameters':done['changed_parameters'],'delta_l2':done['delta_l2'],
        'matched_server_receipts':360,'dataset_manifest_sha256':manifest_hash,
        'evidence_sha256':{p:digest(run/p) for p in paths}}


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--dataset',type=Path,required=True);p.add_argument('--receipts',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    result=audit(args.run,args.dataset,args.receipts)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
