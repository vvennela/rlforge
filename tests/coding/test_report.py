import hashlib
import json
import pytest
from akshara_forge.coding.report import audit


def fixture(tmp_path):
    run=tmp_path/'run';dataset=tmp_path/'dataset';dataset.mkdir();run.mkdir()
    def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v))
    def lines(p,rows):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    for split,n in [('train',80),('heldout',20)]:write(dataset/f'{split}.json',[{'id':f'{split}-{i}'} for i in range(n)])
    manifest={'sha256':{s:hashlib.sha256((dataset/f'{s}.json').read_bytes()).hexdigest() for s in ('train','heldout')}}
    write(dataset/'manifest.json',manifest);fingerprint=hashlib.sha256((dataset/'manifest.json').read_bytes()).hexdigest()
    write(run/'protocol.json',{'manifest':manifest,'steps':80,'group_size':4,'sandbox_wall_timeout_seconds':30,'training_reward_mode':'components','model':'pinned-test-model'})
    receipts=[]
    def row(id,phase,success):
        result={'dataset_manifest_sha256':fingerprint,'wall_timeout_seconds':30,'checks':[success]*24,'field_checks':[{'value':success}]*24,'total':24,'passed':24*success,'success':success,'reward_mode':'components' if phase=='train' else 'tests','reward':float(success),'test_score':float(success),'field_score':float(success)}
        row={'id':id,'completion':'program '+id,'result':result};receipts.append({'phase':phase,**row});return row
    events=[];rollouts=[]
    for i in range(80):
        rollouts.extend({'step':i+1,**row(f'train-{i}','train',bool(j%2))} for j in range(4))
        events.append({'step':i+1,'task':f'train-{i}','rewards':[0,1,0,1],'changed_parameters':123,'delta_l2':.1})
    lines(run/'optimizer-events.jsonl',events);lines(run/'rollouts.jsonl',rollouts)
    phases={}
    for phase in ('before','after'):
        rows=[row(f'heldout-{i}',phase,i<(2 if phase=='before' else 4)) for i in range(20)]
        phases[phase]=rows;lines(run/phase/'episodes.jsonl',rows)
        successes=sum(r['result']['success'] for r in rows)
        write(run/phase/'summary.json',{'complete':True,'completed':20,'expected':20,'successes':successes,'mean_reward':successes/20,'mean_field_score':successes/20})
    write(run/'completion.json',{'pairs':[{'id':b['id'],'before':b['result'],'after':a['result']} for b,a in zip(phases['before'],phases['after'])],'before_success':2,'after_success':4,'heldout':20,'percentage_point_gain':10,'changed_parameters':123,'delta_l2':.1})
    write(run/'adapter/adapter_model.safetensors',{'fixture':'not a real adapter'})
    receipt_file=tmp_path/'receipts.jsonl';lines(receipt_file,receipts)
    return run,dataset,receipt_file


def test_report_requires_complete_paired_run_and_matching_server_receipts(tmp_path):
    run,dataset,receipts=fixture(tmp_path)
    report=audit(run,dataset,receipts)
    assert report['verified'] and report['gain_percentage_points']==10
    assert report['new_successes']==2 and report['regressions']==0
    assert report['matched_server_receipts']==360
    lines=receipts.read_text().splitlines();receipts.write_text('\n'.join(lines[:-1]))
    with pytest.raises(ValueError,match='matching Vultr receipt'):audit(run,dataset,receipts)


@pytest.mark.parametrize('mutation',['duplicate_step','wrong_gain','swapped_holdout','modified_data'])
def test_report_rejects_inconsistent_evidence(tmp_path,mutation):
    run,dataset,receipts=fixture(tmp_path)
    if mutation=='duplicate_step':
        p=run/'optimizer-events.jsonl';lines=p.read_text().splitlines();lines[-1]=lines[-2];p.write_text('\n'.join(lines))
    elif mutation=='wrong_gain':
        p=run/'completion.json';d=json.loads(p.read_text());d['percentage_point_gain']=99;p.write_text(json.dumps(d))
    elif mutation=='swapped_holdout':
        p=run/'after/episodes.jsonl';lines=p.read_text().splitlines();lines[0],lines[1]=lines[1],lines[0];p.write_text('\n'.join(lines))
    else:
        p=dataset/'train.json';p.write_text(p.read_text()+' ')
    with pytest.raises(ValueError):audit(run,dataset,receipts)
