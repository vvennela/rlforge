"""Verify the live sandbox and reward boundary before starting model evaluation."""
import argparse, inspect, json, urllib.request, urllib.error
from pathlib import Path
from .tasks import oracle


def run(root):
    rows=json.loads((root/'dataset-v3/train.json').read_text())+json.loads((root/'dataset-v3/heldout.json').read_text())
    token=(root/'reward-token').read_text().strip();report={'passed':False,'checks':[]}
    def record(test,result):
        report['checks'].append({'test':test,'result':result})
        (root/'preflight-v3.json').write_text(json.dumps(report,indent=2))
    def call(row,code,phase='preflight'):
        req=urllib.request.Request('http://127.0.0.1:8771/grade',data=json.dumps({'id':row['id'],'completion':code,'phase':phase}).encode(),headers={'Authorization':'Bearer '+token,'Content-Type':'application/json'})
        return json.load(urllib.request.urlopen(req,timeout=75))
    for i in [0,20,80,99]:
        row=rows[i];code=inspect.getsource(oracle)+'\ndef solve(problem): return oracle(problem,'+repr(row['configuration'])+')\n'
        result=call(row,code);record('reference '+row['id'],result)
        assert result['passed']==24 and result['success'],result
    for label,code in [('reward_spoof','def solve(p): return {"reward":1,"success":True}'),
       ('private_file','def solve(p):\n return open("'+str(root/'dataset-v3/heldout.json')+'").read()'),
       ('forged_envelope','import os\nos.write(1,b\'{"reward":1,"success":true}\')\nos._exit(0)')]:
        result=call(rows[0],code);record(label,result);assert result['reward']==0 and not result['success'],result
    try:call(rows[80],'def solve(p): return None','train');raise AssertionError('Heldout request accepted')
    except urllib.error.HTTPError as e:assert e.code==400
    report.update(passed=True,heldout_train_blocked=True)
    (root/'preflight-v3.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'passed':True,'reference_programs':4,'tests_each':24,'attacks_rejected':3,'heldout_train_blocked':True}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);run(p.parse_args().root)
