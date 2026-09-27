import hashlib
import json
import pytest
from akshara_forge.compare import compare
from akshara_forge.teacher import grade_run
from akshara_forge.io import write_json,write_jsonl


def test_grader_refuses_changed_reference_dataset(tmp_path):
    root=tmp_path/'root';run=tmp_path/'run'
    problem={'problem_id':'p','paper_id':'paper','prompt':'Question','reference_answer':4}
    digest=hashlib.sha256(json.dumps([problem],sort_keys=True).encode()).hexdigest()
    write_json(run/'config.json',{'papers':['paper'],'split':'test','limit':None,'problem_hash':digest})
    problem['reference_answer']=5
    write_jsonl(root/'data/problems/paper/test.jsonl',[problem])
    with pytest.raises(ValueError,match='differs from the frozen'):grade_run(root,run)


def pair(tmp_path):
    a,b=tmp_path/'a',tmp_path/'b'
    config={'papers':['p'],'split':'test','limit':None,'attempts':1,'problem_hash':'same'}
    receipt={'worker_sha256':'same','ollama_version':'1','model':{'details':{'quantization_level':'Q4'}}}
    rows=[{'problem_id':str(i),'paper_id':'p','attempt':0,'status':'graded','correct':False,'verification':'programmatic_answer'} for i in range(20)]
    for directory in [a,b]:
        write_json(directory/'config.json',config);write_json(directory/'model-receipt.json',receipt)
        write_jsonl(directory/'episodes.jsonl',rows)
    return a,b,rows


def test_paired_comparison_and_missing_episode_rejected(tmp_path):
    a,b,rows=pair(tmp_path);rows[0]['correct']=True;write_jsonl(b/'episodes.jsonl',rows)
    result=compare(a,b);assert result['accuracy_delta']==.05 and result['improved']==1
    write_jsonl(b/'episodes.jsonl',rows[:-1])
    with pytest.raises(ValueError,match='episode sets differ'):compare(a,b)


def test_backend_precision_mismatch_rejected(tmp_path):
    a,b,_=pair(tmp_path)
    receipt=json.loads((b/'model-receipt.json').read_text());receipt['model']['details']['quantization_level']='BF16'
    write_json(b/'model-receipt.json',receipt)
    with pytest.raises(ValueError,match='precision'):compare(a,b)
