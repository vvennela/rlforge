import json
from collections import Counter
import pytest
from akshara_forge.curriculum import slot,schedule
from akshara_forge.coding.curriculum import generate,reference_program
from akshara_forge.coding.feedback import canonical


def test_stage_quotas_and_bounded_schedule():
    for count in (20,100):
        slots=[slot(i,count) for i in range(count)]
        for level in range(1,5):
            assert sum(s['split']=='train' and s['level']==level for s in slots)==count//5
            assert sum(s['split']=='heldout' and s['level']==level for s in slots)==count//20
    rows=[{'id':str(i),'split':'train','curriculum':{'level':i//20+1}} for i in range(80)]
    plan=schedule(rows,24)
    assert [r['curriculum']['level'] for r in plan]==[1]*6+[2]*6+[3]*6+[4]*6
    with pytest.raises(ValueError):schedule([{**rows[0],'split':'heldout'}],24)


def test_real_curriculum_solutions_and_isolation(tmp_path):
    source=tmp_path/'source';source.mkdir();(source/'manifest.json').write_text('{}')
    dest=tmp_path/'data';generate(dest,source)
    rows=[];inputs={}
    for split,count in [('train',80),('heldout',20)]:
        group=json.loads((dest/f'{split}.json').read_text());assert len(group)==count
        assert Counter(r['curriculum']['level'] for r in group)=={i:count//4 for i in range(1,5)}
        inputs[split]=set()
        for row in group:
            private={canonical(t['input']) for t in row['tests']}
            dev={canonical(t['input']) for t in row['development_tests']}
            assert len(private)==24 and len(dev)==6 and not private&dev
            inputs[split].update(private|dev)
            scope={};exec(reference_program(row),scope)
            for t in row['tests']+row['development_tests']:assert scope['solve'](t['input'])==t['expected']
        rows+=group
    assert not inputs['train']&inputs['heldout']
    assert 'single incorrect' in rows[0]['prompt'][1]['content']
    with pytest.raises(ValueError,match='immutable'):generate(dest,source)
