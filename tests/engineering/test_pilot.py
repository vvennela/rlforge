import json
import pytest
from akshara_forge.engineering.pilot import build_dataset, parse, score
from akshara_forge.engineering.worker import apply


def test_frozen_pilot_oracle_and_splits(tmp_path):
    path=tmp_path/'dataset'; build_dataset(path)
    train=json.loads((path/'train.json').read_text()); test=json.loads((path/'heldout.json').read_text())
    assert len(train)==80 and len(test)==20
    assert not {r['geometry_signature'] for r in train}&{r['geometry_signature'] for r in test}
    assert len({json.dumps(r['prompt']) for r in train+test})==100
    for row in train+test:
        assert not score(row,row['initial'])['success']
        assert score(row,row['initial'])['reward']==0
        final=apply(row['initial'],[{'tool':'place','brick':b} for b in row['missing']])
        result=score(row,final)
        assert result['success'] and result['reward']==1
        duplicate={**row['missing'][0],'id':'duplicate'}
        bad=score(row,final+[duplicate])
        assert not bad['success'] and bad['reward']<=.1


def test_pilot_action_boundaries():
    for value in ['{"actions":[{"tool":"remove","id":"b1"}]}','{"actions":[],"reward":1}', 'not json']:
        with pytest.raises((ValueError,KeyError)): parse(value)
