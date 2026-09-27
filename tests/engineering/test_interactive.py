import copy,json
from pathlib import Path
import pytest
from akshara_forge.engineering.interactive import replay,actions,generate
from akshara_forge.engineering.pilot import make_case


def case():
    r=make_case(0,(32,7,23,21,24,0,0,4),'train')
    r['editable_initial']=[];r['target']={'x':7,'y':0,'bottom':9,'top':21,'cross_section':[1,1]}
    return r


def placement(id,x=7,z=9):
    return {'tool':'place','brick':{'id':id,'part':'brick_1x1','x':x,'y':0,'z':z,'rotation':0}}


def test_multiple_placements_and_correction(tmp_path):
    row=case();history=[]
    history.append(json.dumps({'actions':[placement('p1',x=8)]}))
    wrong=replay(row,history,tmp_path,sandbox=False)
    assert not wrong['observation']['success']
    history.append(json.dumps({'actions':[{'tool':'move','id':'p1','x':7,'y':0,'z':9,'rotation':0}]}))
    fixed=replay(row,history,tmp_path,sandbox=False)
    assert fixed['step_reward']>0
    for i,z in enumerate([12,15,18],2):history.append(json.dumps({'actions':[placement(f'p{i}',z=z)]}))
    result=replay(row,history,tmp_path,sandbox=False)
    assert result['observation']['success'] and len(result['observation']['editable_bricks'])==4
    assert result['observation']['turns_used']==5
    assert sum(t['reward'] for t in result['transitions'])==pytest.approx(result['return'])


def test_no_cycle_reward_and_locked_geometry(tmp_path):
    row=case();a=json.dumps({'actions':[placement('p1')]});b=json.dumps({'actions':[{'tool':'remove','id':'p1'}]})
    result=replay(row,[a,b],tmp_path,sandbox=False)
    assert result['return']==pytest.approx(0)
    illegal=json.dumps({'actions':[{'tool':'remove','id':row['initial'][0]['id']}]})
    result=replay(row,[illegal]*8,tmp_path,sandbox=False)
    assert result['observation']['done'] and result['observation']['last_action_error']
    assert result['observation']['potential']==0
    with pytest.raises(ValueError):replay(row,[illegal]*9,tmp_path,sandbox=False)


def test_replay_is_idempotent(tmp_path):
    row=case();history=[json.dumps({'actions':[placement('p1')]})]
    assert replay(row,history,tmp_path,sandbox=False)==replay(row,history,tmp_path,sandbox=False)


def test_policy_gradient_direction():
    torch=pytest.importorskip('torch')
    from akshara_forge.engineering.train_interactive import advantages,policy_objective
    assert advantages([1.,0.])==[1.,-1.]
    assert advantages([.3,.3])==[0.,0.]
    for advantage,sign in [(1.,-1),(-1.,1),(0.,0)]:
        logp=torch.tensor([-.2,-.5],requires_grad=True)
        loss=policy_objective(logp,logp.detach().clone(),advantage,16)
        loss.backward()
        assert torch.all(torch.sign(logp.grad)==sign)


def test_new_bridge_cases_progress_from_one_piece_to_correction(tmp_path):
    old=tmp_path/'old';old.mkdir()
    for split in ('train','heldout'):(old/f'{split}.json').write_text('[]')
    dest=tmp_path/'new';generate(dest,old)
    for split,count in [('train',20),('heldout',5)]:
        rows=json.loads((dest/f'{split}.json').read_text())
        for level in range(1,5):
            group=[r for r in rows if r['curriculum']['level']==level]
            assert len(group)==count and all(len(r['missing'])==level for r in group)
            assert all(bool(r['editable_initial'])==(level==4) for r in group)
            row=group[0];moves=[]
            if row['editable_initial']:moves.append({'tool':'remove','id':'repair_seed'})
            moves += [{'tool':'place','brick':dict(b,id='new_'+b['id'])} for b in row['missing']]
            history=[json.dumps({'actions':moves[i:i+4]}) for i in range(0,len(moves),4)]
            assert replay(row,history,tmp_path/'traces',sandbox=False)['observation']['success']
