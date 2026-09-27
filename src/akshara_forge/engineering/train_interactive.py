"""On-policy, paired-trajectory REINFORCE with leave-one-out baselines.

Only assistant action tokens receive gradients. Environment observations are
context, never training labels. Each batch contains two complete eight-turn
attempts from the same initial state; weights stay fixed during collection.
"""
from __future__ import annotations
import argparse, hashlib, json, os, shutil, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import httpx
from .interactive import SYSTEM,TURNS
from .pilot import write


def remote(row,history,phase):
    token=Path(os.environ['AKSHARA_ENGINEERING_TOKEN_FILE']).read_text().strip()
    # Replay is stateless, so retrying a transport failure cannot duplicate a move.
    for attempt in range(3):
        try:
            with httpx.Client(timeout=90) as c:
                r=c.post(os.environ['AKSHARA_ENGINEERING_REWARD_URL']+'/step',headers={'Authorization':'Bearer '+token},
                         json={'id':row['id'],'history':history,'phase':phase})
                r.raise_for_status();return r.json()
        except (httpx.TransportError,httpx.HTTPStatusError) as exc:
            if isinstance(exc,httpx.HTTPStatusError) and exc.response.status_code not in (500,502,503,504):raise
            print(json.dumps({'event':'reward_service_retry','attempt':attempt+1,'task':row['id'],
                'error':str(exc),'response':exc.response.text[:500] if isinstance(exc,httpx.HTTPStatusError) else None}),flush=True)
            if attempt==2:raise
            time.sleep(2)


def append(path,value):
    with Path(path).open('a') as f:f.write(json.dumps(value,allow_nan=False)+'\n')


def resume_evidence(checkpoint,output,protocol,step):
    """Preserve the parent run and carry forward only evidence through the checkpoint."""
    parent=checkpoint.parent
    previous=json.loads((parent/'protocol.json').read_text())
    for key,value in protocol.items():
        if key!='source_hashes' and previous.get(key)!=value:
            raise ValueError('Resume protocol differs: '+key)
    for name in ('adapter_model.safetensors','optimizer.pt'):
        if not (checkpoint/name).is_file():raise ValueError('Incomplete checkpoint: '+name)
    records=[json.loads(l) for l in (parent/'optimizer-events.jsonl').read_text().splitlines()]
    records=[r for r in records if r['step']<=step]
    if [r['step'] for r in records]!=list(range(1,step+1)):raise ValueError('Checkpoint history is incomplete')
    baseline=json.loads((parent/'before/summary.json').read_text())
    if not baseline['complete']:raise ValueError('Original baseline is incomplete')
    traces=(parent/'training-traces.jsonl').read_text().splitlines()[:step*2]
    if len(traces)!=step*2:raise ValueError('Checkpoint trajectories are incomplete')
    shutil.copytree(parent/'before',output/'before')
    shutil.copy2(parent/'initial-adapter.pt',output/'initial-adapter.pt')
    (output/'optimizer-events.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    (output/'training-traces.jsonl').write_text('\n'.join(traces)+'\n')
    write(output/'resumption.json',{'checkpoint':str(checkpoint),'completed_step':step,
        'next_step':step+1,'restored':['adapter','optimizer','torch_rng','cuda_rng'],
        'parent_evidence_preserved':True,'checkpoint_sha256':{n:hashlib.sha256((checkpoint/n).read_bytes()).hexdigest()
        for n in ('adapter_model.safetensors','optimizer.pt')}})
    return baseline['records'],records[-1]['reward_contrast_updates']


def trajectories(model,tokenizer,rows,phase,sample,output):
    import torch
    episodes=[]
    for row in rows:
        state=remote(row,[],phase)
        episodes.append({'id':row['id'],'history':[],'turns':[],'state':state,
            'messages':[{'role':'system','content':SYSTEM},{'role':'user','content':json.dumps(state['observation'],separators=(',',':'))}]})
    model.eval();model.config.use_cache=True
    for turn in range(TURNS):
        active=[i for i,e in enumerate(episodes) if not e['state']['observation']['done']]
        if not active:break
        texts=[tokenizer.apply_chat_template(episodes[i]['messages'],tokenize=False,add_generation_prompt=True) for i in active]
        inputs=tokenizer(texts,return_tensors='pt',padding=True).to(model.device)
        if inputs['input_ids'].shape[1]>6000:raise RuntimeError('Context bound exceeded; do not silently truncate feedback')
        width=inputs['input_ids'].shape[1]
        with torch.inference_mode():
            generated=model.generate(**inputs,max_new_tokens=256,do_sample=sample,temperature=1.,top_p=1.,top_k=0,
                 repetition_penalty=1.,eos_token_id=tokenizer.eos_token_id,pad_token_id=tokenizer.pad_token_id)
        for j,i in enumerate(active):
            ids=generated[j,width:].tolist()
            if tokenizer.eos_token_id in ids:ids=ids[:ids.index(tokenizer.eos_token_id)+1]
            answer=tokenizer.decode(ids,skip_special_tokens=True)
            prompt=inputs['input_ids'][j][inputs['attention_mask'][j].bool()].tolist()
            episodes[i]['turns'].append({'input_ids':prompt,'output_ids':ids,'text':answer})
            episodes[i]['history'].append(answer)
        with ThreadPoolExecutor(max_workers=2) as pool:
            results=list(pool.map(lambda i:remote(rows[i],episodes[i]['history'],phase),active))
        for i,result in zip(active,results,strict=True):
            e=episodes[i];e['state']=result;e['turns'][-1]['feedback']=result
            e['messages'].extend([{'role':'assistant','content':e['history'][-1]},
                 {'role':'user','content':json.dumps(result['observation'],separators=(',',':'))}])
    model.config.use_cache=False
    for e in episodes:
        e.pop('messages');e['phase']=phase
        append(output,e)
    return episodes


def behavior(episode):
    transitions=episode['state']['transitions']
    placements=sum(sum(a.get('tool')=='place' for a in t['actions']) for t in transitions)
    corrections=sum(any(a.get('tool') in ['move','remove'] for a in t['actions']) and
                    (t['reward']>0 or t['after_extra_cells']<t['before_extra_cells']) for t in transitions)
    obs=episode['state']['observation']
    return {'id':episode['id'],'success':obs['success'],'gap_iou':obs['gap_iou'],'potential':obs['potential'],
            'turns':len(episode['turns']),'placements':placements,'beneficial_corrections':corrections,
            'editable_pieces':len(obs['editable_bricks']),'return':episode['state']['return']}


def evaluation(model,tokenizer,rows,phase,dest):
    dest.mkdir(parents=True,exist_ok=True);records=[]
    for i in range(0,len(rows),2):
        batch=trajectories(model,tokenizer,rows[i:i+2],phase,False,dest/'traces.jsonl')
        records.extend(behavior(e) for e in batch)
        write(dest/'summary.json',{'completed':len(records),'expected':len(rows),'complete':len(records)==len(rows),
            'successes':sum(e['success'] for e in records),'mean_gap_iou':sum(e['gap_iou'] for e in records)/len(records),
            'mean_potential':sum(e['potential'] for e in records)/len(records),'multi_placement_episodes':sum(e['placements']>=2 for e in records),
            'beneficial_corrections':sum(e['beneficial_corrections'] for e in records),'records':records})
    return records


def advantages(returns):
    if len(returns)<2:raise ValueError('Leave-one-out requires at least two attempts')
    return [v-(sum(returns)-v)/(len(returns)-1) for v in returns]


def policy_objective(logp,reference,advantage,normalizer):
    """Caller supplies action-token log probabilities only, never observations."""
    import torch
    log_ratio=reference-logp
    kl=torch.exp(log_ratio)-log_ratio-1
    return (-advantage*logp.sum()+.02*kl.sum())/normalizer


def run(args):
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer,set_seed
    from peft import get_peft_model,LoraConfig,set_peft_model_state_dict
    from safetensors.torch import load_file
    if args.output.exists():raise ValueError('Choose a fresh output directory')
    args.output.mkdir(parents=True)
    manifest=json.loads((args.dataset/'manifest.json').read_text())
    for split,digest in manifest['sha256'].items():
        if hashlib.sha256((args.dataset/f'{split}.json').read_bytes()).hexdigest()!=digest:raise ValueError('Frozen dataset changed')
    training=json.loads((args.dataset/'train.json').read_text())
    quick_rows=[training[i] for i in [0,1,4,5]]
    testing=quick_rows if args.stage=='quick' else json.loads((args.dataset/'heldout.json').read_text())
    if args.stage=='long' and {r['id'] for r in testing}&{r['id'] for r in training}:raise ValueError('Split overlap')
    set_seed(20260928)
    protocol={'stage':args.stage,'steps':args.steps,'manifest':manifest,'model':str(args.model),'seed':20260928,
       'algorithm':'On-policy paired-trajectory REINFORCE with leave-one-out return baseline; sampled KL regularizer',
       'turns':8,'group_size':2,'temperature':1.,'top_p':1.,'top_k':0,'tokens_per_turn':256,
       'learning_rate':1e-5,'kl_coefficient':.02,'lora_rank':8,'lora_alpha':16,'modules':['q_proj','v_proj'],
       'precision':'bfloat16','evaluation':'Greedy; same eight turns, palette, prompt, feedback, and token limits before/after',
       'normalization':'Sum action-token policy loss, divided by fixed group_size*8; no gradient on feedback tokens',
       'checkpoint_selection':'Final fixed-step adapter only; heldout never used by quick run or optimizer',
       'source_hashes':{name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in ['interactive.py','train_interactive.py']}}
    write(args.output/'protocol.json',protocol)
    tokenizer=AutoTokenizer.from_pretrained(args.model);tokenizer.pad_token=tokenizer.eos_token;tokenizer.padding_side='left'
    base=AutoModelForCausalLM.from_pretrained(args.model,torch_dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda')
    model=get_peft_model(base,LoraConfig(r=8,lora_alpha=16,lora_dropout=0.,target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM'))
    model.enable_input_require_grads();model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    initial={n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-5,weight_decay=0.)
    effective=0;step=0
    if args.resume:
        state=torch.load(args.resume/'optimizer.pt',map_location='cpu',weights_only=True)
        step=state['step']
        if not 0<step<args.steps:raise ValueError('Resume checkpoint must precede final step')
        before,effective=resume_evidence(args.resume,args.output,protocol,step)
        initial=torch.load(args.output/'initial-adapter.pt',map_location='cpu',weights_only=True)
        result=set_peft_model_state_dict(model,load_file(str(args.resume/'adapter_model.safetensors')))
        if result.unexpected_keys:raise ValueError('Unexpected adapter keys')
        optimizer.load_state_dict(state['optimizer'])
        torch.set_rng_state(state['rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
        del state
        print(json.dumps({'event':'training_resumed','completed_step':step,'target_steps':args.steps}),flush=True)
    else:
        torch.save(initial,args.output/'initial-adapter.pt')
        before=evaluation(model,tokenizer,testing,'quick' if args.stage=='quick' else 'before',args.output/'before')
    def delta():
        changed=0;squares=0.
        for n,p in model.named_parameters():
            if n in initial:
                d=p.detach().float().cpu()-initial[n].float();changed+=int(torch.count_nonzero(d));squares+=float((d*d).sum())
        return {'changed_parameters':changed,'delta_l2':squares**.5}
    schedule=quick_rows if args.stage=='quick' else sorted(training,key=lambda r:(r['difficulty'],r['id']))
    try:
        for step in range(step+1,args.steps+1):
            started=time.time()
            index=(step-1)%len(schedule) if args.stage=='quick' or step<=len(schedule) else len(schedule)-20+(step-len(schedule)-1)%20
            row=schedule[index]
            attempts=trajectories(model,tokenizer,[row,row],'train',True,args.output/'training-traces.jsonl')
            returns=[e['state']['return'] for e in attempts];adv=advantages(returns)
            optimizer.zero_grad(set_to_none=True);model.train();model.config.use_cache=False;loss_value=0.
            for e,a in zip(attempts,adv,strict=True):
                for turn in e['turns']:
                    # Exactly the policy's generated tokens, including its EOS. Tool observations are never labels.
                    prompt=turn['input_ids'];answer=turn['output_ids']
                    sequence=torch.tensor([prompt+answer],device='cuda');start=len(prompt)-1;end=sequence.shape[1]-1
                    targets=sequence[:,len(prompt):]
                    if not answer:continue
                    with torch.no_grad(),model.disable_adapter():
                        ref_logits=model(sequence,use_cache=False).logits[:,start:end,:].float()
                        ref=ref_logits.log_softmax(-1).gather(-1,targets.unsqueeze(-1)).squeeze(-1)
                        del ref_logits
                    logits=model(sequence,use_cache=False).logits[:,start:end,:].float()
                    logp=logits.log_softmax(-1).gather(-1,targets.unsqueeze(-1)).squeeze(-1)
                    loss=policy_objective(logp,ref,a,len(attempts)*TURNS)
                    if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
                    loss_value+=float(loss.detach());loss.backward()
                    del logits,logp,ref,loss,sequence
            parameters=[p for p in model.parameters() if p.requires_grad]
            norm=float(torch.nn.utils.clip_grad_norm_(parameters,1.))
            if not torch.isfinite(torch.tensor(norm)):raise RuntimeError('Nonfinite gradient')
            nonzero=sum(int(torch.count_nonzero(p.grad)) for p in parameters if p.grad is not None)
            optimizer.step();effective+=int(abs(returns[0]-returns[1])>1e-8)
            record={'step':step,'seconds':time.time()-started,'task':row['id'],'returns':returns,'advantages':adv,
                'gradient_norm_before_clipping':norm,'nonzero_gradient_elements':nonzero,'loss':loss_value,
                'reward_contrast_updates':effective,'episodes':[behavior(e) for e in attempts],**delta()}
            append(args.output/'optimizer-events.jsonl',record)
            print(json.dumps({k:v for k,v in record.items() if k!='episodes'}),flush=True)
            # Durable complete state after every batch; no rollback of several updates on a service outage.
            target=args.output/f'checkpoint-{step}';model.save_pretrained(target);tokenizer.save_pretrained(target)
            torch.save({'optimizer':optimizer.state_dict(),'step':step,'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()},target/'optimizer.pt')
        after=evaluation(model,tokenizer,testing,'quick' if args.stage=='quick' else 'after',args.output/'after')
        model.save_pretrained(args.output/'adapter');tokenizer.save_pretrained(args.output/'adapter')
        summary={'stage':args.stage,'optimizer_steps':step,'reward_contrast_updates':effective,**delta(),
            'evaluation_scope':'training cases only; smoke test' if args.stage=='quick' else '20 frozen heldout cases',
            'before_successes':sum(r['success'] for r in before),'after_successes':sum(r['success'] for r in after),
            'total':len(testing),'paired':[{'before':b,'after':a} for b,a in zip(before,after,strict=True)]}
        # Launch gate verifies behavior, not a heldout uplift or chosen best checkpoint.
        all_episodes=before+after
        for line in (args.output/'training-traces.jsonl').read_text().splitlines():all_episodes.append(behavior(json.loads(line)))
        summary['multi_placement_observed']=any(e['placements']>=2 and e['editable_pieces']>=2 for e in all_episodes)
        summary['beneficial_correction_observed']=any(e['beneficial_corrections']>0 for e in all_episodes)
        summary['quick_gate_pass']=summary['multi_placement_observed'] and summary['beneficial_correction_observed']
        write(args.output/'completion.json',summary)
    except Exception as exc:
        model.save_pretrained(args.output/'interrupted-adapter')
        write(args.output/'interruption.json',{'step':step,'error':str(exc),**delta()});raise


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--steps',type=int,required=True);p.add_argument('--stage',choices=['quick','long'],required=True)
    p.add_argument('--resume',type=Path)
    run(p.parse_args())


if __name__=='__main__':main()
