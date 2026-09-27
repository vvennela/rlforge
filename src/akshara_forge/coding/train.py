"""Qwen coding RL: sampled program groups, sandbox test rewards, frozen paired eval."""
import argparse,hashlib,json,os,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import httpx
from ..engineering.train_interactive import advantages,append
from ..engineering.pilot import write


def reward(row,text,phase):
    token=Path(os.environ['AKSHARA_CODING_TOKEN_FILE']).read_text().strip()
    for attempt in range(3):
        try:
            r=httpx.post(os.environ['AKSHARA_CODING_REWARD_URL']+'/grade',headers={'Authorization':'Bearer '+token},json={'id':row['id'],'completion':text,'phase':phase},timeout=120)
            r.raise_for_status();return r.json()
        except (httpx.TransportError,httpx.HTTPStatusError) as exc:
            if isinstance(exc,httpx.HTTPStatusError) and exc.response.status_code<500:raise
            if attempt==2:raise
            time.sleep(2)


def sample(model,tokenizer,rows,phase,stochastic):
    import torch
    model.eval();model.config.use_cache=True
    texts=[tokenizer.apply_chat_template(row['prompt'],tokenize=False,add_generation_prompt=True) for row in rows]
    inputs=tokenizer(texts,return_tensors='pt',padding=True).to('cuda');width=inputs['input_ids'].shape[1]
    with torch.inference_mode():
        generated=model.generate(**inputs,max_new_tokens=1536,do_sample=stochastic,temperature=.8,top_p=.95,top_k=0,pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id)
    records=[]
    for i,row in enumerate(rows):
        ids=generated[i,width:].tolist()
        if tokenizer.eos_token_id in ids:ids=ids[:ids.index(tokenizer.eos_token_id)+1]
        records.append({'id':row['id'],'completion':tokenizer.decode(ids,skip_special_tokens=True),'input_ids':inputs['input_ids'][i][inputs['attention_mask'][i].bool()].tolist(),'output_ids':ids})
    with ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(lambda x:reward(x[0],x[1]['completion'],phase),zip(rows,records)))
    for record,result in zip(records,results):record['result']=result
    model.config.use_cache=False
    return records


def evaluate(model,tokenizer,rows,dest,phase):
    dest.mkdir(parents=True,exist_ok=True);records=[]
    # Resume evaluation only from a complete prefix, never resample completed cases.
    if (dest/'episodes.jsonl').exists():records=[json.loads(x) for x in (dest/'episodes.jsonl').read_text().splitlines()]
    if [r['id'] for r in records]!=[r['id'] for r in rows[:len(records)]]:raise ValueError('Evaluation prefix mismatch')
    for offset in range(len(records),len(rows),2):
        batch=sample(model,tokenizer,rows[offset:offset+2],phase,False)
        for record in batch:
            record.pop('input_ids');record['generated_tokens']=len(record.pop('output_ids'));append(dest/'episodes.jsonl',record);records.append(record)
        write(dest/'summary.json',{'completed':len(records),'expected':len(rows),'complete':len(records)==len(rows),'successes':sum(r['result']['success'] for r in records),'mean_reward':sum(r['result']['reward'] for r in records)/len(records)})
        print(json.dumps({'event':phase,'completed':len(records),'successes':sum(r['result']['success'] for r in records)}),flush=True)
    return records


def run(args):
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer,set_seed
    from peft import LoraConfig,get_peft_model,set_peft_model_state_dict
    from safetensors.torch import load_file
    if args.output.exists() and not args.resume:raise ValueError('Fresh output required')
    args.output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((args.dataset/'manifest.json').read_text())
    for split,digest in manifest['sha256'].items():
        if hashlib.sha256((args.dataset/f'{split}.json').read_bytes()).hexdigest()!=digest:raise ValueError('Dataset changed')
    training=json.loads((args.dataset/'train.json').read_text());heldout=json.loads((args.dataset/'heldout.json').read_text())
    if {r['id'] for r in training}&{r['id'] for r in heldout}:raise ValueError('Split overlap')
    protocol={'model':str(args.model),'manifest':manifest,'steps':args.steps,'group_size':4,'max_tokens':1536,'seed':20260929,'temperature':.8,'top_p':.95,'top_k':0,'learning_rate':2e-5,'kl':.01,'lora_rank':8,'lora_alpha':16,'modules':['q_proj','v_proj'],'precision':'bfloat16','algorithm':'Group-relative REINFORCE with leave-one-out rewards and sampled KL penalty','reward':f"Fraction of {manifest['tests_per_task']} sandboxed tests passed",'evaluation':'20 frozen specifications; greedy, same 1536-token budget before and after','selection':'Final fixed-step adapter only; no holdout checkpoint selection','source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if args.resume:
        if json.loads((args.output/'protocol.json').read_text())!=protocol:raise ValueError('Resume protocol changed')
    else:write(args.output/'protocol.json',protocol)
    set_seed(protocol['seed']);tokenizer=AutoTokenizer.from_pretrained(args.model);tokenizer.pad_token=tokenizer.eos_token;tokenizer.padding_side='left'
    base=AutoModelForCausalLM.from_pretrained(args.model,torch_dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda')
    model=get_peft_model(base,LoraConfig(r=8,lora_alpha=16,lora_dropout=0.,target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM'))
    model.enable_input_require_grads();model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    parameters=[p for p in model.parameters() if p.requires_grad];optimizer=torch.optim.AdamW(parameters,lr=2e-5,weight_decay=0.)
    initial={n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad};step=0
    if args.resume:
        state=torch.load(args.resume/'optimizer.pt',map_location='cpu',weights_only=True);step=state['step']
        set_peft_model_state_dict(model,load_file(str(args.resume/'adapter_model.safetensors')));optimizer.load_state_dict(state['optimizer']);torch.set_rng_state(state['rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
        initial=torch.load(args.output/'initial-adapter.pt',weights_only=True)
        before=[json.loads(l) for l in (args.output/'before/episodes.jsonl').read_text().splitlines()]
        if len(before)!=20:raise ValueError('Missing baseline')
    else:
        torch.save(initial,args.output/'initial-adapter.pt');before=evaluate(model,tokenizer,heldout,args.output/'before','before')
    def delta():
        changed=0;norm=0.
        for n,p in model.named_parameters():
            if n in initial:
                d=p.detach().float().cpu()-initial[n].float();changed+=int(torch.count_nonzero(d));norm+=float((d*d).sum())
        return {'changed_parameters':changed,'delta_l2':norm**.5}
    def save(name):
        folder=args.output/name;model.save_pretrained(folder);tokenizer.save_pretrained(folder)
        torch.save({'step':step,'optimizer':optimizer.state_dict(),'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()},folder/'optimizer.pt')
    try:
        for step in range(step+1,args.steps+1):
            begin=time.time();row=training[(step-1)%len(training)];records=sample(model,tokenizer,[row]*4,'train',True);returns=[r['result']['reward'] for r in records];adv=advantages(returns)
            optimizer.zero_grad(set_to_none=True);model.train();loss_value=0.
            for record,a in zip(records,adv):
                prompt=record['input_ids'];answer=record['output_ids'];seq=torch.tensor([prompt+answer],device='cuda');targets=seq[:,len(prompt):];start=len(prompt)-1;end=seq.shape[1]-1
                if not answer:continue
                with torch.no_grad(),model.disable_adapter():
                    logits=model(seq,use_cache=False).logits[:,start:end,:].float();ref=logits.log_softmax(-1).gather(-1,targets.unsqueeze(-1)).squeeze(-1);del logits
                logits=model(seq,use_cache=False).logits[:,start:end,:].float();logp=logits.log_softmax(-1).gather(-1,targets.unsqueeze(-1)).squeeze(-1);lr=ref-logp
                loss=(-a*logp+.01*(torch.exp(lr)-lr-1)).mean()/len(records)
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
                loss_value+=float(loss.detach());loss.backward();del logits,logp,ref,loss,seq
            norm=float(torch.nn.utils.clip_grad_norm_(parameters,1.));nonzero=sum(int(torch.count_nonzero(p.grad)) for p in parameters if p.grad is not None)
            if not torch.isfinite(torch.tensor(norm)):raise RuntimeError('Nonfinite gradient')
            optimizer.step()
            for record in records:
                record.pop('input_ids');record.pop('output_ids');append(args.output/'rollouts.jsonl',{'step':step,**record})
            event={'step':step,'task':row['id'],'rewards':returns,'seconds':time.time()-begin,'loss':loss_value,'gradient_norm':norm,'nonzero_gradient_elements':nonzero,**delta()};append(args.output/'optimizer-events.jsonl',event);print(json.dumps(event),flush=True)
            save(f'checkpoint-{step}')
        save('adapter');write(args.output/'training-completion.json',{'steps':step,**delta()})
        after=evaluate(model,tokenizer,heldout,args.output/'after','after')
        b=sum(r['result']['success'] for r in before);a=sum(r['result']['success'] for r in after)
        write(args.output/'completion.json',{'before_success':b,'after_success':a,'heldout':20,'percentage_point_gain':(a-b)*5,'before_mean_reward':sum(r['result']['reward'] for r in before)/20,'after_mean_reward':sum(r['result']['reward'] for r in after)/20,'pairs':[{'id':x['id'],'before':x['result'],'after':y['result']} for x,y in zip(before,after,strict=True)],**delta()})
    except BaseException as exc:
        write(args.output/'interruption.json',{'step':step,'error':str(exc)});raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--steps',type=int,default=80);p.add_argument('--resume',type=Path);run(p.parse_args())
