"""Paired execution-feedback trajectories, followed by bounded LoRA RL updates."""
import argparse,hashlib,json,os,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import httpx
from .feedback import digest
from ..engineering.pilot import write
from ..engineering.train_interactive import append,advantages


def remote(row,text,phase,endpoint):
    token=Path(os.environ['AKSHARA_CODING_TOKEN_FILE']).read_text().strip()
    for retry in range(3):
        try:
            r=httpx.post(os.environ['AKSHARA_CODING_REWARD_URL']+endpoint,headers={'Authorization':'Bearer '+token},json={'id':row['id'],'completion':text,'phase':phase},timeout=120)
            r.raise_for_status();value=r.json()
            if value['manifest_sha256']!=os.environ['AKSHARA_FEEDBACK_MANIFEST']:raise ValueError('Dataset identity mismatch')
            return value
        except (httpx.TransportError,httpx.HTTPStatusError) as e:
            if isinstance(e,httpx.HTTPStatusError) and e.response.status_code<500:raise
            if retry==2:raise
            time.sleep(2)


def trajectories(model,tokenizer,rows,phase,stochastic):
    import torch
    episodes=[{'id':r['id'],'messages':list(r['prompt']),'turns':[],'done':False} for r in rows]
    model.eval();model.config.use_cache=True
    for turn in range(3):
        active=[i for i,e in enumerate(episodes) if not e['done']]
        if not active:break
        text=[tokenizer.apply_chat_template(episodes[i]['messages'],tokenize=False,add_generation_prompt=True) for i in active]
        inputs=tokenizer(text,return_tensors='pt',padding=True).to('cuda');width=inputs['input_ids'].shape[1]
        if width>12000:raise ValueError('Context bound exceeded; no truncation permitted')
        with torch.inference_mode():
            outputs=model.generate(**inputs,max_new_tokens=1536,do_sample=stochastic,temperature=.8,top_p=.95,top_k=0,pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id)
        current=[]
        for k,i in enumerate(active):
            ids=outputs[k,width:].tolist()
            if tokenizer.eos_token_id in ids:ids=ids[:ids.index(tokenizer.eos_token_id)+1]
            record={'turn':turn+1,'completion':tokenizer.decode(ids,skip_special_tokens=True),'input_ids':inputs['input_ids'][k][inputs['attention_mask'][k].bool()].tolist(),'output_ids':ids,'generated_tokens':len(ids)}
            episodes[i]['turns'].append(record);current.append(record)
        with ThreadPoolExecutor(max_workers=2) as pool:
            feedback=list(pool.map(lambda pair:remote(rows[pair[0]],pair[1]['completion'],phase,'/feedback'),zip(active,current)))
        for i,record,result in zip(active,current,feedback):
            episodes[i]['messages'].extend([{'role':'assistant','content':record['completion']},{'role':'user','content':json.dumps(result['feedback'],separators=(',',':'))}])
            record['development']=result;episodes[i]['done']=result['development_success']
    with ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(lambda pair:remote(pair[0],pair[1]['turns'][-1]['completion'],phase,'/score'),zip(rows,episodes)))
    for e,r in zip(episodes,results):e['result']=r
    if phase!='train':
        with ThreadPoolExecutor(max_workers=2) as pool:
            first=list(pool.map(lambda pair:remote(pair[0],pair[1]['turns'][0]['completion'],phase,'/score') if len(pair[1]['turns'])>1 else pair[1]['result'],zip(rows,episodes)))
        for e,r in zip(episodes,first):e['first_result']=r
    model.config.use_cache=False
    return episodes


def public_record(e):
    return {**e,'turns':[{k:v for k,v in t.items() if k not in ['input_ids','output_ids']} for t in e['turns']]}


def evaluation(model,tokenizer,rows,dest,phase):
    dest.mkdir(parents=True,exist_ok=True);records=[]
    for offset in range(0,len(rows),2):
        episodes=trajectories(model,tokenizer,rows[offset:offset+2],phase,False)
        for e in episodes:
            rec=public_record(e);append(dest/'episodes.jsonl',rec);records.append(rec)
        summary={'completed':len(records),'expected':len(rows),'complete':len(records)==len(rows),'first_attempt_successes':sum(e['first_result']['success'] for e in records),'repair_successes':sum(e['result']['success'] for e in records),'first_mean_test_score':sum(e['first_result']['test_score'] for e in records)/len(records),'repair_mean_test_score':sum(e['result']['test_score'] for e in records)/len(records),'repair_mean_field_score':sum(e['result']['field_score'] for e in records)/len(records)}
        write(dest/'summary.json',summary);print(json.dumps({'event':phase,**summary}),flush=True)
    return summary


def run(a):
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM,set_seed
    from peft import LoraConfig,get_peft_model
    if a.output.exists():raise ValueError('Fresh run directory required')
    a.output.mkdir(parents=True)
    manifest=json.loads((a.dataset/'manifest.json').read_text())
    for s,d in manifest['sha256'].items():
        if digest(a.dataset/f'{s}.json')!=d:raise ValueError('Frozen dataset changed')
    os.environ['AKSHARA_FEEDBACK_MANIFEST']=digest(a.dataset/'manifest.json')
    training=json.loads((a.dataset/'train.json').read_text());heldout=json.loads((a.dataset/'heldout.json').read_text())
    # Three-output configurations first, then four-output configurations. Training-only ordering.
    training=sorted(training,key=lambda r:(len(r['configuration']['fields']),r['id']))
    protocol={'model':str(a.model),'dataset':manifest,'max_updates':a.steps,'group_size':2,'turns':3,'max_tokens':1536,'seed':20260930,'precision':'bfloat16','attention':'sdpa','learning_rate':1e-5,'kl':.01,'algorithm':'Paired-trajectory REINFORCE, leave-one-out final private training reward; generated assistant tokens only','reward':'0.5 private-test fraction + 0.5 private-field fraction; development examples are feedback only','selection':'Final completed update before fixed training deadline; never based on held-out scores','training_deadline':a.training_deadline,'hard_deadline':a.deadline,'evaluation':'20 fresh configurations; greedy first attempt and up-to-three-attempt repair; stop only on development success; identical before/after budgets','feedback':'Only development examples, expected outputs, actual outputs and errors enter model context. Private scoring happens after trajectory ends.','training_order':[r['id'] for r in training],'source_sha256':{n:digest(Path(__file__).with_name(n)) for n in ['train_feedback.py','feedback.py','service.py','worker.py','tasks.py']}}
    write(a.output/'protocol.json',protocol);set_seed(20260930)
    tokenizer=AutoTokenizer.from_pretrained(a.model);tokenizer.pad_token=tokenizer.eos_token;tokenizer.padding_side='left'
    base=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda')
    model=get_peft_model(base,LoraConfig(r=8,lora_alpha=16,lora_dropout=0.,target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM'))
    model.enable_input_require_grads();model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    initial={n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad};torch.save(initial,a.output/'initial-adapter.pt')
    params=[p for p in model.parameters() if p.requires_grad];optimizer=torch.optim.AdamW(params,lr=1e-5,weight_decay=0.)
    write(a.output/'runtime.json',{'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'device_memory':torch.cuda.get_device_properties(0).total_memory})
    step=0;contrast=0
    def delta():
        changed=0;norm=0.
        for n,p in model.named_parameters():
            if n in initial:
                d=p.detach().float().cpu()-initial[n].float();changed+=int(torch.count_nonzero(d));norm+=float((d*d).sum())
        return {'changed_parameters':changed,'delta_l2':norm**.5}
    def save(name):
        out=a.output/name;model.save_pretrained(out);tokenizer.save_pretrained(out)
        torch.save({'step':step,'optimizer':optimizer.state_dict(),'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()},out/'optimizer.pt')
    def logp(seq,length):
        # Compute only generated-token logits, avoiding a full-context vocabulary tensor.
        logits=model(seq,use_cache=False,logits_to_keep=length+1).logits[:,:-1,:].float()
        return logits.log_softmax(-1).gather(-1,seq[:,-length:].unsqueeze(-1)).squeeze(-1)
    try:
        before=evaluation(model,tokenizer,heldout,a.output/'before','before')
        for proposed in range(1,a.steps+1):
            # Leave the fixed final-evaluation reserve intact; do not select by score.
            if time.time()>=a.training_deadline:break
            begin=time.time();row=training[(proposed-1)%len(training)]
            episodes=trajectories(model,tokenizer,[row,row],'train',True)
            returns=[e['result']['reward'] for e in episodes];adv=advantages(returns)
            contrast+=len(set(returns))>1
            optimizer.zero_grad(set_to_none=True);model.train();loss_value=0.
            for e,advantage in zip(episodes,adv):
                total=sum(len(t['output_ids']) for t in e['turns'])
                for t in e['turns']:
                    answer=t['output_ids']
                    if not answer:continue
                    seq=torch.tensor([t['input_ids']+answer],device='cuda')
                    with torch.no_grad(),model.disable_adapter():ref=logp(seq,len(answer))
                    actual=logp(seq,len(answer));ratio=ref-actual
                    loss=(-advantage*actual+.01*(torch.exp(ratio)-ratio-1)).sum()/(2*total)
                    if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
                    loss_value+=float(loss.detach());loss.backward();del seq,ref,actual,loss,ratio
            norm=float(torch.nn.utils.clip_grad_norm_(params,1.));nonzero=sum(int(torch.count_nonzero(p.grad)) for p in params if p.grad is not None)
            if not torch.isfinite(torch.tensor(norm)):raise RuntimeError('Nonfinite gradient')
            optimizer.step();step=proposed
            for e in episodes:append(a.output/'training-traces.jsonl',{'step':step,**public_record(e)})
            event={'step':step,'task':row['id'],'rewards':returns,'contrast_batches':contrast,'loss':loss_value,'gradient_norm':norm,'nonzero_gradient_elements':nonzero,'seconds':time.time()-begin,**delta()}
            append(a.output/'optimizer-events.jsonl',event);save('checkpoint-'+str(step));print(json.dumps({'event':'optimizer',**event}),flush=True)
        save('adapter');write(a.output/'training-completion.json',{'updates':step,'contrast_batches':contrast,**delta()})
        after=evaluation(model,tokenizer,heldout,a.output/'after','after')
        write(a.output/'completion.json',{'complete':True,'before':before,'after':after,'updates':step,'contrast_batches':contrast,'repair_percentage_point_gain':5*(after['repair_successes']-before['repair_successes']),**delta()})
    except BaseException as e:
        write(a.output/'interruption.json',{'completed_updates':step,'error':repr(e),'time':time.time()});raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--steps',type=int,default=16);p.add_argument('--training-deadline',type=float,required=True);p.add_argument('--deadline',type=float,required=True);run(p.parse_args())
