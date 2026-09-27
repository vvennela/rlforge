"""GRPO LoRA training on frozen LEGO repairs with Vultr sandbox rewards."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import httpx
from .pilot import write


def request_reward(row_id,completion,phase):
    token=Path(os.environ['AKSHARA_ENGINEERING_TOKEN_FILE']).read_text().strip()
    with httpx.Client(timeout=180) as client:
        r=client.post(os.environ['AKSHARA_ENGINEERING_REWARD_URL']+'/grade',
                      headers={'Authorization':'Bearer '+token},json={'id':row_id,'completion':completion,'phase':phase})
        r.raise_for_status(); return r.json()


def evaluate_model(model,tokenizer,rows,output,phase,max_tokens):
    import torch
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    model.eval(); model.config.use_cache=True
    results=[]
    for row in rows:
        text=tokenizer.apply_chat_template(row['prompt'],tokenize=False,add_generation_prompt=True)
        inputs=tokenizer(text,return_tensors='pt').to(model.device)
        with torch.inference_mode():
            tokens=model.generate(**inputs,max_new_tokens=max_tokens,do_sample=False,pad_token_id=tokenizer.pad_token_id)
        answer=tokenizer.decode(tokens[0,inputs['input_ids'].shape[1]:],skip_special_tokens=True)
        result=request_reward(row['id'],answer,phase)
        record={'id':row['id'],'phase':phase,'completion':answer,'result':result,
                'generated_tokens':tokens.shape[1]-inputs['input_ids'].shape[1]}
        results.append(record)
        with (output/'episodes.jsonl').open('a') as f: f.write(json.dumps(record)+'\n')
        write(output/'summary.json',{'completed':len(results),'expected':len(rows),'complete':len(results)==len(rows),
               'successes':sum(r['result']['success'] for r in results),
               'mean_reward':sum(r['result']['reward'] for r in results)/len(results),
               'mean_gap_iou':sum(r['result']['gap_iou'] for r in results)/len(results)})
    model.config.use_cache=False
    return results


def run(args):
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, set_seed
    from trl import GRPOConfig, GRPOTrainer
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    if args.output.exists(): raise ValueError('Use a fresh run directory')
    args.output.mkdir(parents=True)
    manifest=json.loads((args.dataset/'manifest.json').read_text())
    for split,digest in manifest['sha256'].items():
        if hashlib.sha256((args.dataset/f'{split}.json').read_bytes()).hexdigest()!=digest: raise ValueError('Dataset hash mismatch')
    training=json.loads((args.dataset/'train.json').read_text())
    heldout=json.loads((args.dataset/'heldout.json').read_text())
    if {r['id'] for r in training}&{r['id'] for r in heldout}: raise ValueError('Split overlap')
    set_seed(20260927)
    write(args.output/'protocol.json',{'model':str(args.model),'manifest':manifest,'optimizer_steps':args.steps,
         'learning_rate':2e-5,'num_generations':4,'temperature':0.8,'max_completion_length':768,
         'lora_rank':8,'lora_alpha':16,'lora_modules':['q_proj','v_proj'],'precision':'bfloat16',
         'seed':20260927,'evaluation':'20 frozen held-out prompts, greedy decoding before and after, 768-token budget',
         'selection':'Final fixed-step checkpoint only; held-out scores do not select checkpoints',
         'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         'reward_execution':'Vultr CPU gVisor containers; loopback-only authenticated reward service over SSH tunnels'})
    tokenizer=AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token=tokenizer.eos_token; tokenizer.padding_side='left'
    model=AutoModelForCausalLM.from_pretrained(args.model,torch_dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda')
    before=evaluate_model(model,tokenizer,heldout,args.output/'before','before',768)
    lookup={r['id']:r for r in training}
    def reward(completions,problem_id,**kwargs):
        def one(item):
            completion,identifier=item
            if identifier not in lookup: raise ValueError('Training reward requested for nontraining task')
            text=completion[0]['content'] if isinstance(completion,list) else completion
            result=request_reward(identifier,text,'train')
            return {'id':identifier,'completion':text,'result':result,'global_step':trainer.state.global_step}
        with ThreadPoolExecutor(max_workers=2) as pool:
            records=list(pool.map(one,zip(completions,problem_id,strict=True)))
        with (args.output/'rollouts.jsonl').open('a') as f:
            for record in records: f.write(json.dumps(record)+'\n')
        return [r['result']['reward'] for r in records]
    class Evidence(TrainerCallback):
        def append(self,data):
            with (args.output/'optimizer-events.jsonl').open('a') as f: f.write(json.dumps({'time':time.time(),**data})+'\n')
        def on_pre_optimizer_step(self,args_,state,control,model=None,**kwargs):
            # Trainer has already clipped gradients here; nonzero still proves backward flow.
            grads=[p.grad.detach().float() for p in model.parameters() if p.requires_grad and p.grad is not None]
            norm=sum(float((g*g).sum()) for g in grads)**0.5
            nonzero=sum(int(torch.count_nonzero(g)) for g in grads)
            self.append({'event':'gradient','step':state.global_step+1,'gradient_l2_after_clipping':norm,'nonzero_gradient_elements':nonzero,'finite':all(bool(torch.isfinite(g).all()) for g in grads)})
        def on_step_end(self,args_,state,control,model=None,**kwargs):
            if state.global_step==1 or state.global_step%5==0:
                changed,l2,maximum=delta(model)
                self.append({'event':'weight_delta','step':state.global_step,'changed_elements':changed,'delta_l2':l2,'delta_max_abs':maximum})
        def on_log(self,args_,state,control,logs=None,**kwargs):
            self.append({'event':'metrics','step':state.global_step,'metrics':logs})
    cfg=GRPOConfig(output_dir=str(args.output/'checkpoints'),learning_rate=2e-5,max_steps=args.steps,
          per_device_train_batch_size=1,gradient_accumulation_steps=4,num_generations=4,max_completion_length=768,
          temperature=0.8,bf16=True,gradient_checkpointing=True,seed=20260927,report_to='none',logging_steps=1,
          save_steps=10,save_total_limit=2,remove_unused_columns=False,beta=0.02)
    trainer=GRPOTrainer(model=model,processing_class=tokenizer,args=cfg,
          train_dataset=Dataset.from_list([{'prompt':r['prompt'],'problem_id':r['id']} for r in training]),
          reward_funcs=reward,callbacks=[Evidence()],
          peft_config=LoraConfig(r=8,lora_alpha=16,lora_dropout=0.,target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM'))
    initial={n:p.detach().cpu().clone() for n,p in trainer.model.named_parameters() if p.requires_grad}
    def delta(model):
        changed=0; squares=0.; maximum=0.
        for name,p in model.named_parameters():
            if name not in initial: continue
            d=p.detach().cpu().float()-initial[name].float()
            changed+=int(torch.count_nonzero(d)); squares+=float((d*d).sum()); maximum=max(maximum,float(d.abs().max()))
        return changed,squares**0.5,maximum
    torch.save(initial,args.output/'initial-adapter-tensors.pt')
    write(args.output/'trainable-parameters.json',{'parameters':sum(t.numel() for t in initial.values()),'tensors':len(initial)})
    started=time.time()
    try:
        trainer.train()
        trainer.save_model(str(args.output/'adapter'))
        changed,l2,maximum=delta(trainer.model)
        write(args.output/'training-completion.json',{'optimizer_steps':trainer.state.global_step,'changed_parameter_elements':changed,
              'adapter_delta_l2':l2,'adapter_delta_max_abs':maximum,'seconds':time.time()-started,'weight_updates_performed':changed>0})
        after=evaluate_model(trainer.model,tokenizer,heldout,args.output/'after','after',768)
        before_correct=sum(r['result']['success'] for r in before); after_correct=sum(r['result']['success'] for r in after)
        write(args.output/'comparison.json',{'heldout':len(heldout),'before_success':before_correct,'after_success':after_correct,
             'success_rate_delta':(after_correct-before_correct)/len(heldout),'weight_updates_performed':changed>0,
             'pairs':[{'id':b['id'],'before':b['result'],'after':a['result']} for b,a in zip(before,after,strict=True)],
             'scope':manifest['scope'],'claim':'Single-seed narrow repair pilot; not evidence of general engineering uplift.'})
    except Exception as exc:
        trainer.save_model(str(args.output/'interrupted-adapter'))
        write(args.output/'interruption.json',{'error':str(exc),'optimizer_steps':trainer.state.global_step})
        raise


def main():
    p=argparse.ArgumentParser(); p.add_argument('--dataset',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True); p.add_argument('--steps',type=int,default=40)
    run(p.parse_args())


if __name__=='__main__': main()
