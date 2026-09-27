const $=id=>document.getElementById(id);
let page='upload',active='math',evidence={},selectedFile=null,bridgeState=null,renderer=null,afterRenderer=null,initialRenderer=null,pointer=null,bridgeSelection=null,programSelection=null;
const camera={yaw:-.65,pitch:.55,zoom:1,panX:0,panY:0};
const cases={math:{kicker:'01 / MATHEMATICS',title:'Learn the method.\nSolve the next problem.',copy:'From augmented Lagrangians to checkable exercises. Train on paper-derived tasks, then evaluate on twenty held-out problems.'},coding:{kicker:'02 / CODING',title:'From an algorithm\nto an executable challenge.',copy:'Qwen writes Python implementations of iterative-deepening search. Each program runs against hidden tests in a Vultr sandbox, checking paths, costs, thresholds, traversal order, and edge cases.'},engineering:{kicker:'03 / ENGINEERING',title:'Build. Inspect.\nMake the next move better.',copy:'An agent places, moves, and removes bricks. Every turn returns measured geometry, component coverage, and connection feedback.'}};
function tex(){if(window.renderMathInElement)renderMathInElement(document.body,{delimiters:[{left:'\\[',right:'\\]',display:true},{left:'\\(',right:'\\)',display:false}],throwOnError:false,trust:false});}
function renderCase(){
 const c=cases[active],s=evidence[active]||{};
 $('case-kicker').textContent=c.kicker;$('case-title').innerText=c.title;$('case-copy').textContent=c.copy;$('case-status').textContent=s.status||'Dataset ready';
 $('case-panel').setAttribute('aria-labelledby','tab-'+active);
 for(const key of Object.keys(cases))$(key+'-specimen').hidden=key!==active;
 for(const key of ['before','after']){
  const r=s[key],has=r&&r.completed>0,pct=has?100*r.correct/r.completed:null;
  $(key+'-score').textContent=has?`${r.correct}/${r.completed}`:'—';
  $(key+'-detail').textContent=has?(r.complete?'20 held-out problems':`${r.completed} of 20 evaluated · partial`):(key==='after'?'Evaluation follows training':'Evaluation scheduled');
  $(key+'-bar').style.width=has?pct+'%':'0%';$(key+'-percent').textContent=has?Number(pct.toFixed(1))+'%':'—';
 }
 const ready=s.before?.complete&&s.after?.complete;
 $('delta-score').textContent=ready?`${s.after.correct>=s.before.correct?'+':''}${(s.after.correct-s.before.correct)*5}pp`:'—';
 $('delta-detail').textContent=ready?'held-out accuracy':'Awaiting paired evaluation';
 $('protocol').textContent=s.protocol||'20 held-out problems · fixed before/after evaluation';
 $('run-detail').textContent=active==='engineering'?`${s.steps||0} / ${s.target_steps||50} training batches · ${s.reward_updates||0} batches with reward contrast`:active==='math'?'80 training updates · Qwen 2.5 7B · same frozen completion parser':`${s.steps||0} / ${s.target_steps||80} training batches · four sampled programs per batch · Qwen 2.5 7B`;
 const metric=active==='engineering'?'mean_gap_iou':'mean_reward';
 const metricName=active==='engineering'?'target geometry overlap':'test pass rate';
 if(active!=='math' && s.before?.[metric]!=null){$('run-detail').textContent+=` · Mean ${metricName}: ${(s.before[metric]*100).toFixed(1)}% → ${s.after?.[metric]!=null?(s.after[metric]*100).toFixed(1)+'%'+(s.after.complete?'':` (${s.after.completed}/20, partial)`):'awaiting evaluation'}`}
 if(active==='coding'&&s.before?.mean_field_score!=null){$('run-detail').textContent+=` · Exact field accuracy: ${(100*s.before.mean_field_score).toFixed(1)}% → ${s.after?.mean_field_score!=null?(100*s.after.mean_field_score).toFixed(1)+'%'+(s.after.complete?'':` (${s.after.completed}/20, partial)`):'awaiting evaluation'}`}
 if(s.weight_update?.changed_parameters){$('run-detail').textContent+=` · ${s.weight_update.changed_parameters.toLocaleString()} adapter parameters changed`}
 const programs=active==='coding'?(s.samples||(s.sample?[s.sample]:[])):[];
 const codeSample=programs.find(p=>p.id===programSelection)||programs[0];
 $('program-case-control').hidden=programs.length<2;
 if(active==='coding'){
  const selector=$('program-case');
  const options=programs.map((p,i)=>({id:p.id,label:`Case ${i+1} · ${p.id} · ${p.before.result.passed}/${p.before.result.total} → ${p.after?p.after.result.passed+'/'+p.after.result.total:'pending'}`}));
  const signature=JSON.stringify(options);
  if(selector.dataset.signature!==signature){selector.replaceChildren(...options.map(p=>new Option(p.label,p.id)));selector.dataset.signature=signature}
  if(codeSample){programSelection=codeSample.id;selector.value=codeSample.id}
 }
 $('program-comparison').hidden=active!=='coding'||!codeSample;
 if(active==='coding'&&codeSample){
  $('program-label').textContent='HELD-OUT PROGRAM / '+codeSample.id;
  for(const phase of ['before','after']){
   const attempt=codeSample[phase];$('program-'+phase).textContent=attempt?.completion||'Final evaluation follows training.';
   $('program-'+phase+'-score').textContent=attempt?`${attempt.result.passed}/${attempt.result.total} tests`:'';
   const fields=attempt?.result.field_checks||[];
   $('program-'+phase+'-fields').textContent=fields.length?'Exact fields / '+Object.keys(fields[0]).map(key=>`${key}: ${fields.filter(f=>f[key]===true).length}/${fields.length}`).join(' · '):'';
  }
 }
 if(active==='engineering'){requestAnimationFrame(drawBridge)}
}
async function refresh(){try{const r=await fetch('/api/studio');if(!r.ok)throw Error('Status unavailable');const d=await r.json();evidence=d.evidence;const g=d.generation;document.querySelector('.provider-settings').hidden=g.can_configure===false;$('provider-label').textContent=g.provider==='vultr'?'Vultr Serverless Inference':'Astra · server-side inference';$('provider-status').textContent=g.configured?g.model:'Connect inference key';renderCase()}catch(e){$('provider-status').textContent=e.message}}
function showPage(){
 const requested=location.hash.slice(1);page=requested==='cases'?'math':(['upload',...Object.keys(cases)].includes(requested)?requested:'upload');
 const upload=page==='upload';$('upload-page').hidden=!upload;$('cases').hidden=upload;document.querySelector('[data-upload-only]').hidden=!upload;
 document.querySelectorAll('[data-page]').forEach(b=>{const on=b.dataset.page===page;b.setAttribute('aria-selected',String(on));b.tabIndex=on?0:-1});
 if(!upload){active=page;renderCase()}
 document.title='RLForge — '+(upload?'Environment studio':'Case Study: '+page[0].toUpperCase()+page.slice(1));
 window.scrollTo({top:0,behavior:'instant'});
}
for(const b of document.querySelectorAll('[data-page]'))b.addEventListener('click',()=>{location.hash=b.dataset.page});
document.querySelector('.tabs').addEventListener('keydown',e=>{if(!['ArrowLeft','ArrowRight','Home','End'].includes(e.key))return;e.preventDefault();const keys=['upload',...Object.keys(cases)];let n=keys.indexOf(page);n=e.key==='Home'?0:e.key==='End'?3:(n+(e.key==='ArrowRight'?1:3))%4;location.hash=keys[n];$('tab-'+keys[n]).focus()});
window.addEventListener('hashchange',showPage);showPage();
function choose(file){if(!file)return;selectedFile=file;$('file-label').textContent=`${file.name} · ${(file.size/1024/1024).toFixed(2)} MB`}
$('paper').addEventListener('change',()=>choose($('paper').files[0]));
for(const name of ['dragenter','dragover'])$('dropzone').addEventListener(name,e=>{e.preventDefault();$('dropzone').classList.add('dragover')});
for(const name of ['dragleave','drop'])$('dropzone').addEventListener(name,e=>{e.preventDefault();$('dropzone').classList.remove('dragover')});
$('dropzone').addEventListener('drop',e=>{choose(e.dataTransfer.files[0]);$('paper').required=false});
function jobView(s){$('job').hidden=false;const titles={extracting:'Reading your document',ocr:'Reading scanned pages',artifacts:'Extracting source artifacts',generating:'Generating the environment',reviewing:'Testing adversarial answers',ready:'Your environment is ready',failed:'Generation stopped',awaiting_connection:'Document ready'};$('job-title').textContent=titles[s.status]||s.status;$('job-count').textContent=['ocr','artifacts'].includes(s.status)?`${s.ocr_completed||0} / ${s.ocr_total||0} pages`:`${s.completed||0} / ${s.count||100} problems`;$('job-progress').style.width=s.status==='ready'?'100%':(s.status==='ocr'?Math.max(3,25*(s.ocr_completed||0)/(s.ocr_total||1)):s.status==='artifacts'?25:Math.max(3,(s.ocr?25:0)+(s.ocr?75:100)*(s.completed||0)/(s.count||100)))+'%';$('job-message').textContent=s.message||(s.status==='ready'?'Source-linked problems, training splits, and an executable reward checker. Review the reference answers before training.':'Extracting source evidence and building checkable tasks.');$('download').hidden=!s.download;if(s.download){$('download').href=s.download;$('download').download='environment.zip'}if(s.ocr){$('ocr-result').hidden=false;$('ocr-pages').textContent=s.ocr.pages+' pages';$('ocr-strips').textContent=s.ocr.strips+' strips';$('ocr-artifacts').textContent=s.ocr.artifacts+' artifacts';$('ocr-method').textContent=Object.entries(s.ocr.methods).map(([k,v])=>(k==='tesseract'?'OCR':'Native text')+': '+v+' pages').join(' · ')+' · 200 DPI · source hashes recorded';$('ocr-download').href=s.ocr.download;if($('ocr-page').getAttribute('src')!==s.ocr.preview)$('ocr-page').src=s.ocr.preview}if(s.preview){$('job-preview').replaceChildren();for(const p of s.preview){const el=document.createElement('p');el.textContent=p.prompt;$('job-preview').append(el)}tex()}}
function rememberJob(id){
 const url=new URL(location.href);url.searchParams.set('environment',id);history.replaceState(null,'',url);
}
async function followJob(s){
 jobView(s);
 while(['extracting','ocr','artifacts','generating','reviewing'].includes(s.status)){
  await new Promise(r=>setTimeout(r,1500));
  const poll=await fetch('/api/generation/'+s.id);
  if(!poll.ok)throw Error('Connection interrupted. Reload this page to resume the environment.');
  s=await poll.json();jobView(s);
 }
}
async function resumeJob(){
 const id=new URLSearchParams(location.search).get('environment');
 if(!id||!/^[a-f0-9]{32}$/.test(id))return;
 $('generate').disabled=true;
 try{
  const response=await fetch('/api/generation/'+id);
  if(!response.ok)throw Error('This environment could not be loaded.');
  const s=await response.json();$('file-label').textContent=s.filename;$('count').value=String(s.count);
  await followJob(s);
 }catch(e){jobView({status:'failed',message:e.message})}
 finally{$('generate').disabled=false}
}
$('upload-form').addEventListener('submit',async e=>{e.preventDefault();if(!selectedFile)return;if(selectedFile.size>8*1024*1024){jobView({status:'failed',message:'Upload a file up to 8 MB.'});return} $('generate').disabled=true;$('download').hidden=true;$('ocr-result').hidden=true;$('job-preview').replaceChildren();try{const data=await new Promise((resolve,reject)=>{const r=new FileReader();r.onload=()=>resolve(r.result.split(',')[1]);r.onerror=reject;r.readAsDataURL(selectedFile)});const res=await fetch('/api/generate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({filename:selectedFile.name,data,count:Number($('count').value)})});let s=await res.json();if(!res.ok)throw Error(s.error);rememberJob(s.id);await followJob(s)}catch(e){jobView({status:'failed',message:e.message})}finally{$('generate').disabled=false}});
function drawBridge(){
 if(!renderer||!bridgeState||page!=='engineering')return;
 const samples=evidence.engineering?.samples||[evidence.engineering?.sample].filter(Boolean);
 const sample=samples.find(s=>s.id===bridgeSelection)||samples[0];
 const selector=$('bridge-case');
 if(document.activeElement!==selector){
  selector.replaceChildren(...samples.map((s,i)=>new Option(`Case ${i+1} · ${s.id.split('-').pop()} · ${s.after?(s.before.success?'pass':'fail')+' → '+(s.after.success?'pass':'fail'):'evaluation pending'}`,s.id)));
  if(sample)selector.value=sample.id;
 }
 $('bridge-case-control').hidden=samples.length<2;
 if(sample){
  const scene=phase=>({task:sample.task,palette:sample.palette,bricks:[...sample.fixed,...(sample[phase]?.editable_bricks||[])]});
  $('bridge-sample-label').textContent='HELD-OUT REPAIR / '+sample.id;
  $('bridge-initial-panel').hidden=false;
  initialRenderer?.render({task:sample.task,palette:sample.palette,bricks:[...sample.fixed,...(sample.initial_editable||[])]},camera);
  $('bridge-before-label').textContent=`Base Qwen · ${(100*sample.before.gap_iou).toFixed(1)}% target overlap`;
  $('bridge-after-label').textContent=sample.after?`Trained Qwen · ${(100*sample.after.gap_iou).toFixed(1)}% target overlap`:'After training';
  $('bridge-after-panel').hidden=false;$('paired-bridge').classList.add('has-pair');
  $('bridge-after-pending').hidden=!!sample.after;$('bridge-after').hidden=!sample.after;
  $('bridge-target').textContent=`Target post: (${sample.target.x}, ${sample.target.y}), height ${sample.target.top-sample.target.bottom} plates`;
  for(const phase of ['before','after']){
   const turns=sample.traces?.[phase];
   $('bridge-'+phase+'-trace').textContent=turns?turns.map(t=>`Turn ${t.turn} · reward ${t.reward.toFixed(4)}\n${JSON.stringify(t.actions,null,2)}${t.error?'\nError: '+t.error:''}`).join('\n\n'):'Evaluation pending.';
  }
  renderer.render(scene('before'),camera);if(sample.after&&afterRenderer)afterRenderer.render(scene('after'),camera);
 }else renderer.render(bridgeState,camera);
}
async function initBridge(){try{const r=await fetch('/api/bridge-sample');bridgeState=await r.json();$('bridge').dataset.theme='dark';renderer=new BrickRenderer($('bridge'));$('bridge-after').dataset.theme='dark';afterRenderer=new BrickRenderer($('bridge-after'));$('bridge-initial').dataset.theme='dark';initialRenderer=new BrickRenderer($('bridge-initial'));drawBridge()}catch(e){$('bridge').setAttribute('aria-label',e.message)}}
$('bridge').addEventListener('pointerdown',e=>{pointer={id:e.pointerId,x:e.clientX,y:e.clientY};$('bridge').setPointerCapture(e.pointerId)});
$('bridge').addEventListener('pointermove',e=>{if(!pointer)return;camera.yaw+=(e.clientX-pointer.x)*.009;camera.pitch+=(e.clientY-pointer.y)*.009;pointer.x=e.clientX;pointer.y=e.clientY;drawBridge()});
for(const n of ['pointerup','pointercancel','lostpointercapture'])$('bridge').addEventListener(n,()=>pointer=null);
$('bridge').addEventListener('wheel',e=>{e.preventDefault();camera.zoom=Math.max(.4,Math.min(3,camera.zoom*Math.exp(-e.deltaY*.001)));drawBridge()},{passive:false});
$('bridge').addEventListener('keydown',e=>{if(!['ArrowLeft','ArrowRight','ArrowUp','ArrowDown'].includes(e.key))return;e.preventDefault();camera.yaw+=e.key==='ArrowLeft'?-.15:e.key==='ArrowRight'?.15:0;camera.pitch+=e.key==='ArrowUp'?-.15:e.key==='ArrowDown'?.15:0;drawBridge()});
$('bridge-after').addEventListener('pointerdown',e=>{pointer={id:e.pointerId,x:e.clientX,y:e.clientY};$('bridge-after').setPointerCapture(e.pointerId)});
$('bridge-after').addEventListener('pointermove',e=>{if(!pointer)return;camera.yaw+=(e.clientX-pointer.x)*.009;camera.pitch+=(e.clientY-pointer.y)*.009;pointer.x=e.clientX;pointer.y=e.clientY;drawBridge()});
for(const n of ['pointerup','pointercancel','lostpointercapture'])$('bridge-after').addEventListener(n,()=>pointer=null);
$('bridge-after').addEventListener('wheel',e=>{e.preventDefault();camera.zoom=Math.max(.4,Math.min(3,camera.zoom*Math.exp(-e.deltaY*.001)));drawBridge()},{passive:false});
$('bridge-initial').addEventListener('pointerdown',e=>{pointer={id:e.pointerId,x:e.clientX,y:e.clientY};$('bridge-initial').setPointerCapture(e.pointerId)});
$('bridge-initial').addEventListener('pointermove',e=>{if(!pointer)return;camera.yaw+=(e.clientX-pointer.x)*.009;camera.pitch+=(e.clientY-pointer.y)*.009;pointer.x=e.clientX;pointer.y=e.clientY;drawBridge()});
for(const n of ['pointerup','pointercancel','lostpointercapture'])$('bridge-initial').addEventListener(n,()=>pointer=null);
$('bridge-initial').addEventListener('wheel',e=>{e.preventDefault();camera.zoom=Math.max(.4,Math.min(3,camera.zoom*Math.exp(-e.deltaY*.001)));drawBridge()},{passive:false});
$('program-case').addEventListener('change',e=>{programSelection=e.target.value;renderCase()});
$('bridge-case').addEventListener('change',e=>{bridgeSelection=e.target.value;drawBridge()});
$('reset-view').onclick=()=>{Object.assign(camera,{yaw:-.65,pitch:.55,zoom:1});drawBridge()};window.addEventListener('resize',drawBridge);
refresh();initBridge();tex();resumeJob();setInterval(refresh,15000);

let loadedModels=false;
document.querySelector('.provider-settings').addEventListener('toggle',async e=>{if(!e.target.open||loadedModels)return;try{const r=await fetch('/api/models'),d=await r.json();if(!r.ok)throw Error(d.error);$('provider-model').replaceChildren(new Option('Choose a generation model',''));for(const m of d.models)$('provider-model').add(new Option(m.name,m.id));$('provider-model').value='glm-5.3';loadedModels=true}catch(e){$('connection-message').textContent=e.message}});
$('provider-form').addEventListener('submit',async e=>{e.preventDefault();const button=e.target.querySelector('button');button.disabled=true;try{const r=await fetch('/api/provider',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({key:$('provider-key').value,model:$('provider-model').value})}),d=await r.json();if(!r.ok)throw Error(d.error);$('provider-key').value='';$('connection-message').textContent='Connection saved. Upload a document to generate.';await refresh()}catch(e){$('connection-message').textContent=e.message}finally{button.disabled=false}});

// Reveal each section once; live score refreshes never restart the animation.
if ('IntersectionObserver' in window && !window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
 const revealTargets=document.querySelectorAll('.intro, main .section-label, #upload-form, .case-heading, .case-copy, .metrics, .comparison, .specimen, .case-foot, .method-row, main footer');
 const revealObserver=new IntersectionObserver(entries=>{
  for(const entry of entries)if(entry.isIntersecting){entry.target.classList.add('revealed');revealObserver.unobserve(entry.target)}
 },{threshold:0.08,rootMargin:'0px 0px -28px 0px'});
 for(const el of revealTargets){
  const rect=el.getBoundingClientRect();
  el.classList.add('scroll-reveal');
  if(rect.height && rect.top<window.innerHeight-28 && rect.bottom>0)el.classList.add('revealed');
  else revealObserver.observe(el);
 }
}
