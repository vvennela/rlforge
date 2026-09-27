let state = null;
const camera = {yaw: -.65, pitch: .55, zoom: 1, panX: 0, panY: 0};
let pointer = null;
const canvas = document.getElementById('scene');
let renderer=null,rendererError=null;
try {renderer=new BrickRenderer(canvas)} catch(e) {rendererError=e.message;document.getElementById('error').textContent=e.message}
function draw() {
  if(renderer)renderer.render(state,camera);
  const deg=r=>Math.round(((r*180/Math.PI)%360+360)%360);
  document.getElementById('camera-state').textContent=`Yaw ${deg(camera.yaw)}° · pitch ${deg(camera.pitch)}° · zoom ${camera.zoom.toFixed(1)}× · depth-tested 3D`;
}

function view(name) {
  const presets={iso:[-.65,.55],front:[0,0],side:[Math.PI/2,0],top:[0,Math.PI/2],bottom:[0,-Math.PI/2]};
  [camera.yaw,camera.pitch]=presets[name];camera.zoom=1;camera.panX=camera.panY=0;draw();
}
function zoom(factor){camera.zoom=Math.max(.35,Math.min(4,camera.zoom*factor));draw()}
canvas.addEventListener('pointerdown',e=>{canvas.focus();pointer={id:e.pointerId,x:e.clientX,y:e.clientY,pan:e.shiftKey||e.button===2};canvas.setPointerCapture(e.pointerId)});
canvas.addEventListener('pointermove',e=>{
  if(!pointer||pointer.id!==e.pointerId)return;
  const dx=e.clientX-pointer.x,dy=e.clientY-pointer.y;
  if(pointer.pan){camera.panX+=dx;camera.panY+=dy}else{camera.yaw+=dx*.009;camera.pitch+=dy*.009}
  pointer.x=e.clientX;pointer.y=e.clientY;draw();
});
for(const name of ['pointerup','pointercancel','lostpointercapture'])canvas.addEventListener(name,()=>pointer=null);
canvas.addEventListener('contextmenu',e=>e.preventDefault());
canvas.addEventListener('wheel',e=>{e.preventDefault();zoom(Math.exp(-e.deltaY*.0015))},{passive:false});
canvas.addEventListener('keydown',e=>{
  if(!['ArrowLeft','ArrowRight','ArrowUp','ArrowDown','+','-','0'].includes(e.key))return;e.preventDefault();
  if(e.key==='ArrowLeft')camera.yaw-=.15;if(e.key==='ArrowRight')camera.yaw+=.15;
  if(e.key==='ArrowUp')camera.pitch-=.15;if(e.key==='ArrowDown')camera.pitch+=.15;
  if(e.key==='+')zoom(1.2);if(e.key==='-')zoom(1/1.2);if(e.key==='0')view('iso');draw();
});
window.addEventListener('resize',draw);
function display(s) {
  const changed=state?.task.id!==s.task.id;state=s;const E=s.evaluation;
  const key=s.task.family==='machinery'?'turbine':s.task.family;
  document.getElementById('task-choice').value=key;
  document.getElementById('task-choice').disabled=s.busy;
  document.getElementById('source-title').textContent=s.task.title;
  document.getElementById('source-provenance').textContent=s.task.provenance;
  document.getElementById('source-abstraction').textContent=s.task.abstraction;
  document.getElementById('source-dimensions').textContent=s.task.source_dimensions_inches?Object.entries(s.task.source_dimensions_inches).map(([k,v])=>`${k.replaceAll('_',' ')}: ${v} in`).join(' · ')+' · '+s.task.inches_per_stud+' source inches / stud':'Authored sizes; source supplies component arrangement.';
  document.getElementById('source-link').href=s.task.source_url;
  if(changed){document.getElementById('drawing-image').src='/drawing?task='+key;document.getElementById('drawing-image').alt=s.task.title+' source drawing';document.getElementById('drawing-link').href='/drawing?task='+key;view('iso');document.getElementById('actions').value=JSON.stringify({actions:[]})}

  document.getElementById('mode').textContent=s.mode+(s.busy?' · thinking…':'');
  document.getElementById('error').textContent=s.error||s.last_tool_error||rendererError||'';
  document.getElementById('reward').textContent=E.scalar_reward.toFixed(3);
  document.getElementById('bricks').textContent=s.bricks.length;
  document.getElementById('steps').textContent=s.steps+' / 16';
  document.getElementById('sandbox').textContent=s.sandbox;
  document.getElementById('verdict').textContent=E.success?'Declared components pass':'Incomplete';
  const scores=document.getElementById('scores');scores.replaceChildren();
  Object.entries(E.reward_vector).filter(([k])=>!k.includes('/')).forEach(([k,v])=>{
    const row=document.createElement('div');row.className='row';const label=document.createElement('span');label.textContent=k.replaceAll('_',' ');
    const bar=document.createElement('div');bar.className='bar';const fill=document.createElement('div');fill.className='fill';fill.style.width=v*100+'%';bar.append(fill);
    const value=document.createElement('span');value.textContent=v.toFixed(2);row.append(label,bar,value);scores.append(row);
  });
  const body=document.getElementById('component-body');body.replaceChildren();
  for(const c of E.components||[]){
    const row=document.createElement('tr');let cell=document.createElement('th');cell.scope='row';cell.textContent=c.id.replaceAll('_',' ');
    const dims=document.createElement('small');const d=c.dimensions;
    dims.textContent=`actual ${['x_studs','y_studs','z_plates'].map(k=>d[k].actual??'—').join(' × ')} / target ${['x_studs','y_studs','z_plates'].map(k=>d[k].target).join(' × ')}`;
    cell.append(dims);row.append(cell);
    for(const k of ['presence','position','size','proportions','overlap']){cell=document.createElement('td');cell.textContent=c.scores[k].toFixed(2);if(c.scores[k]<.999)cell.className='bad';row.append(cell)}
    body.append(row);
  }
  document.getElementById('coverage').textContent=`${(E.components||[]).filter(c=>c.complete).length} / ${(E.components||[]).length} components match. Excluded from this specification: ${(E.unmodeled_source_features||[]).join('; ')}. ${(s.task.unsupported_features||[]).join('; ')}`;
  const checks=document.getElementById('checks');checks.replaceChildren();
  Object.entries(E.checks).forEach(([k,v])=>{let e=document.createElement('span');e.className='check'+(v?'':' bad');e.textContent=(v?'✓ ':'× ')+k.replaceAll('_',' ');checks.append(e)});
  document.getElementById('details').textContent=JSON.stringify(E,null,2);
  document.querySelectorAll('button[data-mutation]').forEach(b=>b.disabled=s.busy);
  const agentButton=document.getElementById('run-agent');agentButton.disabled=s.busy||s.agent?.configured===false;agentButton.textContent=s.agent?.configured===false?'Inference key needed':'Run '+(s.agent?.model||'agent')+' repair';draw();
}
async function act(url,data){try{const r=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});const s=await r.json();if(!r.ok)throw Error(s.error);display(s)}catch(e){document.getElementById('error').textContent=e.message}}
function reset(mode){act('/api/reset',{mode,task:document.getElementById('task-choice').value})}
function sendTools(){try{act('/api/step',JSON.parse(document.getElementById('actions').value))}catch(e){document.getElementById('error').textContent=e.message}}
async function poll(){try{const r=await fetch('/api/state');if(r.ok)display(await r.json())}catch(e){document.getElementById('error').textContent='Local environment disconnected'}}
setInterval(poll,2000);poll();
