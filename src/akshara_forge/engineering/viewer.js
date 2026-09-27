let state = null;
const camera = {yaw: -.65, pitch: .55, zoom: 1, panX: 0, panY: 0};
let pointer = null;
const canvas = document.getElementById('scene'), ctx = canvas.getContext('2d');
const dot = (a,b) => a.reduce((s,v,i) => s+v*b[i],0);
function basis() {
  const s=Math.sin(camera.yaw), c=Math.cos(camera.yaw), sp=Math.sin(camera.pitch), cp=Math.cos(camera.pitch);
  return {right:[c,-s,0], up:[-s*sp,-c*sp,cp], toward:[s*cp,c*cp,sp]};
}
function draw() {
  const dpr=devicePixelRatio||1, w=canvas.clientWidth, h=canvas.clientHeight;
  canvas.width=w*dpr; canvas.height=h*dpr; ctx.scale(dpr,dpr); ctx.clearRect(0,0,w,h);
  if (!state) return;
  const B=basis(), units=Math.min(w/48,h/27)*camera.zoom, pivot=[19,3,3.8];
  const P=p => {const v=p.map((n,i)=>n-pivot[i]);return [w/2+camera.panX+dot(v,B.right)*units,h/2+camera.panY-dot(v,B.up)*units]};
  const line=(a,b)=>{ctx.beginPath();ctx.moveTo(...P(a));ctx.lineTo(...P(b));ctx.stroke()};
  ctx.strokeStyle='#d9e0d5';ctx.lineWidth=.5;
  for(let x=-2;x<=42;x+=2)line([x,-2,0],[x,10,0]);
  for(let y=-2;y<=10;y+=2)line([-2,y,0],[42,y,0]);
  const faces=[];
  for(const b of state.bricks) {
    let [dx,dy,dz]=state.palette[b.part];if(b.rotation===90)[dx,dy]=[dy,dx];
    // One plate height is 0.4 studs. Camera preserves physical aspect ratios.
    const x=b.x,y=b.y,z=b.z*.4,X=x+dx,Y=y+dy,Z=z+dz*.4;
    const color=b.z<state.task.grid.deck_z?[156,113,76]:b.z>=state.task.grid.deck_z+2?[194,136,66]:[72,135,108];
    const add=(pts,normal)=>{
      if(dot(normal,B.toward)<=.00001)return;
      const center=pts[0].map((_,i)=>pts.reduce((s,p)=>s+p[i],0)/pts.length);
      const light=.70+.30*Math.max(0,dot(normal,[-.3,-.4,.866]));
      faces.push({pts:pts.map(P),depth:dot(center,B.toward),color:`rgb(${color.map(v=>Math.round(v*light)).join(',')})`});
    };
    add([[x,y,Z],[X,y,Z],[X,Y,Z],[x,Y,Z]],[0,0,1]);
    add([[x,y,z],[x,Y,z],[X,Y,z],[X,y,z]],[0,0,-1]);
    add([[x,y,z],[x,y,Z],[x,Y,Z],[x,Y,z]],[-1,0,0]);
    add([[X,y,z],[X,Y,z],[X,Y,Z],[X,y,Z]],[1,0,0]);
    add([[x,y,z],[X,y,z],[X,y,Z],[x,y,Z]],[0,-1,0]);
    add([[x,Y,z],[x,Y,Z],[X,Y,Z],[X,Y,z]],[0,1,0]);
  }
  faces.sort((a,b)=>a.depth-b.depth);
  for(const f of faces){ctx.beginPath();f.pts.forEach((p,i)=>i?ctx.lineTo(...p):ctx.moveTo(...p));ctx.closePath();ctx.fillStyle=f.color;ctx.fill();ctx.strokeStyle='#233e3a65';ctx.lineWidth=.6;ctx.stroke()}
  if(!state.bricks.length){ctx.fillStyle='#60766c';ctx.font='14px system-ui';ctx.textAlign='center';ctx.fillText('Load a fixture or place bricks',w/2,h/2)}
  const deg=r=>Math.round(((r*180/Math.PI)%360+360)%360);
  document.getElementById('camera-state').textContent=`Yaw ${deg(camera.yaw)}° · pitch ${deg(camera.pitch)}° · zoom ${camera.zoom.toFixed(1)}×`;
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
  state=s;const E=s.evaluation;
  document.getElementById('mode').textContent=s.mode+(s.busy?' · thinking…':'');
  document.getElementById('error').textContent=s.error||s.last_tool_error||'';
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
  document.getElementById('coverage').textContent=`${(E.components||[]).filter(c=>c.complete).length} / ${(E.components||[]).length} components match. Excluded from this specification: ${(E.unmodeled_source_features||[]).join('; ')}. Wires are not established by this drawing.`;
  const checks=document.getElementById('checks');checks.replaceChildren();
  Object.entries(E.checks).forEach(([k,v])=>{let e=document.createElement('span');e.className='check'+(v?'':' bad');e.textContent=(v?'✓ ':'× ')+k.replaceAll('_',' ');checks.append(e)});
  document.getElementById('details').textContent=JSON.stringify(E,null,2);
  document.querySelectorAll('button[data-mutation]').forEach(b=>b.disabled=s.busy);draw();
}
async function act(url,data){try{const r=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});const s=await r.json();if(!r.ok)throw Error(s.error);display(s)}catch(e){document.getElementById('error').textContent=e.message}}
function reset(mode){act('/api/reset',{mode})}
function sendTools(){try{act('/api/step',JSON.parse(document.getElementById('actions').value))}catch(e){document.getElementById('error').textContent=e.message}}
async function poll(){try{const r=await fetch('/api/state');if(r.ok)display(await r.json())}catch(e){document.getElementById('error').textContent='Local environment disconnected'}}
setInterval(poll,2000);poll();
