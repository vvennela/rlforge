/* Depth-buffered rendering: face ordering cannot cut holes through supports. */
class BrickRenderer {
  constructor(canvas) {
    this.canvas=canvas;
    const gl=this.gl=canvas.getContext('webgl',{antialias:true,alpha:false,depth:true});
    if(!gl)throw Error('WebGL is required for the depth-correct 3D viewer.');
    const shader=(type,source)=>{const s=gl.createShader(type);gl.shaderSource(s,source);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw Error(gl.getShaderInfoLog(s));return s};
    const program=this.program=gl.createProgram();
    gl.attachShader(program,shader(gl.VERTEX_SHADER,'attribute vec3 position; attribute vec3 color; varying vec3 tint; void main(){gl_Position=vec4(position,1.0);tint=color;}'));
    gl.attachShader(program,shader(gl.FRAGMENT_SHADER,'precision mediump float; varying vec3 tint; void main(){gl_FragColor=vec4(tint,1.0);}'));
    gl.linkProgram(program);if(!gl.getProgramParameter(program,gl.LINK_STATUS))throw Error(gl.getProgramInfoLog(program));
    this.buffer=gl.createBuffer();this.position=gl.getAttribLocation(program,'position');this.color=gl.getAttribLocation(program,'color');
    gl.enable(gl.DEPTH_TEST);gl.depthFunc(gl.LEQUAL);gl.clearDepth(1);
  }
  render(state,camera) {
    const gl=this.gl,c=this.canvas,dpr=Math.min(devicePixelRatio||1,2),w=c.clientWidth,h=c.clientHeight;
    if(c.width!==Math.round(w*dpr)||c.height!==Math.round(h*dpr)){c.width=Math.round(w*dpr);c.height=Math.round(h*dpr)}
    gl.viewport(0,0,c.width,c.height);gl.clearColor(.965,.973,.947,1);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);
    if(!state)return;
    const dot=(a,b)=>a.reduce((s,v,i)=>s+v*b[i],0),s=Math.sin(camera.yaw),co=Math.cos(camera.yaw),sp=Math.sin(camera.pitch),cp=Math.cos(camera.pitch);
    const right=[co,-s,0],up=[-s*sp,-co*sp,cp],toward=[s*cp,co*cp,sp];
    const extent=state.task.view_bounds||[38,6,22],physical=[extent[0],extent[1],extent[2]*.4];
    const pivot=physical.map(v=>v/2),span=Math.max(...physical),units=Math.min(w/(span*1.35),h/(span*.80))*camera.zoom;
    const project=p=>{const v=p.map((n,i)=>n-pivot[i]);return [(dot(v,right)*units+camera.panX)*2/w,(dot(v,up)*units-camera.panY)*2/h,-dot(v,toward)/150]};
    const triangles=[],edges=[],grid=[];
    const add=(dest,p,color,bias=0)=>{const v=project(p);dest.push(v[0],v[1],v[2]+bias,...color.map(n=>n/255))};
    for(let x=-2;x<=physical[0]+4;x+=2){add(grid,[x,-2,-.02],[216,224,212]);add(grid,[x,physical[1]+4,-.02],[216,224,212])}
    for(let y=-2;y<=physical[1]+4;y+=2){add(grid,[-2,y,-.02],[216,224,212]);add(grid,[physical[0]+4,y,-.02],[216,224,212])}
    for(const b of state.bricks) {
      let [dx,dy,dz]=state.palette[b.part];if(b.rotation===90)[dx,dy]=[dy,dx];
      const x=b.x,y=b.y,z=b.z*.4,X=x+dx,Y=y+dy,Z=z+dz*.4;
      const center=[x+dx/2,y+dy/2,b.z+dz/2];
      const slot=(state.render_components||[]).find(k=>center.every((v,i)=>v>=k.min[i]&&v<=k.max[i]));
      const col=slot?.color || (state.task.grid?(b.z<state.task.grid.deck_z?[156,113,76]:b.z>=state.task.grid.deck_z+2?[194,136,66]:[72,135,108]):[72,135,108]);
      const face=(pts,n)=>{const shade=.65+.35*Math.max(0,dot(n,[-.3,-.4,.866]));const color=col.map(v=>v*shade);
        for(const i of [0,1,2,0,2,3])add(triangles,pts[i],color);
        for(let i=0;i<4;i++){add(edges,pts[i],[43,62,53],-.000025);add(edges,pts[(i+1)%4],[43,62,53],-.000025)}
      };
      face([[x,y,Z],[X,y,Z],[X,Y,Z],[x,Y,Z]],[0,0,1]);
      face([[x,y,z],[x,Y,z],[X,Y,z],[X,y,z]],[0,0,-1]);
      face([[x,y,z],[x,y,Z],[x,Y,Z],[x,Y,z]],[-1,0,0]);
      face([[X,y,z],[X,Y,z],[X,Y,Z],[X,y,Z]],[1,0,0]);
      face([[x,y,z],[X,y,z],[X,y,Z],[x,y,Z]],[0,-1,0]);
      face([[x,Y,z],[x,Y,Z],[X,Y,Z],[X,Y,z]],[0,1,0]);
    }
    gl.useProgram(this.program);gl.bindBuffer(gl.ARRAY_BUFFER,this.buffer);
    gl.enableVertexAttribArray(this.position);gl.enableVertexAttribArray(this.color);
    gl.vertexAttribPointer(this.position,3,gl.FLOAT,false,24,0);gl.vertexAttribPointer(this.color,3,gl.FLOAT,false,24,12);
    const draw=(data,mode)=>{gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(data),gl.DYNAMIC_DRAW);gl.drawArrays(mode,0,data.length/6)};
    gl.depthMask(false);draw(grid,gl.LINES);gl.depthMask(true);
    draw(triangles,gl.TRIANGLES);draw(edges,gl.LINES);
    c.dataset.renderer='webgl-depth-test';
  }
}
