// Anisotropic 3D Gaussian covariance projection and depth-sorted alpha.
// Deterministic sampling and bounded footprints keep this CPU preview responsive.
const LIMIT = 120000, STRIDE = 13;
let data, count = 0, revision = 0;
const covariance = (c, a, b) => c[0]*a[0]*b[0] + c[1]*(a[0]*b[1]+a[1]*b[0]) + c[2]*(a[0]*b[2]+a[2]*b[0]) + c[3]*a[1]*b[1] + c[4]*(a[1]*b[2]+a[2]*b[1]) + c[5]*a[2]*b[2];
function prepare(buffer) {
  if (buffer.byteLength % 32) throw Error('Invalid software Gaussian model');
  const floats = new Float32Array(buffer), bytes = new Uint8Array(buffer), total = buffer.byteLength / 32;
  count = Math.min(total, LIMIT); data = new Float32Array(count * STRIDE);
  for (let j = 0; j < count; j++) {
    const i = Math.floor((j + .5) * total / count), f = i * 8, b = i * 32, o = j * STRIDE; data.set(floats.subarray(f, f + 3), o);
    let [w,x,y,z] = Array.from(bytes.subarray(b+28,b+32), value => (value-128)/128);
    const length = Math.hypot(w,x,y,z) || 1; w/=length; x/=length; y/=length; z/=length;
    const axes = [[1-2*(y*y+z*z),2*(x*y+w*z),2*(x*z-w*y)], [2*(x*y-w*z),1-2*(x*x+z*z),2*(y*z+w*x)], [2*(x*z+w*y),2*(y*z-w*x),1-2*(x*x+y*y)]], c = [0,0,0,0,0,0];
    for (let k = 0; k < 3; k++) {const [ax,ay,az] = axes[k], s = floats[f+3+k] ** 2; c[0]+=s*ax*ax; c[1]+=s*ax*ay; c[2]+=s*ax*az; c[3]+=s*ay*ay; c[4]+=s*ay*az; c[5]+=s*az*az;}
    data.set(c,o+3); data.set(bytes.subarray(b+24,b+28),o+9);
  }
}
function render(message) {
  const {width,height,matrix:m,fov} = message, focal = height / (2*Math.tan(fov*Math.PI/360)), projection = new Float32Array(count*7), visible = [];
  const right=[m[0],m[4],m[8]], up=[m[1],m[5],m[9]], back=[m[2],m[6],m[10]];
  for (let i = 0; i < count; i++) {
    const o=i*STRIDE, px=data[o],py=data[o+1],pz=data[o+2], z=-(m[2]*px+m[6]*py+m[10]*pz+m[14]);
    if (z < .01 || !Number.isFinite(z) || data[o+12]===0) continue;
    const x=m[0]*px+m[4]*py+m[8]*pz+m[12], y=m[1]*px+m[5]*py+m[9]*pz+m[13], screenX=width/2+focal*x/z, screenY=height/2-focal*y/z;
    const jx=right.map((v,k)=>focal/z*(v+x/z*back[k])), jy=up.map((v,k)=>-focal/z*(v+y/z*back[k])), c=data.subarray(o+3,o+9);
    let a=covariance(c,jx,jx)+.3, b=covariance(c,jx,jy), d=covariance(c,jy,jy)+.3;
    if (![a,b,d,screenX,screenY].every(Number.isFinite)) continue;
    const shrink=Math.min(1,96**2/(9*Math.max(a,d))); a*=shrink; b*=shrink; d*=shrink;
    const rx=3*Math.sqrt(a),ry=3*Math.sqrt(d);
    if (screenX+rx<0 || screenX-rx>=width || screenY+ry<0 || screenY-ry>=height) continue;
    projection.set([screenX,screenY,z,a,b,d,data[o+12]/255],i*7); visible.push(i);
  }
  visible.sort((a,b)=>projection[b*7+2]-projection[a*7+2]);
  const pixels=new Uint8ClampedArray(width*height*4);
  for(let i=0;i<pixels.length;i+=4){pixels[i]=24;pixels[i+1]=35;pixels[i+2]=44;pixels[i+3]=255;}
  for (const i of visible) {
    const o=i*7, cx=projection[o],cy=projection[o+1],a=projection[o+3],b=projection[o+4],d=projection[o+5],opacity=projection[o+6],det=a*d-b*b;
    if (det<=0) continue;
    const rx=3*Math.sqrt(a),ry=3*Math.sqrt(d),left=Math.max(0,Math.floor(cx-rx)),right=Math.min(width-1,Math.ceil(cx+rx)),top=Math.max(0,Math.floor(cy-ry)),bottom=Math.min(height-1,Math.ceil(cy+ry)), c=i*STRIDE+9;
    for(let y=top;y<=bottom;y++)for(let x=left;x<=right;x++){
      const dx=x+.5-cx,dy=y+.5-cy,exponent=(d*dx*dx-2*b*dx*dy+a*dy*dy)/(2*det); if(exponent>4.5)continue;
      const alpha=Math.min(.99,opacity*Math.exp(-exponent));if(alpha<1/255)continue; const p=(y*width+x)*4;
      pixels[p]+=alpha*(data[c]-pixels[p]); pixels[p+1]+=alpha*(data[c+1]-pixels[p+1]); pixels[p+2]+=alpha*(data[c+2]-pixels[p+2]);
    }
  }
  postMessage({type:'frame',revision,view:message.view,width,height,pixels:pixels.buffer},[pixels.buffer]);
}
self.onmessage = event => {
  const message=event.data;
  try {
    if(message.type==='clear'){revision=message.revision; data=null; count=0;}
    else if(message.type==='load'){revision=message.revision; prepare(message.buffer); postMessage({type:'loaded',revision,count});}
    else if(message.type==='render' && message.revision===revision && data)render(message);
  } catch(error){postMessage({type:'error',revision,message:error.message});}
};
