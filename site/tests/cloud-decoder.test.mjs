import assert from 'node:assert/strict';
import {readFile,readdir} from 'node:fs/promises';
import {gunzipSync} from 'node:zlib';
import {runInNewContext} from 'node:vm';

const root=new URL('../dist/client/',import.meta.url),context={window:{},document:{body:{dataset:{skipLegacyClouds:'true'}}}};
runInNewContext(await readFile(new URL('cloud-viewer.js',root),'utf8'),context);
const decode=context.window.decodePointCloud;
const buffer=bytes=>bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength);
const original=[.25,-.5,1,121,107,87,-1,2,3,158,149,140],n=2;
const packed=Buffer.alloc(n*15);
for(let i=0;i<n;i++)for(let j=0;j<3;j++){packed.writeFloatLE(original[i*6+j],i*15+j*4);packed[i*15+12+j]=original[i*6+3+j];}
const shuffled=Buffer.alloc(packed.length);
for(let lane=0;lane<15;lane++)for(let i=0;i<n;i++)shuffled[lane*n+i]=packed[i*15+lane];
const decoded=new Float32Array(decode(buffer(packed),{format:'xyz-f32-rgb-u8',displayed_points:n}));
assert.deepEqual(Array.from(new Float32Array(decode(buffer(shuffled),{format:'xyz-f32-rgb-u8-shuffled',displayed_points:n}))),Array.from(decoded));
for(let i=0;i<n;i++)for(let j=0;j<3;j++){assert.equal(decoded[i*6+j],original[i*6+j]);assert.equal(decoded[i*6+3+j],Math.fround(original[i*6+3+j]/255));}
assert.deepEqual(Array.from(new Float32Array(decode(buffer(Buffer.from(decoded.buffer)),{displayed_points:n}))),Array.from(decoded));
assert.throws(()=>decode(buffer(packed),{displayed_points:n}),/Point count mismatch/);
assert.throws(()=>decode(buffer(packed),{format:'xyz-f32-rgb-u8',displayed_points:n+1}),/Point count mismatch/);
assert.throws(()=>decode(buffer(packed),{format:'unsupported'}),/Unsupported/);
const invalid=Buffer.from(packed);invalid.writeFloatLE(NaN,0);
assert.throws(()=>decode(buffer(invalid),{format:'xyz-f32-rgb-u8'}),/Invalid/);

// Exercise actual published catalogs and the same renderer entry point used by
// both comparison panels, including every uncapped 1000-frame sparse component.
const clouds=new Map();
function collect(value){
 if(Array.isArray(value))for(const item of value)collect(item);
 else if(value&&typeof value==='object'){
  if(value.url&&Number.isInteger(value.displayed_points)){
   const old=clouds.get(value.url);if(old)assert.equal(old.format,value.format);clouds.set(value.url,value);
  }
  for(const item of Object.values(value))collect(item);
 }
}
for(const directory of ['','clouds/'])for(const name of await readdir(new URL(directory,root))){
 if(name.endsWith('.json'))collect(JSON.parse(await readFile(new URL(directory+name,root),'utf8')));
}
let points=0,allPoints=0;
for(const cloud of clouds.values()){
 const bytes=await readFile(new URL(cloud.url,root)),raw=cloud.url.endsWith('.gz')?gunzipSync(bytes):bytes;
 const renderer=Object.create(context.window.CloudRenderer.prototype);
 renderer.reset=()=>{};renderer.draw=()=>{};
 renderer.load(buffer(raw),cloud.camera_positions||[],cloud.render_all||false,cloud);
 assert.equal(renderer.count,cloud.displayed_points);
 assert.equal(renderer.softwarePoints.length,cloud.displayed_points*6);
 if(cloud.render_all){assert.equal(renderer.renderAll,true);allPoints++;}
 points+=renderer.count;
}
console.log(`Decoded ${clouds.size} real clouds / ${points.toLocaleString()} points with finite coordinates and original RGB; ${allPoints} uncapped clouds preserved.`);
