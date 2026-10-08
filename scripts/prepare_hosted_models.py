"""Prepare private reconstruction assets without modifying native exports.

Run with the project's gsplat Python environment. The manifest is local/private;
only the derived URL catalog belongs in the owner-restricted Site source.
"""
import argparse
import concurrent.futures
import fcntl
import gzip
import hashlib
import json
import os
import re
import subprocess
import struct
import time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
SITE=ROOT/'living-room-reconstruction/site/dist/client'
STAGE=ROOT/'gsplat_local/hosted_models_20261008'
CONVERTER=ROOT/'.streaming-tools/spark/rust/target/release/build-lod'
def SHA(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def job(path,key):
    return {'path':str(path),'key':key,'size':path.stat().st_size,'sha256':SHA(path),'content_type':'application/octet-stream'}

def ply_info(path):
    properties=[]
    with path.open('rb') as f:
        header=[]
        while True:
            line=f.readline();header.append(line)
            if line==b'end_header\n':break
            if len(header)>200:raise ValueError('Invalid PLY header')
        text=b''.join(header).decode();offset=f.tell()
    count=int(re.search(r'element vertex (\d+)',text).group(1))
    properties=re.findall(r'property float (\S+)',text)
    if 'format binary_little_endian 1.0' not in text or len(properties)!=59 or path.stat().st_size!=offset+count*59*4:raise ValueError('Expected complete SH3 Gaussian PLY')
    return count,properties,offset

def preview(path,output):
    count,properties,offset=ply_info(path);shown=min(count,120000)
    rows=np.linspace(0,count-1,shown,dtype=np.int64)
    raw=np.memmap(path,dtype='<f4',mode='r',offset=offset,shape=(count,59))
    data=np.asarray(raw[rows]);columns={name:data[:,i] for i,name in enumerate(properties)}
    result=np.zeros((shown,32),dtype=np.uint8);floats=result.view('<f4').reshape(shown,8)
    floats[:,:3]=np.column_stack([columns[x] for x in ['x','y','z']])
    floats[:,3:6]=np.exp(np.column_stack([columns[f'scale_{i}'] for i in range(3)]))
    rgb=np.column_stack([columns[f'f_dc_{i}'] for i in range(3)])*0.28209479177387814+0.5
    result[:,24:27]=np.round(np.clip(rgb,0,1)*255).astype(np.uint8)
    result[:,27]=np.round(255/(1+np.exp(-np.clip(columns['opacity'],-80,80)))).astype(np.uint8)
    quat=np.column_stack([columns[f'rot_{i}'] for i in range(4)])
    quat/=np.maximum(np.linalg.norm(quat,axis=1,keepdims=True),1e-20)
    result[:,28:32]=np.round(np.clip(quat*128+128,0,255)).astype(np.uint8)
    if not np.isfinite(floats[:,:6]).all():raise ValueError('Non-finite preview')
    with output.open('wb') as f:
        with gzip.GzipFile(fileobj=f,mode='wb',mtime=0) as z:z.write(result.tobytes())
    return shown

def prepare():
    STAGE.mkdir(exist_ok=True)
    catalogs={
      'original':json.loads((SITE/'gaussians/index.json').read_text()),
      'matching':json.loads((SITE/'matching-gsplat/index.json').read_text())['models'],
      'fix':json.loads((SITE/'fixing-gsplat-assets/index.json').read_text())['models'],
      'insta360':json.loads((SITE/'insta360-recovery-assets/gsplat.json').read_text())['models'],
      'voxel':json.loads((SITE/'voxel-gsplat-assets/index.json').read_text())['models'],
    }
    voxel_paths=json.loads((ROOT/'gsplat_local/voxel_experiments/insta360_4x_20261005/local_catalog.json').read_text())['models']
    entries={};unique={};jobs=[]
    # The matching comparison and repair are imported before original trials.
    for group in ['matching','fix','insta360','voxel','original']:
      for name,model in catalogs[group].items():
        if group=='matching':path=ROOT/f'gsplat_local/matching_20261003/runs/{name}/ply/point_cloud_29999.ply'
        elif group=='fix':path=ROOT/f'gsplat_local/full_resolution_fix_20261004/runs/{name}/ply/point_cloud_29999.ply'
        elif group=='insta360':path=ROOT/f'insta360_local/recovery/gsplat/runs/{name}/ply/point_cloud_29999.ply'
        elif group=='voxel':path=Path(voxel_paths[name]['ply_path'])
        else:path=ROOT/f'gsplat_local/runs/{name}/ply/point_cloud_29999.ply'
        count,_,_=ply_info(path)
        legacy_count=model.get('gaussian_count',model.get('count'))
        # The old 2x browser buffer came from the checkpoint, whereas the native
        # PLY has 815 fewer rows. Preserve both identities rather than pretending
        # that the browser buffer and the saved PLY have identical row counts.
        if count!=legacy_count and not (group=='original' and name=='2x' and count==1539059 and legacy_count==1539874):raise ValueError(f'Catalog/model identity differs: {group}/{name}')
        sha=SHA(path);before=path.stat();key=f'{group}/{name}/{sha[:20]}'
        if sha in unique:key=unique[sha]['key']
        else:
          directory=STAGE/key;directory.mkdir(parents=True,exist_ok=True)
          link=directory/'model.ply'
          if not link.exists():link.symlink_to(path)
          p=directory/'navigation.splat.gz';shown=preview(path,p)
          original_job=job(path,key+'/model.ply');jobs.append(original_job)
          unique[sha]={'key':key,'path':str(path),'link':str(link),'count':count,'preview_count':shown,'preview_job':job(p,key+'/navigation.splat.gz'),'source_size':before.st_size,'source_mtime_ns':before.st_mtime_ns,'sha256':sha}
        u=unique[sha]
        entries[group+'/'+name]={'id':group+'/'+name,'key':key,'label':model.get('label',name),'count':count,'legacy_browser_count':legacy_count if legacy_count!=count else None,'raw_bytes':path.stat().st_size,'raw_sha256':sha,'raw_url':'/models/'+key+'/model.ply','preview_url':'/models/'+key+'/navigation.splat.gz','preview_count':u['preview_count'],'preview_bytes':u['preview_job']['size'],'sh_degree':3,'independent_component':True}
    # Relocate the existing, byte-identical browser chunks to free static space.
    legacy=[]
    legacy_paths=sorted((SITE/'gaussians').glob('*.gsz')) or sorted((STAGE/'legacy_browser_chunks').glob('*.gsz'))
    for p in legacy_paths:legacy.append(job(p,'legacy/gaussians/'+p.name))
    (STAGE/'raw_import.json').write_text(json.dumps({'jobs':jobs+legacy},indent=2)+'\n')
    (STAGE/'models_private.json').write_text(json.dumps({'unique':list(unique.values()),'entries':entries},indent=2)+'\n')
    print(json.dumps({'models':len(entries),'unique_full_models':len(unique),'raw_bytes':sum(j['size'] for j in jobs),'legacy_chunks':len(legacy)}),flush=True)

def convert_one(model):
    link=Path(model['link']);directory=link.parent;log=directory/'conversion.log';rad=directory/'model-lod.rad'
    with (directory/'conversion.lock').open('w') as lock:
      fcntl.flock(lock,fcntl.LOCK_EX)
      if not (directory/'conversion_validated.json').exists():
        print(f'Convert: {model["key"]} · {model["count"]} Gaussians',flush=True)
        # The 6.18M-splat fragment has unusually costly overlap for Bhattacharyya merging.
        # TinyLoD changes only coarse browser hierarchy nodes; preserve every native leaf.
        method='--quick' if model['key'].startswith('matching/sequential_no_loop_component_0/') else '--quality'
        args=[str(CONVERTER),method,'--gsplat','--max-sh=3','--rad-chunked',str(link)]
        t=time.monotonic()
        with log.open('w') as f:subprocess.run(args,stdout=f,stderr=subprocess.STDOUT,check=True)
        text=log.read_text()
        if f'Read: num_splats: {model["count"]} with sh_degree: 3' not in text or not rad.exists() or 'Removed ' in text:raise RuntimeError(f'Conversion identity check failed: {model["key"]}; see local log')
        if Path(model['path']).stat().st_mtime_ns!=model['source_mtime_ns'] or SHA(Path(model['path']))!=model['sha256']:raise RuntimeError('Original export changed')
        (directory/'conversion_validated.json').write_text(json.dumps({'input_count':model['count'],'input_sh_degree':3,'method':('TinyLoD base 1.5' if method=='--quick' else 'Bhattacharyya LoD base 1.75')+', gsplat encoding, SH3, chunked RAD','source_sha256':model['sha256'],'seconds':time.monotonic()-t,'source_unchanged':True},indent=2)+'\n')
      validation=json.loads((directory/'conversion_validated.json').read_text())
      raw=rad.read_bytes();magic,length=struct.unpack('<II',raw[:8]);meta=json.loads(raw[8:8+length])
      description=json.loads(meta['comment'])
      if magic!=0x30444152 or meta.get('maxSh')!=3 or not meta.get('lodTree') or description['input_splat_count']!=model['count'] or description.get('empty_splat_count'):raise RuntimeError('RAD identity differs')
      parts=[rad]
      for chunk in meta['chunks']:
        if Path(chunk['filename']).name!=chunk['filename']:raise RuntimeError('Unsafe RAD chunk reference')
        path=directory/chunk['filename']
        if path.stat().st_size!=chunk['bytes']:raise RuntimeError('RAD chunk length differs')
        with path.open('rb') as f:
          magic,length=struct.unpack('<II',f.read(8));chunk_meta=json.loads(f.read(length))
        if magic!=0x43444152 or chunk_meta.get('maxSh')!=3:raise RuntimeError('RAD chunk identity differs')
        parts.append(path)
      if len(parts)<2 or sum(p.stat().st_size for p in parts[1:])!=meta['allChunkBytes']:raise RuntimeError('Streaming chunks missing')
      jobs=[job(p,model['key']+'/'+p.name) for p in parts]+[model['preview_job']]
      info={'stream_url':'/models/'+model['key']+'/model-lod.rad','stream_bytes':sum(p.stat().st_size for p in parts),'header_bytes':rad.stat().st_size,'chunks':len(parts)-1,'format':'RAD, gsplat encoding, SH3, '+validation['method'], 'lod_method':validation['method'],'stream_sh_degree':3,'source_leaf_count':model['count']}
      print(f'Converted: {model["key"]} · {info["stream_bytes"]/1e6:.1f} MB streaming assets',flush=True)
      return jobs,info

def convert():
    d=json.loads((STAGE/'models_private.json').read_text());jobs=[];catalog={};started=time.monotonic()
    # These are bounded CPU conversions, not GPU training. Two processes fit the
    # desktop's 32GB memory and per-model locks prevent duplicate conversion.
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
      results=pool.map(convert_one,d['unique'])
      for i,(model,(model_jobs,info)) in enumerate(zip(d['unique'],results),1):
        jobs.extend(model_jobs)
        for key,e in d['entries'].items():
          if e['key']==model['key']:catalog[key]=e|info
        (STAGE/'streaming_import.json').write_text(json.dumps({'jobs':jobs},indent=2)+'\n')
        (STAGE/'online_models.json').write_text(json.dumps({'version':1,'models':catalog,'preview_policy':'Navigation fallback is a deterministic evenly distributed sample of at most120000 Gaussian rows with static color. Progressive GPU rendering uses SH3 and adaptive LoD; untouched full PLYs remain downloadable.'},indent=2)+'\n')
        print(f'Validated {i}/{len(d["unique"])} · elapsed {time.monotonic()-started:.1f}s',flush=True)
    print('All progressive model conversions validated.',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--convert',action='store_true');a=p.parse_args()
    convert() if a.convert else prepare()
