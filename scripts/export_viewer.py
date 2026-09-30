"""Export derived point-cloud previews to the PRIVATE site, never source photographs.
The clouds directory is excluded from GitHub source uploads.
"""
from pathlib import Path
import json,re,gzip
import numpy as np
import pycolmap as p
R=Path(__file__).resolve().parents[1];LOCAL=R.parent/'reconstruction_local';OUT=R/'site/dist/clouds';OUT.mkdir(exist_ok=True)
LIMIT=200000

def export_cloud(key,xyz,rgb,cameras):
 n=len(xyz)
 # Deterministic preview sample; keeps all sparse points when within the cap.
 ix=np.linspace(0,n-1,min(n,LIMIT),dtype=np.int64)
 center=np.median(xyz,axis=0);radius=float(np.percentile(np.linalg.norm(xyz-center,axis=1),95));radius=max(radius,1e-6)
 packed=np.concatenate(((xyz[ix]-center)/radius,np.asarray(rgb[ix],dtype=np.float32)/255),axis=1).astype('<f4')
 (OUT/(key+'.bin.gz')).write_bytes(gzip.compress(packed.tobytes(),6))
 camera_positions=[((np.array(c)-center)/radius).round(6).tolist() for c in cameras]
 return dict(url='clouds/'+key+'.bin.gz',total_points=n,displayed_points=len(ix),sampled=n>LIMIT,camera_positions=camera_positions)

def read_ply(path):
 types={'float':'<f4','float32':'<f4','double':'<f8','uchar':'u1','uint8':'u1','int':'<i4','uint':'<u4'}
 with path.open('rb') as f:
  props=[];count=0;vertex=False;fmt=''
  while True:
   line=f.readline().decode('ascii').strip()
   if line.startswith('format '):fmt=line.split()[1]
   if line.startswith('element '):
    a=line.split();vertex=a[1]=='vertex'
    if vertex:count=int(a[2])
   if line.startswith('property ') and vertex:
    a=line.split();props.append((a[2],types[a[1]]))
   if line=='end_header':break
  if fmt!='binary_little_endian':raise ValueError('Expected COLMAP binary little-endian PLY')
  v=np.fromfile(f,dtype=np.dtype(props),count=count)
 xyz=np.column_stack([v[k] for k in ['x','y','z']]);rgb=np.column_stack([v[k] for k in ['red','green','blue']])
 return xyz,rgb

index={'preview_limit':LIMIT,'component_rule':'Substantial: at least 10 registered images and 100 3D points. Smaller non-empty components are also available and explicitly labeled.','alignment':'Components and resolutions use independent coordinate systems. They are displayed separately, not falsely overlaid.','scales':{}}
for scale in ['1x','2x','4x','8x']:
 run=LOCAL/scale;status=json.loads((run/'status.json').read_text()) if (run/'status.json').exists() else {'stages':{}}
 models=[];union=set();component_members=[]
 if status['stages'].get('mapping',{}).get('state')=='complete':
  for path in sorted((run/'sparse').glob('*')):
   if not path.is_dir() or not (path/'points3D.bin').exists():continue
   m=p.Reconstruction(path);names=sorted(im.name for im in m.images.values());ids=[int(re.search(r'(\d+)',n).group(1)) for n in names]
   n=m.num_points3D();item=dict(id=path.name,kind='sparse',registered_images=m.num_reg_images(),frame_ids=ids,total_points=n,substantial=m.num_reg_images()>=10 and n>=100)
   if n:
    points=list(m.points3D.values());xyz=np.array([pt.xyz for pt in points]);rgb=np.array([pt.color for pt in points]);cams=[im.projection_center() for im in m.images.values()]
    item.update(export_cloud(f'{scale}-sparse-{path.name}',xyz,rgb,cams));union.update(ids)
   else:item['reason']='No triangulated points; camera registration alone is not a usable point cloud.'
   models.append(item)
  models.sort(key=lambda x:(x['substantial'],x['registered_images'],x['total_points']),reverse=True)
 dense=None
 if status['stages'].get('fusion',{}).get('state')=='complete':
  xyz,rgb=read_ply(run/'dense/fused.ply')
  if len(xyz):dense=dict(id='dense',kind='dense',**export_cloud(f'{scale}-dense',xyz,rgb,[]))
 index['scales'][scale]=dict(components=models,dense=dense,registered_union=len(union),missing_frame_ids=sorted(set(range(1,501))-union),ready=bool(models))
 print(scale,[(m['id'],m['registered_images'],m['total_points']) for m in models],flush=True)
(OUT/'index.json').write_text(json.dumps(index,indent=2)+'\n')
