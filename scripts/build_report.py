"""Export only aggregate, non-image statistics to the static report."""
from pathlib import Path
from datetime import datetime,timezone
import json,csv,re
R=Path(__file__).resolve().parents[1]
LOCAL=R.parent/'reconstruction_local'
DATA=R.parent/'living_room_frames'
SCALES={'1x':('full_resolution',2160,3840),'2x':('downsample_2x_1080x1920',1080,1920),'4x':('downsample_4x_540x960',540,960),'8x':('downsample_8x_270x480',270,480)}
def read(p,default=None):return json.loads(p.read_text()) if p.exists() else default

def ply_counts(path):
 if not path.exists():return None
 result={}
 with path.open('rb') as f:
  for _ in range(100):
   line=f.readline().decode('ascii',errors='replace').strip()
   if line.startswith('element '):
    _,name,count=line.split();result[name]=int(count)
   if line=='end_header':break
 return result

result={'updated':datetime.now(timezone.utc).isoformat(),'gpu':'NVIDIA GeForce RTX 4090','vram_gb':24,'pycolmap':'4.2.1','repo_url':None,'scales':{}}
if (R/'reports/repository.json').exists():result['repo_url']=read(R/'reports/repository.json')['url']
for scale,(folder,w,h) in SCALES.items():
 run=LOCAL/scale;status=read(run/'status.json',{'stages':{}});metrics=read(run/'sparse_metrics.json',{});models=read(run/'models.json',{}).get('models',[])
 item={'folder':folder,'width':w,'height':h,'count':500,'bytes':sum(p.stat().st_size for p in (DATA/folder).glob('*.png')),'stages':status['stages'],'registered_images':metrics.get('registered_images'),'points3D':metrics.get('points3D'),'mean_reprojection_error':metrics.get('mean_reprojection_error'),'mean_track_length':metrics.get('mean_track_length'),'components':models,'fused':ply_counts(run/'dense/fused.ply'),'mesh':ply_counts(run/'dense/mesh_poisson.ply')}
 # No camera poses, pixel data, photographs, local usernames, or raw logs.
 result['scales'][scale]=item
for dest in [R/'reports/results.json',R/'site/dist/results.json']:
 dest.write_text(json.dumps(result,indent=2)+'\n')
print('Published report data only: scale dimensions, counts, timing, stage states, aggregate metrics.')
