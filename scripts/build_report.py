"""Export only aggregate, non-image statistics to the static report."""
from pathlib import Path
from datetime import datetime,timezone
import json,csv,re,sqlite3
R=Path(__file__).resolve().parents[1]
LOCAL=R.parent/'reconstruction_local'
DATA=R.parent/'living_room_frames'
SCALES={'1x':('full_resolution',2160,3840),'2x':('downsample_2x_1080x1920',1080,1920),'4x':('downsample_4x_540x960',540,960),'8x':('downsample_8x_270x480',270,480)}
def read(p,default=None):return json.loads(p.read_text()) if p.exists() else default

def sift_keypoint_count(database):
 if not database.is_file():return None
 with sqlite3.connect(f'file:{database}?mode=ro',uri=True) as db:
  row=db.execute('SELECT SUM(rows) FROM keypoints').fetchone()
 return int(row[0] or 0)

def ply_counts(path):
 if not path.is_file():return None
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
 wait=read(run/'depth_queue_wait.json',{})
 if 'depth_maps' in status['stages'] and wait:
  depth=status['stages']['depth_maps'];depth['queue_wait_seconds']=wait['seconds']
  if depth.get('seconds') is not None:depth['seconds']=round(max(0,depth['seconds']-wait['seconds']),2)
 if scale=='1x' and (R.parent/'gsplat_local/before_1x_depth.wait').exists() and status['stages'].get('depth_maps',{}).get('state')=='running':
  status['stages']['depth_maps']['state']='queued'
 item={'folder':folder,'width':w,'height':h,'count':500,'bytes':sum(p.stat().st_size for p in (DATA/folder).glob('*.png')),'sift_keypoints':sift_keypoint_count(run/'database.db'),'stages':status['stages'],'registered_images':metrics.get('registered_images'),'points3D':metrics.get('points3D'),'mean_reprojection_error':metrics.get('mean_reprojection_error'),'mean_track_length':metrics.get('mean_track_length'),'components':models,'fused':ply_counts(run/'dense/fused.ply'),'mesh':ply_counts(run/'dense/mesh_poisson.ply')}
 error=item['mean_reprojection_error']
 item['mean_reprojection_error_full_resolution_px']=error*int(scale[:-1]) if error is not None else None
 item['reprojection_error_stage']='after_bundle_adjustment_and_filtering' if status['stages'].get('bundle_adjustment',{}).get('state')=='complete' else None
 config=run/'dense/stereo/patch-match.cfg'
 if config.is_file():
  lines=[line.strip() for line in config.read_text().splitlines() if not line.startswith('#')]
  targets=lines[::2];sources=lines[1::2]
  skipped_names={target for target,source in zip(targets,sources) if not source}
  depth_log=run/'depth_maps.log'
  if depth_log.is_file():
   skipped_names.update(re.findall(r'Ignoring reference image (\S+), because it has no source images',depth_log.read_text()))
  skipped=len(skipped_names)
  item['dense_reference_images']=len(targets)-skipped
  item['dense_skipped_no_sources']=skipped
  item['geometric_depth_maps_written']=len(list((run/'dense/stereo/depth_maps').glob('*.geometric.bin')))
 # No camera poses, pixel data, photographs, local usernames, or raw logs.
 result['scales'][scale]=item
for dest in [R/'reports/results.json',R/'site/dist/results.json']:
 dest.write_text(json.dumps(result,indent=2)+'\n')
print('Published report data only: scale dimensions, SIFT keypoint totals, counts, timing, stage states, aggregate metrics.')
