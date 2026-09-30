"""Publish aggregate experiment status and compact private cloud previews only."""
from pathlib import Path
import json,time,numpy as np
import pycolmap as p
import export_viewer as ev
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'experiments_local';SITE=ROOT/'living-room-reconstruction/site/dist'
# At most 40k points per browser cloud to preserve existing Site assets under hosting limits.
ev.LIMIT=40000
SPECS=[('room_250','250 frames · whole room',250),('room_1000','1,000 frames · whole room',1000),('dining_500','500 frames · dining area',500)]
def main():
 catalog={'preview_limit':40000,'scales':{}};report={'updated_at':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'experiments':[]}
 old=json.loads((SITE/'clouds/index.json').read_text());catalog['scales']['baseline_500']={**old['scales']['4x'],'label':'500 frames · original baseline','input_images':500}
 base_metrics=json.loads((ROOT/'reconstruction_local/4x/sparse_metrics.json').read_text())
 report['baseline']={'count':500,'registered_images':base_metrics['registered_images'],'sparse_points':base_metrics['points3D'],'dense_points':old['scales']['4x']['dense']['total_points'],'reprojection_error':base_metrics['mean_reprojection_error']}
 for key,label,count in SPECS:
  dataset=BASE/key;run=dataset/'colmap/4x';status=json.loads((run/'status.json').read_text()) if (run/'status.json').exists() else {'stages':{}}
  meta=json.loads((dataset/'dataset.json').read_text());models=[];union=set()
  if status['stages'].get('mapping',{}).get('state')=='complete' and status['stages'].get('bundle_adjustment',{}).get('state')!='running':
   for path in sorted((run/'sparse').glob('*')):
    if not (path/'points3D.bin').exists():continue
    m=p.Reconstruction(path);ids=sorted(int(im.name.split('_')[1].split('.')[0]) for im in m.images.values());n=m.num_points3D()
    item={'id':path.name,'kind':'sparse','registered_images':m.num_reg_images(),'frame_ids':ids,'total_points':n,'substantial':m.num_reg_images()>=10 and n>=100}
    if n:
     points=list(m.points3D.values());item.update(ev.export_cloud(f'experiment-{key}-sparse-{path.name}',np.array([pt.xyz for pt in points]),np.array([pt.color for pt in points]),[im.projection_center() for im in m.images.values()]));union.update(ids)
    models.append(item)
  models.sort(key=lambda m:m['registered_images'],reverse=True);dense=None
  if status['stages'].get('fusion',{}).get('state')=='complete':
   xyz,rgb=ev.read_ply(run/'dense/fused.ply');dense={'id':'dense','kind':'dense',**ev.export_cloud(f'experiment-{key}-dense',xyz,rgb,[])}
  catalog['scales'][key]={'label':label,'input_images':count,'components':models,'dense':dense,'registered_union':len(union),'missing_frame_ids':sorted(set(range(1,count+1))-union),'ready':bool(models)}
  metrics=json.loads((run/'sparse_metrics.json').read_text()) if (run/'sparse_metrics.json').exists() else {}
  report['experiments'].append({'key':key,'label':label,'input_images':count,'dataset':meta,'extraction_complete':(dataset/'validation.json').exists(),'stages':status['stages'],'components':[{'id':m['id'],'registered_images':m['registered_images'],'points':m['total_points']} for m in models],'registered_union':len(union),'metrics':{k:metrics[k] for k in ['registered_images','points3D','mean_reprojection_error','mean_track_length'] if k in metrics},'dense_points':dense['total_points'] if dense else None,'folders':{'native':f'experiments_local/{key}/full_resolution','4x':f'experiments_local/{key}/downsample_4x_540x960','colmap':f'experiments_local/{key}/colmap/4x'},'storage_bytes':{f:sum(x.stat().st_size for x in (dataset/f).glob('*.png')) for f in ['full_resolution','downsample_4x_540x960']}})
 (SITE/'clouds/experiments.json').write_text(json.dumps(catalog,indent=2)+'\n')
 (SITE/'experiments.json').write_text(json.dumps(report,indent=2)+'\n')
 (ROOT/'living-room-reconstruction/reports/experiments.json').write_text(json.dumps(report,indent=2)+'\n')
 print([(x['key'],x['metrics'],x['dense_points']) for x in report['experiments']])
if __name__=='__main__':main()
