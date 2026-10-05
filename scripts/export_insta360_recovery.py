"""Stage recovery results privately; never mutate the shared Site checkout.

Default run uses modern PyCOLMAP. --gsplat uses .gsplat-env after training.
"""
from pathlib import Path
import argparse,datetime,json
ROOT=Path(__file__).resolve().parents[2];REC=ROOT/'insta360_local/recovery';STAGE=REC/'publication_stage'
def load(path,default=None):return json.loads(path.read_text()) if path.exists() else default
def write(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2)+'\n')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--gsplat',action='store_true');args=ap.parse_args()
 assets=STAGE/'insta360-recovery-assets';assets.mkdir(parents=True,exist_ok=True)
 if args.gsplat:
  import export_insta360_gsplat as ex
  ex.OUT=REC/'gsplat';ex.SITE=STAGE;ex.ASSETS=assets;ex.REPORT=ROOT/'living-room-reconstruction/reports/insta360_recovery_gsplat_results.json'
  # Existing exporter writes only this report into repository/reports. All
  # private derived images and catalogs use staging, never shared Site.
  ex.main()
  (STAGE/'insta360_gsplat_results.json').replace(STAGE/'insta360_recovery_gsplat_results.json')
  for file in [assets/'gsplat.json']:
   data=json.loads(file.read_text());data['local_viewer']='http://127.0.0.1:8793/'
   for key,model in data['models'].items():
    review=load(REC/'gsplat/runs'/key/'visual_review.json')
    if review:model['visual_quality']=review
    audit=load(REC/'gsplat/runs'/key/'full_validation.json')
    if audit:model['validation_audit']={k:v for k,v in audit.items() if k not in ['native_validation_renders']}
    for view in model['views']:view['render']=view['render'].replace('insta360-assets/','insta360-recovery-assets/')
   write(file,data)
  aggregate=load(STAGE/'insta360_recovery_gsplat_results.json')
  for key,model in data['models'].items():
   for field in ['visual_quality','validation_audit']:
    if field in model:
     aggregate['models'][key][field]={k:v for k,v in model[field].items() if k!='reviewed_validation_frames'}
     if field=='visual_quality':aggregate['models'][key][field]['reviewed_views']=len(model[field].get('reviewed_validation_frames',[]))
  write(STAGE/'insta360_recovery_gsplat_results.json',aggregate);write(ex.REPORT,aggregate)
  return
 import numpy as np,pycolmap as p
 import export_insta360_colmap as ex
 ex.ASSETS=assets
 report={'updated':datetime.datetime.now(datetime.timezone.utc).isoformat(),'queue':load(REC/'queue_status.json',{}),'policy':'Separate screened partial recovery; original 500-view sparse models and all photographs preserved. Intrinsics unchanged; no new BA.','scales':{}}
 catalog={'primary_axis_label':'Recovery resolution','scales':{}}
 for scale in ['1x','4x']:
  run=REC/'colmap'/scale;validation=load(run/'recovery_validation.json');metrics=load(run/'sparse_metrics.json');m=p.Reconstruction(run/'sparse/0');points=list(m.points3D.values());ids=sorted(int(im.name.split('_')[1].split('.')[0]) for im in m.images.values());label=('Full resolution' if scale=='1x' else '4×')+' · screened recovery'
  cloud=ex.cloud(f'{scale}-sparse-0',np.array([v.xyz for v in points]),np.array([v.color for v in points]),[im.projection_center() for im in m.images.values()],None);cloud['url']=cloud['url'].replace('insta360-assets/','insta360-recovery-assets/')
  component={'id':'0','kind':'sparse','registered_images':m.num_reg_images(),'frame_ids':ids,'substantial':True,'mean_reprojection_error':metrics['mean_reprojection_error'],**cloud}
  status=load(run/'status.json',{'stages':{}});dense=None
  if status['stages'].get('fusion',{}).get('state')=='complete':
   xyz,rgb=ex.read_ply(run/'dense/fused.ply');dense={'id':'dense','kind':'dense',**ex.cloud(f'{scale}-dense',xyz,rgb,[],15000)};dense['url']=dense['url'].replace('insta360-assets/','insta360-recovery-assets/')
  catalog['scales'][scale]={'label':label,'input_images':500,'components':[component],'dense':dense,'registered_union':len(ids),'missing_frame_ids':sorted(set(range(1,501))-set(ids)),'ready':True}
  report['scales'][scale]={'label':label,'input_images':500,'usable_images':len(ids),'excluded_images':500-len(ids),'criterion':validation['criterion'],'metrics':metrics,'baseline_metrics':load(ROOT/'insta360_local/colmap'/scale/'sparse_metrics.json'),'stages':status['stages'],'mvs_validation':load(run/'mvs_validation.json'),'dense':ex.ply_counts(run/'dense/fused.ply'),'mesh':ex.ply_counts(run/'dense/mesh_poisson.ply')}
 write(assets/'clouds.json',catalog);write(STAGE/'insta360_recovery_results.json',report);write(ROOT/'living-room-reconstruction/reports/insta360_recovery_results.json',report)
 print(json.dumps({'stage':str(STAGE),'scales':{s:{'cameras':d['usable_images'],'points':d['metrics']['points3D']} for s,d in report['scales'].items()}}))
if __name__=='__main__':main()
