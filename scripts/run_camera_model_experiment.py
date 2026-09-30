"""iPhone-aware estimated-intrinsics alternative; preserve baseline and current CUDA queue."""
from pathlib import Path
import argparse,fcntl,json,os,shutil,sqlite3,subprocess,sys,time
import numpy as np
import pycolmap as p
R=Path(__file__).resolve().parents[2]
def save(path,data):
 temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(path)
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--dataset',choices=['original_500','room_1000'],default='original_500');args=parser.parse_args()
 is1000=args.dataset=='room_1000';count=1000 if is1000 else 500
 source_data=R/'experiments_local/room_1000' if is1000 else R/'living_room_frames'
 source_run=source_data/'colmap/4x' if is1000 else R/'reconstruction_local/4x'
 key='iphone13pro_1000_4x' if is1000 else 'iphone13pro_4x'
 D=R/'experiments_local'/key;RUN=D/'colmap/4x'
 D.mkdir(parents=True,exist_ok=True)
 lock=(D/'queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 for folder in ['full_resolution','downsample_4x_540x960']:
  q=D/folder
  if not q.exists():q.symlink_to(source_data/folder,target_is_directory=True)
 for name in ['manifest.csv','selected_indices.json','validation.json']:
  src=source_data/name
  if src.exists() and not (D/name).exists():shutil.copy2(src,D/name)
 # Nominal 26mm-equivalent / 36mm-width convention is an INITIAL GUESS ONLY.
 # The actual video crop and per-frame stabilization are not calibrated by metadata.
 f=960*26/36;params=[f,f,270.,480.,0.,0.,0.,0.]
 config={'camera_model':'OPENCV','initial_params':params,'parameter_order':'fx fy cx cy k1 k2 p1 p2','calibration':'Estimated; not factory or measured calibration','focal_initialization':'960 * 26 / 36 pixels; nominal full-frame-width approximation, refined by COLMAP','video_metadata':{'lens':'iPhone 13 Pro 26mm','aperture':'f/1.5','software':'Blackmagic Cam 3.5.100017','shutter':'1/250','iso':519,'white_balance_kelvin':5300},'shared_camera':True,'refine_focal_and_distortion':True,'refine_principal_point':False,'matched_inputs':f'Same {count} source frames at 540x960; reuse CUDA features/raw matches, redo geometric verification','confounds':'Model complexity and focal initialization both change. No ground-truth geometry; lower reprojection error alone is not accuracy proof.'}
 save(D/'camera_configuration.json',config)
 save(D/'dataset.json',{'name':key,'count':count,'native_size':[2160,3840],'colmap_size':[540,960],'source':'alexis_living_room.mov','selection':f'Exactly the existing {count} frame IDs; linked immutable input files','source_ranges_seconds':json.loads((source_data/'dataset.json').read_text()).get('source_ranges_seconds',[[0,101.8625]]) if (source_data/'dataset.json').exists() else [[0,101.8625]],'camera':config})
 RUN.mkdir(parents=True,exist_ok=True);sp=RUN/'status.json'
 status=json.loads(sp.read_text()) if sp.exists() else {'scale':'4x','stages':{}}
 if status['stages'].get('matching',{}).get('state')!='complete':
  db=RUN/'database.db';source=sqlite3.connect(f'file:{source_run}/database.db?mode=ro',uri=True);dest=sqlite3.connect(db);source.backup(dest);source.close()
  model_id=p.CameraModelId.OPENCV.value
  dest.execute('UPDATE cameras SET model=?,params=?,prior_focal_length=0',(model_id,np.asarray(params,dtype=np.float64).tobytes()));dest.execute('DELETE FROM two_view_geometries');dest.commit();dest.close()
  shutil.copy2(source_run/'pairs.txt',RUN/'pairs.txt')
  status['stages']['features']={'state':'complete','device':'reused CUDA baseline','seconds':0,'exit_code':0};status['stages']['matching']={'state':'running','device':'CPU geometric verification of reused CUDA matches'};save(sp,status)
  start=time.time();opts=p.TwoViewGeometryOptions();opts.ransac.random_seed=0
  p.verify_matches(db,RUN/'pairs.txt',opts)
  status['stages']['matching'].update(state='complete',seconds=round(time.time()-start,2),exit_code=0);save(sp,status)
 command=[sys.executable,str(R/'living-room-reconstruction/scripts/reconstruct.py'),'--data',str(D),'--output',str(D/'colmap'),'--scales','4x','--camera-model','OPENCV','--camera-params',','.join(map(str,params)),'--num-threads','6']
 for through in ['bundle_adjustment','fusion']:
  if through=='fusion':
   save(D/'queue_status.json',{'state':'waiting_for_existing_colmap_queue','through':through,'pid':os.getpid()})
   while True:
    queue=json.loads((R/'experiments_local/queue_status.json').read_text());gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    predecessor=R/'experiments_local/iphone13pro_4x/queue_status.json'
    camera_done=not is1000 or (predecessor.exists() and json.loads(predecessor.read_text()).get('state')=='complete')
    if queue['state']=='complete' and camera_done and not gpu:break
    time.sleep(30)
  save(D/'queue_status.json',{'state':'running','through':through,'pid':os.getpid()})
  with (D/f'queue_{through}.log').open('a') as log:r=subprocess.run(command+['--through',through],stdout=log,stderr=subprocess.STDOUT)
  status=json.loads(sp.read_text())
  if r.returncode or status['stages'].get(through,{}).get('state')!='complete':
   save(D/'queue_status.json',{'state':'failed','through':through});raise RuntimeError('Camera experiment failed; inspect stage logs')
 save(D/'queue_status.json',{'state':'complete'})
if __name__=='__main__':main()
