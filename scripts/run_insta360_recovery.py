"""Continue only separate screened datasets, waiting for existing GPU queue ownership."""
from pathlib import Path
import fcntl,json,traceback
import run_insta360_pipeline as pipeline
from run_matching_gsplat import save,now,train
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'insta360_local/recovery'
def main():
 pipeline.OUT=OUT
 (OUT/'gsplat').mkdir(exist_ok=True)
 save(OUT/'gsplat/holdout_frames.json',[f'frame_{i:04d}.png' for i in range(1,501,8)])
 save(OUT/'queue_status.json',{'state':'waiting_for_queue','requested_at':now(),'policy':'Separate geometry-screened recovery; never preempt another lock owner.'})
 with (ROOT/'gsplat_local/queue.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  save(OUT/'queue_status.json',{'state':'running','started_at':now()})
  for scale in ['4x','1x']:
   if scale=='4x':pipeline.colmap(scale,['depth_maps','fusion','mesh'])
   for job in pipeline.prepare_gsplat(scale):
    job['label']+=' · geometry-screened partial recovery'
    train(OUT/'gsplat',job)
   if scale=='1x':pipeline.colmap(scale,['depth_maps','fusion','mesh'])
  save(OUT/'queue_status.json',{'state':'complete','finished_at':now()})
if __name__=='__main__':
 try:main()
 except Exception as exc:
  save(OUT/'queue_status.json',{'state':'failed','failed_at':now(),'error':str(exc)})
  raise
