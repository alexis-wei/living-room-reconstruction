"""CPU-only durable staging watcher. Never launches training/CUDA or publishes."""
from pathlib import Path
import datetime,fcntl,json,os,subprocess,time
ROOT=Path(__file__).resolve().parents[2];REC=ROOT/'insta360_local/recovery';HERE=Path(__file__).parent
def load(path):return json.loads(path.read_text())
def save(value):
 tmp=REC/'local_monitor_status.tmp';tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(REC/'local_monitor_status.json')
def main():
 with (REC/'local_monitor.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);env=os.environ.copy();env['PYTHONPATH']=str(ROOT/'.colmap-tools');last=None;milestones=set()
  while True:
   queue=load(REC/'queue_status.json');status=load(REC/'colmap/1x/status.json')['stages'];snapshot={k:v['state'] for k,v in status.items()}
   record={'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':os.getpid(),'queue':queue,'native_stages':snapshot,'state':'monitoring','staged_milestones':sorted(milestones)}
   if snapshot!=last:print(json.dumps(record),flush=True);last=snapshot
   for stage in ['fusion','mesh']:
    if snapshot.get(stage)=='complete' and stage not in milestones:
     subprocess.run(['python3',str(HERE/'validate_insta360_recovery_dense.py'),'--scale','1x'],cwd=ROOT,env=env,check=True)
     subprocess.run(['python3',str(HERE/'export_insta360_recovery.py')],cwd=ROOT,env=env,check=True);milestones.add(stage);record['staged_milestones']=sorted(milestones)
   if queue['state'] in ['complete','failed']:
    record['state']='complete_staged' if queue['state']=='complete' else 'pipeline_failed';save(record);print(json.dumps(record),flush=True);return
   save(record);time.sleep(60)
if __name__=='__main__':
 try:main()
 except Exception as exc:
  save({'state':'monitor_error','error':str(exc),'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat()});raise
