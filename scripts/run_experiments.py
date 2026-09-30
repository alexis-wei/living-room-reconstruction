"""Single-owner sequential COLMAP queue; baseline settings, new datasets, no gsplat."""
import fcntl,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'experiments_local'
KEYS=['room_250','room_1000','dining_500']
def main():
 OUT.mkdir(exist_ok=True)
 lock=(OUT/'queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 env=os.environ.copy();env['PYTHONPATH']=str(ROOT/'.colmap-tools');env['QT_QPA_PLATFORM']='offscreen'
 def status(**kw):
  p=OUT/'queue_status.json';t=p.with_suffix('.tmp');t.write_text(json.dumps({'updated_at':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw},indent=2));t.replace(p)
 status(state='waiting_for_extraction',pid=os.getpid())
 while not (OUT/'extraction_complete.json').exists():time.sleep(10)
 # Obtain calibrated sparse results for all datasets before lengthy dense processing.
 for through in ['bundle_adjustment','fusion']:
  for key in KEYS:
   status(state='running',dataset=key,through=through,pid=os.getpid())
   cmd=[sys.executable,str(ROOT/'living-room-reconstruction/scripts/reconstruct.py'),'--data',str(OUT/key),'--output',str(OUT/key/'colmap'),'--scales','4x','--through',through]
   print('Starting',key,through,flush=True)
   with (OUT/key/f'queue_{through}.log').open('a') as log:result=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
   s=json.loads((OUT/key/'colmap/4x/status.json').read_text())
   if result.returncode or s['stages'].get(through,{}).get('state')!='complete':
    status(state='failed',dataset=key,through=through,pid=os.getpid());raise RuntimeError(f'{key}: {through} failed; inspect stage log')
   print('Finished',key,through,flush=True)
 status(state='complete',datasets=KEYS)
if __name__=='__main__':main()
