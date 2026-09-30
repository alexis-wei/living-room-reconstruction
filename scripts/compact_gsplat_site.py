"""Pack private browser splats with half-precision positions/scales, lossless gzip.
All Gaussians retained. Full-quality checkpoints remain local.
"""
from pathlib import Path
import numpy as np,json,gzip
R=Path(__file__).resolve().parents[1];out=R/'site/dist/gaussians';d=json.loads((out/'index.json').read_text())
dt=np.dtype([('p','<f4',(3,)),('s','<f4',(3,)),('rgba','u1',(4,)),('q','u1',(4,))])
for name,m in d.items():
 new=[];stats=[]
 for url in m['chunks']:
  p=R/'site/dist'/url;a=np.fromfile(p,dtype=dt);center=a['p'].mean(0).astype('f4');unit=max(float(np.max(np.abs(a['p']-center))),float(np.max(a['s'])),1e-12)/1024
  pos=((a['p']-center)/unit).astype('<f2');scale=(a['s']/unit).astype('<f2');assert np.isfinite(pos).all() and np.isfinite(scale).all()
  logs=np.log(np.maximum(scale.astype('f4'),1e-30));lo=np.min(np.where(scale>0,logs,np.inf),axis=0);hi=np.max(logs,axis=0);span=np.maximum(hi-lo,1e-5)
  code=np.where(scale>0,1+np.clip(np.rint((logs-lo)/span*254),0,254),0).astype('u1')
  rgb=a['rgba'][:,:3].astype('u2');c=((rgb[:,0]>>3)<<11)|((rgb[:,1]>>2)<<5)|(rgb[:,2]>>3);colors=np.empty((len(a),3),dtype='u1');colors[:,0]=c&255;colors[:,1]=c>>8;colors[:,2]=a['rgba'][:,3]
  header=np.array([*center,unit,*lo,*span],dtype='<f4').tobytes();raw=header+pos.tobytes()+code.tobytes()+colors.tobytes()+a['q'].tobytes()
  q=p.with_suffix('.gsz');q.write_bytes(gzip.compress(raw,compresslevel=6));new.append('gaussians/'+q.name);stats.append(dict(count=len(a),bytes=q.stat().st_size))
  p.unlink() # Only redundant private browser export, never original model.
 m['encoding']='compact16';m['chunks']=new;m['packed_chunks']=stats;m['download_bytes']=sum(x['bytes'] for x in stats)
 print(name,m['download_bytes'],flush=True)
(out/'index.json').write_text(json.dumps(d,indent=2)+'\n')
