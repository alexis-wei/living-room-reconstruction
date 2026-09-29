import av, cv2, numpy as np, json, csv, time
from pathlib import Path
from PIL import Image, ImageDraw
ROOT=Path(__file__).resolve().parents[2]/'living_room_frames'
SRC=ROOT.parent/'alexis_living_room.mov'
cv2.setNumThreads(4)
def upright(a): return cv2.rotate(a,cv2.ROTATE_90_CLOCKWISE)
def analyze():
 c=av.open(str(SRC)); s=c.streams.video[0];s.thread_type='AUTO'
 rows=[]; thumbs=[]; prev=None; start=time.time()
 for i,f in enumerate(c.decode(s)):
  rgb=f.reformat(width=960,height=540,format='rgb24').to_ndarray();g=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
  lap=cv2.Laplacian(g,cv2.CV_32F)
  patches=[float(lap[y:y+180,x:x+240].var()) for y in range(0,540,180) for x in range(0,960,240)]
  small=cv2.resize(g,(480,270),interpolation=cv2.INTER_AREA)
  motion=0.; matches=0
  if prev is not None:
   pts=cv2.goodFeaturesToTrack(prev,250,.015,12)
   if pts is not None:
    nxt,st,err=cv2.calcOpticalFlowPyrLK(prev,small,pts,None)
    good=st.ravel()==1
    if good.sum()>8:
     ds=np.linalg.norm(nxt[good]-pts[good],axis=2).ravel();motion=float(np.median(ds));matches=int(good.sum())
  rows.append(dict(index=i,pts=f.pts,time=float(f.time),sharpness=float(np.median(patches)),dark=float((g<5).mean()),bright=float((g>250).mean()),motion=motion,tracks=matches))
  thumbs.append(cv2.resize(upright(rgb),(180,320),interpolation=cv2.INTER_AREA))
  prev=small
  if i%240==0: print('analyzed',i,'elapsed',round(time.time()-start),flush=True)
 (ROOT/'analysis.json').write_text(json.dumps(rows)); np.save(ROOT/'review'/'thumbnails.npy',np.array(thumbs))
 print('analysis complete',len(rows),flush=True)
def select():
 rows=json.loads((ROOT/'analysis.json').read_text());n=len(rows)
 # Blend time coverage and measured image motion. Limit outliers caused by occlusion or tracking errors.
 m=np.array([r['motion'] for r in rows]); cap=np.percentile(m,95);m=np.minimum(m,cap)
 weights=.55+ .45*m/max(m.mean(),1e-6);cum=np.cumsum(weights)
 edges=np.searchsorted(cum,np.linspace(0,cum[-1],501));edges[0]=0;edges[-1]=n
 chosen=[]
 for a,b in zip(edges[:-1],edges[1:]):
  a=int(a);b=max(int(b),a+1)
  ids=np.arange(a,b);sharp=np.array([rows[j]['sharpness'] for j in ids]);quality=sharp/max(np.median(sharp),1e-6)
  # Modest preference for bin center keeps spacing regular; exposure penalty only for almost empty frames.
  score=quality-.12*np.abs(ids-(a+b-1)/2)/max((b-a)/2,1)
  for k,j in enumerate(ids):score[k]-=max(0,rows[j]['dark']-.45)*2+max(0,rows[j]['bright']-.45)*2
  chosen.append(int(ids[np.argmax(score)]))
 # QA override: source frame 772 has zero SIFT features (blank wall).
 chosen[152]=775  # Nearby frame with 96 detected features at analysis resolution.
 assert len(set(chosen))==500
 (ROOT/'selected_indices.json').write_text(json.dumps(chosen))
 thumbs=np.load(ROOT/'review'/'thumbnails.npy')
 for page in range(10):
  sheet=Image.new('RGB',(1500,1740),'#151719');draw=ImageDraw.Draw(sheet)
  for j,idx in enumerate(chosen[page*50:(page+1)*50]):
   x=(j%10)*150;y=(j//10)*348
   im=Image.fromarray(thumbs[idx]);im.thumbnail((144,316));sheet.paste(im,(x,y))
   draw.text((x+3,y+318),f'{page*50+j+1:03d} | {rows[idx]["time"]:.2f}s',fill='white')
  sheet.save(ROOT/'review'/f'contact_{page+1:02d}.jpg',quality=92)
 with (ROOT/'manifest.csv').open('w') as out:
  wr=csv.DictWriter(out,fieldnames=['filename',*rows[0].keys()]);wr.writeheader()
  for k,idx in enumerate(chosen):wr.writerow(dict(filename=f'frame_{k+1:04d}.png',**rows[idx]))
 print('selected 500; time gaps',np.percentile(np.diff([rows[i]['time'] for i in chosen]),[0,50,95,100]),flush=True)
def extract():
 chosen=json.loads((ROOT/'selected_indices.json').read_text());lookup={v:k for k,v in enumerate(chosen)}
 c=av.open(str(SRC));s=c.streams.video[0];s.thread_type='AUTO';start=time.time()
 for i,f in enumerate(c.decode(s)):
  if i not in lookup:continue
  k=lookup[i];rgb=upright(f.to_ndarray(format='rgb24'))
  dest=ROOT/'full_resolution'/f'frame_{k+1:04d}.png'
  ok=cv2.imwrite(str(dest),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_PNG_COMPRESSION,3]);assert ok
  if k%25==0: print('saved',k+1,'elapsed',round(time.time()-start),flush=True)
 print('extraction complete',flush=True)
if __name__=='__main__':
 import sys
 for mode in sys.argv[1:]:globals()[mode]()
