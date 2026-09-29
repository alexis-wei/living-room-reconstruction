from pathlib import Path
import av,cv2,numpy as np,json,csv
R=Path(__file__).resolve().parents[2]/'living_room_frames'
cv2.setNumThreads(3)
ids=json.loads((R/'selected_indices.json').read_text());lookup={v:k for k,v in enumerate(ids)}
sift=cv2.SIFT_create(nfeatures=1800);matcher=cv2.BFMatcher();prev=None;out=[]
c=av.open(str(R.parent/'alexis_living_room.mov'));c.streams.video[0].thread_type='AUTO'
for i,f in enumerate(c.decode(video=0)):
 if i not in lookup:continue
 g=f.reformat(width=960,height=540,format='gray').to_ndarray(); kp,des=sift.detectAndCompute(g,None)
 if prev is not None:
  pk,pd=prev;good=[]
  if pd is not None and des is not None:
   good=[a for pair in matcher.knnMatch(pd,des,k=2) if len(pair)==2 for a,b in [pair] if a.distance<.75*b.distance]
  inl=0
  if len(good)>=8:
   a=np.float32([pk[m.queryIdx].pt for m in good]);b=np.float32([kp[m.trainIdx].pt for m in good])
   mat,mask=cv2.findFundamentalMat(a,b,cv2.FM_RANSAC,2.,.99)
   if mask is not None:inl=int(mask.sum())
  out.append(dict(frame=lookup[i]+1,previous_frame=lookup[i],matches=len(good),geometric_inliers=inl))
 prev=(kp,des)
 if lookup[i]%100==0:print('overlap checked',lookup[i]+1,flush=True)
with (R/'overlap_checks.csv').open('w') as fh:
 wr=csv.DictWriter(fh,fieldnames=out[0].keys());wr.writeheader();wr.writerows(out)
print('inlier percentiles',np.percentile([x['geometric_inliers'] for x in out],[0,5,50,95]),'weak',[x for x in out if x['geometric_inliers']<20],flush=True)
