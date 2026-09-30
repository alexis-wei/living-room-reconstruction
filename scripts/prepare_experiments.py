"""Select unique recorded frames; save native PNG masters and direct 4x LANCZOS inputs."""
from pathlib import Path
import csv,json,os,hashlib,time
import av,cv2,numpy as np
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'experiments_local'; BASE=ROOT/'living_room_frames'
SPECS=[('room_250',250,None),('room_1000',1000,None),('dining_500',500,[(34.5,37.0),(61.0,81.0)])]
def choose(rows,ids,count):
 motion=np.array([rows[i]['motion'] for i in ids]);motion=np.minimum(motion,np.percentile(motion,95))
 weights=.55+.45*motion/max(motion.mean(),1e-6);cum=weights.cumsum();n=len(ids)
 edges=[0]
 for k in range(1,count):edges.append(min(n-(count-k),max(edges[-1]+1,int(np.searchsorted(cum,cum[-1]*k/count)))))
 edges.append(n);selected=[]
 for a,b in zip(edges[:-1],edges[1:]):
  candidates=ids[a:b];sharp=np.array([rows[i]['sharpness'] for i in candidates]);score=sharp/max(np.median(sharp),1e-6)-.12*abs(np.arange(a,b)-(a+b-1)/2)/max((b-a)/2,1)
  score-=np.array([max(0,rows[i]['dark']-.45)*2+max(0,rows[i]['bright']-.45)*2 for i in candidates])
  selected.append(candidates[int(score.argmax())])
 assert len(set(selected))==count
 return selected

def write_reviews():
 for key,count,_ in SPECS:
  root=OUT/key
  with (root/'manifest.csv').open() as f:rows=list(csv.DictReader(f))
  cards=''.join(f'<a href="full_resolution/{r["filename"]}"><img loading="lazy" src="downsample_4x_540x960/{r["filename"]}" alt="{r["filename"]}"><span>{r["filename"]} / {float(r["time"]):.2f}s</span></a>' for r in rows)
  (root/'review.html').write_text('<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>'+key+' / Frame review</title><style>body{background:#10202e;color:white;font:16px sans-serif;padding:24px}main{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:16px}a{color:white}img{width:100%;display:block}span{display:block;padding:8px}</style><h1>'+key+f' / {count} frames</h1><p>4× previews. Click to open the native PNG. Source video timestamps shown below each image.</p><main>'+cards+'</main>')

def main():
 OUT.mkdir(exist_ok=True);rows=json.loads((BASE/'analysis.json').read_text());thumbs=np.load(BASE/'review/thumbnails.npy')
 baseline=json.loads((BASE/'selected_indices.json').read_text());known={idx:k+1 for k,idx in enumerate(baseline)};lookup={};metadata={}
 for key,count,ranges in SPECS:
  root=OUT/key;root.mkdir(exist_ok=True)
  for folder in ['full_resolution','downsample_4x_540x960','review']:(root/folder).mkdir(exist_ok=True)
  ids=[i for i,r in enumerate(rows) if ranges is None or any(a<=r['time']<=b for a,b in ranges)];selected=choose(rows,ids,count)
  with (root/'manifest.csv').open('w') as f:
   wr=csv.DictWriter(f,fieldnames=['filename',*rows[0].keys()]);wr.writeheader()
   for k,idx in enumerate(selected):
    name=f'frame_{k+1:04d}.png';wr.writerow({'filename':name,**rows[idx]});lookup.setdefault(idx,[]).append((root,name))
  (root/'selected_indices.json').write_text(json.dumps(selected))
  gaps=np.diff([rows[i]['time'] for i in selected]);metadata[key]={'name':key,'count':count,'source':'alexis_living_room.mov','source_ranges_seconds':ranges or [[rows[0]['time'],rows[-1]['time']]],'native_size':[2160,3840],'colmap_size':[540,960],'selection':'55% time / 45% capped optical flow bins; locally sharp exposure-aware candidate; unique source frames','downsample':'Pillow LANCZOS directly from upright native RGB; no crop or enhancement','gap_seconds':{'median':float(np.median(gaps)),'p95':float(np.percentile(gaps,95)),'maximum':float(max(gaps))},'eligible_source_frames':len(ids)}
  (root/'dataset.json').write_text(json.dumps(metadata[key],indent=2))
  # All selected frames, locally reviewable, labeled with source time.
  for page in range((count+49)//50):
   sheet=Image.new('RGB',(1200,1150),'#151719');d=ImageDraw.Draw(sheet)
   for j,idx in enumerate(selected[page*50:(page+1)*50]):
    x=j%10*120;y=j//10*230;im=Image.fromarray(thumbs[idx]);im.thumbnail((115,205));sheet.paste(im,(x,y));d.text((x,y+207),f'{page*50+j+1}: {rows[idx]["time"]:.2f}s',fill='white')
   sheet.save(root/'review'/f'contact_{page+1:02d}.jpg',quality=90)
 (OUT/'datasets.json').write_text(json.dumps(metadata,indent=2));print('Selected:',{k:v['count'] for k,v in metadata.items()},'unique source frames',len(lookup),flush=True)
 c=av.open(str(ROOT/'alexis_living_room.mov'));s=c.streams.video[0];s.thread_type='AUTO';start=time.time();done=0
 for idx,frame in enumerate(c.decode(s)):
  if idx not in lookup:continue
  targets=lookup[idx];firstroot,firstname=targets[0]
  full=firstroot/'full_resolution'/firstname;small=firstroot/'downsample_4x_540x960'/firstname
  if not full.exists():
   if idx in known:os.link(BASE/'full_resolution'/f'frame_{known[idx]:04d}.png',full)
   else:Image.fromarray(cv2.rotate(frame.to_ndarray(format='rgb24'),cv2.ROTATE_90_CLOCKWISE)).save(full,compress_level=2)
  if not small.exists():
   if idx in known:os.link(BASE/'downsample_4x_540x960'/f'frame_{known[idx]:04d}.png',small)
   else:
    with Image.open(full) as im:im.resize((540,960),getattr(Image, 'Resampling', Image).LANCZOS).save(small,compress_level=3)
  for root,name in targets[1:]:
   for folder,src in [('full_resolution',full),('downsample_4x_540x960',small)]:
    dest=root/folder/name
    if not dest.exists():os.link(src,dest)
  done+=1
  if done%50==0:print('Extracted',done,'/',len(lookup),'elapsed',round(time.time()-start),flush=True)
 for key,count,_ in SPECS:
  root=OUT/key;hashes=[]
  for folder,size in [('full_resolution',(2160,3840)),('downsample_4x_540x960',(540,960))]:
   files=sorted((root/folder).glob('*.png'));assert len(files)==count
   for file in files:
    with Image.open(file) as im:im.load();assert im.size==size and im.mode=='RGB'
    hashes.append((hashlib.sha256(file.read_bytes()).hexdigest(),str(file.relative_to(root))))
  assert len(set(h for h,n in hashes if n.startswith('downsample')))==count
  (root/'checksums.sha256').write_text(''.join(f'{h}  {n}\n' for h,n in hashes))
  (root/'validation.json').write_text(json.dumps({'valid':True,'native_count':count,'4x_count':count,'unique_frames':count,'all_pngs_fully_decoded':True},indent=2))
  print(key,'validated',flush=True)
 write_reviews()
 (OUT/'extraction_complete.json').write_text(json.dumps({'state':'complete','datasets':list(metadata)},indent=2))
if __name__=='__main__':main()
