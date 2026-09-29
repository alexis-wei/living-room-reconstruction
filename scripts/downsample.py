"""Create aligned PNG training scales directly from the approved masters."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import csv, json, hashlib

ROOT=Path(__file__).resolve().parents[2]/'living_room_frames'
SCALES={2:(1080,1920),4:(540,960),8:(270,480)}
DIRS={s:ROOT/f'downsample_{s}x_{w}x{h}' for s,(w,h) in SCALES.items()}
for p in DIRS.values():p.mkdir(exist_ok=True)
masters=sorted((ROOT/'full_resolution').glob('frame_*.png'))
assert len(masters)==500

def make(src):
 results=[]
 with Image.open(src) as original:
  original.load()
  assert original.size==(2160,3840) and original.mode=='RGB'
  for factor,size in SCALES.items():
   # Independent antialiased resize from the master, never a chained reduction.
   resized=original.resize(size,Image.Resampling.LANCZOS)
   dst=DIRS[factor]/src.name
   resized.save(dst,format='PNG',compress_level=3)
   with Image.open(dst) as check:
    check.load()
    assert check.size==size and check.mode=='RGB'
    assert check.tobytes()==resized.tobytes()
   results.append((factor,src.name,dst.stat().st_size,hashlib.sha256(dst.read_bytes()).hexdigest()))
 return results

checks=[]
with ThreadPoolExecutor(max_workers=4) as pool:
 for i,result in enumerate(pool.map(make,masters),1):
  checks.extend(result)
  if i%50==0:print(f'Created and verified all three scales for {i}/500 frames',flush=True)

source_rows=list(csv.DictReader((ROOT/'manifest.csv').open()))
with (ROOT/'dataset_manifest.csv').open('w',newline='') as out:
 fields=['frame_id','source_frame_index','timestamp_seconds','image_1x','image_2x','image_4x','image_8x']
 writer=csv.DictWriter(out,fieldnames=fields);writer.writeheader()
 for row in source_rows:
  name=row['filename']
  writer.writerow(dict(frame_id=Path(name).stem,source_frame_index=row['index'],timestamp_seconds=row['time'],image_1x=f'full_resolution/{name}',**{f'image_{s}x':f'{DIRS[s].name}/{name}' for s in SCALES}))

scales={}
for s,(w,h) in SCALES.items():
 rows=[r for r in checks if r[0]==s]
 assert len(list(DIRS[s].glob('*.png')))==500
 assert {p.name for p in DIRS[s].glob('*.png')}=={p.name for p in masters}
 assert len(set(r[3] for r in rows))==500
 scales[str(s)]=dict(directory=DIRS[s].name,width=w,height=h,count=500,total_bytes=sum(r[2] for r in rows),format='PNG',mode='RGB',bit_depth_per_channel=8)
(ROOT/'downsample_checksums.sha256').write_text(''.join(f'{digest}  {DIRS[s].name}/{name}\n' for s,name,_,digest in checks))
metadata=dict(frame_count=500,master=dict(directory='full_resolution',width=2160,height=3840),scales=scales,resize_filter='Pillow LANCZOS',source='Each scale is independently resized from its full-resolution master.',filename_alignment='Identical filenames refer to identical source frames across every scale.',manifest='dataset_manifest.csv',split='No train/validation/test split assigned.')
(ROOT/'dataset.json').write_text(json.dumps(metadata,indent=2)+'\n')
print(json.dumps(scales,indent=2),flush=True)
