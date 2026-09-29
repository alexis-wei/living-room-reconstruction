"""Publish exactly four user-requested source examples per scale to the private Site."""
from pathlib import Path
import csv,json,shutil,hashlib
from PIL import Image
R=Path(__file__).resolve().parents[1];DATA=R.parent/'living_room_frames';OUT=R/'site/dist/examples'
FRAMES=[(1,'Room overview'),(130,'Sofa and windows'),(175,'Dining area'),(405,'Sideboard detail')]
SCALES={'1x':('full_resolution',2160,3840),'2x':('downsample_2x_1080x1920',1080,1920),'4x':('downsample_4x_540x960',540,960),'8x':('downsample_8x_270x480',270,480)}
rows={r['filename']:r for r in csv.DictReader((DATA/'manifest.csv').open())}
manifest={'selection':'Same four source frames in every resolution; exact PNG copies, no recompression.','scales':{}}
for scale,(folder,w,h) in SCALES.items():
 dest=OUT/scale;dest.mkdir(parents=True,exist_ok=True);items=[]
 for number,title in FRAMES:
  name=f'frame_{number:04d}.png';src=DATA/folder/name;dst=dest/name;shutil.copyfile(src,dst)
  with Image.open(dst) as im:assert im.size==(w,h);im.verify()
  digest=hashlib.sha256(dst.read_bytes()).hexdigest();assert digest==hashlib.sha256(src.read_bytes()).hexdigest()
  items.append(dict(frame=number,filename=name,title=title,time_seconds=float(rows[name]['time']),width=w,height=h,url=f'examples/{scale}/{name}',bytes=dst.stat().st_size,sha256=digest))
 manifest['scales'][scale]=items
(OUT/'index.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Verified16 exact PNG copies with aligned frame IDs at four resolutions.')
