from pathlib import Path
from PIL import Image
import csv,json,hashlib
R=Path(__file__).resolve().parents[2]/'living_room_frames'
rows=list(csv.DictReader((R/'manifest.csv').open()));files=sorted((R/'full_resolution').glob('*.png'));assert len(files)==500,len(files)
checks=[]
for p in files:
 with Image.open(p) as im:
  im.load();assert im.size==(2160,3840) and im.mode=='RGB',(p,im.size,im.mode)
  im.thumbnail((360,640));im.save(R/'review'/(p.stem+'.jpg'),quality=88)
 checks.append((p.name,p.stat().st_size,hashlib.sha256(p.read_bytes()).hexdigest()))
assert len(set(x[2] for x in checks))==500
(R/'checksums.sha256').write_text(''.join(f'{h}  full_resolution/{n}\n' for n,s,h in checks))
size=sum(s for n,s,h in checks)
(R/'validation.json').write_text(json.dumps(dict(count=500,width=2160,height=3840,mode='RGB',unique_file_hashes=500,total_bytes=size),indent=2))
cards=''.join(f'<a class="card" href="full_resolution/{r["filename"]}" target="_blank"><img loading="lazy" src="review/{Path(r["filename"]).stem}.jpg"><span>{int(k+1):03d} · {float(r["time"]):.2f}s</span></a>' for k,r in enumerate(rows))
(R/'review.html').write_text('''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Living room · 500 selected frames</title><style>body{margin:32px;background:#151719;color:#eee;font:16px system-ui}h1{font-size:30px}p{max-width:850px;line-height:1.6}a{color:#bfe4d5}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:16px}.card{background:#25282b;padding:8px;text-decoration:none;border-radius:8px}.card img{width:100%;aspect-ratio:9/16;object-fit:contain}.card span{display:block;padding:8px}nav{margin:24px 0}</style></head><body><h1>Living room / 500 selected frames</h1><p>Full-resolution PNGs: 2160 × 3840 pixels, upright, 8-bit RGB. Click any preview to open its original PNG. These small JPEG previews are for browsing only; they are not reconstruction inputs.</p><p>Selected across the recording using local sharpness and motion-aware spacing. Window-facing frames 220–237 and plain-wall frames around 153–156 deserve attention. Neighbor matches assess continuity, not proven 3D reconstruction quality. Hidden surfaces cannot be recovered from these views.</p><nav><a href="manifest.csv">Timestamps and quality measurements</a> · <a href="overlap_checks.csv">Neighbor overlap checks</a> · <a href="README.md">Method and limitations</a></nav><div class="grid">'''+cards+'</div></body></html>')
print('Validated',len(files),'unique PNGs;',round(size/1e9,2),'GB')
