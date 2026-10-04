"""Local, lossless selection of 500 evenly covered X5 video frames after 10 s.

Run with PYTHONPATH pointing to the existing .video-tools environment. Review
and native PNG files stay outside both the public code repo and private Site.
"""
from pathlib import Path
import csv, hashlib, json, time
import av, cv2, numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'insta360_living_room_footage.mp4'
OUT = ROOT / 'insta360_frames_500'
COUNT, SKIP = 500, 10.0
cv2.setNumThreads(4)
LANCZOS = getattr(Image, 'Resampling', Image).LANCZOS

def digest(data):
    return hashlib.sha256(data).hexdigest()

def main():
    OUT.mkdir(exist_ok=True)
    review = OUT / 'review'; review.mkdir(exist_ok=True)
    if (OUT / 'validation.json').exists():
        print('Previously validated dataset retained; not overwriting.', flush=True)
        return
    rows, descriptors = [], []
    start = time.monotonic()
    with av.open(str(SOURCE)) as container:
        stream = container.streams.video[0]; stream.thread_type = 'AUTO'
        width, height = stream.width, stream.height
        assert (width, height) == (3840, 2160)
        rate = float(stream.average_rate)
        metadata = dict(source=SOURCE.name, source_bytes=SOURCE.stat().st_size,
                        width=width, height=height, rate=rate,
                        codec=stream.codec_context.name, pixel_format=stream.codec_context.format.name,
                        duration_seconds=container.duration / av.time_base,
                        container_metadata=container.metadata)
        for index, frame in enumerate(container.decode(stream)):
            timestamp = float(frame.time)
            if timestamp < SKIP: continue
            rgb = frame.reformat(width=960, height=540, format='rgb24').to_ndarray()
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            lap = cv2.Laplacian(gray, cv2.CV_32F)
            sharp = float(np.median([lap[y:y+180,x:x+240].var()
                                     for y in range(0,540,180) for x in range(0,960,240)]))
            small = cv2.resize(rgb, (32,18), interpolation=cv2.INTER_AREA)
            descriptors.append(small.astype(np.float32).ravel()/255)
            rows.append(dict(source_frame_index=index, pts=int(frame.pts),
                             timestamp_seconds=timestamp, sharpness_preview=sharp,
                             dark_fraction=float((gray<5).mean()),
                             bright_fraction=float((gray>250).mean())))
            if len(rows)%240 == 0:
                print(f'Analyzed {len(rows)} eligible frames in {time.monotonic()-start:.1f}s', flush=True)
    assert len(rows)>=COUNT
    end = rows[-1]['timestamp_seconds'] + 1/rate
    edges = np.linspace(SKIP, end, COUNT+1)
    times = np.array([r['timestamp_seconds'] for r in rows])
    desc = np.asarray(descriptors)
    selection=[]; last=None
    for slot,(left,right) in enumerate(zip(edges[:-1],edges[1:])):
        ids = np.flatnonzero((times>=left)&(times<right))
        assert len(ids), (slot,left,right)
        sharp = np.array([rows[i]['sharpness_preview'] for i in ids])
        quality = np.clip(sharp/max(float(np.median(sharp)),1e-6),0,3)
        novelty = np.zeros(len(ids)) if last is None else np.mean(np.abs(desc[ids]-desc[last]),axis=1)
        novelty = novelty/max(float(novelty.max()),1e-6)
        centered = 1-np.abs(times[ids]-(left+right)/2)/((right-left)/2)
        score = .55*quality + .30*novelty + .15*centered
        score -= np.array([max(0,rows[i]['dark_fraction']-.45)*2+
                           max(0,rows[i]['bright_fraction']-.45)*2 for i in ids])
        last = int(ids[np.argmax(score)])
        selection.append({**rows[last], 'frame_id':slot+1,
                          'filename':f'frame_{slot+1:04d}.png',
                          'bin_start_seconds':float(left),'bin_end_seconds':float(right),
                          'selection_score':float(score.max())})
    assert len({r['source_frame_index'] for r in selection})==COUNT
    assert all(r['timestamp_seconds']>=SKIP for r in selection)
    (OUT/'analysis.json').write_text(json.dumps(rows,indent=2)+'\n')
    (OUT/'selected_frames.json').write_text(json.dumps(selection,indent=2)+'\n')
    full=OUT/'full_resolution'; quarter=OUT/'downsample_4x_960x540'
    full.mkdir(exist_ok=True); quarter.mkdir(exist_ok=True)
    thumbnails=review/'thumbnails'; thumbnails.mkdir(exist_ok=True)
    lookup={r['source_frame_index']:r for r in selection}
    checks=[]; rgb_hashes=set(); preview_images=[]
    with av.open(str(SOURCE)) as container:
        stream=container.streams.video[0];stream.thread_type='AUTO'
        for index,frame in enumerate(container.decode(stream)):
            if index not in lookup: continue
            row=lookup[index]; name=row['filename']
            assert int(frame.pts)==row['pts']
            rgb=frame.to_ndarray(format='rgb24')
            master=Image.fromarray(rgb)
            reduced=master.resize((960,540),LANCZOS)
            for folder, image in [(full,master),(quarter,reduced)]:
                dst=folder/name
                tmp=dst.with_suffix('.tmp.png');image.save(tmp,format='PNG',compress_level=3)
                with Image.open(tmp) as reopened:
                    reopened.load()
                    assert reopened.size==image.size and reopened.mode=='RGB'
                    assert reopened.tobytes()==image.tobytes()
                tmp.replace(dst)
                checks.append(dict(path=str(dst.relative_to(OUT)),bytes=dst.stat().st_size,
                                   sha256=digest(dst.read_bytes())))
            row['decoded_rgb_sha256']=digest(rgb.tobytes())
            rgb_hashes.add(row['decoded_rgb_sha256'])
            thumb=master.resize((256,144),LANCZOS)
            thumb.save(thumbnails/name.replace('.png','.jpg'),quality=92)
            preview_images.append(thumb)
            if row['frame_id']%25==0:
                print(f'Saved and pixel-verified {row["frame_id"]}/500 native + 4x pairs in {time.monotonic()-start:.1f}s',flush=True)
    assert len(rgb_hashes)==COUNT, 'Exact duplicate source frames found; inspect selection before reconstruction'
    names={r['filename'] for r in selection}
    assert {p.name for p in full.glob('*.png')}==names
    assert {p.name for p in quarter.glob('*.png')}==names
    for page in range(10):
        sheet=Image.new('RGB',(1280,1720),'white');draw=ImageDraw.Draw(sheet)
        for j,image in enumerate(preview_images[page*50:(page+1)*50]):
            x=(j%5)*256;y=(j//5)*172;sheet.paste(image,(x,y))
            row=selection[page*50+j]
            draw.text((x+5,y+148),f'{row["frame_id"]:04d} / {row["timestamp_seconds"]:.3f}s',fill='black')
        sheet.save(review/f'contact_{page+1:02d}.jpg',quality=94)
    with (OUT/'manifest.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(selection[0]));writer.writeheader();writer.writerows(selection)
    (OUT/'checksums.sha256').write_text(''.join(f'{r["sha256"]}  {r["path"]}\n' for r in checks))
    gaps=np.diff([r['timestamp_seconds'] for r in selection])
    policy={'count':COUNT,'exclude_before_seconds':SKIP,'eligible_interval_seconds':[SKIP,end],
            'selection':'One frame per equal-duration temporal bin; weighted preview sharpness (0.55), local visual novelty (0.30), bin-center proximity (0.15), exposure penalties.',
            'limitations':'Even temporal coverage and local appearance diversity are not guaranteed 3D viewpoint uniqueness; nearby views and some overlap remain. No synthetic views, frame blending or super-resolution.',
            'full_resolution':{'folder':full.name,'width':width,'height':height,'count':COUNT,'format':'PNG','mode':'RGB','bit_depth':8},
            'downsample_4x':{'folder':quarter.name,'width':960,'height':540,'count':COUNT,'filter':'Pillow LANCZOS directly from native decoded master'},
            'timestamp_gap_seconds':{'min':float(gaps.min()),'median':float(np.median(gaps)),'max':float(gaps.max())},
            'source':metadata,'filename_alignment':'Identical filenames identify identical source frames.',
            'privacy':'All photographs, video and review files stay local.'}
    (OUT/'dataset.json').write_text(json.dumps(policy,indent=2)+'\n')
    validation={'state':'complete','frames_per_resolution':COUNT,'pngs_checked':len(checks),
                'unique_decoded_rgb_frames':len(rgb_hashes),'first_timestamp_seconds':selection[0]['timestamp_seconds'],
                'last_timestamp_seconds':selection[-1]['timestamp_seconds'],
                'full_png_pixels_equal_decoded_rgb':True,'downsample_png_pixels_equal_lanczos_output':True,
                'total_bytes_by_resolution':{folder:sum(r['bytes'] for r in checks if r['path'].startswith(folder+'/'))
                                            for folder in [full.name,quarter.name]}}
    (OUT/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    figures=''.join(f'<figure><a href="../full_resolution/{r["filename"]}"><img loading="lazy" src="thumbnails/{r["filename"].replace(".png",".jpg")}" width="256" height="144"></a><figcaption>{r["filename"]} · {r["timestamp_seconds"]:.3f}s · <a href="../downsample_4x_960x540/{r["filename"]}">4× PNG</a></figcaption></figure>' for r in selection)
    (review/'index.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>Insta360 X5 / 500 selected frames</title><style>body{font:18px Georgia;margin:32px}main{display:grid;grid-template-columns:repeat(auto-fill,minmax(256px,1fr));gap:20px}figure{margin:0}img{max-width:100%;height:auto}figcaption{font-size:14px}</style><h1>Insta360 X5: 500 selected frames</h1><p>First 10 seconds excluded. Native 3840 × 2160 PNGs and aligned 960 × 540 PNGs. Click each thumbnail for its native original.</p><main>'+figures+'</main></html>')
    print(json.dumps(validation,indent=2),flush=True)

if __name__=='__main__':main()
