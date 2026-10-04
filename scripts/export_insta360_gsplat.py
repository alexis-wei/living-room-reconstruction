"""Export X5 derived prediction previews; native renders/full PLYs stay local.

Run in .gsplat-env. Browser previews use a 960-pixel long edge and lossless
WebP; this resizing is disclosed and never changes the native training inputs.
"""
from pathlib import Path
import datetime,json,math,sys
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'insta360_local/gsplat';SITE=ROOT/'living-room-reconstruction/site/dist';ASSETS=SITE/'insta360-assets'
REPORT=ROOT/'living-room-reconstruction/reports/insta360_gsplat_results.json'
sys.path.insert(0,str(ROOT/'.gsplat-src/gsplat/examples'))
from datasets.colmap import Parser

def load(path,default=None):return json.loads(path.read_text()) if path.exists() else default
def write(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)
def validate_ply(file):
    with file.open('rb') as f:
        props=[];count=None;little=False
        while line:=f.readline():
            words=line.decode('ascii').strip().split()
            if words[:2]==['format','binary_little_endian']:little=True
            if words[:2]==['element','vertex']:count=int(words[2])
            if words[:2]==['property','float']:props.append((words[2],'<f4'))
            if words==['end_header']:offset=f.tell();break
    assert little and count and len(props)==59
    dtype=np.dtype(props);assert file.stat().st_size==offset+count*dtype.itemsize
    array=np.memmap(file,dtype=dtype,mode='r',offset=offset,shape=(count,))
    for i in range(0,count,200000):
        chunk=np.asarray(array[i:i+200000]).view('<f4');assert np.isfinite(chunk).all()
    return count

def main():
    ASSETS.mkdir(exist_ok=True)
    manifests=[load(OUT/f'datasets_{s}.json',{'jobs':[],'skipped_components':[]}) for s in ['1x','4x']]
    jobs=[j for m in manifests for j in m['jobs']]
    primary=[j for j in jobs if j['primary_component']]
    common=set(primary[0]['validation_frames']) if primary else set()
    for j in primary[1:]:common &= set(j['validation_frames'])
    shared=sorted(common)
    views=[shared[i] for i in np.linspace(0,len(shared)-1,min(4,len(shared)),dtype=int)] if shared else []
    catalog={'updated':datetime.datetime.now(datetime.timezone.utc).isoformat(),'models':{},'shared_frames':views,
             'local_viewer':'http://127.0.0.1:8791/','settings':load(ROOT/'insta360_local/configuration.json',{}).get('gsplat'),
             'skipped_components':[j for m in manifests for j in m['skipped_components']],
             'preview_policy':'Prediction-only browser images resized to at most960px long edge then saved as lossless WebP. Native predictions and complete SH3 PLYs remain local.'}
    for job in jobs:
        run=OUT/'runs'/job['name'];status=load(run/'status.json',{'state':'queued'})
        item={k:v for k,v in job.items() if k not in ['dataset','training_frames','validation_frames']}
        item.update(state=status['state'],training_images=len(job['training_frames']),validation_images=len(job['validation_frames']),
                    seconds=status.get('seconds'),elapsed_ns=status.get('elapsed_ns'),metrics=status.get('metrics'),views=[])
        if status['state']=='complete':
            assert (run/'ckpts/ckpt_29999_rank0.pt').is_file()
            metrics=status['metrics'];assert all(math.isfinite(metrics[k]) for k in ['psnr','ssim','lpips'])
            file=run/'ply/point_cloud_29999.ply';ledger=run/'ply_validation.json'
            valid=load(ledger)
            if not valid or valid.get('bytes')!=file.stat().st_size:
                valid={'vertices':validate_ply(file),'bytes':file.stat().st_size};write(ledger,valid)
            item.update(gaussian_count=valid['vertices'],trained_gaussian_count=metrics['num_GS'],
                        excluded_invalid_gaussians=metrics['num_GS']-valid['vertices'],model_bytes=valid['bytes'])
            assert item['excluded_invalid_gaussians']>=0
            parser=Parser(job['dataset'],factor=1,normalize=True,test_every=8)
            selected=views if job['primary_component'] else [job['validation_frames'][i] for i in np.linspace(0,len(job['validation_frames'])-1,min(2,len(job['validation_frames'])),dtype=int)]
            for frame in selected:
                if frame not in job['validation_frames']:continue
                index=job['validation_frames'].index(frame)
                with Image.open(run/f'renders/val_step29999_{index:04d}.png') as canvas:
                    assert canvas.width%2==0
                    prediction=canvas.crop((canvas.width//2,0,canvas.width,canvas.height)).convert('RGB')
                    native=prediction.size
                    prediction.thumbnail((960,960),Image.Resampling.LANCZOS)
                    target=ASSETS/f'{job["name"]}-{Path(frame).stem}.webp';tmp=target.with_suffix('.tmp.webp')
                    prediction.save(tmp,'WEBP',lossless=True,method=6)
                    with Image.open(tmp) as check:assert check.tobytes()==prediction.tobytes()
                    tmp.replace(target)
                ix=parser.image_names.index(frame);pose=parser.camtoworlds[ix];K=parser.Ks_dict[parser.camera_ids[ix]]
                item['views'].append({'frame':frame,'render':'insta360-assets/'+target.name,'width':prediction.width,'height':prediction.height,
                                       'native_width':native[0],'native_height':native[1],
                                       'position':pose[:3,3].tolist(),'forward':pose[:3,2].tolist(),'up':(-pose[:3,1]).tolist(),
                                       'fov':float(2*np.arctan(native[1]/(2*K[1,1]))*180/np.pi)})
            item['scene_scale']=float(parser.scene_scale)
        catalog['models'][job['name']]=item
    write(ASSETS/'gsplat.json',catalog)
    used={Path(v['render']).name for m in catalog['models'].values() for v in m['views']}
    # Only obsolete browser prediction copies owned by this exporter; originals
    # and all complete model/native render files remain outside the Site.
    for file in ASSETS.glob('*_component_*-frame_*.webp'):
        if file.name not in used:file.unlink()
    aggregate={k:v for k,v in catalog.items() if k not in ['models','local_viewer']}
    aggregate['models']={key:{k:v for k,v in item.items() if k not in ['views','scene_scale']} for key,item in catalog['models'].items()}
    write(SITE/'insta360_gsplat_results.json',aggregate)
    write(REPORT,aggregate)
    print(json.dumps({key:row['state'] for key,row in catalog['models'].items()}))

if __name__=='__main__':main()
