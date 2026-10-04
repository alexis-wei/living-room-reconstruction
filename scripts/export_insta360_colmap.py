"""Export aggregate X5 results and complete sparse clouds and sampled PRIVATE dense previews."""
from pathlib import Path
import datetime, gzip, json, sqlite3
import numpy as np
import pycolmap as p
from export_viewer import read_ply

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'insta360_local';SITE=ROOT/'living-room-reconstruction/site/dist'
ASSETS=SITE/'insta360-assets'
SPARSE_LIMIT,DENSE_LIMIT=None,15000

def load(path,default=None):return json.loads(path.read_text()) if path.exists() else default
def ply_counts(path):
    if not path.is_file():return None
    counts={}
    with path.open('rb') as stream:
        for _ in range(100):
            line=stream.readline().decode('ascii').strip().split()
            if line[:1]==['element']:counts[line[1]]=int(line[2])
            if line==['end_header']:break
    return counts
def write(path,data):
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(data,indent=2)+'\n');temporary.replace(path)

def cloud(key,xyz,rgb,cameras,limit):
    n=len(xyz);assert n and np.isfinite(xyz).all()
    ix=np.arange(n,dtype=np.int64) if limit is None else np.linspace(0,n-1,min(n,limit),dtype=np.int64)
    center=np.median(xyz,axis=0);radius=max(float(np.percentile(np.linalg.norm(xyz-center,axis=1),95)),1e-6)
    packed=np.column_stack(((xyz[ix]-center)/radius,rgb[ix]/255)).astype('<f4')
    assert np.isfinite(packed).all()
    file=ASSETS/(key+'.bin.gz');tmp=file.with_suffix('.tmp')
    compact=np.empty(len(ix),dtype=[('xyz','<f4',(3,)),('rgb','u1',(3,))]);compact['xyz']=packed[:,:3];compact['rgb']=rgb[ix]
    assert np.array_equal(compact['xyz'],packed[:,:3])
    tmp.write_bytes(gzip.compress(compact.tobytes(),9));tmp.replace(file)
    assert len(gzip.decompress(file.read_bytes()))==len(ix)*15
    return {'url':'insta360-assets/'+file.name,'total_points':n,'displayed_points':len(ix),
            'sampled':n>len(ix),'render_all':True,'format':'xyz-f32-rgb-u8',
            'camera_positions':[((np.asarray(c)-center)/radius).tolist() for c in cameras]}

def main():
    ASSETS.mkdir(exist_ok=True)
    report={'updated':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'dataset':load(ROOT/'insta360_frames_500/dataset.json'),
            'validation':load(ROOT/'insta360_frames_500/validation.json'),
            'settings':load(OUT/'configuration.json'),
            'queue':load(OUT/'queue_status.json',{'state':'not_started'}),'scales':{}}
    catalog={'primary_axis_label':'Image resolution','scales':{}}
    for scale,w,h,folder in [('1x',3840,2160,'full_resolution'),('4x',960,540,'downsample_4x_960x540')]:
        run=OUT/'colmap'/scale
        status=load(run/'status.json',{'stages':{}})
        selected=load(run/'models.json',{}).get('selected_model')
        record={'label':'Full resolution' if scale=='1x' else '4× downsample',
                'width':w,'height':h,'input_images':500,'folder':folder,
                'storage_bytes':sum(f.stat().st_size for f in (ROOT/'insta360_frames_500'/folder).glob('*.png')),
                'stages':status['stages'],'selected_model':selected,'components':[],'sift_keypoints':None,
                'metrics':load(run/'sparse_metrics.json',{}),'dense':ply_counts(run/'dense/fused.ply'),'mesh':ply_counts(run/'dense/mesh_poisson.ply')}
        audit=load(run/'depth_scale_audit.json',{})
        if audit:
            record['geometry_quality']={'state':'degenerate_camera_groups_detected',
                'cameras_with_median_observed_depth_below_1e_minus_5':len(audit.get('degenerate_frames',[])),
                'warning':'Some camera/point groups collapse to near-zero local depth relative to the overall scene. Registration and reprojection error do not demonstrate physical accuracy; dense processing may fail.'}
        mapping_log=run/'mapping.log'
        record['solver_warning_count']=sum('Linear solver failure' in line for line in mapping_log.open()) if mapping_log.exists() else 0
        if record['metrics'].get('mean_reprojection_error') is not None:
            record['metrics']['full_resolution_equivalent_error_px']=record['metrics']['mean_reprojection_error']*int(scale[:-1])
        if status['stages'].get('features',{}).get('state')=='complete':
            with sqlite3.connect(f'file:{run/"database.db"}?mode=ro',uri=True) as db:
                record['sift_keypoints']=db.execute('SELECT SUM(rows) FROM keypoints').fetchone()[0]
                if status['stages'].get('matching',{}).get('state')=='complete':
                    record['matching_counts']={'matched_pairs':db.execute('SELECT COUNT(*) FROM matches WHERE rows>0').fetchone()[0],
                                               'verified_pairs':db.execute('SELECT COUNT(*) FROM two_view_geometries WHERE rows>0').fetchone()[0],
                                               'verified_inliers':db.execute('SELECT SUM(rows) FROM two_view_geometries').fetchone()[0]}
        models=[];union=set()
        if status['stages'].get('mapping',{}).get('state')=='complete' and status['stages'].get('bundle_adjustment',{}).get('state')!='running':
            for path in sorted((run/'sparse').glob('*')):
                if not (path/'cameras.bin').is_file():continue
                rec=p.Reconstruction(path);n=rec.num_points3D()
                ids=sorted(int(im.name.split('_')[1].split('.')[0]) for im in rec.images.values())
                item={'id':path.name,'kind':'sparse','registered_images':rec.num_reg_images(),
                      'frame_ids':ids,'total_points':n,'substantial':rec.num_reg_images()>=10 and n>=100,
                      'mean_reprojection_error':rec.compute_mean_reprojection_error() if n else None}
                if n:
                    pts=list(rec.points3D.values())
                    item.update(cloud(f'{scale}-sparse-{path.name}',np.asarray([v.xyz for v in pts]),
                                      np.asarray([v.color for v in pts]),[im.projection_center() for im in rec.images.values()],SPARSE_LIMIT))
                    union.update(ids)
                models.append(item)
                record['components'].append({k:v for k,v in item.items() if k not in ['frame_ids','camera_positions','url']})
            models.sort(key=lambda m:(m['id']!=str(selected),-m['registered_images']))
        dense=None
        if status['stages'].get('fusion',{}).get('state')=='complete':
            xyz,rgb=read_ply(run/'dense/fused.ply')
            if len(xyz):dense={'id':'dense','kind':'dense',**cloud(f'{scale}-dense',xyz,rgb,[],DENSE_LIMIT)}
        catalog['scales'][scale]={'label':record['label'],'input_images':500,'components':models,'dense':dense,
                                  'registered_union':len(union),'missing_frame_ids':sorted(set(range(1,501))-union),'ready':bool(models)}
        record['registered_union']=len(union) if status['stages'].get('mapping',{}).get('state')=='complete' else None
        report['scales'][scale]=record
    write(ASSETS/'clouds.json',catalog)
    write(SITE/'insta360_results.json',report)
    write(ROOT/'living-room-reconstruction/reports/insta360_results.json',report)
    print(json.dumps({s:{'features':d['sift_keypoints'],'components':len(d['components']),'dense':d['dense']} for s,d in report['scales'].items()}))

if __name__=='__main__':main()
