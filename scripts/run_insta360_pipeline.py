"""Serial RTX4090 COLMAP and gsplat queue; original project runs untouched.

CPU launches this script. Each stage records an accurate monotonic timer and
skips only verified completed work. Source selection must pass validation first.
"""
from pathlib import Path
import datetime, fcntl, json, math, os, subprocess, time
from train_gsplat_queue import environment
from run_matching_gsplat import other_gpu_jobs, train, save, now

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).parent
OUT=ROOT/'insta360_local'
SPECS={'4x':(960,540),'1x':(3840,2160)}
COLMAP_STAGES=['features','matching','mapping','bundle_adjustment','undistortion','depth_maps','fusion','mesh']

def colmap(scale, stages):
    folder=OUT/'colmap'/scale;folder.mkdir(parents=True,exist_ok=True)
    status_path=folder/'status.json'
    status=json.loads(status_path.read_text()) if status_path.exists() else {'scale':scale,'stages':{}}
    for stage in stages:
        old=status['stages'].get(stage,{})
        if old.get('state')=='complete':continue
        if old.get('state')=='failed':raise RuntimeError(f'Preserve and diagnose failed {scale}/{stage} before retry')
        if stage in ['features','matching','depth_maps']:
            status['stages'][stage]={'state':'waiting_for_gpu','device':'CUDA','requested_at':now()};save(status_path,status)
            while other_gpu_jobs():time.sleep(15)
        entry={'state':'running','started_at':now(),'device':'CUDA' if stage in ['features','matching','depth_maps'] else 'CPU'}
        status['stages'][stage]=entry;save(status_path,status)
        print(f'{scale}/{stage} started',flush=True)
        env=os.environ.copy();env['PYTHONPATH']=str(ROOT/'.colmap-tools')
        if stage=='fusion':env['COLMAP_FUSION_CACHE_GB']='12'
        start=time.perf_counter_ns()
        with (folder/f'{stage}.log').open('w') as log:
            result=subprocess.run(['python3',str(HERE/'run_insta360_colmap_stage.py'),'--scale',scale,'--stage',stage],env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        elapsed=time.perf_counter_ns()-start
        entry.update(state='complete' if result.returncode==0 else 'failed',finished_at=now(),elapsed_ns=elapsed,seconds=elapsed/1e9,exit_code=result.returncode)
        save(status_path,status);print(f'{scale}/{stage} {entry["state"]} {entry["seconds"]:.6f}s',flush=True)
        if result.returncode:raise RuntimeError(f'{scale}/{stage} failed')

def prepare_gsplat(scale):
    # COLMAP undistortion supplies pinhole images AND the correspondingly
    # transformed camera matrices; never feed fisheye pixels to pinhole cameras.
    run=OUT/'colmap'/scale
    component_list=json.loads((run/'models.json').read_text())
    jobs=[];skipped=[]
    env=os.environ.copy();env['PYTHONPATH']=str(ROOT/'.colmap-tools')
    for item in component_list['models']:
        key=f'{scale}_component_{item["id"]}'
        common={'name':key,'scale':scale,'label':'Full resolution' if scale=='1x' else '4× downsample',
                'component':str(item['id']),'registered_images':item['registered_images'],
                'sparse_points':item['points3D'],'primary_component':item['id']==component_list['selected_model']}
        if common['primary_component']:
            refined=json.loads((run/'sparse_metrics.json').read_text())
            common.update(registered_images=refined['registered_images'],sparse_points=refined['points3D'])
        if item['registered_images']<10 or item['points3D']<100:
            skipped.append({**common,'reason':'Fewer than 10 cameras or 100 points; COLMAP result retained, insufficient for reliable gsplat.'});continue
        target=OUT/'gsplat/datasets'/key;target.mkdir(parents=True,exist_ok=True)
        if common['primary_component']:
            for name in ['images','sparse']:
                link=target/name;destination=run/'dense'/name
                if not link.exists():link.symlink_to(destination,target_is_directory=True)
                assert link.resolve()==destination.resolve()
        elif not (target/'sparse/cameras.bin').exists():
            code='import pycolmap as p;from pathlib import Path;import sys;o=p.UndistortCameraOptions();o.max_image_size=int(sys.argv[4]);o.max_scale=1.;p.undistort_images(Path(sys.argv[1]),Path(sys.argv[2]),Path(sys.argv[3]),undistort_options=o,num_patch_match_src_images=10,num_threads=8)'
            images=ROOT/'insta360_frames_500'/('full_resolution' if scale=='1x' else 'downsample_4x_960x540')
            subprocess.run(['python3','-c',code,str(target),str(run/'sparse'/str(item['id'])),str(images),str(SPECS[scale][0])],env=env,check=True)
        common['dataset']=str(target)
        # Validate with the trainer's own legacy parser in its separate env.
        validation=OUT/'gsplat'/f'dataset_validation_{key}.json'
        code='import sys,json,numpy as np;from pathlib import Path;sys.path.insert(0,sys.argv[3]);from datasets.colmap import Parser;p=Parser(sys.argv[1],factor=1,normalize=True,test_every=8);assert np.isfinite(p.camtoworlds).all() and np.isfinite(p.points).all();held=set(json.loads(Path(sys.argv[4]).read_text()));train=[n for n in p.image_names if n not in held];val=[n for n in p.image_names if n in held];assert len(train)>=2 and val;Path(sys.argv[2]).write_text(json.dumps(dict(training_frames=train,validation_frames=val,undistorted_sizes=[list(v) for v in p.imsize_dict.values()],registered_images=len(p.image_names),points=len(p.points)),indent=2))'
        subprocess.run([str(ROOT/'.gsplat-env/bin/python'),'-c',code,str(target),str(validation),str(ROOT/'.gsplat-src/gsplat/examples'),str(OUT/'gsplat/holdout_frames.json')],env=environment(),check=True)
        parsed=json.loads(validation.read_text())
        assert parsed['registered_images']==common['registered_images']
        assert parsed['points']==common['sparse_points']
        common.update({k:parsed[k] for k in ['training_frames','validation_frames','undistorted_sizes']})
        jobs.append(common)
    jobs.sort(key=lambda j:(not j['primary_component'],-j['registered_images']))
    save(OUT/'gsplat'/f'datasets_{scale}.json',{'jobs':jobs,'skipped_components':skipped})
    return jobs

def main():
    OUT.mkdir(exist_ok=True); (OUT/'gsplat').mkdir(exist_ok=True)
    assert json.loads((ROOT/'insta360_frames_500/validation.json').read_text())['state']=='complete'
    with (ROOT/'gsplat_local/queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        config={'source':'insta360_living_room_footage.mp4','trim_before_seconds':10,'frames':500,
                'camera_model':'OPENCV_FISHEYE','parameter_order':['fx','fy','cx','cy','k1','k2','k3','k4'],
                'intrinsics_status':'Estimated for this flat MegaView export, not factory or checkerboard calibration.',
                'initialization':'Equidistant f=width/radians(170), fx=fy; centered principal point, k1..k4=0. Nominal FOV interpretation is an assumption.',
                'shared_intrinsics':True,'principal_point_refined':False,'focal_and_distortion_refined':True,
                'matching':'15 neighbors, offsets 20/40/80/160/320, anchors every 20 frames; CUDA guided matches and CPU geometry verification.',
                'features':'SIFT max8192 at native dimensions per scale; CUDA','mapping':'CPU incremental mapper seed0,12threads; global BA and observation filtering4px/1.5deg',
                'dense':'COLMAP fisheye-to-pinhole undistortion at native long edge;10sourceviews;CUDA geometric PatchMatch5iterations;CPU fusion12GBcache;Poisson depth10',
                'gsplat':{'version':'1.5.3','steps':30000,'data_factor':1,'batch_size':1,'packed':True,'sh_degree':3,'pose_opt':False,'appearance_opt':False,'ssim_lambda':.2,'seed':42,'holdout':'Fixed source frames1,9,...497; SfM used all views. Images are original-derived calibrated undistorted pixels with matching pinhole cameras.'},
                'sources':['https://www.insta360.com/blog/tips/understanding-8K-360-video.html','https://onlinemanual.insta360.com/studio/en-us/operation-guide/edit-function/adjust-the-perspective','https://colmap.github.io/cameras.html','https://docs.opencv.org/4.x/db/d58/group__calib3d__fisheye.html'],
                'initial_camera_params':{scale:[w/math.radians(170),w/math.radians(170),w/2,h/2,0,0,0,0] for scale,(w,h) in SPECS.items()},
                'privacy':'Native images, full clouds, meshes and checkpoints remain local. Private Site gets derived previews only.'}
        save(OUT/'configuration.json',config)
        save(OUT/'gsplat/holdout_frames.json',[f'frame_{i:04d}.png' for i in range(1,501,8)])
        os.environ['GSPLAT_HOLDOUT_FILE']=str(OUT/'gsplat/holdout_frames.json')
        # train() sets its own holdout path from the given root.
        save(OUT/'queue_status.json',{'state':'running','started_at':now(),'order':'Sparse both scales;4x dense+gsplat;full undistortion+gsplat;full dense'})
        for scale in SPECS:colmap(scale,COLMAP_STAGES[:4])
        for scale in SPECS:
            # gsplat needs refined cameras and calibrated photographs, not a
            # dense cloud. Deliver native Gaussian training before the much
            # longer native stereo stage without overlapping CUDA jobs.
            colmap(scale,COLMAP_STAGES[4:] if scale=='4x' else ['undistortion'])
            for job in prepare_gsplat(scale):
                train(OUT/'gsplat',job)
            if scale=='1x':colmap(scale,['depth_maps','fusion','mesh'])
        save(OUT/'queue_status.json',{'state':'complete','finished_at':now()})
        print('Insta360 COLMAP and gsplat queue complete',flush=True)

if __name__=='__main__':
    try:main()
    except Exception as exc:
        if OUT.exists():save(OUT/'queue_status.json',{'state':'failed','failed_at':now(),'error':str(exc)})
        raise
