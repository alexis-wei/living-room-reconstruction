"""Local-only COLMAP pipeline. Run using the pinned pycolmap-cuda12 environment."""
import argparse, json, os, subprocess, sys, time, sqlite3, shutil, math
from pathlib import Path

REPO=Path(__file__).resolve().parents[1]
DEFAULT_DATA=REPO.parent/'living_room_frames'
SCALES={'1x':('full_resolution',2160,3840),'2x':('downsample_2x_1080x1920',1080,1920),'4x':('downsample_4x_540x960',540,960),'8x':('downsample_8x_270x480',270,480)}
STAGES=['features','matching','mapping','bundle_adjustment','undistortion','depth_maps','fusion','mesh']

def atomic_json(path,data):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2,default=str)+'\n');tmp.replace(path)

def execute_stage(args):
 import pycolmap as p
 scale=args.scale; folder,w,h=SCALES[scale];images=args.data/folder;run=args.output/scale;run.mkdir(parents=True,exist_ok=True)
 db=run/'database.db';sparse=run/'sparse';dense=run/'dense';sparse.mkdir(exist_ok=True)
 stage=args.stage
 if stage=='features':
  assert p.has_cuda,'CUDA-enabled PyCOLMAP required'
  opts=p.FeatureExtractionOptions();opts.max_image_size=max(w,h);opts.num_threads=args.num_threads;opts.use_gpu=True;opts.gpu_index='0';opts.sift.max_num_features=8192
  reader=p.ImageReaderOptions();reader.camera_model=args.camera_model
  if args.camera_params:reader.camera_params=args.camera_params
  p.extract_features(db,images,camera_mode=p.CameraMode.SINGLE,reader_options=reader,extraction_options=opts,device=p.Device.cuda)
 elif stage=='matching':
  names=sorted(x.name for x in images.glob('frame_*.png'));pairs=set()
  for i in range(len(names)):
   for d in list(range(1,16))+[20,40,80,160,320]:
    if i+d<len(names):pairs.add((i,i+d))
   for j in range(0,len(names),20):
    if i!=j:pairs.add(tuple(sorted((i,j))))
  pairfile=run/'pairs.txt';pairfile.write_text(''.join(f'{names[i]} {names[j]}\n' for i,j in sorted(pairs)))
  opts=p.FeatureMatchingOptions();opts.use_gpu=True;opts.gpu_index='0';opts.num_threads=args.num_threads;opts.guided_matching=True
  pairing=p.ImportedPairingOptions();pairing.match_list_path=pairfile;pairing.block_size=100
  p.match_image_pairs(db,matching_options=opts,pairing_options=pairing,device=p.Device.cuda)
 elif stage=='mapping':
  opts=p.IncrementalPipelineOptions();opts.num_threads=args.num_threads;opts.random_seed=0;opts.ba_use_gpu=False
  models=p.incremental_mapping(db,images,sparse,options=opts)
  if not models:raise RuntimeError('COLMAP did not recover a sparse model')
  ranked=sorted(models.items(),key=lambda kv:kv[1].num_reg_images(),reverse=True)
  chosen,model=ranked[0]
  stats={'selected_model':int(chosen),'models':[{'id':int(k),'registered_images':m.num_reg_images(),'points3D':m.num_points3D(),'mean_reprojection_error':m.compute_mean_reprojection_error()} for k,m in ranked],'input_images':len(list(images.glob('*.png')))}
  atomic_json(run/'models.json',stats)
  model.export_PLY(run/'sparse_points.ply')
 elif stage=='bundle_adjustment':
  sel=json.loads((run/'models.json').read_text())['selected_model'];mp=sparse/str(sel);m=p.Reconstruction(mp)
  backup=run/'sparse_before_bundle_adjustment'/str(sel)
  if not backup.exists():shutil.copytree(mp,backup)
  opts=p.BundleAdjustmentOptions();opts.ceres.use_gpu=False;opts.ceres.solver_options.num_threads=args.num_threads
  p.bundle_adjustment(m,opts)
  # Global BA can move observations behind cameras. Reapply mapper-quality
  # filtering before reporting errors or using the model for dense stereo.
  filtered=p.ObservationManager(m).filter_all_points3D(4.0,1.5)
  m.update_point_3d_errors()
  if not m.num_points3D() or not math.isfinite(m.compute_mean_reprojection_error()):raise RuntimeError('Invalid sparse geometry after bundle adjustment')
  atomic_json(run/'bundle_adjustment_filter.json',dict(filtered_observations=filtered,max_reprojection_error_px=4.0,min_triangulation_angle_degrees=1.5))
  m.write(mp);m.export_PLY(run/'sparse_points.ply')
  atomic_json(run/'sparse_metrics.json',dict(registered_images=m.num_reg_images(),points3D=m.num_points3D(),mean_reprojection_error=m.compute_mean_reprojection_error(),mean_track_length=m.compute_mean_track_length(),camera_model=str(next(iter(m.cameras.values())).model),cameras=[cam.todict() for cam in m.cameras.values()],unregistered_images=sorted(set(x.name for x in images.glob('*.png'))-set(im.name for im in m.images.values()))))
 elif stage=='undistortion':
  sel=json.loads((run/'models.json').read_text())['selected_model'];opts=p.UndistortCameraOptions();opts.max_image_size=max(w,h);opts.max_scale=1.0
  p.undistort_images(dense,sparse/str(sel),images,num_patch_match_src_images=10,undistort_options=opts,num_threads=12)
 elif stage=='depth_maps':
  # A local scheduler may reserve the next GPU slot for completed-scale gsplat
  # training. This does not interrupt a PatchMatch process already running.
  gate=args.output.parent/'gsplat_local'/'before_1x_depth.wait'
  if scale=='1x' and gate.exists():
   waiting=time.time();print('Waiting for the completed-scale gsplat queue before 1x CUDA stereo.',flush=True)
   while gate.exists():time.sleep(10)
   atomic_json(run/'depth_queue_wait.json',{'seconds':round(time.time()-waiting,2)})
  opts=p.PatchMatchOptions();opts.gpu_index='0';opts.max_image_size=max(w,h);opts.geom_consistency=True;opts.cache_size=4.;opts.num_threads=8
  p.patch_match_stereo(dense,options=opts)
 elif stage=='fusion':
  opts=p.StereoFusionOptions();opts.num_threads=8;opts.max_image_size=max(w,h);opts.use_cache=True;opts.cache_size=float(os.environ.get('COLMAP_FUSION_CACHE_GB', '4'))
  p.stereo_fusion(dense/'fused.ply',dense,input_type='geometric',options=opts,output_type='ply')
  if not (dense/'fused.ply').exists():raise RuntimeError('No fused point cloud written')
 elif stage=='mesh':
  opts=p.PoissonMeshingOptions();opts.depth=10;opts.num_threads=args.num_threads
  p.poisson_meshing(dense/'fused.ply',dense/'mesh_poisson.ply',options=opts)
  if not (dense/'mesh_poisson.ply').exists():raise RuntimeError('No mesh written')
 else:raise ValueError(stage)

def orchestrate(args):
 args.output.mkdir(parents=True,exist_ok=True)
 try:gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],text=True).strip()
 except Exception:gpu='unavailable'
 import pycolmap as p
 config={'pycolmap':p.__version__,'gpu':gpu,'image_policy':'Local files only; no uploads. Each resolution is reconstructed independently.','features':{'max_features':8192,'device':'CUDA','single_camera':args.camera_model,'initial_camera_params':args.camera_params},'matching':'15 sequential neighbors + exponential offsets + cross-sequence anchors every 20 images, guided matching on CUDA','mapping':'Incremental + Ceres bundle adjustment on CPU','dense':'10 source views, 5 PatchMatch iterations, geometric consistency on CUDA, native resolution ceiling','fusion':'CPU, 4 GB cache','mesh':'CPU Poisson depth 10','dense_component_policy':'Largest registered component. Other sparse models retained and reported.'}
 atomic_json(args.output/'configuration.json',config)
 for scale in args.scales:
  run=args.output/scale;run.mkdir(exist_ok=True);statuspath=run/'status.json'
  status=json.loads(statuspath.read_text()) if statuspath.exists() else {'scale':scale,'stages':{}}
  for stage in STAGES[:STAGES.index(args.through)+1]:
   if status['stages'].get(stage,{}).get('state')=='complete':continue
   start=time.time();entry={'state':'running','started_at':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'device':'CUDA' if stage in ['features','matching','depth_maps'] else 'CPU'}
   status['stages'][stage]=entry;atomic_json(statuspath,status)
   print(f'{scale}: {stage} started ({entry["device"]})',flush=True)
   logpath=run/f'{stage}.log'
   cmd=[sys.executable,str(Path(__file__).resolve()),'--data',str(args.data),'--output',str(args.output),'--scale',scale,'--stage',stage,'--camera-model',args.camera_model,'--num-threads',str(args.num_threads)]+(['--camera-params',args.camera_params] if args.camera_params else [])
   with logpath.open('w') as log:
    result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
   entry.update(state='complete' if result.returncode==0 else 'failed',seconds=round(time.time()-start,2),exit_code=result.returncode)
   atomic_json(statuspath,status)
   print(f'{scale}: {stage} {entry["state"]} in {entry["seconds"]}s',flush=True)
   if result.returncode:break
 print('Pipeline finished; inspect per-scale status for successes/failures.',flush=True)

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,default=DEFAULT_DATA);parser.add_argument('--output',type=Path,default=REPO.parent/'reconstruction_local');parser.add_argument('--scales',nargs='+',choices=list(SCALES),default=['8x','4x','2x','1x']);parser.add_argument('--through',choices=STAGES,default='mesh');parser.add_argument('--scale',choices=list(SCALES));parser.add_argument('--stage',choices=STAGES)
 parser.add_argument('--camera-model',choices=['SIMPLE_RADIAL','OPENCV'],default='SIMPLE_RADIAL');parser.add_argument('--camera-params');parser.add_argument('--num-threads',type=int,default=12)
 args=parser.parse_args()
 if args.stage:execute_stage(args)
 else:orchestrate(args)
