"""Prepare separate geometry-screened copies; original 500-view models untouched."""
from pathlib import Path
import json, numpy as np, pycolmap as p
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'insta360_local/recovery'
def main():
 for scale in ['4x','1x']:
  run=OUT/'colmap'/scale;run.mkdir(parents=True,exist_ok=True)
  if (run/'recovery_validation.json').exists():continue
  source=ROOT/'insta360_local/colmap'/scale/'sparse/0';m=p.Reconstruction(source)
  centers=np.array([im.projection_center() for im in m.images.values()]);radius=float(np.quantile(np.linalg.norm(centers-np.median(centers,axis=0),axis=1),.95));threshold=radius*1e-5
  excluded=[];drop=[]
  for im in list(m.images.values()):
   xyz=np.array([m.points3D[o.point3D_id].xyz for o in im.points2D if o.has_point3D()]);z=(im.cam_from_world()*xyz)[:,2] if len(xyz) else np.array([]);positive=z[z>0]
   median=float(np.median(positive)) if len(positive) else 0.
   if len(positive)<100 or median<threshold:
    excluded.append({'name':im.name,'median_depth':median,'positive_observations':len(positive)});drop.append(im.frame_id)
  for frame_id in drop:m.deregister_frame(frame_id)
  for pid,point in list(m.points3D.items()):
   if point.track.length()<2:m.delete_point3D(pid)
  p.ObservationManager(m).filter_all_points3D(4.,1.5)
  m.normalize();m.update_point_3d_errors()
  sparse=run/'sparse/0';sparse.mkdir(parents=True,exist_ok=True);m.write(sparse)
  # Reload to verify only registered views serialize, with finite geometry.
  m=p.Reconstruction(sparse);assert all(v.track.length()>=2 and np.isfinite(v.xyz).all() for v in m.points3D.values())
  metrics={'registered_images':m.num_reg_images(),'points3D':m.num_points3D(),'mean_reprojection_error':m.compute_mean_reprojection_error(),'mean_track_length':m.compute_mean_track_length(),'camera_model':'OPENCV_FISHEYE'}
  (run/'models.json').write_text(json.dumps({'selected_model':0,'models':[{'id':0,**metrics}],'input_images':500},indent=2))
  (run/'sparse_metrics.json').write_text(json.dumps(metrics,indent=2))
  report={'source':str(source),'criterion':'median positive observed depth < 1e-5 * original camera-radius95, or fewer than100 positive observations','radius95':radius,'depth_threshold':threshold,'excluded':excluded,'metrics':metrics,'normalization':'Global similarity only; no new bundle adjustment or intrinsics change','original_model_unchanged':True}
  (run/'recovery_validation.json').write_text(json.dumps(report,indent=2))
  print(scale,metrics,'excluded',len(excluded),flush=True)
if __name__=='__main__':main()
