"""Read-only native map/header and complete dense/mesh vertex validation."""
from pathlib import Path
import argparse,json,numpy as np,pycolmap as p
from export_viewer import read_ply
from export_insta360_colmap import ply_counts
ROOT=Path(__file__).resolve().parents[2]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--scale',choices=['1x','4x'],required=True);args=ap.parse_args();r=ROOT/'insta360_local/recovery/colmap'/args.scale;m=p.Reconstruction(r/'dense/sparse');images={im.name:m.cameras[im.camera_id] for im in m.images.values()};checked=0
 for kind,ch in [('depth_maps',1),('normal_maps',3)]:
  for suffix in ['photometric','geometric']:
   folder=r/'dense/stereo'/kind;files=list(folder.glob('*.'+suffix+'.bin'));assert len(files)==len(images)
   for name,cam in images.items():
    f=folder/(name+'.'+suffix+'.bin')
    with f.open('rb') as stream:
     header=b''
     while header.count(b'&')<3:
      byte=stream.read(1);assert byte and len(header)<100;header+=byte
    w,h,c=map(int,header[:-1].split(b'&'));assert (w,h,c)==(cam.width,cam.height,ch);assert f.stat().st_size==len(header)+w*h*c*4;checked+=1
 result={'registered_references':len(images),'map_files_with_verified_native_headers_and_sizes':checked}
 for filename,key in [('fused.ply','dense'),('mesh_poisson.ply','mesh')]:
  file=r/'dense'/filename
  if file.is_file():
   xyz,rgb=read_ply(file);counts=ply_counts(file);assert len(xyz)==counts['vertex'] and len(xyz)>0 and np.isfinite(xyz).all();result[key]={'counts':counts,'all_vertex_coordinates_finite':True,'bytes':file.stat().st_size}
 (r/'dense_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
