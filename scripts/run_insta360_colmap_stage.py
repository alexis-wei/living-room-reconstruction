"""Reuse the documented COLMAP stages at the X5's actual image dimensions."""
import argparse, json, math, shutil
from pathlib import Path
import reconstruct

ROOT=Path(__file__).resolve().parents[2]
reconstruct.SCALES={'1x':('full_resolution',3840,2160),
                   '4x':('downsample_4x_960x540',960,540)}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scale',choices=list(reconstruct.SCALES),required=True)
    parser.add_argument('--stage',choices=reconstruct.STAGES,required=True)
    args=parser.parse_args()
    width,height=reconstruct.SCALES[args.scale][1:]
    # Nominal 170-degree MegaView description, interpreted horizontally ONLY
    # as an equidistant initialization. Not a measured physical calibration.
    f=width/math.radians(170)
    args.camera_model='OPENCV_FISHEYE'
    args.camera_params=','.join(str(x) for x in [f,f,width/2,height/2,0,0,0,0])
    args.data=ROOT/'insta360_frames_500'
    args.output=ROOT/'insta360_local/colmap'
    args.num_threads=12
    if args.stage=='depth_maps':
        import pycolmap as p
        model=p.Reconstruction(args.output/args.scale/'dense/sparse')
        # Two passes: depth(1 float) + normals(3 floats), float32 per pixel.
        expected=sum(model.cameras[im.camera_id].width*model.cameras[im.camera_id].height*32
                     for im in model.images.values())
        stereo=args.output/args.scale/'dense/stereo'
        existing=sum(file.stat().st_size for folder in ['depth_maps','normal_maps']
                     for file in (stereo/folder).glob('*.bin'))
        free=shutil.disk_usage(ROOT).free
        check={'expected_depth_normal_bytes':expected,'existing_depth_normal_bytes':existing,
               'free_bytes':free,'minimum_spare_bytes':10*(1<<30)}
        (args.output/args.scale/'depth_storage_preflight.json').write_text(json.dumps(check,indent=2)+'\n')
        if free<max(0,expected-existing)+check['minimum_spare_bytes']:
            raise RuntimeError('Insufficient free space for native depth/normal maps and a10GiB reserve; preserve completed work and provide storage before resuming.')
    reconstruct.execute_stage(args)

if __name__=='__main__':main()
