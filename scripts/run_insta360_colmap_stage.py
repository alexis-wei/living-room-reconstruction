"""Reuse the documented COLMAP stages at the X5's actual image dimensions."""
import argparse, math
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
    reconstruct.execute_stage(args)

if __name__=='__main__':main()
