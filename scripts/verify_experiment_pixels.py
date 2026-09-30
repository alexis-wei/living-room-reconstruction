"""Spot-check source decoding and exact direct-downsample pixels after extraction."""
from pathlib import Path
import av,cv2,json,numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'experiments_local'
def main():
 targets={};results=[]
 for key in ['room_250','room_1000','dining_500']:
  root=BASE/key;ids=json.loads((root/'selected_indices.json').read_text())
  for k in [0,len(ids)//2,len(ids)-1]:targets.setdefault(ids[k],[]).append((key,k+1))
 c=av.open(str(ROOT/'alexis_living_room.mov'));c.streams.video[0].thread_type='AUTO'
 for idx,frame in enumerate(c.decode(video=0)):
  if idx not in targets:continue
  rgb=cv2.rotate(frame.to_ndarray(format='rgb24'),cv2.ROTATE_90_CLOCKWISE)
  small=np.array(Image.fromarray(rgb).resize((540,960),getattr(Image,'Resampling',Image).LANCZOS))
  for key,num in targets[idx]:
   for folder,expected in [('full_resolution',rgb),('downsample_4x_540x960',small)]:
    with Image.open(BASE/key/folder/f'frame_{num:04d}.png') as im:assert np.array_equal(np.array(im),expected),(key,num,folder)
   results.append({'dataset':key,'frame':num,'source_index':idx,'native_exact':True,'4x_exact':True})
 assert len(results)==9
 (BASE/'pixel_verification.json').write_text(json.dumps(results,indent=2));print('9 source-frame pairs match fresh native decoding and direct LANCZOS exactly.')
if __name__=='__main__':main()
