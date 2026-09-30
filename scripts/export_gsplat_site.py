"""Export all trained Gaussian geometry in chunked SPLAT format and genuine model renders.
Private Site only; original checkpoints/SH3 PLY stay local and unchanged.
"""
from pathlib import Path
import sys,json,os
import numpy as np
import torch
import shlex
import torch.utils.cpp_extension as extension
_prepare=extension._prepare_ldflags
def _quoted(*args,**kwargs):
 return [shlex.quote(x) if x.startswith("-L") and " " in x else x for x in _prepare(*args,**kwargs)]
extension._prepare_ldflags=_quoted
import imageio.v2 as imageio
W=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(W/'.gsplat-src/gsplat/examples'))
from datasets.colmap import Parser
from gsplat import rasterization
OUT=W/'living-room-reconstruction/site/dist/gaussians';OUT.mkdir(exist_ok=True)
manifest={}
for name in ['8x','4x','2x','1x','8x_component_2']:
 run=W/'gsplat_local/runs'/name
 data=torch.load(run/'ckpts/ckpt_29999_rank0.pt',map_location='cpu',weights_only=False)['splats']
 a={k:v.detach().numpy() for k,v in data.items()};n=len(a['means'])
 assert all(np.isfinite(v).all() for k,v in a.items() if k!='scales')
 assert not np.isnan(a['scales']).any() and not np.isposinf(a['scales']).any() # -inf log-scale means zero extent
 q=a['quats']/np.maximum(np.linalg.norm(a['quats'],axis=1,keepdims=True),1e-20)
 rgba=np.concatenate([np.clip(.5+.28209479177387814*a['sh0'][:,0,:],0,1),1/(1+np.exp(-np.clip(a['opacities'].reshape(-1,1),-80,80)))],axis=1)
 dtype=np.dtype([('pos','<f4',(3,)),('scale','<f4',(3,)),('rgba','u1',(4,)),('quat','u1',(4,))]);assert dtype.itemsize==32
 chunks=[]
 for start in range(0,n,500000):
  end=min(n,start+500000);v=np.empty(end-start,dtype=dtype)
  v['pos']=a['means'][start:end];v['scale']=np.exp(a['scales'][start:end]);v['rgba']=(rgba[start:end]*255).round().astype('u1');v['quat']=np.clip(q[start:end]*128+128,0,255).astype('u1')
  assert np.isfinite(v['scale']).all()
  dest=OUT/f'{name}-{start//500000:02d}.splat';dest.write_bytes(v.tobytes());chunks.append('gaussians/'+dest.name)
 parser=Parser(str(W/'gsplat_local/datasets'/name),factor=1,normalize=True,test_every=8)
 views=[];selected=[]
 for target in ['frame_0001.png','frame_0130.png','frame_0175.png','frame_0405.png']:
  if target in parser.image_names:selected.append(parser.image_names.index(target))
 if len(selected)<4:selected=list(dict.fromkeys(selected+np.linspace(0,len(parser.image_names)-1,4,dtype=int).tolist()))[:4]
 gpu={k:v.cuda() for k,v in data.items()};colors=torch.cat([gpu['sh0'],gpu['shN']],dim=1)
 for ix in selected:
  c=parser.camtoworlds[ix];cid=parser.camera_ids[ix];K=parser.Ks_dict[cid].copy();width,height=parser.imsize_dict[cid];factor=min(1,960/max(width,height));width=round(width*factor);height=round(height*factor);K[:2]*=factor
  with torch.no_grad():
   rgb,_,_=rasterization(means=gpu['means'],quats=gpu['quats'],scales=torch.exp(gpu['scales']),opacities=torch.sigmoid(gpu['opacities']),colors=colors,viewmats=torch.tensor(np.linalg.inv(c)[None],device='cuda',dtype=torch.float32),Ks=torch.tensor(K[None],device='cuda',dtype=torch.float32),width=width,height=height,sh_degree=3,packed=True,near_plane=.01,far_plane=1e10)
  frame=parser.image_names[ix];dest=OUT/f'{name}-{Path(frame).stem}-render.png';imageio.imwrite(dest,np.uint8(np.clip(rgb[0].cpu().numpy(),0,1)*255))
  views.append(dict(frame=frame,position=c[:3,3].tolist(),forward=c[:3,2].tolist(),up=(-c[:3,1]).tolist(),fov=float(2*np.arctan(height/(2*K[1,1]))*180/np.pi),render='gaussians/'+dest.name))
 manifest[name]=dict(count=n,bytes=n*32,chunks=chunks,views=views,center=np.median(a['means'],axis=0).tolist(),radius=float(np.quantile(np.linalg.norm(a['means']-np.median(a['means'],axis=0),axis=1),.9)))
 del gpu,colors,data;torch.cuda.empty_cache();print(name,n,len(chunks),flush=True)
(OUT/'index.json').write_text(json.dumps(manifest,indent=2)+'\n')
