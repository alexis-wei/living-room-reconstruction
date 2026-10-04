"""CPU-only audit of completed recovery checkpoints, full PLY and all holdouts."""
from pathlib import Path
import json,math,torch,numpy as np
from PIL import Image
from export_insta360_gsplat import validate_ply
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'insta360_local/recovery/gsplat'
def main():
 for scale in ['4x','1x']:
  manifest=OUT/f'datasets_{scale}.json'
  if not manifest.exists():continue
  for job in json.loads(manifest.read_text())['jobs']:
   run=OUT/'runs'/job['name'];status=json.loads((run/'status.json').read_text())
   if status['state']!='complete':continue
   checkpoint=run/'ckpts/ckpt_29999_rank0.pt';data=torch.load(checkpoint,map_location='cpu',weights_only=False)
   splats=data['splats'];assert set(['means','scales','quats','opacities','sh0','shN']).issubset(splats)
   n=len(splats['means']);assert splats['means'].shape==(n,3) and splats['shN'].shape==(n,15,3)
   invalid={}
   for name,tensor in splats.items():
    assert len(tensor)==n
    bad=0
    for part in tensor.split(100000):bad+=int((~torch.isfinite(part)).sum())
    invalid[name]=bad
   vertices=validate_ply(run/'ply/point_cloud_29999.ply');assert vertices<=n
   assert int(status['metrics']['num_GS'])==n
   assert all(math.isfinite(status['metrics'][k]) for k in ['psnr','ssim','lpips'])
   sizes={tuple(x) for x in job['undistorted_sizes']};renders=[]
   for index,frame in enumerate(job['validation_frames']):
    path=run/f'renders/val_step29999_{index:04d}.png'
    with Image.open(path) as image:
     image.load();assert image.width%2==0;native=(image.width//2,image.height);assert native in sizes,(native,sizes)
     prediction=np.asarray(image)[:,image.width//2:];assert prediction.size and np.isfinite(prediction).all()
     renders.append({'frame':frame,'native_prediction_size':native,'bytes':path.stat().st_size})
   assert not set(job['validation_frames'])&set(job['training_frames'])
   result={'state':'complete','checkpoint_gaussians':n,'checkpoint_nonfinite_values':invalid,'checkpoint_all_finite':not any(invalid.values()),'full_ply_gaussians':vertices,'all_ply_values_finite':True,'sh_degree':3,'native_validation_renders':renders,'training_images':len(job['training_frames']),'validation_images':len(renders),'metrics':status['metrics'],'quality_note':'Finite outputs and completed optimization do not establish faithful room geometry; inspect held-out predictions.'}
   (run/'full_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(job['name'],n,vertices,'all holdouts',len(renders),invalid,flush=True)
if __name__=='__main__':main()
