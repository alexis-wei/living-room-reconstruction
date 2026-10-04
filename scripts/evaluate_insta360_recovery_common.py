"""CPU common-heldout comparison from saved native PNG predictions (8-bit)."""
from pathlib import Path
import json,os
ROOT=Path(__file__).resolve().parents[2]
os.environ.setdefault('TORCH_HOME',str(ROOT/'.gsplat-cache/torch'))
import numpy as np,torch
from PIL import Image
from torchmetrics.image import PeakSignalNoiseRatio,StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
OUT=ROOT/'insta360_local/recovery/gsplat'
def main():
 torch.set_num_threads(4)
 jobs={s:next(j for j in json.loads((OUT/f'datasets_{s}.json').read_text())['jobs'] if j['primary_component']) for s in ['4x','1x']}
 shared=sorted(set(jobs['4x']['validation_frames'])&set(jobs['1x']['validation_frames']));assert shared
 metrics={'psnr':PeakSignalNoiseRatio(data_range=1.),'ssim':StructuralSimilarityIndexMeasure(data_range=1.),'lpips':LearnedPerceptualImagePatchSimilarity(net_type='alex',normalize=True)}
 report={'common_frame_count':len(shared),'frames':shared,'protocol':'Same held-out source frames; native resolution per model. CPU metrics from saved8-bit PNG reference/prediction halves, distinct from trainer float-render metrics. Different resolution and undistortion remain confounders.','models':{}}
 with torch.no_grad():
  for scale,job in jobs.items():
   run=OUT/'runs'/job['name'];assert json.loads((run/'status.json').read_text())['state']=='complete';rows=[]
   for frame in shared:
    index=job['validation_frames'].index(frame)
    with Image.open(run/f'renders/val_step29999_{index:04d}.png') as image:a=np.array(image.convert('RGB'),copy=True)
    tensor=torch.from_numpy(a).permute(2,0,1).unsqueeze(0).float()/255;w=tensor.shape[-1]//2;reference,prediction=tensor[:,:,:,:w],tensor[:,:,:,w:]
    row={'frame':frame}
    for name,metric in metrics.items():row[name]=float(metric(prediction,reference));metric.reset()
    rows.append(row);print(scale,frame,flush=True)
   report['models'][scale]={'frames':rows,'mean':{k:float(np.mean([v[k] for v in rows])) for k in metrics}}
 (OUT/'common_validation.json').write_text(json.dumps(report,indent=2)+'\n')
 aggregate={k:v for k,v in report.items() if k not in ['frames','models']};aggregate['models']={k:{'mean':v['mean']} for k,v in report['models'].items()}
 (ROOT/'living-room-reconstruction/reports/insta360_recovery_common.json').write_text(json.dumps(aggregate,indent=2)+'\n')
if __name__=='__main__':main()
