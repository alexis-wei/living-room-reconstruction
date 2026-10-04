const stages=[
 ['features','Find distinctive features','Detect SIFT keypoints and descriptors in every image. Each set is processed at its own native resolution with one shared camera model.','CUDA','8,192 maximum features · SIMPLE_RADIAL · one camera per set','pycolmap.extract_features'],
 ['matching','Match views & verify geometry','Match adjacent frames and cross-sequence anchor views. Guided matching and geometric verification reject inconsistent correspondences.','CUDA + CPU verification','15 neighbors · exponential offsets · anchors every 20 images','pycolmap.match_image_pairs'],
 ['mapping','Recover cameras & sparse structure','Incremental structure from motion estimates camera poses and triangulates 3D points, refining them repeatedly. Separate components are retained when views cannot be joined.','CPU','Incremental mapper · 12 threads · random seed 0','pycolmap.incremental_mapping'],
 ['bundle_adjustment','Refine the largest component','Jointly optimize camera parameters and 3D points in the largest recovered component. The report counts images in that component, not a sum across overlapping models.','CPU','Ceres · focal length and distortion refined · principal point fixed','pycolmap.bundle_adjustment'],
 ['undistortion','Prepare the dense workspace','Remove estimated lens distortion and prepare registered images for stereo. These are derived local images; the original PNGs remain unchanged.','CPU','Native long-edge ceiling · no upscaling · 10 source views','pycolmap.undistort_images'],
 ['depth_maps','Estimate depth & surface normals','CUDA PatchMatch estimates per-pixel depth and normals. Geometric consistency checks agreement across neighboring views.','CUDA','5 iterations · 10 source views · geometric consistency · 4 GB cache','pycolmap.patch_match_stereo'],
 ['fusion','Fuse the dense point cloud','Combine consistent depth observations into a dense colored point cloud. Reprojection, depth, and normal checks filter conflicting samples.','CPU','Geometric fusion · 8 threads · 4 GB cache','pycolmap.stereo_fusion'],
 ['mesh','Reconstruct the surface','Poisson meshing converts the fused oriented points into a surface. Holes, reflections, or missing observations can still produce artifacts.','CPU','Poisson · octree depth 10 · color enabled','pycolmap.poisson_meshing']
];
let report,examples,scale='1x',active=0;
const el=id=>document.getElementById(id),fmt=n=>n==null?'—':n.toLocaleString(),duration=s=>s==null?'Not finished':s<60?`${s.toFixed(1)} seconds`:`${(s/60).toFixed(1)} minutes`;
function renderExamples(){
 if(!examples)return;
 const items=examples.scales[scale];el('examples-scale').textContent=`${scale==='1x'?'FULL RESOLUTION':scale.toUpperCase()+' DOWNSAMPLE'} · ${items[0].width} × ${items[0].height}`;
 el('example-grid').replaceChildren(...items.map(item=>{
  const figure=document.createElement('figure'),a=document.createElement('a'),img=document.createElement('img'),cap=document.createElement('figcaption');
  a.href=item.url;a.target='_blank';a.rel='noopener';a.title=`Open ${item.filename} at ${item.width} × ${item.height}`;
  img.src=item.url;img.alt=`${item.title}, source frame ${item.frame}, ${scale} resolution`;img.width=item.width;img.height=item.height;img.loading='lazy';
  cap.innerHTML=`<strong>${item.title}</strong><span>${item.filename} · ${item.time_seconds.toFixed(2)}s</span><span>${item.width} × ${item.height} · ${(item.bytes/1e6).toFixed(2)} MB · ${item.format||'PNG'}</span>`;
  a.append(img);figure.append(a,cap);return figure;
 }));
}
fetch('examples/index.json').then(r=>{if(!r.ok)throw Error('Examples unavailable');return r.json()}).then(data=>{examples=data;renderExamples()}).catch(()=>{el('example-grid').textContent='Source examples could not be loaded.'});
function render(){
 renderExamples();
 const d=report.scales[scale];el('dimensions').textContent=`${d.width} × ${d.height}`;el('registered').textContent=d.registered_images==null?'Pending':`${d.registered_images} / 500`;el('points').textContent=fmt(d.points3D);
 const states=Object.values(d.stages),done=states.filter(s=>s.state==='complete').length;el('runstatus').textContent=`${done} / 8 STEPS COMPLETE`;
 el('steps').replaceChildren(...stages.map(([key,title,desc,device],i)=>{const s=d.stages[key]||{state:'pending'};const b=document.createElement('button');b.className=`step ${i===active?'active':''}`;b.setAttribute('aria-pressed',i===active);b.innerHTML=`<span class="number">${String(i+1).padStart(2,'0')}</span><span><b>${title}</b><small>${device}</small></span><span class="badge ${s.state}">${s.state}</span>`;b.onclick=()=>{active=i;render()};return b}));
 const [key,title,desc,device,settings,api]=stages[active],s=d.stages[key]||{state:'pending'};
 let measured='This stage has not completed.';
 if(s.state==='complete'){
  measured=`Completed in ${duration(s.seconds)}.`;
  if(['mapping','bundle_adjustment'].includes(key))measured+=` Largest component: ${fmt(d.registered_images)} registered images and ${fmt(d.points3D)} sparse points. Mean reprojection error: ${d.mean_reprojection_error?.toFixed(3)??'—'} pixels.`;
  if(key==='fusion')measured+=` Fused vertices: ${fmt(d.fused?.vertex)}.`;
  if(key==='mesh')measured+=` Mesh: ${fmt(d.mesh?.vertex)} vertices, ${fmt(d.mesh?.face)} faces.`;
 }else if(s.state==='running')measured='Running at the time of this report snapshot. The page does not stream live progress.';
 else if(s.state==='failed')measured='The stage failed. Local logs retain the diagnostic details; later stages are not marked complete.';
 else if(s.state==='queued')measured='Queued for the GPU while the prior workload completes; CUDA processing has not started.';
 if(s.queue_wait_seconds)measured+=` A separate ${duration(s.queue_wait_seconds)} GPU queue wait is excluded from the processing time.`;
 if(key==='depth_maps' && d.dense_reference_images!=null)measured+=` ${d.dense_reference_images} reference views have stereo sources; ${d.dense_skipped_no_sources} registered views are skipped because no usable source views remain. ${d.geometric_depth_maps_written} geometric depth maps have been written.`;
 el('detail').innerHTML=`<p class="eyebrow">STEP ${String(active+1).padStart(2,'0')} / ${scale.toUpperCase()}</p><h3>${title}</h3><p>${desc}</p><dl><dt>Compute device</dt><dd>${device}</dd><dt>Settings</dt><dd>${settings}</dd><dt>Measured result</dt><dd>${measured}</dd><dt>Local output</dt><dd>reconstruction_local/${scale}/</dd></dl><code>${api}(…)</code>`;
 const inputLabel=scale==='8x'?'8× downsampled inputs (270 × 480 px; the smallest images in this comparison)':`${scale} inputs (${d.width} × ${d.height} px)`;
 el('coverage').textContent=d.registered_images==null?'Sparse reconstruction metrics will appear once this resolution finishes camera recovery.':`The main ${scale} component—largest by registered-image count, using ${inputLabel}—contains ${d.registered_images} of 500 images (${(d.registered_images/5).toFixed(1)}%). “Largest” means the component with the most registered images, not the highest-resolution input or the most 3D points. COLMAP returned ${d.components.length} sparse components, including small or degenerate results. Dense processing uses the main component; this does not establish complete room coverage.`;
 if(d.dense_skipped_no_sources)el('coverage').textContent+=` Dense stereo skips ${d.dense_skipped_no_sources} registered views with no usable source images, leaving ${d.dense_reference_images} reference views. Camera registration alone does not establish dense coverage.`;
 document.querySelectorAll('.tabs button').forEach(b=>b.setAttribute('aria-pressed',b.dataset.scale===scale));
}
fetch('results.json').then(r=>{if(!r.ok)throw Error('Report unavailable');return r.json()}).then(data=>{
 report=data;el('updated').textContent=`Snapshot: ${new Date(data.updated).toLocaleString()}`;
 el('folders').innerHTML=Object.entries(data.scales).map(([k,d])=>`<tr><td><b>${k==='1x'?'Full resolution':k+' downsample'}</b></td><td>${d.folder}</td><td>${d.width} × ${d.height}</td><td>${(d.width*d.height/1e6).toFixed(2)} MP</td><td>${(d.bytes/1e9).toFixed(2)} GB</td></tr>`).join('');
 if(data.repo_url){el('repo-link').href=data.repo_url;el('repo-link').hidden=false}
 document.querySelectorAll('.tabs button').forEach(b=>b.onclick=()=>{scale=b.dataset.scale;render()});render();
}).catch(e=>{el('updated').textContent='Unable to load results. Open this report through its web server.';console.error(e)});
