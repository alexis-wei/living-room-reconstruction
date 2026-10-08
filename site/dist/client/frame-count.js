(()=>{
const body=document.getElementById('frame-count-results');
if(!body)return;
const number=value=>value==null?'—':Number(value).toLocaleString();
fetch('experiments.json',{cache:'no-store'}).then(response=>{
 if(!response.ok)throw Error('Frame-count results are unavailable.');
 return response.json();
}).then(data=>{
 const wholeRoom=['room_250','baseline_500','room_1000'].map(key=>key==='baseline_500'?{
  key,label:'500 frames · original baseline',input_images:data.baseline.count,
  metrics:{registered_images:data.baseline.registered_images,points3D:data.baseline.sparse_points},
  registered_union:data.baseline.registered_images,dense_points:data.baseline.dense_points
 }:data.experiments.find(experiment=>experiment.key===key)).filter(Boolean);
 body.replaceChildren(...wholeRoom.map(experiment=>{
  const row=document.createElement('tr');
  const largest=experiment.metrics.registered_images;
  const total=experiment.registered_union??largest;
  const values=[experiment.label,number(experiment.input_images),`${number(largest)} / ${number(experiment.input_images)}`,`${number(total)} / ${number(experiment.input_images)}`,number(experiment.metrics.points3D),number(experiment.dense_points)];
  for(const value of values){const cell=document.createElement('td');cell.textContent=value;row.append(cell)}
  return row;
 }));
}).catch(error=>{body.innerHTML=`<tr><td colspan="6">${error.message}</td></tr>`});
})();
