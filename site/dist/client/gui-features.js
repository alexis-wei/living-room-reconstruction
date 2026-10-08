(() => {
  const ns = 'http://www.w3.org/2000/svg';
  const fmt = (n, digits = 0) => Number(n).toLocaleString(undefined, { maximumFractionDigits: digits });
  const duration = seconds => `${Math.floor(seconds/60)}:${(seconds%60).toFixed(2).padStart(5,'0')}`;
  const el = (name, attrs = {}, text = '') => {
    const node = document.createElementNS(ns, name);
    Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, value));
    node.textContent = text;
    return node;
  };
  const nice = (n) => Math.max(1, Math.ceil(n / 4 / (10 ** Math.floor(Math.log10(Math.max(n, 1) / 4)))) * (10 ** Math.floor(Math.log10(Math.max(n, 1) / 4))) * 4);
  const width = 1000, height = 300;
  const m = { left: 80, right: 60, top: 28, bottom: 65 };
  const pw = width - m.left - m.right, ph = height - m.top - m.bottom;

  function axes(svg, ymax, xmax, xLabel, yLabel, ticks) {
    svg.replaceChildren();
    svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
    const y = (value) => m.top + ph - value / ymax * ph;
    for (let i = 0; i <= 4; i++) {
      const value = ymax * i / 4, yy = y(value);
      svg.append(el('line', { x1: m.left, y1: yy, x2: width-m.right, y2: yy, class: 'grid' }));
      svg.append(el('text', { x: m.left-10, y: yy+4, 'text-anchor': 'end' }, fmt(value)));
    }
    svg.append(el('line', { x1: m.left, y1: m.top, x2: m.left, y2: m.top+ph, class: 'axis' }));
    svg.append(el('line', { x1: m.left, y1: m.top+ph, x2: width-m.right, y2: m.top+ph, class: 'axis' }));
    svg.append(el('text', { x: m.left, y: 16 }, yLabel));
    ticks.forEach(([value, label]) => svg.append(el('text', { x: m.left + value/xmax*pw, y: m.top+ph+25, 'text-anchor': 'middle' }, label)));
    svg.append(el('text', { x: m.left+pw/2, y: height-6, 'text-anchor': 'middle' }, xLabel));
    return y;
  }

  function frameBars(svg, dataset, maxFeatures, selected) {
    const n = dataset.expected_images;
    const y = axes(svg, maxFeatures, n, 'Source frame in chronological order', 'Stored SIFT features', [[1,'1'],[100,'100'],[200,'200'],[300,'300'],[400,'400'],[500,'500']]);
    dataset.frames.forEach((frame, i) => {
      if (frame.features === null) return;
      const x = m.left + i/n*pw;
      const bar = el('rect', { x, y: y(frame.features), width: Math.max(.8, pw/n-.2), height: Math.max(.7, m.top+ph-y(frame.features)), class: frame.frame===selected?'feature-bar selected':'feature-bar' });
      bar.append(el('title', {}, `Frame ${String(frame.frame).padStart(4,'0')}: ${fmt(frame.features)} stored features`));
      svg.append(bar);
    });
    const frame = dataset.frames.find(item => item.frame === selected);
    if (frame && frame.features !== null) {
      const x = m.left + (dataset.frames.indexOf(frame)+.5)/n*pw;
      svg.append(el('line', { x1:x, x2:x, y1:m.top, y2:m.top+ph, class:'feature-inspector' }));
    }
    svg.setAttribute('aria-label', `${dataset.label}: ${dataset.processed_images} measured images. Feature counts by chronological frame; maximum axis ${fmt(maxFeatures)}.`);
  }

  function histogram(svg, dataset, maxFeatures, binWidth, maxFrequency) {
    const bins = Array.from({ length: Math.ceil(maxFeatures/binWidth) }, () => 0);
    dataset.frames.forEach(frame => { if(frame.features !== null) bins[Math.min(bins.length-1,Math.floor(frame.features/binWidth))]++; });
    const y = axes(svg, maxFrequency, maxFeatures, 'Stored SIFT features per image', 'Number of images', [0,.25,.5,.75,1].map(f=>[f*maxFeatures,fmt(f*maxFeatures)]));
    bins.forEach((count,i) => {
      if(!count)return;
      const bar = el('rect', { x:m.left+i/binCount()*pw+1, y:y(count), width:pw/binCount()-2, height:m.top+ph-y(count), class:'feature-bar' });
      bar.append(el('title',{},`${fmt(i*binWidth)}–${fmt((i+1)*binWidth-1)} features: ${count} images`));
      svg.append(bar);
    });
    function binCount(){return bins.length;}
    svg.setAttribute('aria-label', `${dataset.label}: histogram of features per image with shared bins of ${fmt(binWidth)} features.`);
  }

  function meanChart(svg, datasets) {
    const available = datasets.filter(d=>d.statistics);
    const ymax = nice(Math.max(1,...available.map(d=>Math.max(d.statistics.mean,d.statistics.p75)))*1.1);
    const y = axes(svg, ymax, 7, 'Downsample factor (linear spacing)', 'Mean features per processed image', datasets.map(d=>[d.factor-1,d.factor===1?'1× · full':`${d.factor}× downsized`]));
    const x = d => m.left + (d.factor-1)/7*pw;
    let previous;
    datasets.forEach(dataset => {
      const s=dataset.statistics;
      if(!s){svg.append(el('text',{x:x(dataset),y:m.top+ph-10,'text-anchor':'middle',class:'feature-pending'},'Pending'));previous=null;return;}
      if(previous){svg.append(el('line',{x1:x(previous),y1:y(previous.statistics.mean),x2:x(dataset),y2:y(s.mean),class:'series'}));}
      svg.append(el('line',{x1:x(dataset),x2:x(dataset),y1:y(s.p25),y2:y(s.p75),class:'feature-range'}));
      for(const value of [s.p25,s.p75]) svg.append(el('line',{x1:x(dataset)-6,x2:x(dataset)+6,y1:y(value),y2:y(value),class:'feature-range'}));
      const circle=el('circle',{cx:x(dataset),cy:y(s.mean),r:5,class:dataset.complete?'point':'point partial'});
      circle.append(el('title',{},`${dataset.label}: mean ${fmt(s.mean,2)}, median ${fmt(s.median,2)}, ${dataset.processed_images}/${dataset.expected_images} images measured`));svg.append(circle);
      svg.append(el('text',{x:x(dataset),y:y(s.mean)-12,'text-anchor':'middle',class:'point-label'},fmt(s.mean,2)));
      previous=dataset;
    });
    svg.setAttribute('aria-label', `Mean feature count at numeric downsample factors 1, 2, 4 and 8. ${available.map(d=>`${d.label}: ${fmt(d.statistics.mean,2)}`).join('; ')}.${available.length<datasets.length?' Other scales are pending.':''}`);
  }

  fetch('gui-features.json',{cache:'no-store'}).then(response=>{if(!response.ok)throw Error('Feature-count snapshot is unavailable.');return response.json();}).then(report=>{
    const datasets=report.scales;
    document.getElementById('gui-feature-status').textContent=`Recorded ${new Date(report.generated_at).toLocaleString()} · ${datasets.filter(d=>d.complete).length}/4 feature-extraction sets complete.`;
    const table=document.getElementById('gui-feature-summary');
    table.replaceChildren();
    datasets.forEach(dataset=>{
      const tr=document.createElement('tr'),s=dataset.statistics;
      [dataset.label,`${dataset.width} × ${dataset.height}`,`${dataset.processed_images}/${dataset.expected_images}`,s?fmt(s.mean,2):'Pending',s?fmt(s.median,2):'Pending',s?`${fmt(s.minimum)}–${fmt(s.maximum)}`:'Pending',s?fmt(s.total):'Pending'].forEach(value=>{const td=document.createElement('td');td.textContent=value;tr.append(td);});table.append(tr);
    });
    meanChart(document.getElementById('gui-mean-chart'),datasets);
    const timings = report.timing_baseline;
    const timingTable = document.getElementById('gui-stage-timings');
    const breakdown = document.getElementById('gui-sparse-breakdown');
    if (timings) {
      document.getElementById('gui-timing-hardware').textContent = `${timings.gpu} · features and matching on CUDA; sparse optimization on CPU.`;
      timings.scales.forEach(dataset => {
        const row=document.createElement('tr');
        [dataset.label,...['features','matching','sparse_total','three_stage_total'].map(key=>duration(dataset[key]))].forEach(value=>{const td=document.createElement('td');td.textContent=value;row.append(td);});timingTable.append(row);
        const detail=document.createElement('tr');
        [dataset.label,duration(dataset.mapping),duration(dataset.bundle_adjustment)].forEach(value=>{const td=document.createElement('td');td.textContent=value;detail.append(td);});breakdown.append(detail);
      });
    }
    const grid=document.getElementById('gui-feature-panels');
    const panels=datasets.map(dataset=>{
      const panel=document.createElement('figure');panel.className='count-chart-panel gui-feature-panel';
      const caption=document.createElement('figcaption'),title=document.createElement('strong'),state=document.createElement('span');
      title.textContent=dataset.label;state.textContent=`${dataset.width} × ${dataset.height} · ${dataset.processed_images}/${dataset.expected_images} measured`;caption.append(title,state);
      const svg=el('svg',{class:'count-line-chart',role:'img'}),readout=document.createElement('p');readout.className='chart-values';
      panel.append(caption,svg,readout);grid.append(panel);return {dataset,svg,readout};
    });
    const maxRaw=Math.max(1,...datasets.flatMap(d=>d.frames.map(f=>f.features||0)));
    const binWidth=Math.max(100,Math.ceil(maxRaw/20/100)*100),maxFeatures=Math.ceil(maxRaw/binWidth)*binWidth;
    const frequencies=datasets.map(d=>{const bins=Array(Math.ceil(maxFeatures/binWidth)).fill(0);d.frames.forEach(f=>{if(f.features!==null)bins[Math.min(bins.length-1,Math.floor(f.features/binWidth))]++;});return Math.max(0,...bins);});
    const mode=document.getElementById('gui-chart-mode'),frame=document.getElementById('gui-inspect-frame');
    const render=()=>{
      const selected=Math.max(1,Math.min(500,Number(frame.value)||1));frame.value=selected;
      panels.forEach(({dataset,svg,readout})=>{
        if(mode.value==='histogram')histogram(svg,dataset,maxFeatures,binWidth,nice(Math.max(1,...frequencies)));else frameBars(svg,dataset,maxFeatures,selected);
        const point=dataset.frames.find(f=>f.frame===selected),s=dataset.statistics;
        readout.textContent=s?`Mean ${fmt(s.mean,2)} · median ${fmt(s.median,2)} · middle 50% ${fmt(s.p25)}–${fmt(s.p75)} features.\nFrame ${String(selected).padStart(4,'0')}: ${point?.features===null?'pending':fmt(point?.features||0)}${point?.features===null?'':' features'}.${dataset.complete?'':' Partial snapshot: unprocessed images are excluded from statistics.'}`:'Awaiting feature extraction. Pending images are not counted as zero.';
      });
    };
    mode.onchange=render;frame.onchange=render;render();
    const settings=document.getElementById('gui-saved-settings');
    datasets.forEach(dataset=>{const heading=document.createElement('h3'),pre=document.createElement('pre');heading.textContent=dataset.label;pre.textContent=JSON.stringify(dataset.saved_settings,null,2);settings.append(heading,pre);});
  }).catch(error=>{document.getElementById('gui-feature-status').textContent=error.message;});
})();
