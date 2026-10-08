(() => {
  const section = document.querySelector('[data-line-chart]');
  if (!section) return;

  const svgNS = 'http://www.w3.org/2000/svg';
  const short = (value) => {
    if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(value >= 10_000_000 ? 0 : 1)}M`;
    if (value >= 10_000) return `${(value / 1_000).toFixed(0)}k`;
    if (value >= 1_000) return `${(value / 1_000).toFixed(value >= 10_000 ? 0 : 1)}k`;
    return Math.round(value).toLocaleString();
  };
  const exact = (value) => Math.round(value).toLocaleString();
  const svgElement = (name, attrs = {}, value = '') => {
    const node = document.createElementNS(svgNS, name);
    for (const [key, item] of Object.entries(attrs)) node.setAttribute(key, item);
    if (value) node.textContent = value;
    return node;
  };
  const niceMaximum = (value) => {
    const target = Math.max(value * 1.08, 1);
    const power = 10 ** Math.floor(Math.log10(target / 4));
    const unit = target / (power * 4);
    const factor = [1, 1.25, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10].find((n) => n >= unit) || 10;
    return power * factor * 4;
  };
  const pearson = (xs, ys) => {
    if (xs.length < 2 || xs.length !== ys.length) return Number.NaN;
    const meanX = xs.reduce((sum, value) => sum + value, 0) / xs.length;
    const meanY = ys.reduce((sum, value) => sum + value, 0) / ys.length;
    const covariance = xs.reduce((sum, value, index) => sum + (value - meanX) * (ys[index] - meanY), 0);
    const scaleX = Math.sqrt(xs.reduce((sum, value) => sum + (value - meanX) ** 2, 0));
    const scaleY = Math.sqrt(ys.reduce((sum, value) => sum + (value - meanY) ** 2, 0));
    return covariance / (scaleX * scaleY);
  };

  function drawChart(panel, metric, samples, xTitle) {
    const svg = panel.querySelector('svg');
    const values = samples.map((sample) => sample[metric.key]);
    if (values.some((value) => !Number.isFinite(value))) {
      panel.querySelector('.chart-values').textContent = 'This metric is not available for every set.';
      return false;
    }

    const width = 820;
    const height = 320;
    const margin = { top: 32, right: 72, bottom: 78, left: 92 };
    const plotWidth = width - margin.left - margin.right;
    const plotHeight = height - margin.top - margin.bottom;
    const max = niceMaximum(Math.max(...values));
    const numericX = samples.map((sample) => sample.xValue);
    const hasNumericX = numericX.every(Number.isFinite);
    const minX = hasNumericX ? Math.min(...numericX) : 0;
    const maxX = hasNumericX ? Math.max(...numericX) : 0;
    const x = (index) => {
      if (samples.length === 1) return margin.left + plotWidth / 2;
      if (hasNumericX && maxX > minX) return margin.left + ((numericX[index] - minX) / (maxX - minX)) * plotWidth;
      return margin.left + (plotWidth * index) / (samples.length - 1);
    };
    const y = (value) => margin.top + plotHeight - (value / max) * plotHeight;

    svg.replaceChildren();
    svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
    svg.setAttribute('aria-label', `${metric.title}. ${samples.map((sample, index) => `${sample.primary}: ${exact(values[index])}`).join('; ')}.`);
    svg.append(svgElement('title', {}, metric.title));
    svg.append(svgElement('desc', {}, `${metric.description} Horizontal axis: ${xTitle}. Vertical axis: count, shown with a separate scale for this metric.`));

    for (let tick = 0; tick <= 4; tick += 1) {
      const value = (max * tick) / 4;
      const yy = y(value);
      svg.append(svgElement('line', { x1: margin.left, y1: yy, x2: width - margin.right, y2: yy, class: 'grid' }));
      svg.append(svgElement('text', { x: margin.left - 12, y: yy + 4, 'text-anchor': 'end' }, short(value)));
    }
    svg.append(svgElement('line', { x1: margin.left, y1: margin.top, x2: margin.left, y2: margin.top + plotHeight, class: 'axis' }));
    svg.append(svgElement('line', { x1: margin.left, y1: margin.top + plotHeight, x2: width - margin.right, y2: margin.top + plotHeight, class: 'axis' }));
    svg.append(svgElement('text', { x: margin.left, y: margin.top - 12, class: 'axis-title' }, 'Count'));

    const path = samples.map((sample, index) => `${index ? 'L' : 'M'} ${x(index)} ${y(sample[metric.key])}`).join(' ');
    svg.append(svgElement('path', { d: path, class: 'series' }));

    samples.forEach((sample, index) => {
      const xx = x(index);
      const yy = y(sample[metric.key]);
      svg.append(svgElement('circle', { cx: xx, cy: yy, r: 5, class: 'point' }));
      svg.append(svgElement('text', { x: xx, y: yy - 11, 'text-anchor': 'middle', class: 'point-label' }, short(sample[metric.key])));
      svg.append(svgElement('text', { x: xx, y: margin.top + plotHeight + 25, 'text-anchor': 'middle', class: 'x-primary' }, sample.primary));
      if (sample.secondary) svg.append(svgElement('text', { x: xx, y: margin.top + plotHeight + 45, 'text-anchor': 'middle', class: 'x-secondary' }, sample.secondary));
    });
    svg.append(svgElement('text', { x: width / 2, y: height - 5, 'text-anchor': 'middle', class: 'axis-title' }, xTitle));

    const correlation = pearson(numericX, values);
    const correlationLabel = Number.isFinite(correlation)
      ? `Pearson r vs ${xTitle.toLowerCase()}: ${correlation >= 0 ? '+' : ''}${correlation.toFixed(3)} (n=${samples.length})`
      : 'Pearson r unavailable';
    panel.querySelector('.chart-values').textContent = `${samples.map((sample) => `${sample.primary}: ${exact(sample[metric.key])}`).join(' · ')}\n${correlationLabel}`;
    return true;
  }

  const metrics = {
    features: { key: 'features', title: 'SIFT keypoints stored', description: 'Total detected SIFT keypoints stored in the COLMAP database.' },
    sparse: { key: 'sparse', title: 'Sparse 3D points', description: 'Three-dimensional points in the main registered component.' },
    dense: { key: 'dense', title: 'Dense fused points', description: 'Vertices in the full fused dense point cloud.' },
  };

  async function loadSamples() {
    const type = section.dataset.lineChart;
    if (type === 'resolution') {
      const report = await fetch('results.json', { cache: 'no-store' }).then((response) => {
        if (!response.ok) throw new Error('Resolution measurements could not be loaded.');
        return response.json();
      });
      const names = [
        ['1x', 'Full resolution', '2160 × 3840'],
        ['2x', '2× downsized', '1080 × 1920'],
        ['4x', '4× downsized', '540 × 960'],
        ['8x', '8× downsized', '270 × 480'],
      ];
      return names.map(([key, primary, secondary], index) => {
        const scale = report.scales[key];
        return { primary, secondary, xValue: [1, 2, 4, 8][index], features: scale.sift_keypoints, sparse: scale.points3D, dense: scale.fused?.vertex };
      });
    }
    if (type === 'frame-count') {
      const report = await fetch('experiments.json', { cache: 'no-store' }).then((response) => {
        if (!response.ok) throw new Error('Frame-count measurements could not be loaded.');
        return response.json();
      });
      const room250 = report.experiments.find((item) => item.key === 'room_250');
      const room1000 = report.experiments.find((item) => item.key === 'room_1000');
      const baseline = report.baseline;
      return [
        { primary: '250 frames', xValue: 250, features: room250?.sift_keypoints, sparse: room250?.metrics.points3D, dense: room250?.dense_points },
        { primary: '500 frames', xValue: 500, features: baseline.sift_keypoints, sparse: baseline.sparse_points, dense: baseline.dense_points },
        { primary: '1,000 frames', xValue: 1000, features: room1000?.sift_keypoints, sparse: room1000?.metrics.points3D, dense: room1000?.dense_points },
      ];
    }
    throw new Error('Unknown chart configuration.');
  }

  loadSamples().then((samples) => {
    let complete = true;
    for (const panel of section.querySelectorAll('[data-metric]')) {
      const metric = metrics[panel.dataset.metric];
      complete = drawChart(panel, metric, samples, section.dataset.lineChart === 'resolution' ? 'Downsample factor' : 'Input image count') && complete;
    }
    const status = section.querySelector('.chart-status');
    status.textContent = complete ? 'Counts are measured from the local COLMAP databases and completed PLY outputs.' : 'One or more totals are unavailable; see the text below each chart.';
  }).catch((error) => {
    section.querySelector('.chart-status').textContent = error.message;
  });
})();
