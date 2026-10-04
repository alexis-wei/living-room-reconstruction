(() => {
  const chapters = [
    { id: 0, title: 'Compare any two', href: 'compare.html' },
    { id: 1, title: 'COLMAP · four resolutions', href: 'colmap.html#colmap-study' },
    { id: 2, title: 'First gsplat trials', href: 'gsplat.html' },
    { id: 3, title: '250 / 500 / 1,000 frames', href: 'experiments.html#frame-count-study' },
    { id: 4, title: 'Camera intrinsics', href: 'camera-comparison.html' },
    { id: 5, title: 'Exhaustive vs. sequential', href: 'matching.html' },
    { id: 6, title: 'COLMAP GUI experiments', href: 'gui-experiments.html' },
    { id: 7, title: 'Insta360 X5 · full vs. 4×', href: 'insta360.html' },
    { id: 8, title: 'Fixing gsplat · full resolution', href: 'fixing-gsplat.html' },
  ];
  const header = document.querySelector('.site-header');
  if (header && !header.querySelector('.always-compare')) {
    const compare = document.createElement('a');
    compare.className = 'always-compare';
    compare.href = 'compare.html';
    compare.textContent = '00 · Compare results';
    header.append(compare);
  }
  const current = document.body.dataset.chapter === undefined ? -1 : Number(document.body.dataset.chapter);
  const nav = document.querySelector('.study-navigation');
  const index = chapters.findIndex(chapter => chapter.id === current);
  if (!nav || index < 0) return;
  const previous = chapters[index - 1];
  const next = chapters[index + 1];
  const select = document.createElement('select');
  select.id = 'chapter-select';
  select.setAttribute('aria-label', 'Choose a study chapter');
  chapters.forEach(chapter => {
    const option = document.createElement('option');
    option.value = chapter.href;
    option.textContent = `${String(chapter.id).padStart(2, '0')} · ${chapter.title}`;
    option.selected = chapter.id === current;
    select.append(option);
  });
  select.addEventListener('change', () => { location.href = select.value; });
  const link = (label, chapter, className) => {
    if (!chapter) {
      const span = document.createElement('span');
      span.className = `${className} is-disabled`;
      span.setAttribute('aria-disabled', 'true');
      span.textContent = label;
      return span;
    }
    const a = document.createElement('a');
    a.className = className;
    a.href = chapter.href;
    a.textContent = label;
    return a;
  };
  const contents = document.createElement('a');
  contents.href = 'index.html';
  contents.className = 'nav-contents';
  contents.textContent = 'Contents';
  const picker = document.createElement('label');
  picker.htmlFor = select.id;
  picker.append(document.createTextNode('Chapter '), select);
  nav.replaceChildren(contents, link('← Previous', previous, 'nav-prev'), picker, link('Next →', next, 'nav-next'));
})();
