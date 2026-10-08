"""Audit an explicit source allowlist; do not publish the enclosing photo workspace."""
from pathlib import Path
import json
R=Path(__file__).resolve().parents[1]
allowed=[R/'README.md',R/'.gitignore',R/'requirements.txt',R/'requirements-gsplat.lock.txt',*sorted((R/'scripts').glob('*.py')),*sorted((R/'reports').glob('*.json')),*sorted((R/'reports').glob('*.csv')),*sorted(p for p in (R/'site/dist').glob('*') if p.is_file()),*sorted(p for p in (R/'site/dist/client').glob('*') if p.suffix in {'.html','.css','.js','.mjs'}),R/'site/worker.js',R/'site/build-worker.mjs',R/'site/tests/worker.test.mjs']
for p in allowed:
 assert p.is_file(),p
 assert p.suffix in {'.py','.md','.txt','.json','.csv','.html','.css','.js','.mjs'} or p.name=='.gitignore',p
 data=p.read_text();assert ('data:'+'image/') not in data and ('BEGIN '+'PRIVATE KEY') not in data,p
 assert p.stat().st_size<1_000_000,(p,'unexpectedly large text file')
print(json.dumps([str(p.relative_to(R)) for p in allowed],indent=2))
