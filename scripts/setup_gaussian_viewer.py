"""Fetch pinned viewer libraries into the private Site checkout."""
from pathlib import Path
from urllib.request import urlretrieve
OUT=Path(__file__).resolve().parents[1]/'site/dist/vendor'
OUT.mkdir(exist_ok=True)
urls={
 'spark.module.js':'https://sparkjs.dev/releases/spark/2.3.0/spark.module.js',
 'three.module.js':'https://cdn.jsdelivr.net/npm/three@0.180.0/build/three.module.js',
 'three.core.js':'https://cdn.jsdelivr.net/npm/three@0.180.0/build/three.core.js',
 'OrbitControls.js':'https://cdn.jsdelivr.net/npm/three@0.180.0/examples/jsm/controls/OrbitControls.js',
 'Pass.js':'https://cdn.jsdelivr.net/npm/three@0.180.0/examples/jsm/postprocessing/Pass.js',
}
for name,url in urls.items():urlretrieve(url,OUT/name)
print('Viewer libraries ready.')
