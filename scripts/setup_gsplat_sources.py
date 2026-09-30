"""Download pinned official gsplat source and a project-local CUDA compiler."""
import hashlib
import json
import shutil
import tarfile
import urllib.request
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
CACHE = WORKSPACE / '.gsplat-cache'
SRC = WORKSPACE / '.gsplat-src'
CUDA = WORKSPACE / '.cuda-12.4'

def fetch(url, destination):
    if not destination.exists():
        print('Downloading', url, flush=True)
        urllib.request.urlretrieve(url, destination)
    return destination

def extract(archive, destination):
    with tarfile.open(archive) as tar:
        root = Path(tar.getmembers()[0].name).parts[0]
        temporary = CACHE / (root + '-unpack')
        temporary.mkdir(exist_ok=True)
        for entry in tar.getmembers():
            if entry.name.startswith('/') or '..' in Path(entry.name).parts:
                raise RuntimeError('Unsafe archive path')
        tar.extractall(temporary)
        shutil.copytree(temporary / root, destination, dirs_exist_ok=True)

def main():
    for path in [CACHE, SRC, CUDA]: path.mkdir(exist_ok=True)
    ref = json.load(urllib.request.urlopen('https://api.github.com/repos/nerfstudio-project/gsplat/git/ref/tags/v1.5.3'))['object']
    if ref['type'] == 'tag': ref = json.load(urllib.request.urlopen(ref['url']))['object']
    commit = ref['sha']
    extract(fetch(f'https://codeload.github.com/nerfstudio-project/gsplat/tar.gz/{commit}', CACHE / f'gsplat-{commit}.tar.gz'), SRC / 'gsplat')
    base = 'https://developer.download.nvidia.com/compute/cuda/redist/'
    manifest = json.load(urllib.request.urlopen(base + 'redistrib_12.4.1.json'))
    packages = {}
    for name in ['cuda_nvcc', 'cuda_cudart', 'cuda_cccl']:
        item = manifest[name]['linux-x86_64']
        archive = fetch(base + item['relative_path'], CACHE / Path(item['relative_path']).name)
        if hashlib.sha256(archive.read_bytes()).hexdigest() != item['sha256']:
            raise RuntimeError('CUDA archive checksum mismatch: ' + name)
        extract(archive, CUDA)
        packages[name] = item
    record = {'gsplat_release': '1.5.3', 'gsplat_commit': commit, 'cuda_toolkit': '12.4.1', 'cuda_packages': packages}
    (WORKSPACE / 'gsplat_local' / 'source_versions.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record), flush=True)

if __name__ == '__main__': main()
