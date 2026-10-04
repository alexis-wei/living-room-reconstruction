"""Official trainer entry with PyTorch 2.6 build-path quoting compatibility."""
import runpy
import json
import os
import shlex
import sys
from pathlib import Path
import torch.utils.cpp_extension as extension

# Upstream PyTorch's Ninja generator leaves -L paths unquoted. The project has
# a space in its name; quote only library-search flags, without changing CUDA
# code or training logic. This adapter is also used for CPU-only compilation.
original_prepare_ldflags = extension._prepare_ldflags

def quoted_ldflags(*args, **kwargs):
    return [shlex.quote(flag) if flag.startswith('-L') and ' ' in flag else flag
            for flag in original_prepare_ldflags(*args, **kwargs)]

extension._prepare_ldflags = quoted_ldflags
if '--compile-only' in sys.argv:
    from gsplat.cuda._backend import _C
    assert _C is not None
    print('gsplat CUDA extension compiled')
else:
    trainer = Path(__file__).resolve().parents[2] / '.gsplat-src/gsplat/examples/simple_trainer.py'
    sys.path.insert(0, str(trainer.parent))
    # Matching trials hold out the same source frames even when COLMAP recovers
    # different subsets. Leave the original trainer split unchanged otherwise.
    if os.environ.get('GSPLAT_HOLDOUT_FILE'):
        import numpy as np
        from datasets.colmap import Dataset
        held_out = set(json.loads(Path(os.environ['GSPLAT_HOLDOUT_FILE']).read_text()))
        original_dataset_init = Dataset.__init__

        def fixed_split(self, parser, split='train', **kwargs):
            original_dataset_init(self, parser, split=split, **kwargs)
            use_validation = split != 'train'
            self.indices = np.asarray([i for i, name in enumerate(parser.image_names)
                                       if (name in held_out) == use_validation], dtype=int)
            if not len(self.indices):
                raise ValueError(f'No images in the fixed {split} split')

        Dataset.__init__ = fixed_split
    sys.argv[0] = str(trainer)
    runpy.run_path(str(trainer), run_name='__main__')
