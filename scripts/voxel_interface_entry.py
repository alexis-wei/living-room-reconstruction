"""Local UI adapter around the unmodified official gsplat example trainer.

Only localhost binding, initial pause, camera bookmarks and status reporting
are added. Rasterization, optimization and densification stay upstream.
"""
import json
import math
import os
import runpy
import sys
import threading
import time
from pathlib import Path

import numpy as np
import viser
import viser.transforms as vt

WORKSPACE = Path(__file__).resolve().parents[2]
EXAMPLES = WORKSPACE / '.gsplat-src/gsplat/examples'
sys.path.insert(0, str(EXAMPLES))
import gsplat_viewer

OriginalServer = viser.ViserServer
OriginalViewer = gsplat_viewer.GsplatViewer


class LocalServer(OriginalServer):
    def __init__(self, *args, **kwargs):
        kwargs['host'] = '127.0.0.1'
        super().__init__(*args, **kwargs)


class LocalViewer(OriginalViewer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        runner = getattr(kwargs.get('render_fn'), '__self__', None)
        self.server.gui.set_panel_label('gsplat · voxel input experiment')
        if self.mode == 'training' and os.environ.get('GSPLAT_START_PAUSED') == '1':
            self.state = 'paused'
            self._training_tab_handles['pause_train_button'].visible = False
            self._training_tab_handles['resume_train_button'].visible = True
        if runner is not None and hasattr(runner, 'parser'):
            self._add_camera_bookmarks(runner)
        status_path = os.environ.get('GSPLAT_INTERACTIVE_STATUS')
        if status_path:
            threading.Thread(target=self._report_status, args=(Path(status_path),), daemon=True).start()
        print(f'Official gsplat viewer ready; state={self.state}', flush=True)

    def _add_camera_bookmarks(self, runner):
        parser = runner.parser
        names = parser.image_names
        selected = 'frame_0337.png' if 'frame_0337.png' in names else names[0]
        with self.server.gui.add_folder('Input experiment'):
            self.server.gui.add_markdown(
                f'**{len(parser.points):,} initial sparse points**, {len(names)} cameras. '
                'Same undistorted 4× photos; data factor 1. Voxel sizes are model units, not meters. '
                'Resume starts training; densification can increase the point count.'
            )
            bookmark = self.server.gui.add_dropdown('Recovered camera', options=names, initial_value=selected)

        def set_camera(client):
            index = names.index(bookmark.value)
            matrix = parser.camtoworlds[index]
            camera_id = parser.camera_ids[index]
            _, height = parser.imsize_dict[camera_id]
            fy = parser.Ks_dict[camera_id][1, 1]
            with client.atomic():
                client.camera.wxyz = vt.SO3.from_matrix(matrix[:3, :3]).wxyz
                client.camera.position = matrix[:3, 3]
                client.camera.fov = float(2 * math.atan(height / (2 * fy)))

        @self.server.on_client_connect
        def on_connect(client):
            set_camera(client)

        @bookmark.on_update
        def on_bookmark(_):
            for client in self.server.get_clients().values():
                set_camera(client)

    def _report_status(self, path):
        while True:
            record = json.loads(path.read_text())
            record.update(state=self.state, step=int(self._step), trainer_pid=os.getpid(),
                          status_updated_unix=time.time())
            if self.state == 'completed':
                stats = sorted((self.output_dir / 'stats').glob('val_step*.json'))
                record['validation_metrics'] = {p.name: json.loads(p.read_text()) for p in stats}
                record['checkpoints'] = [str(p) for p in sorted((self.output_dir / 'ckpts').glob('*.pt'))]
            temporary = path.with_suffix('.ui.tmp')
            temporary.write_text(json.dumps(record, indent=2) + '\n')
            temporary.replace(path)
            time.sleep(2)


viser.ViserServer = LocalServer
gsplat_viewer.GsplatViewer = LocalViewer
runpy.run_path(str(Path(__file__).with_name('gsplat_entry.py')), run_name='__main__')
