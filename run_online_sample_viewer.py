from multiprocessing import freeze_support
from pathlib import Path
from functools import wraps
import pathlib
import platform

import torch

# config.yml was created on Windows and embeds pathlib.WindowsPath objects in YAML.
# On Linux, WindowsPath cannot be instantiated, so we remap it to the native Path
# (PosixPath on Linux) so the YAML loader can construct the paths correctly.
if platform.system() != "Windows":
    pathlib.WindowsPath = Path  # type: ignore[attr-defined]

from nerfstudio.scripts.viewer.run_viewer import _start_viewer
from nerfstudio.utils.eval_utils import eval_setup


def main() -> None:
    torch_load = torch.load

    @wraps(torch_load)
    def load_local_checkpoint(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return torch_load(*args, **kwargs)

    torch.load = load_local_checkpoint

    config_path = Path("outputs-online-sample/unnamed/nerfacto/2026-06-02_120616/config.yml")
    config, pipeline, _, step = eval_setup(config_path, eval_num_rays_per_chunk=None, test_mode="inference")
    config.vis = "viewer"
    config.viewer.websocket_port = 7008
    config.viewer.num_rays_per_chunk = 512
    _start_viewer(config, pipeline, step)


if __name__ == "__main__":
    freeze_support()
    main()
