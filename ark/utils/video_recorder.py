from __future__ import annotations

from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np


class VideoRecorder:
    """
    Video recorder for Ark environments.
    """

    def __init__(
        self,
        out_path: str | Path,
        fps: int = 20,
        obs_rgb_key: str = "rgb",
    ) -> None:
        self.out_path = Path(out_path)
        self.fps = fps
        self.obs_rgb_key = obs_rgb_key
        self._writer = None

    def start(self) -> None:
        if self._writer is None:
            self.out_path.parent.mkdir(parents=True, exist_ok=True)
            self._writer = imageio.get_writer(self.out_path, fps=self.fps)

    def add_frame(self, obs: dict[str, Any]) -> None:
        """
        Extract an RGB frame from observation dict and append to the video.
        """
        if self._writer is None:
            self.start()

        frame = obs.get(self.obs_rgb_key)
        if frame is None:
            # print("empty  frame")
            return

        arr = np.asarray(frame)
        # Handle batched obs by taking first element
        if arr.ndim == 4:
            arr = arr[0]

        # Normalize if float images are in [0, 1]
        if arr.dtype != np.uint8:
            if arr.max() <= 1.0:
                arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
            else:
                arr = arr.astype(np.uint8)

        # imageio expects HWC
        if arr.shape[-1] != 3 and arr.shape[0] == 3:
            arr = np.transpose(arr, (1, 2, 0))

        self._writer.append_data(arr)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None
