import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from gymnasium.spaces import Box, Dict as GymDict
from typing import Any, SupportsFloat


class Image(Box):
    __slots__ = ("height", "width", "color_channels")

    def __init__(
        self,
        height: int,
        width: int,
        color_channels: int,
        dtype: type[np.integer[Any]] | type[np.floating[Any]],
        high: SupportsFloat | NDArray[Any] | None = None,
        seed: int | np.random.Generator | None = None,
    ):
        if height < 1 or width < 1:
            raise ValueError("height and width must be positive.")
        if color_channels < 1:
            raise ValueError("color_channels must be positive.")

        self.height = height
        self.width = width
        self.color_channels = color_channels

        if color_channels == 1:
            shape = (height, width)
        else:
            shape = (height, width, color_channels)

        low = np.zeros(shape, dtype=dtype)
        if high is None:
            high = np.iinfo(dtype).max if np.issubdtype(dtype, np.integer) else 1.0

        super().__init__(
            low=low,
            high=high,
            shape=shape,
            dtype=dtype,
            seed=seed,
        )

    def view(self, sample: NDArray[Any]):
        raise NotImplementedError("ImageSpace does not support viewing.")


class GrayscaleImage(Image):

    def __init__(
        self,
        height: int,
        width: int,
        dtype: type[np.integer[Any]] | type[np.floating[Any]],
        high: SupportsFloat | NDArray[Any] | None = None,
        seed: int | np.random.Generator | None = None,
    ):
        super().__init__(
            height=height,
            width=width,
            color_channels=1,
            dtype=dtype,
            high=high,
            seed=seed,
        )

    def view(self, sample: NDArray[Any]) -> tuple[plt.Figure, plt.Axes]:
        sample = np.asarray(sample, dtype=self.dtype)
        if not self.contains(sample):
            raise ValueError("Sample is not contained in the space.")
        fig, ax = plt.subplots()
        ax.imshow(
            sample,
            cmap="gray",
            vmin=float(np.min(self.low)),
            vmax=float(np.max(self.high)),
        )
        ax.axis("off")
        return fig, ax


class RGBImage(Image):

    def __init__(
        self,
        height: int,
        width: int,
        dtype: type[np.integer[Any]] | type[np.floating[Any]],
        high: SupportsFloat | NDArray[Any] | None = None,
        seed: int | np.random.Generator | None = None,
    ):
        super().__init__(
            height=height,
            width=width,
            color_channels=3,
            dtype=dtype,
            high=high,
            seed=seed,
        )

    def view(self, sample: NDArray[Any]) -> tuple[plt.Figure, plt.Axes]:
        sample = np.asarray(sample, dtype=self.dtype)
        if not self.contains(sample):
            raise ValueError("Sample is not contained in the space.")
        fig, ax = plt.subplots()
        ax.imshow(sample)
        ax.axis("off")
        return fig, ax


class DepthImage(Image):

    def __init__(
        self,
        height: int,
        width: int,
        dtype: type[np.integer[Any]] | type[np.floating[Any]],
        high: SupportsFloat | NDArray[Any] | None = None,
        seed: int | np.random.Generator | None = None,
    ):
        super().__init__(
            height=height,
            width=width,
            color_channels=1,
            dtype=dtype,
            high=high,
            seed=seed,
        )

    def view(self, sample: NDArray[Any]) -> tuple[plt.Figure, plt.Axes]:
        sample = np.asarray(sample, dtype=self.dtype)
        if not self.contains(sample):
            raise ValueError("Sample is not contained in the space.")
        fig, ax = plt.subplots()
        ax.imshow(sample, cmap="viridis")
        # Add color map legend
        cbar = fig.colorbar(ax.images[0], ax=ax)
        cbar.set_label("Depth")
        ax.axis("off")
        return fig, ax


class RGBDImage(GymDict):

    def __init__(
        self,
        height: int,
        width: int,
        rgb_dtype: type[np.integer[Any]] | type[np.floating[Any]],
        depth_dtype: type[np.integer[Any]] | type[np.floating[Any]],
        rgb_high: SupportsFloat | NDArray[Any] | None = None,
        depth_high: SupportsFloat | NDArray[Any] | None = None,
        seed: int | np.random.Generator | None = None,
    ):
        super().__init__(
            {
                "rgb": RGBImage(
                    height=height,
                    width=width,
                    dtype=rgb_dtype,
                    high=rgb_high,
                    seed=seed,
                ),
                "depth": DepthImage(
                    height=height,
                    width=width,
                    dtype=depth_dtype,
                    high=depth_high,
                    seed=seed,
                ),
            },
            seed=seed,
        )

    def view(
        self, sample: NDArray[Any] | dict[str, NDArray[Any]], vstack: bool = False
    ) -> tuple[plt.Figure, plt.Axes]:
        if isinstance(sample, dict):
            rgb_sample = sample.get("rgb")
            depth_sample = sample.get("depth")
        elif (
            isinstance(sample, np.ndarray) and sample.ndim == 3 and sample.shape[2] == 4
        ):
            rgb_sample = sample[:, :, :3]
            depth_sample = sample[:, :, 3]
        else:
            raise ValueError(
                "Sample must be a dict with 'rgb' and 'depth' keys or a 3D array with 4 channels."
            )

        if not self.spaces["rgb"].contains(rgb_sample):
            raise ValueError("RGB sample is not contained in the space.")
        if not self.spaces["depth"].contains(depth_sample):
            raise ValueError("Depth sample is not contained in the space.")

        fig, axes = plt.subplots(1, 2) if not vstack else plt.subplots(2, 1)

        axes[0].imshow(rgb_sample)
        axes[0].set_title("RGB Image")
        axes[0].axis("off")

        im = axes[1].imshow(depth_sample, cmap="viridis")
        axes[1].set_title("Depth Image")
        axes[1].axis("off")

        # Add color map legend for depth image
        cbar = fig.colorbar(im, ax=axes[1])
        cbar.set_label("Depth")

        return fig, axes
