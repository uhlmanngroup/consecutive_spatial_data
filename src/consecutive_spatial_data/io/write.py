import math

import numpy
import tifffile
from tqdm import tqdm
from xarray import DataArray, DataTree
import dask_image.ndinterp

class MultiscaleImageWriter:
    photometric = "minisblack"
    # compression = "zlib"
    compression = None
    resolutionunit = "CENTIMETER"
    dtype = numpy.uint8

    def __init__(self, image: DataTree, tile_width: int, pixel_size: float):
        self.image = image
        self.tile_width = tile_width
        self.pixel_size = pixel_size

        # Get scale names from the DataTree (ie. scale0, scale1 etc.)
        self.scale_names = list(self.image.children)
        self.channel_names = list(map(str, self.image[self.scale_names[0]].c.values))
        self.metadata = {
            "SignificantBits": 8,
            "PhysicalSizeX": pixel_size,
            "PhysicalSizeXUnit": "µm",
            "PhysicalSizeY": pixel_size,
            "PhysicalSizeYUnit": "µm",
            "Channel": {"Name": self.channel_names},
        }
        self.data = None

        self.lazy = True
        self.ram_threshold_gb = None

    def _n_tiles_axis(self, xarr: DataArray, axis: int) -> int:
        """Find the number of tiles required per axis"""
        return math.ceil(xarr.shape[axis] / self.tile_width)

    def _get_tiles(self, xarr: DataArray):
        """Create a tile generator"""
        for c in range(xarr.shape[0]):
            for index_y in range(self._n_tiles_axis(xarr, 1)):
                for index_x in range(self._n_tiles_axis(xarr, 2)):
                    tile = xarr[
                        c,
                        self.tile_width * index_y : self.tile_width * (index_y + 1),
                        self.tile_width * index_x : self.tile_width * (index_x + 1),
                    ]
                    yield self._scale(tile.values)

    def _should_load_memory(self, shape: tuple[int, int, int], dtype: numpy.dtype):
        if not self.lazy:
            return True

        if self.ram_threshold_gb is None:
            return False

        itemsize = max(numpy.dtype(dtype).itemsize, numpy.dtype(self.dtype).itemsize)
        size = shape[0] * shape[1] * shape[2] * itemsize

        return size <= self.ram_threshold_gb * self.tile_width**3

    def _scale(self, array: numpy.ndarray):
        """Scale to a desired dtype"""
        return scale_dtype(array, self.dtype)

    def _write_image_level(self, tif: tifffile.TiffWriter, scale_index: int, **kwargs):
        xarr: DataArray = next(iter(self.image[self.scale_names[scale_index]].values()))
        resolution = 1e4 * 2**scale_index / self.pixel_size

        if not self._should_load_memory(xarr.shape, xarr.dtype):
            # Dynamically write tiles without loading into memory
            n_tiles = (
                xarr.shape[0]
                * self._n_tiles_axis(xarr, 1)
                * self._n_tiles_axis(xarr, 2)
            )
            data = self._get_tiles(xarr)
            data = iter(tqdm(data, total=n_tiles, desc="Writing tiles"))
        else:
            # Load array into memory and scale
            if self.data is not None:
                self.data = resize_numpy(self.data, 2, xarr.dims, xarr.shape)
            else:
                self.data = self._scale(xarr.values)

            data = self.data

        tif.write(
            data,
            tile=(self.tile_width, self.tile_width),
            resolution=(resolution, resolution),
            metadata=self.metadata,
            shape=xarr.shape,
            dtype=self.dtype,
            photometric=self.photometric,
            compression=self.compression,
            resolutionunit=self.resolutionunit,
            **kwargs,
        )

    def __len__(self):
        return len(self.scale_names)

    def write(self, path, lazy=True, ram_threshold_gb=None):
        self.lazy = lazy
        self.ram_threshold_gb = ram_threshold_gb

        with tifffile.TiffWriter(path, bigtiff=True) as tif:
            self._write_image_level(tif, 0, subifds=len(self) - 1)

            for i in range(1, len(self)):
                self._write_image_level(tif, i, subfiletype=1)

def scale_dtype(arr: numpy.ndarray, dtype: numpy.dtype) -> numpy.ndarray:
    """Change the dtype of an array but keep the scale compared to the type maximum value.

    !!! note "Example"
        For an array of dtype `uint8` being transformed to `numpy.uint16`, the value `255` will become `65535`

    Args:
        arr: A `numpy` array
        dtype: Target `numpy` data type

    Returns:
        A scaled `numpy` array with the dtype provided.
    """
    assert_is_integer_dtype(arr.dtype)
    assert_is_integer_dtype(dtype)

    if arr.dtype == dtype:
        return arr

    factor = numpy.iinfo(dtype).max / numpy.iinfo(arr.dtype).max
    return (arr * factor).astype(dtype)

def assert_is_integer_dtype(dtype: numpy.dtype):
    assert numpy.issubdtype(
        dtype, numpy.integer
    ), f"Expecting image to have an integer dtype, but found {dtype}"


def resize_numpy(
    arr: numpy.ndarray, scale_factor: float, dims: list[str], output_shape: list[int]
) -> numpy.ndarray:
    """Resize a numpy image

    Args:
        arr: a `numpy` array
        scale_factor: Scale factor of resizing, e.g. `2` will decrease the width by 2
        dims: List of dimension names. Only `"x"` and `"y"` are resized.
        output_shape: Size of the output array

    Returns:
        Resized array
    """
    resize_dims = [dim in ["x", "y"] for dim in dims]
    transform = numpy.diag(
        [scale_factor if resize_dim else 1 for resize_dim in resize_dims]
    )

    return dask_image.ndinterp.affine_transform(
        arr, matrix=transform, output_shape=output_shape
    ).compute()
