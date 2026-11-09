"""
Unit tests for the windowed inference module.

See Also
--------
docs/references.md#ai-usage : AI assistance log
"""

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from mangrove_mapper.inference import (
    NODATA,
    extent_tile,
    iter_windows,
    predict_tile,
)

RES = 10  # 10 m pixels, so one pixel is 0.01 ha


def _write(path, arr, descriptions=None, nodata=None):
    """Write an (bands, H, W) array as a UTM 21N GeoTiff on a 10 m grid."""
    count, height, width = arr.shape
    profile = {
        "driver": "GTiff", "count": count, "height": height, "width": width,
        "dtype": arr.dtype, "crs": "EPSG:32621",
        "transform": from_origin(500000, 500000, RES, RES),
    }
    if nodata is not None:
        profile["nodata"] = nodata
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr)
        for i, name in enumerate(descriptions or [], start=1):
            dst.set_band_description(i, name)
    return str(path)


class _StubModel:
    """Returns a fixed P(mangrove) for every row, like a trained classifier."""

    classes_ = [0, 1]

    def __init__(self, proba=0.5):
        self.proba = proba

    def predict_proba(self, x):
        p = np.full(x.shape[0], self.proba)
        return np.column_stack([1 - p, p])


@pytest.fixture
def habitat_stack(tmp_path):
    # NDVI/mNDWI chosen so every pixel is habitat except the bottom-left one
    ndvi = np.array([[0.6, 0.6], [0.1, 0.6]], dtype=np.float32)
    mndwi = np.full((2, 2), -0.2, dtype=np.float32)
    return _write(tmp_path / "stack.tif", np.stack([ndvi, mndwi]),
                  ["s2_NDVI", "s2_mNDWI"])


class TestIterWindows:
    class _Grid:
        def __init__(self, width, height):
            self.width, self.height = width, height

    @pytest.mark.parametrize(("width", "height", "block"), [
        (3317, 3458, 512),   # the region-2 tile
        (100, 100, 7),       # block that divides neither side
        (513, 511, 512),     # one px either side of a single block
        (1, 1, 512),         # raster smaller than the block
    ])
    def test_covers_every_pixel_exactly_once(self, width, height, block):
        seen = np.zeros((height, width), dtype=np.int32)
        for win in iter_windows(self._Grid(width, height), block):
            seen[win.row_off:win.row_off + win.height,
                 win.col_off:win.col_off + win.width] += 1

        assert (seen == 1).all()

    def test_windows_never_exceed_the_block(self):
        wins = list(iter_windows(self._Grid(100, 100), 32))

        assert all(w.width <= 32 and w.height <= 32 for w in wins)


class TestPredictTile:
    def test_raises_when_stack_is_missing_a_feature(self, habitat_stack, tmp_path):
        with pytest.raises(ValueError, match="missing bands"):
            predict_tile(_StubModel(), ["s2_NDVI", "alos_HH"],
                         habitat_stack, tmp_path / "p.tif")

    def test_raises_on_unexpected_class_order(self, habitat_stack, tmp_path):
        model = _StubModel()
        model.classes_ = [1, 0]

        with pytest.raises(ValueError, match="unexpected class order"):
            predict_tile(model, ["s2_NDVI", "s2_mNDWI"],
                         habitat_stack, tmp_path / "p.tif")


class TestExtentTile:
    @pytest.fixture
    def tiles(self, tmp_path, habitat_stack):
        # p = 0.9 everywhere except the bottom-right pixel, which is nodata
        prob = np.array([[[229, 229], [229, NODATA]]], dtype=np.uint8)
        # top-right is mudflat
        gsw = np.array([[[0, 80], [0, 0]]], dtype=np.uint8)
        return {
            "prob_path": _write(tmp_path / "prob.tif", prob, nodata=NODATA),
            "stack_path": habitat_stack,
            "gsw_path": _write(tmp_path / "gsw.tif", gsw),
            "extent_path": str(tmp_path / "extent.tif"),
        }

    def test_rejects_a_percentage_threshold(self, tiles):
        # a T50 map is 0.5; 50 would otherwise yield a silently empty map
        with pytest.raises(ValueError, match="probability in"):
            extent_tile(threshold=50, **tiles)

    def test_raises_when_stack_is_missing_habitat_bands(self, tiles):
        tiles["stack_path"] = tiles["prob_path"]  # no s2_NDVI / s2_mNDWI

        with pytest.raises(ValueError, match="missing bands"):
            extent_tile(threshold=0.5, **tiles)
