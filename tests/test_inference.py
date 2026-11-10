"""
Unit tests for the windowed inference module.

See Also
--------
docs/references.md#ai-usage : AI assistance log
"""

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import transform as warp_transform
from shapely.geometry import box

from mangrove_mapper.inference import (
    NODATA,
    apply_landward_mask,
    distance_to_water,
    extent_tile,
    iter_windows,
    landward_cutoff,
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


ORIGIN = (500000, 500000)  # top-left of every test grid, UTM 21N


def _write_on(path, arr, res, nodata=None, origin=ORIGIN):
    """Write a single (H, W) band as a UTM 21N GeoTiff with `res` m pixels."""
    profile = {
        "driver": "GTiff", "count": 1, "height": arr.shape[0], "width": arr.shape[1],
        "dtype": arr.dtype, "crs": "EPSG:32621", "transform": from_origin(*origin, res, res),
    }
    if nodata is not None:
        profile["nodata"] = nodata
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr, 1)
    return str(path)


def _land(path, xmin, ymin, xmax, ymax):
    """A land polygon file (UTM 21N) covering the given box."""
    gpd.GeoDataFrame(geometry=[box(xmin, ymin, xmax, ymax)], crs="EPSG:32621").to_file(path)
    return str(path)


def _lonlat_of_pixel(row, col, res):
    """EPSG:4326 coordinates of a pixel centre on a test grid."""
    x = ORIGIN[0] + (col + 0.5) * res
    y = ORIGIN[1] - (row + 0.5) * res
    lon, lat = warp_transform("EPSG:32621", "EPSG:4326", [x], [y])
    return lon[0], lat[0]


def _read(path):
    with rasterio.open(path) as src:
        return src.read(1)


class TestDistanceToWater:
    RES = 30

    @pytest.fixture
    def everywhere_land(self, tmp_path):
        x0, y0 = ORIGIN
        return _land(tmp_path / "land.geojson", x0 - 1000, y0 - 1000, x0 + 1000, y0 + 1000)

    def test_distance_in_metres_from_permanent_water(self, tmp_path, everywhere_land):
        occ = np.zeros((5, 5), dtype=np.uint8)
        occ[2, 2] = 100
        gsw = _write_on(tmp_path / "gsw.tif", occ, self.RES)

        dist = _read(distance_to_water(gsw, everywhere_land, tmp_path / "dist.tif"))

        assert dist[2, 2] == 0
        assert dist[2, 3] == pytest.approx(30)
        assert dist[3, 3] == pytest.approx(30 * np.sqrt(2))
        assert dist[0, 0] == pytest.approx(60 * np.sqrt(2))

    def test_occurrence_threshold(self, tmp_path, everywhere_land):
        occ = np.zeros((3, 3), dtype=np.uint8)
        occ[1, 1] = 95
        occ[0, 0] = 100
        gsw = _write_on(tmp_path / "gsw.tif", occ, self.RES)

        strict = _read(distance_to_water(gsw, everywhere_land, tmp_path / "strict.tif"))
        loose = _read(distance_to_water(gsw, everywhere_land, tmp_path / "loose.tif",
                                        occurrence=90))

        # 95% is not permanent water by default, but is at a 90% threshold
        assert strict[1, 1] > 0
        assert loose[1, 1] == 0

    def test_outside_land_counts_as_sea(self, tmp_path):
        occ = np.zeros((3, 4), dtype=np.uint8)
        gsw = _write_on(tmp_path / "gsw.tif", occ, self.RES)
        x0, y0 = ORIGIN
        # land covers the left two columns only; the rest is sea
        land = _land(tmp_path / "land.geojson", x0 - 100, y0 - 200, x0 + 60, y0 + 100)

        dist = _read(distance_to_water(gsw, land, tmp_path / "dist.tif"))

        assert (dist[:, 2:] == 0).all()
        assert dist[:, 1] == pytest.approx([30] * 3)
        assert dist[:, 0] == pytest.approx([60] * 3)

    def test_output_is_float32_on_gsw_grid(self, tmp_path, everywhere_land):
        gsw = _write_on(tmp_path / "gsw.tif", np.zeros((4, 6), dtype=np.uint8), self.RES)

        out = distance_to_water(gsw, everywhere_land, tmp_path / "dist.tif")

        with rasterio.open(gsw) as g, rasterio.open(out) as d:
            assert d.dtypes[0] == "float32"
            assert (d.width, d.height, d.transform, d.crs) == \
                (g.width, g.height, g.transform, g.crs)


class TestLandwardCutoff:
    RES = 30

    @pytest.fixture
    def dist(self, tmp_path):
        # distance grows by 100 m per column: 0, 100, ..., 900
        arr = np.tile(np.arange(10, dtype=np.float32) * 100, (3, 1))
        return _write_on(tmp_path / "dist.tif", arr, self.RES)

    def test_percentile_of_distance_at_points(self, dist):
        coords = [_lonlat_of_pixel(1, c, self.RES) for c in range(10)]

        assert landward_cutoff(coords, dist, q=100) == pytest.approx(900)
        assert landward_cutoff(coords, dist, q=50) == pytest.approx(450)
        assert landward_cutoff(coords, dist) == pytest.approx(np.percentile(np.arange(10) * 100, 99))

    def test_point_outside_raster_raises(self, dist):
        coords = [_lonlat_of_pixel(1, 2, self.RES), _lonlat_of_pixel(1, 50, self.RES)]

        with pytest.raises(ValueError, match="1 of 2 points"):
            landward_cutoff(coords, dist)

    def test_empty_coords_raise(self, dist):
        with pytest.raises(ValueError, match="no points"):
            landward_cutoff(np.empty((0, 2)), dist)


class TestApplyLandwardMask:
    @pytest.fixture
    def extent(self, tmp_path):
        arr = np.array([[1, 1, 1, 1],
                        [0, 0, 1, 1],
                        [NODATA, 1, 0, 1]], dtype=np.uint8)
        return _write_on(tmp_path / "extent.tif", arr, RES, nodata=NODATA)

    @pytest.fixture
    def dist(self, tmp_path):
        # same 10 m grid: the right half is beyond a 500 m cutoff
        arr = np.array([[0, 400, 600, 900]] * 3, dtype=np.float32)
        return _write_on(tmp_path / "dist.tif", arr, RES)

    def test_only_far_mangrove_is_removed(self, tmp_path, extent, dist):
        out = tmp_path / "masked.tif"

        apply_landward_mask(extent, dist, 500, out)

        assert (_read(out) == np.array([[1, 1, 0, 0],
                                        [0, 0, 0, 0],
                                        [NODATA, 1, 0, 0]], dtype=np.uint8)).all()

    def test_domain_and_grid_unchanged(self, tmp_path, extent, dist):
        out = tmp_path / "masked.tif"

        apply_landward_mask(extent, dist, 500, out)

        before, after = _read(extent), _read(out)
        # nodata stays nodata and nothing becomes mangrove
        assert ((before == NODATA) == (after == NODATA)).all()
        assert not ((after == 1) & (before != 1)).any()
        with rasterio.open(extent) as e, rasterio.open(out) as m:
            assert (m.transform, m.crs, m.nodata) == (e.transform, e.crs, e.nodata)

    def test_returns_kept_and_removed_hectares(self, tmp_path, extent, dist):
        kept, removed = apply_landward_mask(extent, dist, 500, tmp_path / "masked.tif")

        # 10 m pixels are 0.01 ha: 3 mangrove pixels kept, 5 removed
        assert kept == pytest.approx(0.03)
        assert removed == pytest.approx(0.05)

    def test_warps_coarser_distance_grid(self, tmp_path, extent):
        # one 30 m distance pixel covers the extent's first three columns
        coarse = _write_on(tmp_path / "dist30.tif",
                           np.array([[100, 900]], dtype=np.float32), 30)
        out = tmp_path / "masked.tif"

        apply_landward_mask(extent, coarse, 500, out)

        after = _read(out)
        assert (after[:, :3] == _read(extent)[:, :3]).all()
        assert not (after[:, 3] == 1).any()

    def test_pixels_without_distance_are_kept(self, tmp_path, extent):
        # distance raster covers only the first two columns
        partial = _write_on(tmp_path / "dist.tif", np.full((3, 2), 900, dtype=np.float32), RES)
        out = tmp_path / "masked.tif"

        apply_landward_mask(extent, partial, 500, out)

        after = _read(out)
        assert not (after[:, :2] == 1).any()
        assert (after[:, 2:] == _read(extent)[:, 2:]).all()

    def test_no_partial_file_left_behind(self, tmp_path, extent, dist):
        out = tmp_path / "masked.tif"

        apply_landward_mask(extent, dist, 500, out)

        assert out.exists()
        assert not (tmp_path / "masked.tif.part").exists()

    @pytest.mark.parametrize("cutoff", [0, -10])
    def test_rejects_non_positive_cutoff(self, tmp_path, extent, dist, cutoff):
        with pytest.raises(ValueError, match="cutoff"):
            apply_landward_mask(extent, dist, cutoff, tmp_path / "masked.tif")
