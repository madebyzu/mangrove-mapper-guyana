import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.merge import merge
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform as warp_transform
from rasterio.windows import Window
from scipy.ndimage import distance_transform_edt

INFERENCE_MASK_THRESHOLD = 40
PERMANENT_WATER = 100  # GSW occurrence (%) counted as water by the landward mask
# NDVI/MNDWI mask values determined by Nedd et al. (2023)
NDVI_MIN = 0.25
MNDWI_MAX = -0.50
PROB_SCALE = 254  # 0 - 254 are valid prob scale values
NODATA = 255  # reserved for no data
PROFILE_OPTS = {"compress": "deflate", "tiled": True,
                "blockxsize": 256, "blockysize": 256}  # deflate allows for lossless compression


def iter_windows(src, block):
    """
    Row-ordered windows of at most `block` x `block` px tiling the raster `src`.

    Parameters
    ----------
    src : rasterio.DatasetReader
        Opened raster whose `width` and `height` define the grid to tile.
    block : int
        Maximum side length of each window, in pixels. Sets memory of
        when read is called.

    Yields
    ------
    rasterio.windows.Window
        Windows in row-major order, each at most `block` x `block` px.

    See Also
    --------
    docs/references.md#documentation : Rasterio Documentation Windowed reading and writing
    https://rasterio.readthedocs.io/en/latest/topics/windowed-rw.html

    """
    for row in range(0, src.height, block):
        h = min(block, src.height - row)
        for col in range(0, src.width, block):
            # yield returns a generator for lazy tiling (one window at a time, preserving memory)
            yield Window(col, row, min(block, src.width - col), h)


def predict_tile(model, feats, stack_path, prob_path, block=512):
    """
    Predicts the mangrove probabilities over a windowed single tile
    and writes those scaled probability values to `prob_path`

    Parameters
    ----------
    model : estimator
        Trained classifier using `predict_proba`, with classes [0, 1].
    feats : list of str
        Band names used by the model, in the order it was trained on.
    stack_path : pathlib.Path or str
        Path to feature stack GeoTiff for every pixel in scene.
    prob_path : pathlib.Path or str
        Output path to write probability raster. Single uint8 band, where
        0 - 254 is p(mangrove) x PROB_SCALE and 255 is nodata.
    block : int
        Maximum side length of each window, in pixels. Defaults to 512.

    Returns
    -------
    pathlib.Path or str
        Output path where probability raster has been written.

    Raises
    ------
    ValueError
        If the stack is missing any band in `feats`.

    See Also
    --------
    docs/references.md#documentation : Rasterio Documentation Nodata Masks
    https://rasterio.readthedocs.io/en/latest/topics/masks.html

    """
    with rasterio.open(stack_path) as src:
        names = list(src.descriptions)  # feature names
        missing = [f for f in feats if f not in names]
        if missing:
            raise ValueError(
                f"{Path(stack_path).name} is missing bands: {missing}")
        # rasterio bands are 1-based
        idx = [names.index(f) + 1 for f in feats]

        # Depending on how the feature stack was exported, the band order may already match.
        # If so, the whole pixel-interleaved block can be read in one go.
        # If not, say a re-export reorders bands, it falls back to the indexed read.
        ordered = names == list(feats)

        classes = list(getattr(model, "classes_", [0, 1]))
        if classes != [0, 1]:
            raise ValueError(f"unexpected class order: {classes}")

        prof = src.profile.copy()
        # one uint8 band out, not the float32 multi-band stack that came in
        prof.update(count=1, dtype="uint8", nodata=NODATA, **PROFILE_OPTS)

        with rasterio.open(prob_path, "w", **prof) as dst:
            dst.set_band_description(1, f"p_mangrove_x{PROB_SCALE}")
            for win in iter_windows(src, block):
                block_arr = (src.read(window=win) if ordered
                             else src.read(idx, window=win)).astype(np.float32)
                n_band, h, w = block_arr.shape
                flat = block_arr.reshape(n_band, -1).T  # (h*w, n_band)

                valid = np.isfinite(flat).all(
                    axis=1) & ~(flat == 0).all(axis=1)
                out = np.full(flat.shape[0], NODATA, dtype=np.uint8)
                if valid.any():
                    # column 1 is the mangrove class
                    proba = model.predict_proba(flat[valid])[:, 1]
                    # quantize to 0 - 254, leaving 255 free for nodata
                    out[valid] = np.round(proba * PROB_SCALE).astype(np.uint8)
                dst.write(out.reshape(h, w), 1, window=win)
    return prob_path


def extent_tile(prob_path, stack_path, gsw_path, threshold, extent_path, block=512,
                ndvi_min=NDVI_MIN, mndwi_max=MNDWI_MAX):
    """
    Re-threshold a probability tile into an extent tile. Returns the tile's mangrove area in hectares.

    Parameters
    ----------
    prob_path : pathlib.Path or str
        Probability raster from `predict_tile`. Single uint8 band, where
        0 - 254 is p(mangrove) x PROB_SCALE and 255 is nodata.
    stack_path : pathlib.Path or str
        Path to feature stack GeoTiff, on the same grid as `prob_path`.
        Read for the s2_NDVI and s2_mNDWI habitat bands.
    gsw_path : pathlib.Path or str
        Path to Global Surface Water (GSW) GeoTiff. Warped onto the
        probability grid, so it must have no nodata set: uncovered area
        then fills with 0 and reads as dry land.
    threshold : float
        Probability cut in 0 - 1, not percent. A T50 map is threshold=0.5.
    extent_path : pathlib.Path or str
        Path to output extent map after thresholding is applied. Single
        uint8 band, where 1 is mangrove, 0 is non-mangrove, 255 is nodata.
    block : int
        Maximum side length of each window, in pixels. Defaults to 512.

    Returns
    -------
    float
        mangrove area of a single tile in hectares, assuming a projected
        CRS in metres.

    Raises
    ------
    ValueError
        If the stack is missing the s2_NDVI or s2_mNDWI band.

    Notes
    -----
    `ndvi_min` and `mndwi_max` default to the surface-reflectance values reported in
    Nedd et al. (2023). They are configurable because they are Sentinel-2 Level
    specific and need to be recalibrated when using top-of-atmosphere reflectance in
    order to obtain accurate results.

    See Also
    --------
    docs/references.md#documentation : Rasterio Documentation Virtual Warping
    https://rasterio.readthedocs.io/en/latest/topics/virtual-warping.html
    """

    if not 0 < threshold <= 1:
        raise ValueError(
            f"threshold must be a probability in (0, 1], got {threshold}")

    with rasterio.open(prob_path) as psrc, rasterio.open(stack_path) as src:
        names = list(src.descriptions)
        missing = [b for b in ("s2_NDVI", "s2_mNDWI") if b not in names]
        if missing:
            raise ValueError(
                f"{Path(stack_path).name} is missing bands: {missing}")
        # rasterio bands are 1-based
        ndvi_b, mndwi_b = names.index(
            "s2_NDVI") + 1, names.index("s2_mNDWI") + 1

        prof = psrc.profile.copy()
        prof.update(**PROFILE_OPTS)
        n_mangrove = 0

        # discrete resampling from GSW's native 30 m to 10 m resolution
        with rasterio.open(gsw_path) as gsrc, \
                WarpedVRT(gsrc, crs=psrc.crs, transform=psrc.transform,
                          width=psrc.width, height=psrc.height,
                          resampling=Resampling.nearest) as gsw, \
                rasterio.open(extent_path, "w", **prof) as dst:

            dst.set_band_description(1, f"mangrove_extent_t{threshold}")
            for win in iter_windows(psrc, block):
                pu8 = psrc.read(1, window=win)
                valid = pu8 != NODATA
                # converting back to probabilities, with a small quantization error
                proba = pu8.astype(np.float32) / PROB_SCALE

                ndvi = src.read(ndvi_b, window=win).astype(np.float32)
                mndwi = src.read(mndwi_b, window=win).astype(np.float32)
                # mask applied from Nedd et al. (2023)
                habitat = (ndvi > ndvi_min) & (mndwi > mndwi_max)
                # wet >= 40% of the GSW record, so intertidal mudflat not mangrove
                mudflat = gsw.read(1, window=win) >= INFERENCE_MASK_THRESHOLD

                extent = np.full(pu8.shape, NODATA, dtype=np.uint8)
                extent[valid] = (proba[valid] >= threshold).astype(np.uint8)
                # deliberately 0, not nodata: keeps them in the estimator's domain
                extent[valid & (mudflat | ~habitat)] = 0
                dst.write(extent, 1, window=win)
                # updating the number of mangroves after thresholding
                n_mangrove += int((extent == 1).sum())
    # pixel resolution (w x h) 10 x 10 = 100 m^2 / px
    # 1 ha = 10,000 m^2
    px_ha = abs(prof["transform"].a) * abs(prof["transform"].e) / 1e4
    return n_mangrove * px_ha


def extent_from_prob(prob_path, base_extent_path, threshold, extent_path, block=512):
    """
    Re-threshold a probability raster into an extent map

    Parameters
    ----------
    prob_path : pathlib.Path or str
        Probability raster from `predict_tile`. uint8 band, where
        0 - 254 is p(mangrove) x PROB_SCALE and 255 is nodata.
    base_extent_path : pathlib.Path or str
        extent map on the same grid as `prob_path` with feature bands and water masks applied
    threshold : float
        Probability cut between 0 and 1, (note: threshold must be at 0.5 or higher)
    extent_path : pathlib.Path or str
        Output extent map.
    block : int
        Maximum side length of each window, in pixels. Defaults to 512.

    Returns
    -------
    tuple of float
        (mangrove_ha, domain_ha): the extent's mangrove area and its total
        classified area, both in hectares. domain_ha is the base map's domain
        and the denominator for the Olofsson stratum weights.

    Raises
    ------
    ValueError
        If `threshold` is not in [0, 1], or `prob` and the base extent differ in grid.
    """

    if not 0 < threshold <= 1:
        raise ValueError(
            f"threshold must be a probability in [0, 1], got {threshold}")

    # converting threshold to match other values in probability raster (will likely be a float not int)
    cut = threshold * PROB_SCALE
    with rasterio.open(prob_path) as psrc, rasterio.open(base_extent_path) as bsrc:
        if (psrc.width, psrc.height) != (bsrc.width, bsrc.height) or \
                psrc.transform != bsrc.transform:
            raise ValueError("prob and base extent must share the same grid")

        prof = psrc.profile.copy()
        prof.update(count=1, dtype="uint8", nodata=NODATA, **PROFILE_OPTS)
        n_mangrove, total = 0, 0

        with rasterio.open(extent_path, "w", **prof) as dst:
            dst.set_band_description(1, f"mangrove_extent_t{threshold}")
            for win in iter_windows(psrc, block):
                pu8 = psrc.read(1, window=win)
                base = bsrc.read(1, window=win)

                # ensuring type is uint8 will save memory
                extent = np.full(pu8.shape, NODATA, dtype=np.uint8)
                # base carries the domain (nodata) and the mudflat/habitat masks
                extent[base != NODATA] = 0
                # we want to apply threshold to valid mangrove pixels only using the base extent
                # the result is a stricter set of mangroves compared to the base
                extent[(base == 1) & (pu8 >= cut)] = 1
                dst.write(extent, 1, window=win)
                n_mangrove += int((extent == 1).sum())
                total += int((extent != NODATA).sum())

    # pixel resolution (w x h) 10 x 10 = 100 m^2 / px
    # 1 ha = 10,000 m^2
    px_ha = abs(prof["transform"].a) * abs(prof["transform"].e) / 1e4
    return n_mangrove * px_ha, total * px_ha


def snap_to_domain(product_path, model_path, out_path, block=512):
    """
    Reproject a product extent onto the model's grid and clip it to the model domain.

    A published product (e.g. WorldCover) sits on its own grid and covers its own
    footprint. To score it against the model on equal terms, it is resampled (nearest,
    to keep the 0/1 classes intact) onto the model extent's grid and set to nodata
    everywhere the model is nodata, so both maps share exactly one classified domain.

    Parameters
    ----------
    product_path : pathlib.Path or str
        Product extent raster, 1 mangrove / 0 non-mangrove, on any grid or CRS
        (e.g. the exported `fusion.product_label`). Must cover the model domain.
    model_path : pathlib.Path or str
        Model extent map defining the target grid and domain. Single uint8 band:
        1 mangrove, 0 non-mangrove, 255 nodata.
    out_path : pathlib.Path or str
        Output snapped product extent, on the model's grid. Single uint8 band:
        1 mangrove, 0 non-mangrove, 255 nodata (outside the model domain).
    block : int
        Maximum side length of each window, in pixels. Defaults to 512.

    Returns
    -------
    pathlib.Path or str
        `out_path`.

    See Also
    --------
    docs/references.md#documentation : Rasterio Documentation Virtual Warping
    https://rasterio.readthedocs.io/en/latest/topics/virtual-warping.html
    docs/references.md#ai-usage : AI assistance log
    """
    with rasterio.open(model_path) as msrc:
        prof = msrc.profile.copy()
        prof.update(count=1, dtype="uint8", nodata=NODATA, **PROFILE_OPTS)

        # nearest keeps the categorical 0/1 classes; the VRT regrids the product
        # onto the model's exact pixels regardless of its native grid or CRS
        with rasterio.open(product_path) as psrc, \
                WarpedVRT(psrc, crs=msrc.crs, transform=msrc.transform,
                          width=msrc.width, height=msrc.height,
                          resampling=Resampling.nearest) as product, \
                rasterio.open(out_path, "w", **prof) as dst:

            dst.set_band_description(1, "mangrove_extent_product")
            for win in iter_windows(msrc, block):
                model = msrc.read(1, window=win)
                # outside the model domain -> nodata, so both maps share one domain
                out = np.full(model.shape, NODATA, dtype=np.uint8)
                in_domain = model != NODATA
                out[in_domain] = product.read(1, window=win)[in_domain]
                dst.write(out, 1, window=win)
    return out_path


def mosaic(tile_paths, out_path, nodata=NODATA):
    """
    Stitch grid-aligned tiles into one raster, preserving every band.

    Tiles that share one grid (e.g. per-tile maps from `predict_tile` /
    `extent_tile`, or a tiled `export_image_tiled`) are exact neighbours, so
    merging is a straight copy, not a resample. All bands and their descriptions
    carry through, so a single-band probability map and a multi-band feature
    stack both work. The whole mosaic is held in memory once, then written with
    BIGTIFF enabled for the large coast-wide output.

    Parameters
    ----------
    tile_paths : iterable of pathlib.Path or str
        Tiles on a shared grid, e.g. ``sorted(dir.glob(pattern))``.
    out_path : pathlib.Path or str
        Output mosaic path.
    nodata : int or None
        Nodata value used to combine the tiles and set on the output. Pass None
        to carry no nodata (e.g. a float feature stack). Default 255.

    Returns
    -------
    pathlib.Path or str
        `out_path`.

    Raises
    ------
    FileNotFoundError
        If `tile_paths` is empty.

    See Also
    --------
    docs/references.md#documentation : rasterio.merge. Rasterio Documentation.
    docs/references.md#ai-usage : AI assistance log
    """
    tiles = sorted(tile_paths)
    if not tiles:
        raise FileNotFoundError("no tiles to mosaic")

    srcs = [rasterio.open(p) for p in tiles]
    try:
        # copies valid pixels from input files (srcs) to an output file
        # outfile => new 3d array (n_bands, h, w) and transform object
        arr, transform = merge(srcs, nodata=nodata)
        prof = srcs[0].profile.copy()
        prof.update(height=arr.shape[1], width=arr.shape[2], transform=transform,
                    count=arr.shape[0], nodata=nodata, BIGTIFF="IF_SAFER", **PROFILE_OPTS)
        descs = srcs[0].descriptions  # carry the band names onto the mosaic
    finally:
        for s in srcs:
            s.close()

    with rasterio.open(out_path, "w", **prof) as dst:
        dst.write(arr)  # every band; arr is (n_bands, h, w)
        for band, name in enumerate(descs, start=1):
            if name:
                dst.set_band_description(band, name)
    return out_path


def distance_to_water(gsw_path, land_path, out_path, occurrence=PERMANENT_WATER):
    """
    Distance in metres from every pixel to the nearest water, on the GSW grid.

    Water is any pixel with GSW occurrence >= `occurrence` (permanent water by
    default) or outside the land polygons (the sea). This is the distance
    raster the landward mask is cut from.

    Parameters
    ----------
    gsw_path : pathlib.Path or str
        Global Surface Water occurrence GeoTiff (0 - 100 %) in a projected CRS
        in metres. Its grid is the output grid.
    land_path : pathlib.Path or str
        Polygon file (e.g. data/guyana_regions.geojson) whose union is land.
    out_path : pathlib.Path or str
        Output float32 GeoTiff of distance to water in metres; 0 on water.
    occurrence : int
        GSW occurrence (%) at or above which a pixel counts as water. Default 100.

    Returns
    -------
    pathlib.Path or str
        `out_path`.

    Notes
    -----
    The whole raster is held in memory, because an exact distance transform
    needs every pixel at once. At 30 m the Guyana coast fits comfortably.
    """
    with rasterio.open(gsw_path) as src:
        occ, prof = src.read(1), src.profile.copy()
    land = geometry_mask(gpd.read_file(land_path).to_crs(prof["crs"]).geometry,
                         out_shape=occ.shape, transform=prof["transform"], invert=True)
    water = ~land | (occ >= occurrence)
    # the EDT measures distance to the nearest zero, so water must be the zeros
    dist = distance_transform_edt(~water, sampling=(abs(prof["transform"].e),
                                                    abs(prof["transform"].a)))
    prof.update(count=1, dtype="float32", nodata=None, **PROFILE_OPTS)
    with rasterio.open(out_path, "w", **prof) as dst:
        dst.write(dist.astype(np.float32), 1)
        dst.set_band_description(1, "distance_to_water_m")
    return out_path


def landward_cutoff(coords, dist_path, q=99):
    """
    The landward-mask cutoff: a percentile of training mangrove's distance to water.

    Parameters
    ----------
    coords : array-like
        (n, 2) [lon, lat] pairs in EPSG:4326 of the training mangrove points,
        e.g. ``extract_coords(train[train.label == 1])``.
    dist_path : pathlib.Path or str
        Distance raster from `distance_to_water`.
    q : float
        Percentile in 0 - 100. Default 99, so 1% of training mangrove would be
        masked.

    Returns
    -------
    float
        Cutoff distance in metres.

    Raises
    ------
    ValueError
        If `coords` is empty, or any point falls outside the distance raster.

    Notes
    -----
    The cutoff depends on the training set, so store it with the model
    (``MangroveClassifier.save_model(..., metadata={"landward_cutoff_m": cut})``)
    rather than recomputing it.
    """
    coords = np.asarray(coords, dtype=float).reshape(-1, 2)
    if len(coords) == 0:
        raise ValueError("no points to compute the cutoff from")
    with rasterio.open(dist_path) as src:
        xs, ys = warp_transform("EPSG:4326", src.crs, coords[:, 0], coords[:, 1])
        rows, cols = rasterio.transform.rowcol(src.transform, xs, ys)
        rows, cols = np.asarray(rows), np.asarray(cols)
        outside = (rows < 0) | (rows >= src.height) | (cols < 0) | (cols >= src.width)
        if outside.any():
            raise ValueError(
                f"{int(outside.sum())} of {len(coords)} points fall outside {Path(dist_path).name}")
        dist = src.read(1)[rows, cols]
    return float(np.percentile(dist, q))


def apply_landward_mask(extent_path, dist_path, cutoff, out_path, block=512):
    """
    Sets mapped mangrove more than `cutoff` m from water to non-mangrove.

    Only mangrove pixels change (1 -> 0). Non-mangrove and nodata are copied
    as they are, so the masked map keeps the original's domain and the
    Olofsson stratum areas stay comparable.

    Parameters
    ----------
    extent_path : pathlib.Path or str
        Extent map: single uint8 band, 1 mangrove, 0 non-mangrove, 255 nodata.
    dist_path : pathlib.Path or str
        Distance raster from `distance_to_water`, on any grid or CRS. Warped
        (nearest) onto the extent grid; pixels it does not cover are left
        unchanged.
    cutoff : float
        Maximum distance to water in metres, e.g. from `landward_cutoff`.
    out_path : pathlib.Path or str
        Output masked extent map, same grid and encoding as `extent_path`.
    block : int
        Maximum side length of each window, in pixels. Defaults to 512.

    Returns
    -------
    tuple of float
        (mangrove_ha, removed_ha): mangrove left after masking and mangrove
        removed by it, in hectares, assuming a projected CRS in metres.

    Raises
    ------
    ValueError
        If `cutoff` is not positive.

    Notes
    -----
    The map is written to ``<out_path>.part`` and renamed only once complete,
    so an interrupted run never leaves a partial file under `out_path`.

    See Also
    --------
    docs/references.md#documentation : Rasterio Documentation Virtual Warping
    https://rasterio.readthedocs.io/en/latest/topics/virtual-warping.html
    """
    if not cutoff > 0:
        raise ValueError(f"cutoff must be a positive distance in metres, got {cutoff}")

    part = f"{out_path}.part"
    kept, removed = 0, 0
    with rasterio.open(extent_path) as esrc:
        prof = esrc.profile.copy()
        prof.update(**PROFILE_OPTS)
        # uncovered pixels read as NaN, and NaN > cutoff is False, so they are kept
        with rasterio.open(dist_path) as dsrc, \
                WarpedVRT(dsrc, crs=esrc.crs, transform=esrc.transform,
                          width=esrc.width, height=esrc.height, dtype="float32",
                          nodata=np.nan, resampling=Resampling.nearest) as dist, \
                rasterio.open(part, "w", **prof) as dst:
            dst.set_band_description(1, esrc.descriptions[0] or "mangrove_extent")
            for win in iter_windows(esrc, block):
                extent = esrc.read(1, window=win)
                far = (extent == 1) & (dist.read(1, window=win) > cutoff)
                extent[far] = 0
                dst.write(extent, 1, window=win)
                kept += int((extent == 1).sum())
                removed += int(far.sum())
    os.replace(part, out_path)

    px_ha = abs(prof["transform"].a) * abs(prof["transform"].e) / 1e4
    return kept * px_ha, removed * px_ha


__all__ = ["apply_landward_mask", "distance_to_water", "extent_from_prob", "extent_tile",
           "iter_windows", "landward_cutoff", "mosaic", "predict_tile", "snap_to_domain"]
