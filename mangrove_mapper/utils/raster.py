import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject


def on_grid(path, H, W, T, dst_crs, resampling=Resampling.nearest, src_nodata="declared"):
    """
    Warp a raster onto a reference grid, without clipping it to any domain.

    Reads one band of `path` and resamples it onto the (H, W) grid described by
    `T` and `dst_crs`, so several layers can be sampled together as aligned
    arrays.

    Parameters
    ----------
    path : pathlib.Path or str
        Raster to read.
    H, W : int
        Height and width of the reference grid, in pixels.
    T : affine.Affine
        Affine transform of the reference grid.
    dst_crs : rasterio.crs.CRS or str
        CRS of the reference grid.
    resampling : rasterio.enums.Resampling
        Nearest for categorical layers, whose class labels must not be
        averaged, and bilinear for continuous ones. Default nearest.
    src_nodata : int, float, None or 'declared'
        Nodata value of the source. The default reads the file's own
        declaration; pass a value to override one that is wrong.

    Returns
    -------
    numpy.ndarray
        Float32 array of shape (H, W), NaN where the source has no data.

    See Also
    --------
    mangrove_mapper.inference.snap_to_domain : the same regridding, but clipped
        to a model extent's classified domain and forced to uint8. Use that to
        score a product against the model on equal terms; use this when the
        values outside that domain are the ones you need, as they are when
        sampling landward of the mangrove belt for risk assessment.

    docs/references.md#documentation : Rasterio Documentation Reprojection
    https://rasterio.readthedocs.io/en/latest/topics/reproject.html
    """
    out = np.full((H, W), np.nan, dtype="float32")
    with rasterio.open(path) as src:
        nd = src.nodata if src_nodata == "declared" else src_nodata
        reproject(rasterio.band(src, 1), out, dst_transform=T, dst_crs=dst_crs,
                  src_nodata=nd, dst_nodata=np.nan, resampling=resampling)
    return out
