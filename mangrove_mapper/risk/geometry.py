"""
Coastline geometry for the fringe risk layer.

Uses the water mask to build transects with an origin on the coastline and a
unit normal pointing inland. All risk index readings like fringe width,
year-to-year edge shifts etc. are sampled by traveling along each of those rays.

Distances are in metres which explains the conversion to map coordinates rather than
pixel indices for transect casting.

See Also
--------
docs/references.md#journal-articles--papers : Best et al., 2022
docs/references.md#ai-usage : AI assistance log
"""

import matplotlib.pyplot as plt
import numpy as np
from rasterio import transform
from scipy import ndimage as ndi


def sea_mask(water, res):
    """
    Isolate the ocean from a water mask, and measure distance to it.

    The largest connected component of `water` is taken to be the sea; every
    other component is discarded, which is what excludes other bodies of water like rivers
    which are not connected to the coast for the Global Surface Water occurrence threshold defined.

    Parameters
    ----------
    water : numpy.ndarray of bool
        True where a pixel counts as permanent water.
    res : float
        Pixel size in metres, used to convert the distance pixels into metres.

    Returns
    -------
    sea : numpy.ndarray of bool
        The largest connected water body.
    dist_sea_m : numpy.ndarray of float
        Distance from every pixel to the nearest sea pixel, in metres. Sea
        pixels are 0.
    ncomp : int
        Number of connected water components found, before selection.

    Notes
    -----
    The "largest component is the ocean" assumption works while the raster
    covers enough open water.  It doesn't work well when a mask
    clipped to a narrow AOI. The ocean is still likely the largest component but
    it may result in a shorter coastline, which may not be intended.
    Double check the `ncomp` number and the pixel count to verify.

    See Also
    --------
    docs/references.md#documentation : SciPy Documentation
    """
    # labels the contiguous water components and counts their number
    lbl, ncomp = ndi.label(water)
    sizes = ndi.sum(np.ones_like(lbl), lbl, index=np.arange(1, ncomp + 1))
    sea = lbl == (int(np.argmax(sizes)) + 1)
    # distance from every pixel to the nearest sea pixel, in metres
    # turns bool array into array of distances in meters
    dist_sea_m = ndi.distance_transform_edt(~sea) * res

    return sea, dist_sea_m, ncomp


def trace_coastline(sea, aff_T, smooth_win, spacing_m):
    """
    Trace the coastline and place transect origins along it at fixed spacing.

    The coastline is the 0.5 contour of the sea mask, the level exactly between
    not-sea (0) and sea (1). Marching squares interpolates the crossing, so
    vertices fall between pixel centres and the line is sub-pixel rather than
    a staircase. It is then smoothed and resampled at even arc-length intervals.

    Parameters
    ----------
    sea : numpy.ndarray of bool
        Ocean mask from `sea_mask`.
    aff_T : affine.Affine
        Transform of the grid `sea` sits on, mapping (col, row) to map
        coordinates.
    smooth_win : int
        Box-filter width in vertices, passed to `smooth`.
    spacing_m : float
        Alongshore spacing between transect origins, in metres.

    Returns
    -------
    pts : numpy.ndarray
        (n, 2) transect origins as (easting, northing) in map units.
    pos_m : numpy.ndarray
        (n,) origin position relative to the start of the traced coastline in metres.
    cu_len : numpy.ndarray
        Cumulative arc length at every polyline vertex; `cu_len[-1]` is the total
        coastline length.

    Notes
    -----
    Only the longest contour segment is kept, and the rest are silently dropped.
    """
    # creating a list of separate polylines (x, y) at level 0.5 using matplotlib
    cs = plt.contour(sea.astype(float), levels=[0.5])
    # matplotlib is used as a geometry tool not for plotting, so closing to avoid leakage.
    plt.close()
    # longest contour is index 0
    segs = sorted(cs.allsegs[0], key=len, reverse=True)
    # converts longest contour from px to map coords
    coast_xy = np.array([transform.xy(aff_T, r, c, offset="center")
                         for c, r in segs[0]])
    coast_s = smooth(coast_xy, smooth_win)
    # list of vertices where vertex cu_len[i] is # many metres along the coeastline.
    cu_len = np.concatenate(
        [[0], np.cumsum(np.hypot(*np.diff(coast_s, axis=0).T))])
    pos_m = np.arange(0, cu_len[-1], spacing_m)
    pts = np.column_stack([np.interp(pos_m, cu_len, coast_s[:, 0]),  # easting
                           # northing
                           np.interp(pos_m, cu_len, coast_s[:, 1])])

    return pts, pos_m, cu_len


def shore_normals(pts, dist_sea_m, inv, probe=50.0):
    """
    Unit vectors perpendicular to the coastline, oriented landward.

    The alongshore tangent is differentiated from the origin polyline and
    rotated 90 degrees. That leaves the sign ambiguous. The rotation points
    inland or out to sea depending which way the coastline happened to be traced
    so each normal is resolved by sampling distance-to-sea a short step
    either side and keeping whichever points away from the water.

    Parameters
    ----------
    pts : numpy.ndarray
        (n, 2) transect origins in map coordinates.
    dist_sea_m : numpy.ndarray
        Distance-to-sea raster from `sea_mask`.
    inv : affine.Affine
        Inverse transform of the grid, mapping map coordinates to (col, row).
    probe : float
        How far either side to sample when resolving the sign, in metres.
        Default 50.

    Returns
    -------
    norm : numpy.ndarray
        (n, 2) unit vectors pointing inland.
    n_flipped : int
        How many normals had their sign reversed. Note: a consistently
        traced coastline gives 0 or n, and a value in between means the rotation
        disagrees with the geometry somewhere along the coast.

    Notes
    -----
    Without the flip roughly half the transects would march out to sea and
    measure nothing, which is why the sign is resolved per transect rather than
    assumed from the winding direction.
    """
    tang = np.gradient(
        pts, axis=0)  # the smoother the coastline trace, the better
    tang /= (np.hypot(*tang.T)[:, None] + 1e-9)  # unit length
    # rotating 90 degrees to get the norm from tang
    # (x, y) => (-y, x)
    norm = np.column_stack([-tang[:, 1], tang[:, 0]])
    # one sample distance to check norm direction; needs to be array for `sample_along` function
    probe_d = np.array([probe])

    n_flipped = 0
    for i in range(len(pts)):
        # forward f and backward b directions along the normals at 50 m
        f = np.nan_to_num(sample_along(
            dist_sea_m, pts[i], norm[i], probe_d, inv)[0], nan=-1)
        b = np.nan_to_num(sample_along(
            dist_sea_m, pts[i], -norm[i], probe_d, inv)[0], nan=-1)
        # at 50 m along the normal in the backward and forward direction,
        # if backward direction is further from the sea than the forward,
        # the backward direction is inland so norms needs a flip
        if b > f:
            norm[i] = -norm[i]
            n_flipped += 1
    return norm, n_flipped


def sample_along(array, origin, direction, distances, inv):
    """
    Sample a raster along a ray, vectorised over the whole ray.

    Builds every sample coordinate at once, converts the batch to row/col in one
    affine multiply, and reads them with arrays of integers

    Parameters
    ----------
    array : numpy.ndarray
        Raster to read, on the grid `inv` describes. Boolean arrays should be
        cast to a numeric dtype by the caller.
    origin : numpy.ndarray
        (2,) start point in map coordinates.
    direction : numpy.ndarray
        (2,) unit direction. Negate it to travel the other way.
    distances : numpy.ndarray
        (k,) distances along the ray, in metres.
    inv : affine.Affine
        Inverse transform, mapping map coordinates to (col, row).

    Returns
    -------
    numpy.ndarray
        (k,) sampled values, NaN wherever the ray leaves the grid.

    Notes
    -----

    Coordinates are rounded to the nearest pixel centre, so the sampling
    resolution is the pixel, not `distances`. A finer step re-reads the same
    pixels rather than resolving anything smaller.
    """

    xy = origin[None, :] + distances[:, None] * direction[None, :]
    cols, rows = inv * (xy[:, 0], xy[:, 1])
    r, c = np.rint(rows).astype(int), np.rint(
        cols).astype(int)  # rounding to whole nums
    inside = (r >= 0) & (r < array.shape[0]) & (
        c >= 0) & (c < array.shape[1])  # valid bounds
    out = np.full(distances.size, np.nan)
    out[inside] = array[r[inside], c[inside]]
    return out


def seaward_fetch(pts, norm, sea, inv, **config):
    """
    Measure seaward open water per transect, and flag transect as open-coast or not.

    Restricts the analysis to the open Atlantic front excluding estuary or creek.

    Parameters
    ----------
    pts : numpy.ndarray
        (n, 2) transect origins in map coordinates.
    norm : numpy.ndarray
        (n, 2) landward unit normals. The ray is cast along `-norm`, seaward.
    sea : numpy.ndarray of bool
        Ocean mask. The raw mask, not one with creeks filled -- closing the
        channels first would hand every estuary bank an artificial fetch.
    inv : affine.Affine
        Inverse transform, mapping map coordinates to (col, row).
    **config
        Requires `rmax_m` (how far seaward to look, metres), `step_m` (sampling
        step, metres) and `fetch_min_m` (open-coast threshold, metres).

    Returns
    -------
    fetch_m : numpy.ndarray
        (n,) metres of open water within `rmax_m` seaward of each origin.
    open_coast : numpy.ndarray of bool
        (n,) True where `fetch_m >= fetch_min_m`.
    """
    fetch_steps = np.arange(
        0, config['rmax_m'] + config['step_m'], config['step_m'])

    sea_f = sea.astype("float32")
    fetch_m = np.array([np.nansum(sample_along(sea_f, pts[i], -norm[i], fetch_steps, inv)) * config['step_m']
                        # sums the 0s or 1s from each seaward step and excludes nans that are off-grid
                        for i in range(len(pts))])
    open_coast = fetch_m >= config['fetch_min_m']

    print(f"open-coast transects (fetch >= {config['fetch_min_m']} m): "
          f"{open_coast.sum()}/{len(pts)} ({100 * open_coast.mean():.0f}%)")
    print(f"median fetch: open {np.median(fetch_m[open_coast]):.0f} m | "
          f"excluded {np.median(fetch_m[~open_coast]):.0f} m")

    return fetch_m, open_coast


def smooth(poly, win):
    """
    Box-filter a polyline, leaving its endpoints untouched.

    The traced coastline is a pixel-scale zigzag. Left by itself, the tangent from
    `np.gradient` jitters and the shore-normals could point in unexpected directions,
    so the line is smoothed before transect origins are placed on it.

    Parameters
    ----------
    poly : numpy.ndarray
        (n, 2) vertices.
    win : int
        Filter width in vertices. Values below 3 return the input unchanged.

    Returns
    -------
    numpy.ndarray
        (n, 2) smoothed vertices.

    Notes
    -----
    The first and last `win` vertices are left alone. `np.convolve(..., "same")`
    pads with zeros at the edges.
    """
    if win < 3:
        return poly
    k = np.ones(win) / win
    out = poly.copy()
    out[win:-win, 0] = np.convolve(poly[:, 0], k, "same")[win:-win]
    out[win:-win, 1] = np.convolve(poly[:, 1], k, "same")[win:-win]
    return out


def build_transects(water, aff_T, **config):
    """
    Build the transect set from a water mask, in one call.

    Parameters
    ----------
    water : numpy.ndarray of bool
        True where a pixel counts as permanent water.
    aff_T : affine.Affine
        Transform of the grid. Pixel size is taken from it, so the resolution
        cannot disagree with the grid it describes.
    **config
        Requires `smooth_win` and `spacing_m`. Extra keys are ignored.

    Returns
    -------
    pts : numpy.ndarray
        (n, 2) transect origins in map coordinates.
    norm : numpy.ndarray
        (n, 2) landward unit normals.
    sea : numpy.ndarray of bool
        Ocean mask, returned because callers plot against it.
    pos_m : numpy.ndarray
            (n,) origin position relative to the start of the traced coastline in metres.
    """
    res = aff_T.a
    sea, dist_sea_m, ncomp = sea_mask(water, res)
    print(
        f"water components: {ncomp}   ocean = the largest ({sea.sum():,} px)")
    pts, pos_m, len_cu = trace_coastline(
        sea, aff_T, config["smooth_win"], config["spacing_m"])
    print(
        f"coastline {len_cu[-1] / 1000:.1f} km -> {len(pts)} transect origins at {config['spacing_m']} m")
    norm, n_flipped = shore_normals(pts, dist_sea_m, ~aff_T)
    print(f"flipped {n_flipped}/{len(pts)} normals to point landward")

    return pts, norm, sea, pos_m
