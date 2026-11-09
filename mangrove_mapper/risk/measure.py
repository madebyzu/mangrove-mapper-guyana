"""
Per-transect measurements for the fringe risk layer.

All measurements are sampled along transects, which were built by the `geometry` functions.

Distances are in metres and measurments will be quantized (10 m pixels) so that is the finest detail,
an ode to using finest remote sensing data available rather than coarser 30 m resolution +.

This technique of detecting edge shifts between years is part of standard shoreline anaylsis like
net shoreline movement. A common software associated with the process is known as 
DSAS (Digital Shoreline Analysis System).

See Also
--------
docs/references.md#journal-articles--papers : Best et al., 2022
docs/references.md#journal-articles--papers : Oyedotun & Johnson-Bhola, 2019
docs/references.md#ai-usage : AI assistance log
"""

import numpy as np

from .geometry import sample_along


def belt_and_mud(mangrove, pts, norm, inv, **config):
    """
    Measure the mudflat offset and the first mangrove belt on every transect.

    Marching landward the order is coastline then mudflat then mangrove then inland. 
    `mudflat` is the distance from the coastline to the first mangrove pixel, which is 
    where the belt starts. The attenuation zone we are choosing to focus on is only the mangrove
    portion, not the mudflat region (which is not exact, could be open water too).

    Parameters
    ----------
    mangrove : numpy.ndarray of bool
        Mangrove extent on the grid.
    pts : numpy.ndarray
        (n, 2) transect origins in map coordinates.
    norm : numpy.ndarray
        (n, 2) landward unit normals.
    inv : affine.Affine
        Inverse transform, mapping map coordinates to (col, row).
    **config
        Requires `max_len_m` (how far landward to look), `step_m` (sampling
        step) and `gap_tol` (consecutive misses tolerated inside one belt).

    Returns
    -------
    Wbelt : numpy.ndarray
        (n,) belt width in metres. Zero where the transect crosses no mangrove.
    mudflat : numpy.ndarray
        (n,) distance from the coastline to the first mangrove, in metres.
        NaN where the transect crosses no mangrove.

    Notes
    -----
    Both arrays are full length, so they stay aligned with `pts`, `norm` and
    `open_coast` and can be indexed by transect.
    """
    # mudflat is the distance from coastline to the first mangrove.
    # Wbelt of i starts from that point
    mangrove_f = mangrove.astype("float32")
    ds_steps = np.arange(0, config['max_len_m'] +
                         config['step_m'], config['step_m'])
    Wbelt = np.zeros(len(pts))
    mudflat = np.full(len(pts), np.nan)
    # sample_along can only return 0 or 1, nan values exist where rays leave the grid,
    # > 0.5 returns a boolean when values are in between
    for i in range(len(pts)):
        hits = sample_along(
            mangrove_f, pts[i], norm[i], ds_steps, inv
        ) > 0.5

        idx = np.where(hits)[0]
        if idx.size == 0:
            continue  # no mangrove = fringe absent
        # seaward mudflat = coastline start
        start = idx[0]
        end, gap = start, 0
        for j in range(start, len(hits)):
            if hits[j]:
                end, gap = j, 0
            else:
                gap += 1
                if gap > config['gap_tol']:
                    break
        Wbelt[i] = (end - start + 1) * config['step_m']
        mudflat[i] = start * config['step_m']
    return Wbelt, mudflat


def belt_mean(raster, Wbelt, mudflat, pts, norm, inv, i, **config):
    """
    Average a per-pixel raster over one transect's mangrove belt.

    Spectral/Backscatter indices are per-pixel, but the index (evi, hv) is per-transect, so each
    has to be reduced along the belt it crosses before it can be part of the risk equation. 
    Sampling starts at the landward edge of the mudflat and ends at
    the landward edge of the belt, so only mangrove pixels contribute.

    Parameters
    ----------
    raster : numpy.ndarray
        Per-pixel layer on the grid, e.g. canopy stress (evi).
    Wbelt, mudflat : numpy.ndarray
        (n,) belt widths and mudflat offsets from `belt_and_mud`.
    pts, norm : numpy.ndarray
        (n, 2) transect origins and landward unit normals.
    inv : affine.Affine
        Inverse transform, converting map coordinates to (col, row).
    i : int
        Transect index.
    **config
        Requires `step_m`.

    Returns
    -------
    float
        Mean over the belt, or NaN where the transect has no belt or the belt
        holds no finite values.

    Notes
    -----
    Non-finite samples are dropped before averaging, so a belt that runs partly
    off the grid, or over pixels masked out of the proxy, is scored on the
    valid part rather than being lost entirely.
    """
    if Wbelt[i] == 0 or np.isnan(mudflat[i]):
        return np.nan
    ds = np.arange(mudflat[i], mudflat[i] + Wbelt[i] +
                   config['step_m'], config['step_m'])
    vals = sample_along(raster, pts[i], norm[i], ds, inv)
    vals = vals[np.isfinite(vals)]
    return float(vals.mean()) if vals.size else np.nan


def edge_shift(prev_raster, pts, norm, open_coast, mudflat, inv, **config):
    """
    Movement of the seaward mangrove edge between two extent maps.

    The previous year's map is sampled along the exact same transects, and its
    first-mangrove distance (mudflat) is differenced against this year's `mudflat`.

    Parameters
    ----------
    prev_raster : numpy.ndarray of bool
        Mangrove extent for the previous year, on the same grid.
    pts, norm : numpy.ndarray
        (n, 2) transect origins and landward unit normals.
    open_coast : numpy.ndarray of bool
        (n,) which transects to measure. Estuary banks are skipped because they don't
        correspond with with the same ocean wave-attentuation phenomenon (from my understanding).
    mudflat : numpy.ndarray
        (n,) this year's mudflat derived from `belt_and_mud`.
    inv : affine.Affine
        Inverse transform, converting map coordinates to (col, row).
    **config
        Requires `max_len_m` and `step_m`.

    Returns
    -------
    numpy.ndarray
        (n,) signed shift in metres. Positive is seaward advance
        (accretion), negative is landward retreat (erosion). NaN where the
        transect was not measured, or where either year has no mangrove on it.

    Notes
    -----
    Both maps must come from the same fitted model, the same grid and the same
    threshold. 
    """
    edge_shift = np.full(len(pts), np.nan)
    prev_f = prev_raster.astype("float32")
    ds_steps = np.arange(0, config['max_len_m'] +
                         config['step_m'], config['step_m'])
    for i in np.where(open_coast)[0]:
        hits = sample_along(prev_f, pts[i], norm[i], ds_steps, inv) > 0.5
        idx = np.where(hits)[0]
        if idx.size and np.isfinite(mudflat[i]):
            # + = accretion, - = erosion
            edge_shift[i] = idx[0] * config['step_m'] - mudflat[i]
    return edge_shift


def landward_steps(i, Wbelt, mudflat, **config):
    """
    Sample distances for the thin area behind one transect's belt.

    Begins beyond the belt landward, and samples what the mangrove is protecting (i.e. human capital).

    Parameters
    ----------
    i : int
        Transect index.
    Wbelt, mudflat : numpy.ndarray
        (n,) belt widths and mudflat offsets from `belt_and_mud`.
    **config
        Requires `exposure_depth_m` (how far landward to reach) and `step_m`.

    Returns
    -------
    numpy.ndarray
        Distances from the transect origin, in metres.

    Notes
    -----
    Where the transect crosses no mangrove, `mudflat` is NaN and the corridor
    starts at the coastline instead. Those transects have nothing in front of
    them, so the corridor begins at the shore.
    """
    start = (mudflat[i] + Wbelt[i]) if np.isfinite(mudflat[i]) else 0.0
    return np.arange(start, start + config['exposure_depth_m'] + config['step_m'], config['step_m'])


def landward_agg(raster, i, pts, norm, Wbelt, mudflat, inv, classes=None, agg="max", **config):
    """
    Reduce the landward corridor of one transect to a single number.

    Two modes, because the two exposure members ask different questions. With
    `classes`, it returns the fraction of the corridor falling in those
    classes -- the right quantity for land cover, where the share of the strip
    that is farmland is what matters. Without, it aggregates the values, and
    reports the max population density (ppl/ha) for that corridor behind the belt.

    Parameters
    ----------
    raster : numpy.ndarray
        Per-pixel layer on the working grid.
    i : int
        Transect index.
    pts, norm : numpy.ndarray
        (n, 2) transect origins and landward unit normals.
    Wbelt, mudflat : numpy.ndarray
        (n,) belt widths and mudflat offsets, used to find where the belt ends.
    inv : affine.Affine
        Inverse transform, converting map coordinates to (col, row).
    classes : sequence of int, optional
        Class values to count. When given, the value returned is the fraction of
        finite samples in those classes and `agg` is ignored.
    agg : {'max', 'mean'}
        Decides how to reduce the values when `classes` is None. Default 'max'.
    **config
        Requires `exposure_depth_m` and `step_m`.

    Returns
    -------
    float
        Corridor fraction or aggregate, NaN if no finite samples were found.
    """
    vals = sample_along(raster, pts[i], norm[i],
                        landward_steps(i, Wbelt, mudflat, **config), inv)
    vals = vals[np.isfinite(vals)]
    if not vals.size:
        return np.nan
    if classes is not None:
        return float(np.mean(np.isin(np.rint(vals), classes)))
    return float(vals.max()) if agg == "max" else float(vals.mean())
