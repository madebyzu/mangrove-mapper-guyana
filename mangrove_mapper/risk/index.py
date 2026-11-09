"""
Turning per-transect measurements into risk scores.

Calculates all risk elements and prepares them for aggregation via geometric mean
as a final composite indicator.

Note:
Two kinds of scale live here, when comparing region (relative and absolute).
`robust_stress` is relative as it rescales against this region's
own mangroves since "stressed" is in the context of only healthy fringes nearby
at least from a remote sensing pov I would argue. The exposure inputs are absolute.
For example,  a fully forested hinterland must score the same in every region, so they
use fixed, stated caps rather than percentiles.

See Also
--------
docs/references.md#journal-articles--papers Joint Research Centre, 2008
docs/references.md#ai-usage : AI assistance log
"""

import numpy as np
import rasterio


def compute_wave_atten(Wbelt, **config):
    """
    Residual wave height behind a mangrove belt.

    Applies `r = exp(-alpha * W)`, the attenuation equation `r = 1` means no protection;
    `r` approaching 0 means the belt removes essentially all of the incoming wave height.

    Parameters
    ----------
    Wbelt : numpy.ndarray
        (n,) mangrove belt widths in metres, from `measure.belt_and_mud`.
    **config
        Requires `alpha_fringe`, the fractional wave-height loss per metre.

    Returns
    -------
    numpy.ndarray
        (n,) residual wave height, between 0 and 1.

    See Also
    --------
    docs/references.md#journal-articles--papers : Mazda et al., 2006
    """
    r = np.exp(-config["alpha_fringe"] * Wbelt)
    return r


def read_bands(path, bands_needed, W, H, GRID_CRS):
    """
    Read named bands from the feature stack

    Parameters
    ----------
    path : pathlib.Path or str
        Multi-band feature stack.
    bands_needed : sequence of str
        Band descriptions to pull, e.g. ['s2_B2', 'alos_HV'].
    W, H : int
        Expected width and height of the grid.
    GRID_CRS : rasterio.crs.CRS
        Expected CRS of the grid.

    Returns
    -------
    dict of str to numpy.ndarray
        Band name to float32 array.

    Raises
    ------
    ValueError
        If the stack does not match the working grid.

    """
    with rasterio.open(path) as ds:
        if (ds.width, ds.height) != (W, H) or ds.crs != GRID_CRS:
            raise ValueError("stack grid does not match the working grid")
        desc = list(ds.descriptions)
        return {n: ds.read(desc.index(n) + 1).astype("float32") for n in bands_needed}


def compute_evi(stack_path, W, H, GRID_CRS):
    """
    Enhanced Vegetation Index from the bands of the model's own feature stack.

    EVI is computed rather than read because the stack carries no EVI band. It
    is used in preference to NDVI because it is less likely to saturate in tropical
    environments.

    Parameters
    ----------
    stack_path : pathlib.Path or str
        Feature stack holding `s2_B2`, `s2_B4` and `s2_B8`.
    W, H : int
        Working grid dimensions.
    GRID_CRS : rasterio.crs.CRS
        Working grid CRS.

    Returns
    -------
    numpy.ndarray
        EVI per pixel, NaN where the denominator is non-finite.

    Notes
    -----
    Sentinel-2 bands in this stack are already 0-1 surface reflectance, so no
    scaling factor is applied.

    The denominator can go negative for unusual reflectance combinations, which
    is why non-finite results are set to NaN rather than left to propagate.

    See Also
    --------
    docs/references.md#journal-articles--papers : Zeng et al., 2022
    """
    s2_b = read_bands(stack_path, ["s2_B2", "s2_B4", "s2_B8"], W, H, GRID_CRS)
    blue, red, nir = s2_b['s2_B2'], s2_b['s2_B4'], s2_b['s2_B8']
    C1, C2, L = 6.0, 7.5, 1.0

    with np.errstate(invalid="ignore", divide="ignore"):
        evi = 2.5 * (nir - red) / (nir + C1 * red - C2 * blue + L)
    evi[~np.isfinite(evi)] = np.nan
    return evi


def retrieve_hv(stack_path, W, H, GRID_CRS):
    """
    ALOS-PALSAR HV backscatter, read from the feature stack.

    Cross-polarised L-band backscatter responds to vegetation volume, so it
    stands in for structural condition alongside the optical measure. It is
    pulled directly rather than recomputed, since it is already a model feature
    and already in dB.

    Parameters
    ----------
    stack_path : pathlib.Path or str
        Feature stack holding `alos_HV`.
    W, H : int
        Working grid dimensions.
    GRID_CRS : rasterio.crs.CRS
        Working grid CRS.

    Returns
    -------
    numpy.ndarray
        HV backscatter in dB, per pixel.

    Notes
    -----
    ALOS PALSAR is natively 25 m and was resampled into this 10 m stack, so
    neighbouring pixels are not independent observations. It is a relative
    structural proxy.
    """
    alos_b = read_bands(stack_path, ["alos_HV"], W, H, GRID_CRS)
    hv_db = alos_b["alos_HV"]
    return hv_db


def robust_stress(index, mask, lower=2, upper=98):
    """
    Rescale an index over a mask so that higher means more stress.

    Percentile clipping rather than min/max, so a single extreme pixel cannot
    compress the whole scale, and inverted so the output is risk-oriented and
    can be combined directly with the other factors.

    Parameters
    ----------
    index : numpy.ndarray
        Raw per-pixel index, e.g. EVI or HV in dB.
    mask : numpy.ndarray of bool
        Pixels to derive the scale from and to score.
    lower, upper : float
        Percentile clip points. Defaults 2 and 98.

    Returns
    -------
    stress : numpy.ndarray
        0-1 within the mask, NaN outside. Higher means relatively more stress.
    limits : tuple of float
        The (lower, upper) values used, so they can be reported.

    Notes
    -----
    This scale is relative to the masked sample. The same canopy scores
    differently in different regions or years. For any cross-region or cross-year comparison
    the limits can be documented, which is why they are returned
    rather than discarded.
    """
    lo, hi = np.nanpercentile(index[mask], [lower, upper])
    health = np.clip((index - lo) / (hi - lo), 0, 1)
    return np.where(mask, 1 - health, np.nan), (float(lo), float(hi))


def wmean(cols, weights):
    """
    Renormalised weighted arithmetic mean over the finite entries of each row.

    A low score on one member can be offset by a high score on
    another.

    Parameters
    ----------
    cols : list of numpy.ndarray
        Equal-length factors, all oriented so higher means higher vulnerability.
    weights : sequence of float
        One weight per column. Need not sum to 1.

    Returns
    -------
    numpy.ndarray
        Weighted mean per row, NaN where every column is missing.

    Notes
    -----
    A missing factor drops its weight and the row is scored on what is known,
    rather than being filled with zero.
    """
    M = np.vstack(cols).T
    w = np.asarray(weights, float)
    fin = np.isfinite(M)
    wg = fin * w
    den = wg.sum(1)
    num = np.nansum(np.where(fin, M, 0.0) * wg, 1)

    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def wgeomean(cols, weights):
    """
    Renormalised weighted geometric mean over the finite entries of each row.

    Multiplicative, so each leg affects overall score in the event of outliers 
    rather than compensating.

    Parameters
    ----------
    cols : list of numpy.ndarray
        Equal-length legs, all strictly positive and oriented so higher is worse.
    weights : sequence of float
        One weight per leg. Need not sum to 1.

    Returns
    -------
    numpy.ndarray
        Weighted geometric mean per row, NaN where every leg is missing.

    Notes
    -----
    Every input must be strictly positive A zero would send its logarithm
    to negative infinity and veto the whole product, and two of the three legs
    are legitimately zero across much of the coast. `modifier` is what
    guarantees positivity; this function assumes it has already been applied.

    Missing legs are renormalised away exactly as in `wmean`.
    """
    M = np.vstack(cols).T
    w = np.asarray(weights, float)
    fin = np.isfinite(M)
    wg = fin * w
    den = wg.sum(1)
    lx = np.where(fin, np.log(np.where(fin, M, 1.0)), 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, np.exp((lx * wg).sum(1) / den), np.nan)


def modifier(x, floor):
    """
    Rescale a 0-1 leg onto [floor, 1] so a zero damps rather than vetoes.

    In a multiplicative index a leg of exactly zero zeroes the result. Taken
    literally that would assert something too strong.

    Parameters
    ----------
    x : numpy.ndarray
        Leg score in 0-1, higher meaning riskier.
    floor : float
        Value a zero maps to. A stated planning judgment, not a fitted value.

    Returns
    -------
    numpy.ndarray
        Rescaled leg, spanning [floor, 1].

    Notes
    -----
    The alternative, dropping a zero leg and renormalising the others, makes
    the index non-monotonic. The freed weight is redistributed onto the
    surviving legs, which raises the score, so a transect with zero exposure
    ends up outranking one with a sliver of farmland behind it. More at stake
    reading as less risk. A modifier range keeps the index continuous and
    monotonic in every leg.
    """
    return floor + (1 - floor) * x


def compute_erosion_risk(edge_shift, open_coast, upper=98):
    """
    Convert signed edge movement into a 0-1 erosion hazard.

    Only landward retreat is a hazard. Accretion clamps to zero, so stable or
    prograding scores present no erosion risk.

    Parameters
    ----------
    edge_shift : numpy.ndarray
        (n,) signed shift in metres from `measure.edge_shift`. Positive is
        seaward advance.
    open_coast : numpy.ndarray of bool
        (n,) transects the scale is derived from.
    upper : float
        Percentile of retreat mapped to 1.0. Default 98.

    Returns
    -------
    erosion_risk : numpy.ndarray
        (n,) 0-1, higher meaning more retreat. NaN where no shift was measured.
    hi_ret : float
        The retreat distance that maps to 1.0, in metres, so the scale can be
        reported or frozen.
    """
    oc_edges = open_coast & np.isfinite(edge_shift)
    retreat = np.clip(-edge_shift, 0, None)  # clamping to zero first
    hi_ret = np.nanpercentile(retreat[oc_edges], upper)
    erosion_risk = np.clip(retreat / (hi_ret + 1e-9), 0,
                           1)  # then applying scaling
    erosion_risk[~np.isfinite(edge_shift)] = np.nan
    return erosion_risk, hi_ret
