import numpy as np

from .geometry import build_transects, seaward_fetch
from .index import (
    compute_erosion_risk,
    compute_evi,
    compute_wave_atten,
    modifier,
    retrieve_hv,
    robust_stress,
    wgeomean,
    wmean,
)
from .measure import belt_and_mud, belt_mean, edge_shift, landward_agg


def build_factors(mangrove, water, stack_path, aff_T, grid_crs,
                  prev_mangrove=None, landcover=None, population=None, **config):
    """
    Cast transects and measure every risk factor along them, for one year.

    Parameters
    ----------
    mangrove : numpy.ndarray of bool
        Extent map for the year being scored.
    water : numpy.ndarray of bool
        Permanent water mask on the same grid.
    stack_path : pathlib.Path or str
        Feature stack for the same year, read for EVI and ALOS HV.
    aff_T : affine.Affine
        Transform of the grid.
    grid_crs : rasterio.crs.CRS
        CRS of the grid.
    prev_mangrove : numpy.ndarray of bool, optional
        Previous year's extent map, for erosion. Omitted leaves erosion NaN.
    landcover : numpy.ndarray, optional
        Class labels on the same grid. Omitted leaves landcover NaN.
    population : numpy.ndarray, optional
        People per hectare on the same grid. Omitted leaves population NaN.
    **config
        Risk config

    Returns
    -------
    dict
        Geometry (`pts`, `norm`, `sea`, `pos_m`, `fetch_m`, `open_coast`), belt
        measurements (`Wbelt`, `mudflat`), the six factors (`fringe`, `evi`,
        `hv`, `erosion`, `landcover`, `population`), the raw quantities they
        were scaled from (`edge_shift`, `pop_per_ha`) and the scaling limits
        (`evi_lim`, `hv_lim`, `hi_ret`).

    Notes
    -----
    `region2_risk_index_figures` notebook covers some of the inner workings of these functions
    briefly.
    """
    H, W = mangrove.shape
    inv = ~aff_T

    pts, norm, sea, pos_m = build_transects(water, aff_T, **config)
    fetch_m, open_coast = seaward_fetch(pts, norm, sea, inv, **config)
    n = len(pts)
    idx_open = np.where(open_coast)[0]

    Wbelt, mudflat = belt_and_mud(mangrove, pts, norm, inv, **config)

    fringe = compute_wave_atten(Wbelt, **config)
    evi = compute_evi(stack_path, W, H, grid_crs)
    hv_db = retrieve_hv(stack_path, W, H, grid_crs)
    evi_stress, evi_lim = robust_stress(evi, mangrove & np.isfinite(evi))
    hv_weakness, hv_lim = robust_stress(hv_db, mangrove & np.isfinite(hv_db))

    evi_t, hv_t = np.full(n, np.nan), np.full(n, np.nan)
    for i in idx_open:
        evi_t[i] = belt_mean(evi_stress, Wbelt, mudflat,
                             pts, norm, inv, i, **config)
        hv_t[i] = belt_mean(hv_weakness, Wbelt, mudflat,
                            pts, norm, inv, i, **config)

    shift = np.full(n, np.nan)
    erosion_risk, hi_ret = np.full(n, np.nan), np.nan
    if prev_mangrove is not None:
        shift = edge_shift(prev_mangrove, pts, norm,
                           open_coast, mudflat, inv, **config)
        erosion_risk, hi_ret = compute_erosion_risk(shift, open_coast)

    landcover_exp, pop_per_ha = np.full(n, np.nan), np.full(n, np.nan)
    if landcover is not None:
        for i in idx_open:
            landcover_exp[i] = landward_agg(
                landcover, i, pts, norm, Wbelt, mudflat, inv,
                classes=config['exposure_classes'], agg='mean', **config)
    if population is not None:
        for i in idx_open:
            pop_per_ha[i] = landward_agg(
                population, i, pts, norm, Wbelt, mudflat, inv, agg='max', **config)
    # Neither is rescaled: both are already absolute and zero-anchored.
    pop_exp = np.clip(pop_per_ha / config['pop_full_exposure_per_ha'], 0, 1)

    return {
        'pts': pts, 'norm': norm, 'sea': sea, 'pos_m': pos_m,
        'fetch_m': fetch_m, 'open_coast': open_coast,
        'Wbelt': Wbelt, 'mudflat': mudflat,
        'fringe': fringe, 'evi': evi_t, 'hv': hv_t,
        'erosion': erosion_risk, 'landcover': landcover_exp,
        'population': pop_exp,
        'edge_shift': shift, 'pop_per_ha': pop_per_ha,
        'evi_lim': evi_lim, 'hv_lim': hv_lim, 'hi_ret': hi_ret
    }


def compose_index(factors, **config):
    """
    Combine measured factors into the three legs and the composite index.

    Parameters
    ----------
    factors : dict
        Output of `build_factors`.
    **config
        `risk.index` config, for `belt_weights`, `leg_weights`,
        `exposure_min_multiplier` and `erosion_min_multiplier`.

    Returns
    -------
    dict
        `risk_index`, the raw legs (`leg_belt`, `leg_exposure`, `leg_erosion`)
        and the modifiers used for exposure and erosion.
    """
    belt_w, leg_w = config['belt_weights'], config['leg_weights']

    leg_belt = wmean([factors['fringe'], factors['evi'], factors['hv']],
                     [belt_w['fringe'], belt_w['evi'], belt_w['hv']])
    # crops and people are distinct assets, either alone is at stake (like our Dantzigt-Fairfield case study)
    leg_exposure = np.fmax(factors['landcover'], factors['population'])
    leg_erosion = factors['erosion']

    exposure_mod = modifier(leg_exposure, config['exposure_min_multiplier'])
    erosion_mod = modifier(leg_erosion, config['erosion_min_multiplier'])

    risk_index = wgeomean([leg_belt, exposure_mod, erosion_mod],
                          [leg_w['belt'], leg_w['exposure'], leg_w['erosion']])
    risk_index[~factors['open_coast']] = np.nan

    return {'risk_index': risk_index, 'leg_belt': leg_belt,
            'leg_exposure': leg_exposure, 'leg_erosion': leg_erosion,
            'exposure_mod': exposure_mod, 'erosion_mod': erosion_mod}


__all__ = ["build_factors", "compose_index"]
