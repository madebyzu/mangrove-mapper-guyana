from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from sklearn.metrics import average_precision_score, roc_auc_score

N_PER_CLASS = 200                 # points per stratum to interpret in Planet/CEO
SEED = 48
CLASSES = {0: 'non_mangrove', 1: 'mangrove'}   # nodata (255) is excluded
OUT_DIR = Path(__file__).parents[1] / 'accuracy_assessment'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def build_accuracy_assessment_sheet(sample_map_path, file_name, n_per_class=N_PER_CLASS, seed=SEED, classes=CLASSES):
    """
    Creates the independent accuracy-assessment CSV: a stratified random sample
    of plots drawn from an extent map, each with a plot id and lon/lat, plus
    empty columns for the interpreter(s) to fill on two platforms (Planet or Collect Earth Online [CEO])

    Note:
    The CSV is written to OUT_DIR (mangrove_mapper/accuracy_assessment/) but the file name is determined by the user.

    Parameters
    ----------
    sample_map_path : str or pathlib.Path
        Extent map to draw the sample from.
    file_name : str
        Name of the output CSV, created inside OUT_DIR.
    n_per_class : int, optional
        Plots drawn per class. Equal allocation across strata. Defaults to 200.
    seed : int, optional
        Seed for the random sampler; keep fixed for reproducibility.
        Default 42.
    classes : dict, optional
        Maps the pixel values to sample from to readable names. Only the keys
        are used, so any value not listed (nodata 255) is excluded from the
        sample. Default {0: 'non_mangrove', 1: 'mangrove'}.

    Returns
    -------
    pathlib.Path
        Path to the CSV that was written.

    Notes
    -----
    PLOTID, LON and LAT are capitalised because CEO requires
    it on import. `planet_label` and `ceo_label` are left empty for one or two
    interpreters to fill.

    See Also
    --------
    docs/references.md#documentation : Python quickstart. Rasterio Documentation.
    docs/references.md#documentation : Transformer. pyproj Stable Documentation.
    """
    with rasterio.open(sample_map_path) as s:
        ext = s.read(1)  # extent map
        transform, crs = s.transform, s.crs

    rng = np.random.default_rng(seed=seed)
    # function to convert UTM meters easting/northings to WGS84 lon/lat coordinates, ordered x (lon), y (lat)
    to_latlon = Transformer.from_crs(crs, 'EPSG:4326', always_xy=True)
    rows = []
    for cls in classes:
        # row and col where pixel matches class
        ys, xs = np.where(ext == cls)
        # randomly samples `n_per_class` points or however maany are available without duplication (no replacement)
        pick = rng.choice(len(ys), size=min(
            n_per_class, len(ys)), replace=False)

        for row, col in zip(ys[pick], xs[pick]):
            #  converts row/col pixel index to map coordinates of the pixel center
            # --> eastings (x) and northings (y) in m
            x, y = rasterio.transform.xy(transform, int(row), int(col))
            lon, lat = to_latlon.transform(x, y)  # conversion to lon, lat
            rows.append({'LON': lon, 'LAT': lat})
    # Collect Earth Online (CEO) accuracy assessments require these columns (plotid, lon, lat) to be CAPITALIZED
    df = pd.DataFrame(rows)
    # filling unique plot ids incrementing from 0 to total number of samples - 1
    df.insert(0, 'PLOTID', range(len(df)))
    # fill by interpreting Planet imagery
    df['planet_label'] = ''
    df['entry_date_planet'] = ''
    # fill from CEO interpretation
    df['ceo_label'] = ''
    df['entry_date_ceo'] = ''
    out_file = OUT_DIR / file_name
    df.to_csv(out_file, index=False)
    print(f'Accuracy Assessment CSV created at: {out_file}')
    return out_file


def evaluate_model(accuracy_assessment_sheet_path, model_name, model_extent_map_path, write=False, classes=CLASSES):
    """
    Reads a model's class at each interpreted plot and returns the figures needed for
    an unbiased area estimate.

    Looks up the extent map at every plot's lon/lat, stores the result in a
    `model_name` column, then measures how much of the map each class covers.
    The plots are already fixed, so this is the inverse of
    build_accuracy_assessment_sheet: that goes from map to points, this goes from
    points to map.

    Parameters
    ----------
    accuracy_assessment_sheet_path : str or pathlib.Path
        Accuracy-assessment CSV holding PLOTID/LON/LAT and the interpreted
        reference labels.
    model_name : str
        Column the map's classes are written under, e.g. 'random_forest_full'.
    model_extent_map_path : str or pathlib.Path
        Extent map to evaluate.
    write : bool, optional
        Save the sheet back to local drive with the new column. Default False.
    classes : dict, optional
        Pixel values counted as mapped classes. Only the keys are used, so any
        value not listed (nodata 255) is excluded from the domain.
        Default {0: 'non_mangrove', 1: 'mangrove'}.

    Returns
    -------
    pandas.DataFrame
        The sheet with `model_name` added.
    dict
        Figures for the estimator:

        - ``stratum_px`` : pixels the map assigns to each class
        - ``weights`` : each stratum's share of the classified domain
        - ``px_area_m2`` : ground area of a single pixel
        - ``domain_ha`` : total classified area (in hectares), nodata excluded
        - ``n_unmapped_plots`` : plots that landed on no class (see Notes)

    Notes
    -----
    Plots falling outside the raster, or on nodata, are stored as <NA> rather
    than 0, so they drop out of the assessment instead of counting as
    non-mangrove. `n_unmapped_plots` reports how many.

    See Also
    --------
    docs/references.md#documentation : Python quickstart. Rasterio Documentation.
    docs/references.md#documentation : Transformer. pyproj Stable Documentation.
    docs/references.md#journal-articles--papers : Olofsson et al., 2014
    """
    df = pd.read_csv(accuracy_assessment_sheet_path)
    with rasterio.open(model_extent_map_path) as s:
        ext = s.read(1)  # extent map
        # transform = (a, b, c, d, e, f)
        # where a: x pixel scale (10), b: 0 rotation, c: easting coordinate from raster,
        # d: 0 skew, e: y pixel scale (-10), f: northing coordinate from raster
        transform, crs, height, width, nodata = s.transform, s.crs, s.height, s.width, s.nodata

    # function to convert WGS84 lon/lat coordinates to UTM meters easting/northings
    to_map = Transformer.from_crs('EPSG:4326', crs, always_xy=True)
    x, y = to_map.transform(df['LON'].values, df['LAT'].values)

    # converts to pixel row and cols to use on extent map
    rows, cols = rasterio.transform.rowcol(transform, x, y)
    # casting to numpy arrays just in case for optimized vector funcs
    rows, cols = np.asarray(rows), np.asarray(cols)

    # indices of valid pixels inside the map's bounds
    val_pixels = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
    values = np.full(len(df), np.nan)
    values[val_pixels] = ext[rows[val_pixels], cols[val_pixels]]

    if nodata is not None:
        values[values == nodata] = np.nan

    # converting floats to 0 and 1s and NaNs
    df[model_name] = pd.Series(values).astype('Int64')

    if write:
        df.to_csv(accuracy_assessment_sheet_path, index=False)

    stratum_px = {cls: int((ext == cls).sum()) for cls in classes}
    # transform.a pixel width; transform.e pixel height
    px_area = abs(transform.a) * abs(transform.e)  # m^2 per pixel
    total = sum(stratum_px.values())
    weights = {cls: n / total for cls, n in stratum_px.items()}

    metadata = {
        'stratum_px': stratum_px,
        'weights': weights,
        'px_area_m2': px_area,
        'domain_ha': total * px_area / 1e4,
        'n_unmapped_plots': int(df[model_name].isna().sum())
    }
    return df, metadata


def auc_scores(accuracy_assessment_sheet_path, prob_map_path,
               ref_cols=('ceo_label', 'planet_label'), prob_scale=254, nodata=255):
    """
    Threshold-free ranking scores (ROC-AUC and PR-AUC) for a probability map.

    The probability map is the uint8 ``p_mangrove x prob_scale`` layer written at
    inference. Values are sampled at each plot's lon/lat and scored against every
    reference column in accuracy assessment. Essentially, this is a true head-to-head between models:
    unlike the thresholded accuracies, it does not depend on a cut-off.

    Parameters
    ----------
    accuracy_assessment_sheet_path : str or pathlib.Path
        Accuracy sheet holding PLOTID/LON/LAT and the reference label columns.
    prob_map_path : str or pathlib.Path
        Probability map (uint8, P(mangrove) x `prob_scale`; `nodata` = no data).
    ref_cols : iterable of str, optional
        Reference columns to score against. Default ('ceo_label', 'planet_label').
    prob_scale : int, optional
        Factor the probabilities were multiplied by on write. Default 254.
    nodata : int, optional
        Nodata value in the probability map. Default 255.

    Returns
    -------
    dict
        {ref_col: {'roc_auc', 'pr_auc', 'n', 'base_rate'}}. Plots off the map, on
        nodata, or missing the reference label are dropped per reference; a
        reference with a single class present yields NaN AUCs.

    Notes
    -----
    ROC-AUC is prevalence-independent; PR-AUC (average precision) is prevalence-
    sensitive and the more honest score when mangrove is the rare class, so
    ``base_rate`` is returned as well.

    See Also
    --------
    docs/references.md#documentation : Transformer. pyproj Stable Documentation.
    """
    df = pd.read_csv(accuracy_assessment_sheet_path)
    with rasterio.open(prob_map_path) as s:
        transform, crs, height, width = s.transform, s.crs, s.height, s.width

        # Converts UTM map to row/col coordinates
        to_map = Transformer.from_crs('EPSG:4326', crs, always_xy=True)
        x, y = to_map.transform(df['LON'].values, df['LAT'].values)
        rows, cols = rasterio.transform.rowcol(transform, x, y)
        rows, cols = np.asarray(rows), np.asarray(cols)

        # Converts valid pixel in the probability map from integers back to probabilities
        in_bounds = (rows >= 0) & (rows < height) & (
            cols >= 0) & (cols < width)
        prob = np.full(len(df), np.nan)
        # sample() reads only the in-bounds pixels, never the full raster
        if in_bounds.any():
            coords = zip(np.asarray(x)[in_bounds], np.asarray(y)[in_bounds])
            prob[in_bounds] = [v[0] for v in s.sample(coords)]

    prob[prob == nodata] = np.nan
    prob = prob / prob_scale

    scores = {}

    for col in ref_cols:
        ref = pd.to_numeric(df[col], errors='coerce').values
        keep = np.isfinite(prob) & np.isfinite(ref)
        y_true, y_score = ref[keep].astype(int), prob[keep]
        # checks mangrove and non-mangrove class exists and no weird edge cases present
        single_class = len(np.unique(y_true)) < 2
        scores[col] = {
            'roc_auc': float('nan') if single_class else float(roc_auc_score(y_true, y_score)),
            'pr_auc': float('nan') if single_class else float(average_precision_score(y_true, y_score)),
            'n': int(keep.sum()),
            # Determines class distribution of mangroves/non-mangroves
            'base_rate': float(y_true.mean()) if keep.sum() else float('nan'),
        }
    return scores


def olofsson_estimate(df, map_col, weights, domain_ha=None,
                      ref_col='planet_label', classes=CLASSES, z=1.96):
    """
    Stratified accuracy and error-adjusted area estimator (Olofsson et al., 2014).

    Combines the reference and map-class entries in the
    accuracy-assessment sheet with each stratum's area weight from the extent
    map to caclculate user's, producer's and overall accuracy, and (optionally)
    error-adjusted class area, each with a 95% confidence interval (z = 1.96).

    The map class defines the stratum for every plot, so the sample is treated
    as stratified-by-map (Olofsson's design). Every quantity below is expressed
    in the stratum weights W_i, so absolute pixel counts are not needed.

    Parameters
    ----------
    df : pandas.DataFrame
        Accuracy sheet holding the reference labels and the map's class column,
        e.g. the frame returned by evaluate_model.
    map_col : str
        Column giving the map's class (0/1) at each plot -- the strata. This is
        the `model_name` column written by evaluate_model.
    weights : dict
        {class: W_i}, each stratum's share of the mapped domain. Use
        `metadata['weights']` from evaluate_model (or fixed weights to compare
        several thresholded maps on a common basis).
    domain_ha : float, optional
        Total classified area in hectares (`metadata['domain_ha']`). If given,
        error-adjusted class areas (with CIs) are added to the per-class table.
    ref_col : str, optional
        Reference-label column filled by the interpreter. Default 'planet_label'.
    classes : dict, optional
        Class values to include; only the keys are used. Default {0, 1}.
    z : float, optional
        Normal quantile for the confidence interval. Default 1.96 (95%).

    Returns
    -------
    dict
        - ``confusion`` : DataFrame of counts n_ij (rows = map, cols = reference)
        - ``n`` : plots used (rows with both labels present)
        - ``overall_accuracy`` / ``overall_accuracy_ci``
        - ``per_class`` : DataFrame indexed by class name with user's/producer's
          accuracy (+/- CI), area proportion (+/- CI), and, when `domain_ha` is
          given, error-adjusted ``area_ha`` (+/- CI)
        - ``weights`` : the weights used

    Notes
    -----
    Plots missing either label (blank reference or <NA> map value) are dropped,
    so the reference column must be current for `map_col`

    See Also
    --------
    docs/references.md#journal-articles--papers : Olofsson et al., 2014
    docs/references.md#websites : Costa, (2024)
    https://github.com/cran/mapaccuracy/blob/master/R/olofsson.R
    docs/references.md#ai-usage : AI assistance log
    """
    keys = list(classes)
    names = [classes[k] for k in keys]

    # keep only plots that carry BOTH a reference and a map label
    pair = df[[ref_col, map_col]].apply(
        pd.to_numeric, errors='coerce').dropna()
    ref = pair[ref_col].astype(int)
    mp = pair[map_col].astype(int)

    # confusion counts n_ij: row i = map class (stratum), col j = reference class
    n = pd.DataFrame(
        [[int(((mp == i) & (ref == j)).sum()) for j in keys] for i in keys],
        index=keys, columns=keys)

    ni = n.sum(axis=1)                       # n_i+  plots per stratum
    denom = (ni - 1).clip(lower=1)           # (n_i+ - 1), guarded against /0
    W = pd.Series(weights, dtype=float).reindex(keys)
    q = n.div(ni, axis=0)                    # q_ij = n_ij / n_i+
    p = q.mul(W, axis=0)                      # p_ij = W_i * q_ij

    U = pd.Series({i: q.loc[i, i] for i in keys})          # user's accuracy
    p_area = p.sum(axis=0)                                  # p_+j (area props)
    P = pd.Series({j: p.loc[j, j] / p_area[j] for j in keys})  # producer's acc
    overall = float(sum(p.loc[i, i] for i in keys))        # overall accuracy

    # standard errors (Olofsson et al. 2014, eqs. 5-7, 10)
    se_overall = float(sum(W[i]**2 * U[i] * (1 - U[i]) / denom[i]
                           for i in keys)) ** 0.5
    se_U = pd.Series({i: (U[i] * (1 - U[i]) / denom[i]) ** 0.5 for i in keys})
    se_area = pd.Series({
        j: sum(W[i]**2 * q.loc[i, j] * (1 - q.loc[i, j]) / denom[i]
               for i in keys) ** 0.5
        for j in keys})
    se_P = {}
    for j in keys:
        t1 = W[j]**2 * (1 - P[j])**2 * U[j] * (1 - U[j]) / denom[j]
        t2 = P[j]**2 * sum(W[i]**2 * q.loc[i, j] * (1 - q.loc[i, j]) / denom[i]
                           for i in keys if i != j)
        se_P[j] = (t1 + t2) ** 0.5 / p_area[j]
    se_P = pd.Series(se_P)

    per_class = pd.DataFrame({
        'users_acc': U.values, 'users_ci': (z * se_U).values,
        'producers_acc': P.values, 'producers_ci': (z * se_P).values,
        'area_prop': p_area.values, 'area_prop_ci': (z * se_area).values,
    }, index=names)
    if domain_ha is not None:
        per_class['area_ha'] = p_area.values * domain_ha
        per_class['area_ha_ci'] = (z * se_area).values * domain_ha

    return {
        'confusion': n.rename(index=classes, columns=classes),
        'n': int(ni.sum()),
        'overall_accuracy': overall,
        'overall_accuracy_ci': z * se_overall,
        'per_class': per_class,
        'weights': dict(W),
    }


def summarise_estimates(results, focus='mangrove', digits=3):
    """
    Flatten several olofsson_estimate() results into one table of results

    Presentation only -- useful to feed numbers into CSV, plots or LaTeX. One row per label (model/threshold), with
    overall accuracy and the `focus` class's user's/producer's accuracy and
    (when present) error-adjusted area, each as a "value ± CI" string.

    Parameters
    ----------
    results : dict
        {label: result}, where each result is an olofsson_estimate() return.
    focus : str, optional
        Class name whose accuracies/area are tabulated. Default 'mangrove'.
    digits : int, optional
        Decimal places for the value ± CI cells. Default 3.

    Returns
    -------
    pandas.DataFrame
        One row per label; string cells "value ± CI".
    """
    def plus_minus(v, ci):
        return f'{v:.{digits}f} ± {ci:.{digits}f}'

    rows = {}
    for name, res in results.items():
        focus_stats = res['per_class'].loc[focus]
        row = {
            'n': res['n'],
            'overall acc.': plus_minus(res['overall_accuracy'], res['overall_accuracy_ci']),
            f'{focus} UA': plus_minus(focus_stats['users_acc'], focus_stats['users_ci']),
            f'{focus} PA': plus_minus(focus_stats['producers_acc'], focus_stats['producers_ci']),
        }
        if 'area_ha' in focus_stats:
            row[f'{focus} area (ha)'] = plus_minus(
                focus_stats['area_ha'], focus_stats['area_ha_ci'])
        rows[name] = row

    table = pd.DataFrame(rows).T  # each row a model, and metrics as cols
    return table
