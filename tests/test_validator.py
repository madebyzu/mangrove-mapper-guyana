"""
Unit tests for the accuracy-assessment module: map sampling at the interpreted
plots (evaluate_model), threshold-free ranking scores (auc_scores) and the
stratified estimator (olofsson_estimate). All tests run on small synthetic
GeoTiffs and hand-built sheets, so no real extent map or Earth Engine (GEE)
objects are involved.

See Also
--------
docs/references.md#journal-articles--papers : Olofsson et al., 2014
docs/references.md#websites : Costa, (2024)
https://search.r-project.org/CRAN/refmans/mapaccuracy/html/olofsson.html 
https://github.com/cran/mapaccuracy/blob/master/R/olofsson.R#L205C3-L233C1
"""

import numpy as np
import pandas as pd
import pytest
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin

from mangrove_mapper.validation.validator import (
    auc_scores,
    evaluate_model,
    olofsson_estimate,
)

RES = 10                  # 10 m pixels, so one pixel is 0.01 ha
CRS = "EPSG:32621"        # UTM 21N, the grid the Guyana maps are exported on
ORIGIN = (500000, 500000)  # top-left easting/northing of every fixture raster
NODATA = 255

# 4 x 4 test map: 4 mangrove px, 11 non-mangrove px, 1 nodata px, so the
# stratum weights are 4/15 and 11/15 and the classified domain is 0.15 ha
EXTENT = np.array([
    [1, 1, 0, 0],
    [1, 1, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, NODATA],
], dtype=np.uint8)


def _write(path, arr, nodata=NODATA):
    """Write a single-band (H, W) array as a UTM 21N GeoTiff on a 10 m grid."""
    height, width = arr.shape
    profile = {
        "driver": "GTiff", "count": 1, "height": height, "width": width,
        "dtype": arr.dtype, "crs": CRS, "nodata": nodata,
        "transform": from_origin(*ORIGIN, RES, RES),
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr, 1)
    return str(path)


def _lonlat(row, col):
    """Lon/lat of the centre of pixel (row, col) on the fixture grid."""
    # from top-left corner
    # moving to the right by col * RES
    x = ORIGIN[0] + col * RES + RES / 2
    # moving to down by row * RES
    y = ORIGIN[1] - row * RES - RES / 2
    return Transformer.from_crs(CRS, "EPSG:4326", always_xy=True).transform(x, y)


def _sheet(path, plots, **label_cols):
    """
    Write an accuracy-assessment CSV for plots given as (row, col) pixels.

    `label_cols` supplies the interpreted columns, e.g.
    ``planet_label=[1, 1, 0, ""]``; a blank string is an uninterpreted plot.
    Pixels outside the 4 x 4 grid are allowed and stay off-map on purpose.
    """
    lon, lat = zip(*(_lonlat(r, c) for r, c in plots))
    df = pd.DataFrame({"PLOTID": range(len(plots)), "LON": lon, "LAT": lat})
    for name, values in label_cols.items():
        df[name] = values
    df.to_csv(path, index=False)
    return str(path)

# Olofsson et al. (2014), Section 5 worked example pg.52-55

# A hypothetical 2000-2010 forest-change map with two change and two stable
# classes, assessed by stratified random sampling with the map classes as
# strata, similar design as the mangrove assessment, so using this worked example
# for code verfication.


OLOFSSON_CLASSES = {1: 'deforestation', 2: 'forest_gain',
                    3: 'stable_forest', 4: 'stable_non_forest'}

# Table 8: sample counts n_{ij}, {(map class, reference class): count}
OLOFSSON_COUNTS = {

    (1, 1): 66, (1, 2): 0, (1, 3): 5, (1, 4): 4,      # = 75
    (2, 1): 0, (2, 2): 55, (2, 3): 8, (2, 4): 12,     # = 75
    (3, 1): 1, (3, 2): 0, (3, 3): 153, (3, 4): 11,    # = 165
    (4, 1): 2, (4, 2): 1, (4, 3): 9, (4, 4): 313,     # = 325
}                                                     # = 640
TOTAL_COUNTS = 640

# Table 8: mapped area of each stratum, 200,000 / 150,000 / 3,200,000 /
# 6,450,000 of 10,000,000 Landsat pixels
OLOFSSON_WEIGHTS = {1: 0.020, 2: 0.015, 3: 0.320, 4: 0.645}

# 10,000,000 Landsat pixels at 30 m, i.e. 0.09 ha each
OLOFSSON_DOMAIN_HA = 10_000_000 * 30**2 / 1e4

# Section 5.2.1 (accuracies, +/- 95% CI), Table 9 column totals (area
# proportions) and Section 5.2.2 (error-adjusted areas). Column names and
# order match the per_class frame olofsson_estimate returns, so the whole
# table can be compared in one go. The paper reports accuracies to 2 decimal places, so
# compare those with abs=0.005 and the areas with abs=1 (ha).
OLOFSSON_EXPECTED = pd.DataFrame({
    'users_acc': [0.88, 0.73, 0.93, 0.96],
    'users_ci': [0.07, 0.10, 0.04, 0.02],
    'producers_acc': [0.75, 0.85, 0.93, 0.96],
    # as printed; two of these are not reproducible, see OLOFSSON_EQ7_PA_CI
    'producers_ci': [0.21, 0.23, 0.03, 0.01],
    'area_prop': [0.0235, 0.0130, 0.3175, 0.6460],
    'area_ha': [21158, 11686, 285770, 581386],
    'area_ha_ci': [6158, 3756, 15510, 16282],
}, index=list(OLOFSSON_CLASSES.values()))

OLOFSSON_OVERALL = (0.95, 0.02)   # overall accuracy +/- 95% CI

# Producer's accuracy CIs from Eq. (7) applied to Table 8's counts.
# Deforestation and stable forest reproduce Section 5.2.1 exactly, but forest
# gain evaluates to 0.254 where the paper prints 0.23, and stable non-forest
# to 0.018 where it prints 0.01. Rounding does not close the gap and the
# CRAN package mapaccuracy::olofsson() implements Eq. (7) identically to validator.py,
# so the equation is taken as the authority for these two classes.
OLOFSSON_EQ7_PA_CI = [0.21, 0.25, 0.03, 0.02]


@pytest.fixture
def olofsson_sheet():
    """Table 8's counts as a plot-level frame, one row per sample unit."""
    return _sheet_frame(OLOFSSON_COUNTS)


def _sheet_frame(confusion, map_col="model", ref_col="planet_label"):
    """
    Build an evaluate_model-style frame from a confusion matrix.

    `confusion` is {(map_class, ref_class): count}, so a case can be stated as
    the counts Olofsson's equations consume without going through a raster.
    """
    rows = [{map_col: m, ref_col: r}
            for (m, r), count in confusion.items() for _ in range(count)]
    df = pd.DataFrame(rows, columns=[map_col, ref_col])
    df[map_col] = df[map_col].astype("Int64")
    return df


@pytest.fixture
def extent_map(tmp_path):
    return _write(tmp_path / "extent.tif", EXTENT)


@pytest.fixture
def prob_map(tmp_path):
    # Each entry will be divided by PROB_SCALE (254) to get P(mangrove)
    # P(mangrove) x 254: high on the mangrove block, low elsewhere, and nodata
    # on the pixel the extent map also masks out
    prob = np.array([
        [229, 229, 25, 25],
        [229, 229, 25, 25],
        [25, 25, 25, 25],
        [25, 25, 25, NODATA],
    ], dtype=np.uint8)
    return _write(tmp_path / "prob.tif", prob)


@pytest.fixture
def sheet(tmp_path):
    """
    Six plots as (row, col) pixels: two on mangrove, two on non-mangrove, one
    on the nodata pixel and one just past the edge of the raster.

    The blank ceo_label sits on plot 3, a valid pixel, so that dropping it is
    visible in the scores; a blank on an already-dropped plot would prove
    nothing.
    """
    plots = [(0, 0), (1, 1), (2, 2), (3, 0), (3, 3), (4, 4)]
    return _sheet(
        tmp_path / "sheet.csv", plots,
        planet_label=[1, 1, 0, 0, 0, 1],
        ceo_label=[1, 0, 0, "", 0, 1],
    )


class TestEvaluateModel:
    def test_samples_the_map_class_at_each_plot(self, sheet, extent_map):
        df, _ = evaluate_model(sheet, 'model', extent_map)

        # plots 0-3 sit on mangrove, mangrove, non-mangrove, non-mangrove
        assert df['model'].tolist()[:4] == [1, 1, 0, 0]

    def test_off_map_plots_are_na_not_zero(self, sheet, extent_map):
        df, _ = evaluate_model(sheet, 'model', extent_map)

        # plot 5 is one pixel past the corner; scoring it 0 would silently
        # add a non-mangrove plot the map never classified
        assert pd.isna(df['model'].iloc[5])

    def test_nodata_plots_are_na_not_zero(self, sheet, extent_map):
        df, _ = evaluate_model(sheet, 'model', extent_map)

        assert pd.isna(df['model'].iloc[4])

    def test_counts_unmapped_plots(self, sheet, extent_map):
        _, metadata = evaluate_model(sheet, 'model', extent_map)

        # the nodata plot and the off-map plot
        assert metadata['n_unmapped_plots'] == 2

    def test_stratum_pixel_counts_exclude_nodata(self, sheet, extent_map):
        _, metadata = evaluate_model(sheet, 'model', extent_map)

        # 16 pixels in the raster, one of them nodata
        assert metadata['stratum_px'] == {0: 11, 1: 4}

    def test_weights_are_stratum_shares_summing_to_one(self, sheet, extent_map):
        _, metadata = evaluate_model(sheet, 'model', extent_map)

        assert metadata['weights'] == pytest.approx({0: 11 / 15, 1: 4 / 15})
        assert sum(metadata['weights'].values()) == pytest.approx(1.0)

    def test_pixel_area_and_domain_area(self, sheet, extent_map):
        _, metadata = evaluate_model(sheet, 'model', extent_map)

        assert metadata['px_area_m2'] == pytest.approx(RES ** 2)
        # 15 classified pixels of 100 m^2 each
        assert metadata['domain_ha'] == pytest.approx(0.15)

    def test_write_persists_the_model_column(self, sheet, extent_map):
        evaluate_model(sheet, 'model', extent_map, write=True)

        assert 'model' in pd.read_csv(sheet).columns

    def test_does_not_write_by_default(self, sheet, extent_map):
        evaluate_model(sheet, 'model', extent_map)

        assert 'model' not in pd.read_csv(sheet).columns


class TestAucScores:
    def test_scores_every_reference_column(self, sheet, prob_map):
        scores = auc_scores(sheet, prob_map)

        assert set(scores) == {'planet_label', 'ceo_label'}

    def test_perfectly_ranked_map_scores_one(self, sheet, prob_map):
        scores = auc_scores(sheet, prob_map)

        # every scored plot labelled mangrove carries the higher probability
        assert scores['planet_label']['roc_auc'] == pytest.approx(1.0)
        assert scores['planet_label']['pr_auc'] == pytest.approx(1.0)

    def test_drops_plots_off_the_map_or_on_nodata(self, sheet, prob_map):
        scores = auc_scores(sheet, prob_map)

        # six plots, less the nodata pixel and the one past the edge
        assert scores['planet_label']['n'] == 4

    def test_drops_plots_missing_the_reference_label(self, sheet, prob_map):
        scores = auc_scores(sheet, prob_map)

        # ceo_label is blank on plot 3, which planet_label scores
        assert scores['ceo_label']['n'] == scores['planet_label']['n'] - 1

    def test_base_rate_is_the_reference_prevalence(self, sheet, prob_map):
        scores = auc_scores(sheet, prob_map)

        assert scores['planet_label']['base_rate'] == pytest.approx(0.5)
        assert scores['ceo_label']['base_rate'] == pytest.approx(1 / 3)


class TestOlofssonEstimate:

    def test_matches_published_worked_example(self, olofsson_sheet):
        # Olofsson et al. (2014), sec. 5: check UA/PA/OA and the adjusted area
        # against the published numbers
        res = olofsson_estimate(olofsson_sheet,
                                map_col='model', ref_col='planet_label', domain_ha=OLOFSSON_DOMAIN_HA,
                                weights=OLOFSSON_WEIGHTS, classes=OLOFSSON_CLASSES)
        acc = ['users_acc', 'users_ci', 'producers_acc']

        assert res['n'] == TOTAL_COUNTS

        assert res['per_class'][acc].to_numpy() == pytest.approx(
            OLOFSSON_EXPECTED[acc].to_numpy(), abs=0.005)

        assert res['per_class']['producers_ci'].to_numpy() == pytest.approx(
            OLOFSSON_EQ7_PA_CI, abs=0.005)

        assert res['per_class']['area_prop'].to_numpy() == pytest.approx(
            OLOFSSON_EXPECTED['area_prop'].to_numpy(), abs=5e-5)

        assert res['per_class'][['area_ha', 'area_ha_ci']].to_numpy() == pytest.approx(
            OLOFSSON_EXPECTED[['area_ha', 'area_ha_ci']].to_numpy(), abs=1)

        assert (res['overall_accuracy'], res['overall_accuracy_ci']) == pytest.approx(
            OLOFSSON_OVERALL, abs=0.005)


class TestEndToEnd:
    def test_evaluate_model_feeds_olofsson_estimate(self, sheet, extent_map):
        # metadata['weights'] and metadata['domain_ha'] flow straight into the
        # estimator, which is how the notebooks call the two functions
        df, metadata = evaluate_model(sheet, 'model', extent_map)

        res = olofsson_estimate(df, 'model', metadata['weights'],
                                domain_ha=metadata['domain_ha'])

        # the four scored plots all agree with planet_label, so a perfect map
        # puts each class's estimated area at its stratum's share of the domain
        assert res['n'] == 4
        assert res['overall_accuracy'] == pytest.approx(1.0)
        assert res['per_class']['area_ha'].tolist() == pytest.approx([
            0.11, 0.04])
