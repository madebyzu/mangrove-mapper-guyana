"""
Unit tests for the MangroveClassifier model wrapper. All tests run on small
synthetic datasets, so no Google Earth Engine (GEE) objects are involved.

See Also
--------
docs/references.md#ai-usage : AI assistance log
"""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from mangrove_mapper.models.mangrove_classifier import MangroveClassifier


@pytest.fixture
def mc():
    return MangroveClassifier()


@pytest.fixture
def small_dataset():
    # linearly separable-ish binary problem, enough samples for CV metrics
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 3))
    y = (X[:, 0] + rng.normal(scale=0.3, size=80) > 0).astype(int)
    return X, y


def _spatial_dataset(n_per_region=40):
    # two spatially separated regions ~50 km apart (0.5 deg at the equator)
    rng = np.random.default_rng(1)
    coords_a = rng.uniform(0.0, 0.1, size=(n_per_region, 2))
    coords_b = rng.uniform(0.5, 0.6, size=(n_per_region, 2))
    all_coords = np.vstack([coords_a, coords_b])
    region_labels = np.array(['A'] * n_per_region + ['B'] * n_per_region)
    X = rng.normal(size=(2 * n_per_region, 3))
    y = (X[:, 0] > 0).astype(int)
    return X, y, region_labels, all_coords


def _planted_dataset(n_per_region=120):
    # signal_a drives the label and signal_b is its correlated copy;
    # noise_1/noise_2 are correlated with each other but not the label;
    # noise_solo belongs to no feature group (exercises the singleton path)
    rng = np.random.default_rng(0)
    n = 2 * n_per_region
    signal_a = rng.normal(size=n)
    noise_base = rng.normal(size=n)
    data = pd.DataFrame({
        'signal_a': signal_a,
        'signal_b': signal_a + rng.normal(scale=0.1, size=n),
        'noise_1': noise_base,
        'noise_2': noise_base + rng.normal(scale=0.1, size=n),
        'noise_solo': rng.normal(size=n),
        'label': (signal_a > 0).astype(int),
        'region': ['A'] * n_per_region + ['B'] * n_per_region,
    })
    coords_a = rng.uniform(0.0, 0.1, size=(n_per_region, 2))
    coords_b = rng.uniform(0.5, 0.6, size=(n_per_region, 2))
    all_coords = np.vstack([coords_a, coords_b])
    # '.geo' mirrors all_coords, as in a real GEE export ([lon, lat] order)
    data['.geo'] = [json.dumps({'type': 'Point', 'coordinates': list(pt)})
                    for pt in all_coords]
    return data, data['region'].values, all_coords


class TestFeaturesExcept:
    def test_drops_non_feature_columns(self, mc):
        df = pd.DataFrame({
            'B2': [1.0, 2.0], 'B3': [3.0, 4.0],
            'label': [0, 1], '.geo': ['x', 'y'],
            'system:index': ['0', '1'], 'region': ['A', 'B']})

        out = mc.features_except(df)

        assert out.shape == (2, 2)

    def test_drops_requested_bands_as_well(self, mc):
        df = pd.DataFrame({
            'B2': [1.0], 'B3': [2.0],
            'label': [0], '.geo': ['x'],
            'system:index': ['0'], 'region': ['A']})

        out = mc.features_except(df, drop_bands=('B3',))

        assert out.shape == (1, 1)


class TestCreateClassifier:
    def test_defaults_to_random_forest(self, mc):
        clf = mc.create_classifier(None, random_state=42)

        assert isinstance(clf, RandomForestClassifier)

    def test_creates_xgboost(self, mc):
        clf = mc.create_classifier('xgboost', random_state=42)

        assert isinstance(clf, XGBClassifier)

    def test_unet_not_implemented_yet(self, mc):
        with pytest.raises(NotImplementedError):
            mc.create_classifier('unet', random_state=42)

    def test_unknown_estimator_raises(self, mc):
        with pytest.raises(ValueError, match='base_estimator'):
            mc.create_classifier('svm', random_state=42)


class TestSaveLoadModel:
    def test_random_forest_round_trip(self, mc, small_dataset, tmp_path):
        X, y = small_dataset
        clf = mc.create_classifier('random_forest', random_state=42).fit(X, y)

        path = mc.save_model(clf, 'rf', output_dir=tmp_path)
        loaded = mc.load_model(path)

        assert path.suffix == '.joblib'
        assert (loaded.predict(X) == clf.predict(X)).all()

    def test_xgboost_round_trip_uses_native_json(self, mc, small_dataset, tmp_path):
        X, y = small_dataset
        clf = mc.create_classifier('xgboost', random_state=42).fit(X, y)

        path = mc.save_model(clf, 'xgb', output_dir=tmp_path)
        loaded = mc.load_model(path)

        assert path.suffix == '.json'
        assert (loaded.predict(X) == clf.predict(X)).all()

    def test_best_params_sidecar_handles_numpy_scalars(self, mc, small_dataset, tmp_path):
        X, y = small_dataset
        clf = mc.create_classifier('random_forest', random_state=42).fit(X, y)

        mc.save_model(clf, 'rf', output_dir=tmp_path,
                      best_params={'n_estimators': np.int64(100),
                                   'learning_rate': np.float64(0.1)})

        params = json.loads((tmp_path / 'rf_best_params.json').read_text())
        assert params == {'n_estimators': 100, 'learning_rate': 0.1}

    def test_no_sidecar_without_best_params(self, mc, small_dataset, tmp_path):
        X, y = small_dataset
        clf = mc.create_classifier('random_forest', random_state=42).fit(X, y)

        mc.save_model(clf, 'rf', output_dir=tmp_path)

        assert not (tmp_path / 'rf_best_params.json').exists()

    def test_creates_missing_output_dir(self, mc, small_dataset, tmp_path):
        X, y = small_dataset
        clf = mc.create_classifier('random_forest', random_state=42).fit(X, y)
        nested = tmp_path / 'does' / 'not' / 'exist'

        mc.save_model(clf, 'rf', output_dir=nested)

        assert nested.exists()


class TestRunSpatialCV:
    def test_one_fold_per_region_with_metrics(self, mc):
        X, y, region_labels, all_coords = _spatial_dataset()

        res = mc.run_spatial_cv(X, y, region_labels, all_coords,
                                regions=['A', 'B'], buffer=3.0, verbose=False)

        assert list(res.index) == ['A', 'B']
        assert {'f1', 'user_accuracy', 'producer_accuracy',
                'n_dropped'} <= set(res.columns)
        assert ((res['f1'] >= 0) & (res['f1'] <= 1)).all()

    def test_no_drops_when_regions_far_apart(self, mc):
        # regions are ~50 km apart, so a 3 km buffer removes nothing
        X, y, region_labels, all_coords = _spatial_dataset()

        res = mc.run_spatial_cv(X, y, region_labels, all_coords,
                                regions=['A', 'B'], buffer=3.0, verbose=False)

        assert res['n_dropped'].sum() == 0

    def test_buffer_drops_train_points_near_test_region(self, mc):
        X, y, region_labels, all_coords = _spatial_dataset()
        # plant one region-A training point inside region B's cluster
        all_coords = np.vstack([all_coords, [0.55, 0.55]])
        region_labels = np.append(region_labels, 'A')
        X = np.vstack([X, X[0]])
        y = np.append(y, y[0])

        res = mc.run_spatial_cv(X, y, region_labels, all_coords,
                                regions=['B'], buffer=10.0, verbose=False)

        assert res.loc['B', 'n_dropped'] >= 1


class TestSummarise:
    def test_prints_metrics_but_not_best_params(self, mc, capsys):
        results = pd.DataFrame({
            'f1': [0.9, 0.8],
            'best_params': [{'n_estimators': 100}, None],
        }, index=['A', 'B'])

        mc.summarise('My Model', results)

        out = capsys.readouterr().out
        assert 'My Model' in out
        assert 'f1' in out
        assert 'n_estimators' not in out


class TestGroupedPermutationImportance:
    GROUPS = {'signal': ['signal_a', 'signal_b'],
              'corr_noise': ['noise_1', 'noise_2']}

    def test_group_ranks_first(self, mc):
        data, region_labels, all_coords = _planted_dataset()

        imp = mc.grouped_permutation_importance(
            data, region_labels, all_coords, regions=['A', 'B'],
            feature_groups=self.GROUPS, n_repeats=10, verbose=False)

        assert imp.index[0] == 'signal', "informative group should rank first"
        assert imp.loc['signal', 'mean'] > 0.2
        # correlated-but-irrelevant group must not gain importance from grouping
        assert abs(imp.loc['corr_noise', 'mean']) < 0.05
        assert abs(imp.loc['noise_solo', 'mean']) < 0.05

    def test_one_row_per_group_and_singleton(self, mc):
        data, region_labels, all_coords = _planted_dataset()

        imp = mc.grouped_permutation_importance(
            data, region_labels, all_coords, regions=['A', 'B'],
            feature_groups=self.GROUPS, n_repeats=2, verbose=False)

        assert set(imp.index) == {'signal', 'corr_noise', 'noise_solo'}
        assert list(imp.columns) == ['A', 'B', 'mean']

    def test_unknown_group_column_raises(self, mc):
        data, region_labels, all_coords = _planted_dataset()

        with pytest.raises(ValueError, match='does_not_exist'):
            mc.grouped_permutation_importance(
                data, region_labels, all_coords, regions=['A', 'B'],
                feature_groups={'bad': ['does_not_exist']}, verbose=False)
