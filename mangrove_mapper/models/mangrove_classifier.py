# User and producer accuracy are more commonly used in remote sensing rather than precision and recall,
# so I've renamed them accordingly
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score as user_score
from sklearn.metrics import recall_score as producer_score
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from xgboost import XGBClassifier

from ..utils.regions import REGIONS, extract_coords

# trained-model files
TRAINED_MODELS_DIR = Path(__file__).resolve().parents[2] / 'trained_models'


class MangroveClassifier:
    """
    Trains and evaluates mangrove classification models on multi-sensor
    samples exported from GEE, using leave-one-region-out spatial
    cross-validation with a coastal buffer to prevent spatial leakage
    between folds.

    Supports random forest and XGBoost (UNet planned), hyperparameter
    tuning, grouped permutation importance, and model persistence.

    See Also
    --------
    docs/references.md#ai-usage : AI assistance log
    """

    def __init__(self):
        # columns that are never features
        self.NON_FEATURES = ['label', '.geo', 'system:index', 'region']

        # Model Hyperparameter tuning
        # See Banerjee, (2020) and XGBoost parameters in references.md for more info.
        self.XGB_PARAMS_DIST = {
            'n_estimators':     [200, 400],
            'max_depth':        [1, 2, 3, 4, 6, 8],
            'learning_rate':    [0.01, 0.03, 0.1, 0.3],
            'subsample':        [0.5, 0.6, 0.7, 0.85, 1.0],
            'colsample_bytree': [0.5, 0.6, 0.7, 0.85, 1.0],
            'min_child_weight': [1, 3, 5]
        }

        self.RF_PARAMS_DIST = {
            'n_estimators':      [100, 300, 400, 500],
            'max_depth':         [None, 10, 20, 30],
            'max_features':      ['sqrt', 'log2', 0.3, 0.5, 0.8, 1.0],
            'min_samples_leaf':  [1, 2, 5, 10, 12],
            'class_weight':      [None, 'balanced'],
        }

    def features_except(self, df, drop_bands=()):
        """
        Extracts the feature matrix by dropping the non-feature columns
        (NON_FEATURES). Also used to perform ablation studies by dropping
        selected feature bands.

        Parameters
        ----------
        df : pandas.DataFrame
            Samples with feature columns plus the NON_FEATURES metadata
            columns.
        drop_bands : tuple or list of str
            Additional feature bands to drop.

        Returns
        -------
        numpy.ndarray
            Feature matrix, columns in the DataFrame's original order.
        """
        return df.drop(columns=self.NON_FEATURES + list(drop_bands)).values

    def run_spatial_cv(
            self, X, y, region_labels, all_coords, regions=None,
            base_estimator=None, param_dist=None, buffer=3.0,
            tune=False, random_state=42, verbose=True):
        """
        Evaluates a classifier with leave-one-region-out spatial CV. For
        each held-out region, training points within `buffer` km of the
        test region are dropped to prevent spatial leakage.

        Parameters
        ----------
        X : numpy.ndarray
            Feature matrix.
        y : numpy.ndarray
            Binary labels (0 = non-mangrove, 1 = mangrove).
        region_labels : numpy.ndarray
            Per-sample region name, aligned with X rows.
        all_coords : numpy.ndarray
            (n_samples, 2) array of [lon, lat] pairs, aligned with X rows.
        regions : list of str, optional
            Regions to hold out, one fold each. Defaults to REGIONS.
        base_estimator : str, optional
            'random_forest' (default) or 'xgboost'.
        param_dist : dict, optional
            Hyperparameter distributions to sample; required when tune=True.
        buffer : float
            Exclusion buffer around the held-out region, in km.
        tune : bool
            If True, tune hyperparameters within each fold's training
            regions (RandomizedSearchCV over GroupKFold) before scoring.
        random_state : int
            Seed for classifier and search reproducibility.
        verbose : bool
            Print per-fold progress and scores.

        Returns
        -------
        pandas.DataFrame
            One row per held-out region: n_dropped, f1 (macro),
            user_accuracy (precision), producer_accuracy (recall) and
            best_params (None unless tuned).
        """
        if regions is None:
            regions = REGIONS
        clf = self.create_classifier(
            base_estimator, random_state=random_state)
        res = []

        for i, region in enumerate(regions, 1):
            t0 = time.perf_counter()
            if verbose:
                print(f"[{i}/{len(regions)}] {region:35s}",
                      end="", flush=True)
            test_mask = region == region_labels
            train_mask = region != region_labels
            valid_train_idx, n_dropped = self._buffered_train_idx(
                all_coords, test_mask, train_mask, buffer)
            if tune:
                search = RandomizedSearchCV(
                    clf,
                    param_dist,
                    n_jobs=-1,
                    n_iter=60,
                    scoring='f1_macro',
                    random_state=random_state,
                    cv=GroupKFold(n_splits=5)
                )
                search.fit(X[valid_train_idx], y[valid_train_idx],
                           groups=region_labels[valid_train_idx])
                clf = search.best_estimator_
                best_params = search.best_params_
            else:
                # fresh classifier state for each fold
                clf = clone(clf).fit(X[valid_train_idx], y[valid_train_idx])
                best_params = None

            y_pred = clf.predict(X[test_mask])
            fold = {
                'region':            region,
                'n_dropped':         n_dropped,
                'f1':                f1_score(y[test_mask], y_pred, average='macro'),
                'user_accuracy':     user_score(y[test_mask], y_pred, pos_label=1),
                'producer_accuracy': producer_score(y[test_mask], y_pred, pos_label=1),
                'best_params':       best_params,
            }
            res.append(fold)
            if verbose:
                cv_note = f"CV={search.best_score_:.3f}  " if tune else ""
                print(f"f1={fold['f1']:.3f}  {cv_note}({time.perf_counter() - t0:4.0f}s)",
                      flush=True)
        return pd.DataFrame(res).set_index('region')

    def _buffered_train_idx(self, all_coords, test_mask, train_mask, buffer):
        """
        Selects the training indices at least `buffer` km away from every
        held-out test point, so near-duplicate neighbours cannot leak
        across the fold boundary.

        Parameters
        ----------
        all_coords : numpy.ndarray
            (n_samples, 2) array of [lon, lat] pairs.
        test_mask : numpy.ndarray
            Boolean mask of held-out samples.
        train_mask : numpy.ndarray
            Boolean mask of candidate training samples.
        buffer : float
            Exclusion radius in km, converted to degrees internally.

        Returns
        -------
        tuple of (numpy.ndarray, int)
            Indices of retained training samples, and the number dropped.
        """
        # circumference of the Earth divided by 360 degrees:
        # 40,075 km ÷ 360° ≈ 111.3 km per degree
        buffer_deg = buffer / 111.0

        # drop training points too close to the held-out region's points
        tree = cKDTree(all_coords[test_mask])
        too_close = np.array([
            len(tree.query_ball_point(pt, r=buffer_deg)) > 0
            for pt in all_coords[train_mask]
        ])
        return np.where(train_mask)[0][~too_close], int(too_close.sum())

    def create_classifier(self, base_estimator, random_state):
        """
        Builds an unfitted classifier of the requested family.

        Parameters
        ----------
        base_estimator : str or None
            'random_forest' (default when None), 'xgboost', or 'unet'
            (not yet implemented).
        random_state : int
            Seed passed to the classifier.

        Returns
        -------
        estimator
            Unfitted scikit-learn compatible classifier.

        Raises
        ------
        NotImplementedError
            If 'unet' is requested.
        ValueError
            If base_estimator is not a recognised name.

        See Also
        --------
        docs/references.md#documentation : XGBoost parameters
        """
        if base_estimator is None:
            base_estimator = 'random_forest'
        if base_estimator in ['random_forest', 'xgboost', 'unet']:
            match base_estimator:
                case "random_forest":
                    clf = RandomForestClassifier(
                        n_estimators=100,
                        random_state=random_state,
                        n_jobs=1)
                case "xgboost":
                    clf = XGBClassifier(
                        # minimizes logloss producing probabilities (via sigmoid fn)
                        # other options do not have this same functionality
                        objective='binary:logistic',
                        # logloss is a continuous fn as well and better suited for class imbalance
                        eval_metric='logloss',
                        # fastest optimizer
                        tree_method='hist',
                        random_state=random_state,
                        n_jobs=1
                    )
                case "unet":
                    # TODO: add unet code
                    raise NotImplementedError(
                        "unet is not implemented yet")
        else:
            raise ValueError(
                f"base_estimator must be one of ['random_forest', 'xgboost', 'unet'], "
                f"got '{base_estimator}'")
        return clf

    def build_final_model(self, X, y, groups, base_estimator=None, param_dist=None,
                          n_iter=40, random_state=42, verbose=True, save_as=None):
        """
        Tunes and fits the final model on all regions, using
        leave-one-region-out folds (GroupKFold) and macro-F1 so the search
        optimises the same objective run_spatial_cv reports.

        Parameters
        ----------
        X : numpy.ndarray
            Feature matrix.
        y : numpy.ndarray
            Binary labels.
        groups : array-like
            Per-sample region name, used as the CV groups.
        base_estimator : str, optional
            'random_forest' (default) or 'xgboost'.
        param_dist : dict
            Hyperparameter distributions to sample.
        n_iter : int
            Number of sampled parameter candidates.
        random_state : int
            Seed for search reproducibility.
        verbose : bool
            Print the chosen hyperparameters.
        save_as : str, optional
            If given, save the best model and its tuned params under this
            name via save_model.

        Returns
        -------
        tuple of (estimator, dict)
            The fitted best estimator and its tuned hyperparameters.
        """
        clf = self.create_classifier(
            base_estimator, random_state=random_state)
        search = RandomizedSearchCV(estimator=clf,
                                    param_distributions=param_dist,
                                    n_jobs=-1,
                                    scoring='f1_macro',
                                    random_state=random_state,
                                    n_iter=n_iter,
                                    # leave-one-region-out folds
                                    cv=GroupKFold(n_splits=len(np.unique(groups))))
        search.fit(X, y, groups=groups)
        if verbose:
            print("Chosen hyperparameters:", search.best_params_)
        if save_as:
            self.save_model(search.best_estimator_, save_as,
                            best_params=search.best_params_)
        return search.best_estimator_, search.best_params_

    def save_model(self, model, name, best_params=None, output_dir=TRAINED_MODELS_DIR,
                   metadata=None):
        """
        Serializes a fitted model, plus an optional sidecar JSON recording
        the tuned hyperparameters. XGBoost models use the native JSON
        format (stable across xgboost versions); anything else is pickled
        with joblib.

        Parameters
        ----------
        model : estimator
            Fitted classifier.
        name : str
            Base file name, without extension.
        best_params : dict, optional
            Tuned hyperparameters, written to <name>_best_params.json.
        output_dir : Path or str
            Output directory, created if missing. Defaults to
            <repo>/trained_models.
        metadata : dict, optional
            Values derived from the training set that inference needs, such
            as {'landward_cutoff_m': 2084.0}. Written to <name>_metadata.json,
            kept apart from best_params because load_model passes those to
            set_params. Read back with load_metadata.

        Returns
        -------
        Path
            Path of the saved model file.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if isinstance(model, XGBClassifier):
            model_path = output_dir / f'{name}.json'
            model.save_model(model_path)
        else:
            model_path = output_dir / f'{name}.joblib'
            joblib.dump(model, model_path)
        if best_params is not None:
            # numpy scalars sampled from param_dist are not JSON serializable
            clean = {k: v.item() if hasattr(v, 'item') else v
                     for k, v in best_params.items()}
            with open(output_dir / f'{name}_best_params.json', 'w') as f:
                json.dump(clean, f, indent=2)
        if metadata is not None:
            clean = {k: v.item() if hasattr(v, 'item') else v
                     for k, v in metadata.items()}
            with open(output_dir / f'{name}_metadata.json', 'w') as f:
                json.dump(clean, f, indent=2)
        return model_path

    def load_metadata(self, path):
        """
        Loads the metadata sidecar written by save_model for a model file.

        Parameters
        ----------
        path : Path or str
            Path of the saved model file (not the sidecar).

        Returns
        -------
        dict
            The saved metadata, or an empty dict if the model has none.
        """
        path = Path(path)
        meta_path = path.parent / f'{path.stem}_metadata.json'
        if not meta_path.exists():
            return {}
        with open(meta_path) as f:
            return json.load(f)

    def load_model(self, path):
        """
        Loads a model saved by save_model, dispatching on the extension:
        '.json' loads an XGBoost booster (restoring tuned hyperparameters
        from the sidecar JSON when present), anything else is loaded with
        joblib. The returned model is fitted and ready to predict.

        Parameters
        ----------
        path : Path or str
            Path of the saved model file.

        Returns
        -------
        estimator
            Fitted classifier.
        """
        path = Path(path)
        if path.suffix == '.json':
            model = XGBClassifier()
            model.load_model(path)
            # Only stores the booster, the params must be added in separately
            params_path = path.parent / f'{path.stem}_best_params.json'
            if params_path.exists():
                with open(params_path) as f:
                    model.set_params(**json.load(f))
            return model
        return joblib.load(path)

    def summarise(self, name, results_df):
        """
        Prints a titled results table (excluding best_params) followed by
        the mean of each metric across the region hold-outs.

        Parameters
        ----------
        name : str
            Title printed above the table.
        results_df : pandas.DataFrame
            Output of run_spatial_cv.
        """
        cols = [c for c in results_df.columns if c != 'best_params']
        print("\033[1m" + name + "\033[0m")  # bolding the titles
        print(results_df[cols].round(3).to_string())
        print("\nMean across region hold-outs:")
        print(results_df[cols].mean(
            numeric_only=True).round(3).to_string(), "\n")

    def grouped_permutation_importance(
            self, data, region_labels, all_coords=None, regions=None,
            feature_groups=None, base_estimator=None, buffer=3.0,
            n_repeats=20, random_state=42, verbose=True):
        """
        Permutation importance per feature group, under the same
        region-held-out, buffered setup as run_spatial_cv. Each group's
        columns are permuted together (one shared row order), so correlated
        features are scored jointly; ungrouped features are permuted alone.
        Importance is the drop in held-out macro-F1, averaged over
        n_repeats.

        Parameters
        ----------
        data : pandas.DataFrame
            Samples with feature columns, 'label', and the other
            NON_FEATURES columns.
        region_labels : array-like
            Per-sample region name.
        all_coords : numpy.ndarray, optional
            [lon, lat] per sample; derived from '.geo' if not given.
        regions : list of str, optional
            Regions to hold out. Defaults to REGIONS.
        feature_groups : dict, optional
            Group name -> list of correlated feature columns.
        base_estimator : str, optional
            'random_forest' (default) or 'xgboost'.
        buffer : float
            Exclusion buffer in km.
        n_repeats : int
            Permutations averaged per group and region.
        random_state : int
            Seed, re-applied per region.
        verbose : bool
            Print per-region progress.

        Returns
        -------
        pandas.DataFrame
            Groups as rows, regions plus 'mean' as columns; higher F1 drop
            means more important. Sorted by 'mean' descending.
        """
        if regions is None:
            regions = REGIONS
        if all_coords is None:
            all_coords = extract_coords(data)
        feature_groups = feature_groups or {}
        feat_names = [c for c in data.columns if c not in self.NON_FEATURES]
        missing = sorted({c for cols in feature_groups.values()
                          for c in cols} - set(feat_names))
        if missing:
            raise ValueError(
                f"feature_groups references columns not in data: {missing}")

        col_idx = {name: j for j, name in enumerate(feat_names)}
        X = data[feat_names].values
        y = data['label'].values
        region_labels = np.asarray(region_labels)

        clf = self.create_classifier(base_estimator, random_state=random_state)

        grouped_cols = {g: np.array([col_idx[c] for c in cols])
                        for g, cols in feature_groups.items()}
        in_a_group = {c for cols in feature_groups.values() for c in cols}
        for name in feat_names:
            if name not in in_a_group:
                grouped_cols[name] = np.array([col_idx[name]])

        per_region = {}
        for region in regions:
            if verbose:
                print(f"  {region} ...", flush=True)
            test_mask = region_labels == region
            train_mask = region_labels != region

            train_idx, _ = self._buffered_train_idx(
                all_coords, test_mask, train_mask, buffer)

            fold_clf = clone(clf).fit(X[train_idx], y[train_idx])

            X_test, y_test = X[test_mask], y[test_mask]
            baseline = f1_score(
                y_test, fold_clf.predict(X_test), average='macro')

            # random state set for reproducibility, consider modifying for more randomized results
            rng = np.random.default_rng(random_state)
            importances = {}
            for gname, cols in grouped_cols.items():
                drops = np.empty(n_repeats)
                for r in range(n_repeats):
                    Xp = X_test.copy()
                    # ONE order for the whole group
                    perm = rng.permutation(len(Xp))
                    # preserves intra-group correlation
                    Xp[:, cols] = Xp[perm][:, cols]
                    drops[r] = baseline - f1_score(
                        y_test, fold_clf.predict(Xp), average='macro')
                importances[gname] = drops.mean()
            per_region[region] = importances

        imp = pd.DataFrame(per_region)
        imp['mean'] = imp.mean(axis=1)
        return imp.sort_values('mean', ascending=False)
