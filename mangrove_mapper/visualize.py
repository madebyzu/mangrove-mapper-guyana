"""
Plotting utilities for spatial samples, learning curves, and feature
importance.

See Also
--------
docs/references.md#ai-usage : AI assistance log
"""

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import shap
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold, learning_curve
from xgboost import XGBClassifier

from .utils.regions import REGIONS, extract_coords, tag_regions


def plot_spatial_samples_over_regions(data_path, regions_data_path=None, figsize=(8, 6),
                                      ax=None):
    """
    Plots the training samples at their coordinates, coloured by the coastal
    region

    Parameters
    ----------
    data_path : str or Path
        Path to the samples file exported from GEE. Must contain a '.geo'
        column of GeoJSON strings.
    regions_data_path : str or Path, optional
        Path to the regions GeoJSON. Exported via export_regions_map() if
        not provided, which requires an authenticated GEE session.
    figsize : tuple of float
        Matplotlib figure height and width in inches. Ignored when `ax` is given.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.

    Returns
    -------
    matplotlib.axes.Axes
        The axes drawn on.
    """
    data = gpd.read_file(data_path)
    data = tag_regions(data, regions_data_path)

    # extract after tagging so lon/lat stay aligned with 'region'
    all_coords = extract_coords(data)
    lon, lat = all_coords[:, 0], all_coords[:, 1]

    # only create and show a figure when the caller did not supply an axes
    owns_figure = ax is None
    if owns_figure:
        _, ax = plt.subplots(figsize=figsize)

    present_regions = set(data['region'].dropna().unique())
    for region in REGIONS:
        if region not in present_regions:
            continue
        mask = data['region'] == region
        ax.scatter(lon[mask], lat[mask], s=2, label=region)

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Spatial Samples (6 Coastline Regions)')
    ax.legend(markerscale=5)

    if owns_figure:
        ax.figure.tight_layout()
        plt.show()
    return ax


def plot_learning_curve(estimator, X, y, region_labels, title, ax):
    """
    Plots a learning curve (f1_macro vs training-set size) where each
    validation fold is a held-out region (GroupKFold), so the curve reflects
    performance on unseen regions rather than unseen points.

    Parameters
    ----------
    estimator : estimator
        Unfitted scikit-learn compatible classifier.
    X : numpy.ndarray
        matrix of bands and indices
    y : numpy.ndarray
        0 - nonmangrove, 1 - mangrove
    region_labels : array-like
        Per-sample region name, used as the CV groups.
    title : str
        Axes title.
    ax : matplotlib.axes.Axes
        Axes to draw on
    """
    sizes, train, val = learning_curve(
        estimator, X, y, groups=region_labels,
        cv=GroupKFold(n_splits=6), scoring='f1_macro',
        train_sizes=np.linspace(0.2, 1.0, 6), n_jobs=-1)
    ax.plot(sizes, train.mean(1), 'o-', label='train')
    ax.plot(sizes, val.mean(1),   'o-', label='validation (held-out region)')
    ax.fill_between(sizes, val.mean(1) - val.std(1),
                    val.mean(1) + val.std(1), alpha=0.15)
    ax.set_xlabel('training samples')
    ax.set_ylabel('f1_macro')
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)


def plot_shap_bar(data, feature_names, model, max_display=20):
    """
    Plots a SHAP bar chart of mean absolute feature contributions for a
    fitted tree-based model.

    Parameters
    ----------
    data : pandas.DataFrame
        Samples to explain.
    feature_names : list of str
        Feature columns, in the order the model was trained on.
    model : estimator
        Fitted tree-based model (e.g. RandomForest or XGBoost).
    max_display : int
        Maximum number of features shown.
    """
    X_df = data[feature_names]
    explainer = shap.Explainer(model=model)
    results = explainer(X_df)
    shap.plots.bar(results, max_display=max_display, show=True)


def plot_xgb_trees_learning_curve(X, y, region_labels, XGB_BEST, val_region='East Berbice-Corentyne'):
    """
    Plots XGBoost train vs held-out-region f1_macro, Marks the tuned
    n_estimators alongside the observed validation-F1 peak.

    Parameters
    ----------
    X : numpy.ndarray
        Feature matrix.
    y : numpy.ndarray
        Binary labels.
    region_labels : array-like
        Per-sample region name; samples in val_region form the held-out set.
    XGB_BEST : dict
        Tuned XGBoost hyperparameters. n_estimators is overridden to 800
        for the diagnostic fit so the curve extends past the tuned value.
    val_region : str
        Region to hold out, must be one of REGIONS.
    """
    if val_region not in REGIONS:
        raise ValueError(
            f'region specified not valid. Please select one of the following regions: {REGIONS}')

    region_labels = np.asarray(region_labels)
    train, val = region_labels != val_region, region_labels == val_region

    diag = dict(XGB_BEST)
    diag['n_estimators'] = 800
    model = XGBClassifier(objective='binary:logistic', eval_metric='logloss',
                          tree_method='hist', random_state=42, n_jobs=1, **diag)

    model.fit(X[train], y[train], eval_set=[
        (X[train], y[train]), (X[val], y[val])], verbose=False)

    steps = list(range(10, 801, 10))
    train_f1 = [f1_score(y[train], model.predict(X[train], iteration_range=(0, k)), average='macro')
                for k in steps]
    val_f1 = [f1_score(y[val], model.predict(X[val], iteration_range=(0, k)), average='macro')
              for k in steps]

    best_round = steps[int(np.argmax(val_f1))]
    plt.figure(figsize=(7, 5))
    plt.plot(steps, train_f1, label='train')
    plt.plot(steps, val_f1,   label=f'validation ({val_region})')
    plt.axvline(XGB_BEST['n_estimators'], ls='--',
                c='gray',  label=f"tuned n_estimators={XGB_BEST['n_estimators']}")
    plt.axvline(best_round,               ls=':',  c='green',
                label=f'val F1 peak = {best_round}')
    plt.xlabel('boosting round')
    plt.ylabel('f1_macro')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.title('XGBoost train vs held-out F1')
    plt.show()
