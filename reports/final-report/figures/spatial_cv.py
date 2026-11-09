import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mangrove_mapper import MangroveMapper
from mangrove_mapper.utils.regions import REGIONS, tag_regions
from mangrove_mapper.visualize import plot_spatial_samples_over_regions

"""
Builds spatial cross validation plot as a figure.
"""

REPO_ROOT = Path(__file__).resolve().parents[3]
TRAINING_CSV = REPO_ROOT / "data/training_data_coastwide_wc.csv"
REGIONS_PATH = REPO_ROOT / "data/guyana_regions.geojson"

# REGIONS is ordered west to east (top to bottom)
REGION_NUMBERS = {name: i + 1 for i, name in enumerate(REGIONS)}

CLASSES = {0: "Non-mangrove", 1: "Mangrove"}

CLASS_COLORS = {0: "#898781", 1: "#008300"}


def build_figure(project_name=None, training_data=TRAINING_CSV,
                 regions_data_path=REGIONS_PATH, figsize=(12, 5.5),
                 balance=True):
    """
    Draw the spatial samples over the six coastal cross-validation regions.

    Parameters
    ----------
    project_name : str, optional
        Google Cloud project. Only needed when the samples are missing and have
        to be re-exported from Earth Engine.
    training_data : str or pathlib.Path, optional
        Samples exported from GEE, with a '.geo' column of GeoJSON strings.
        Re-exported through MangroveMapper.get_sensor_data if the file is not
        present. Defaults to data/6000-fixed-trimmed.csv.
    regions_data_path : str or pathlib.Path, optional
        Region polygons GeoJSON. Exported from GEE by tag_regions if missing,
        which needs an authenticated session.
    figsize : tuple of float, optional
        Figure size (width, height) in inches. Default (12, 5.5).
    balance : bool, optional
        If True, add panel (b), the per-fold class balance. Default True.

    Returns
    -------
    matplotlib.figure.Figure
        The figure drawn by plot_spatial_samples_over_regions, ready to save.

    Raises
    ------
    FileNotFoundError
        If the samples are missing and no project_name was given to rebuild them.
    """
    training_data = Path(training_data)
    if not training_data.exists():
        if project_name is None:
            raise FileNotFoundError(
                f"{training_data} not found. Pass flag '--project' with the project_name to re-export it "
                "from Earth Engine.")
        MangroveMapper(project_name=project_name).get_sensor_data(
            training_data.stem, retrieve_from_drive=True, target_dataset="ESA/WorldCover/v100", mangrove_class=95)

    if not balance:
        ax = plot_spatial_samples_over_regions(
            training_data, regions_data_path, figsize=figsize)
        return ax.figure

    # two plots, for map and bar chart
    fig, (ax_map, ax_counts) = plt.subplots(
        # 55% of plot width as map, 45% as bar chart
        1, 2, figsize=figsize, gridspec_kw={"width_ratios": [1.2, 1]})

    plot_spatial_samples_over_regions(
        training_data, regions_data_path, ax=ax_map)
    ax_map.set_title("A")
    print(class_balance(training_data, regions_data_path))
    panel_class_balance(ax_counts,
                        class_balance(training_data, regions_data_path))

    fig.tight_layout()
    return fig


def class_balance(training_data=TRAINING_CSV, regions_data_path=REGIONS_PATH):
    """
    Count samples per region per class, ordered Barima (west) to Corentyne (east).

    Parameters
    ----------
    training_data : str or pathlib.Path, optional
        Samples exported from GEE.
    regions_data_path : str or pathlib.Path, optional
        Region polygons GeoJSON also from GEE

    Returns
    -------
    pandas.DataFrame
        Rows include six coastal regions west to east, columns the binary
        labels (0 non-mangrove, 1 mangrove).
    """
    data = tag_regions(pd.read_csv(training_data),
                       regions_data_path=regions_data_path)
    counts = pd.crosstab(data["region"], data["label"])
    return counts.reindex(REGIONS).reindex(columns=[0, 1]).fillna(0).astype(int)


def panel_class_balance(ax, counts):
    """
    Stacked bar of samples per fold, split by class.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    counts : pandas.DataFrame
        From class_balance().
    """
    y = np.arange(len(counts))  # each bar number 0-indexed
    left = np.zeros(len(counts))  # where the bar starts on the x-axis
    for label in (0, 1):
        # height = bar thickness
        ax.barh(y, counts[label], left=left, height=0.62,
                color=CLASS_COLORS[label], label=CLASSES[label], zorder=3)
        # creates the larger grey (non-mangrove) then the smaller green (mangrove) bar stack stacked after
        # converting the pd.Series to numpy array to determine the amount of each class.
        left = left + counts[label].values

    ax.set_yticks(y)
    ax.set_yticklabels([f"{REGION_NUMBERS[r]}  {r}" for r in counts.index])
    # puts the westward region Barima-Waini at the top followed by other regions eastward
    ax.invert_yaxis()
    ax.set_xlabel("Training samples")
    # preserving vertical gridlines, horizontal not needed
    ax.xaxis.grid(True, color="#e1e0d9", linewidth=0.6)
    ax.set_axisbelow(True)  # moving grid underneath bars
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0, labelsize=8.5)

    # idxmin rather than a hardcoded region, so a re-export stays correct.
    scarcest = counts[1].idxmin()  # currently Demerara-Mahaica
    row = list(counts.index).index(scarcest)
    # helpful to show the actual samples for the region with fewest
    ax.annotate(f"{counts.loc[scarcest, 1]} mangrove samples",
                (counts.loc[scarcest].sum(), row), xytext=(6, 0),
                textcoords="offset points", va="center",
                fontsize=8.5, fontweight="bold")

    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    ax.set_title("B")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Build the spatial CV figure.")
    parser.add_argument(
        "--project",
        dest="project_name",
        default=None, help="Google Cloud Project. "
        "Only needed if the training samples are mising and they need to be re-exported.")
    args = parser.parse_args()
    fig = build_figure(project_name=args.project_name)
    out = Path(__file__).with_name("spatial_cv_fig.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"wrote {out}")
