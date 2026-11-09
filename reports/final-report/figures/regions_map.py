import math
from pathlib import Path

import contextily as cx
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib_map_utils import north_arrow, scale_bar
from pyproj import Transformer

from mangrove_mapper.utils.regions import export_regions_map, export_south_america, load_regions

"""
Build the "Regions of Guyana" study-area figure for the final report (Fig. 1)
Built on top of the region polygons exported by mangrove_mapper.utils.regions.

See Also
--------
docs/references.md#documentation : Transformer. pyproj Stable Documentation.
docs/references.md#documentation : Introductory guide. Contextily Documentation.
docs/references.md#documentation : GeoPandas Stable Documentation.
(Adding Scale Bars and North Arrows to a Matplotlib Plot)
"""

REPO_ROOT = Path(__file__).resolve().parents[3]
REGIONS_PATH = REPO_ROOT / "data/guyana_regions.geojson"

# Web Mercator, to line up with contextily basemap tiles.
WEB_MERCATOR = "EPSG:3857"

REGION_NUMBERS = {
    "Barima-Waini": 1,
    "Pomeroon-Supenaam": 2,
    "Essequibo Islands-West Demerara": 3,
    "Demerara-Mahaica": 4,
    "Mahaica-Berbice": 5,
    "East Berbice-Corentyne": 6,
    "Cuyuni-Mazaruni": 7,
    "Potaro-Siparuni": 8,
    "Upper Takutu-Upper Essequibo": 9,
    "Upper Demerara-Berbice": 10,
}

COASTAL_NUMBERS = set(range(1, 7))

# Geographic window (west, south, east, north) in degrees
# Frames the coastal north with a margin into Venezuela (W), Suriname (E) and the Atlantic (N).
EXTENT_WGS84 = (-62.6, 4.4, -55.4, 9.4)

# Label-free natural-colour basemap (closest to NatGeo basemaps I could find)
BASEMAP = cx.providers.Esri.WorldPhysical

# (lon, lat). from Wikipedia (converted from degree/mins/secs to lon/lat degrees)
CAPITAL = ("Georgetown", -58.1550, 6.8011)

STUDY_AREAS = [
    ("Area of validation", -58.80461650901522, 7.643615939250853),
    ("Area of application", -57.8568, 6.63109),
]

# Similar fonts and colors to ArcGISPro
MAP_FONT = "Optima"
WATER_FONT = "Georgia"

# Make the generic families resolve to our map fonts, so libraries that only
# accept generic names (e.g. matplotlib-map-utils) still render in Optima.
plt.rcParams["font.sans-serif"] = [MAP_FONT, "DejaVu Sans"]
plt.rcParams["font.serif"] = [WATER_FONT, "DejaVu Serif"]
COUNTRY_COLOR = "#4d4033"   # muted brownish-grey for neighbouring countries
WATER_COLOR = "#33607f"     # desaturated blue for the ocean

# Neighbour / ocean labels: (text, lon, lat, type)
CONTEXT_LABELS = [
    ("VENEZUELA", -61.7, 7.0, "country"),
    # Note basemap doesn't have a boundary separating VZ and BR, eyeballing the locations
    ("BRAZIL", -61.9, 5.0, "country"),
    ("SURINAME", -56.2, 5.5, "country"),
    ("ATLANTIC OCEAN", -57.6, 8.8, "water"),
]


def prep_regions(path=REGIONS_PATH):
    """
    Load the regions polygon and set attributes for plotting.

    Parameters
    ----------
    path : pathlib.Path
        Path to regions map (Default data/guyana_regions.geojson)

    Returns
    -------
    geopandas.GeoDataFrame in Web Mercator with additional columns:
    - number  : ADM1 region number (1-10)
    - coastal : True for the six coastal regions
    - color   : Fill color (light blue for coastal, grey for interior)
    sorted by region number
    """
    path = Path(path)
    if not path.exists():
        # Rebuild from GEE (needs an initialized Earth Engine session).
        path = export_regions_map(name=path.name, output_dir=path.parent)

    gdf = load_regions(path)
    gdf["number"] = gdf["shapeName"].map(REGION_NUMBERS)
    gdf["coastal"] = gdf["number"].isin(COASTAL_NUMBERS)
    gdf = gdf.sort_values("number").reset_index(drop=True)

    # reprojecting EPSG:4326 from the geojson to EPSG:3857 to match the contextily basemaps
    return gdf.to_crs(WEB_MERCATOR)


def _extent_mercator(extent_wgs84=EXTENT_WGS84):
    """(west, south, east, north) degrees -> (xmin, xmax, ymin, ymax) in 3857."""
    w, s, e, n = extent_wgs84
    to_merc = Transformer.from_crs("EPSG:4326", WEB_MERCATOR, always_xy=True)
    xmin, ymin = to_merc.transform(w, s)
    xmax, ymax = to_merc.transform(e, n)
    return xmin, xmax, ymin, ymax


# White halo to match copyright text at the bottom of basemap
_HALO = [pe.withStroke(linewidth=1.5, foreground="white")]

# Region 6 (East Berbice-Corentyne) stretches far south; its auto-point falls
# below the frame, so pin its number to the coastal (northern) portion.
# Region 3 touched up to align better with other region labels
LABEL_OVERRIDES = {3: (-58.41, 6.6), 6: (-57.4, 6.0)}


def add_region_numbers(ax, gdf, coastal_only=True, overrides=LABEL_OVERRIDES):
    """Place each region's number at an interior point (or an override)."""
    to_merc = Transformer.from_crs("EPSG:4326", WEB_MERCATOR, always_xy=True)
    for _, r in gdf.iterrows():
        n = int(r["number"])
        if coastal_only and not r["coastal"]:
            continue
        if n in overrides:
            x, y = to_merc.transform(*overrides[n])
        else:
            p = r.geometry.representative_point()   # guaranteed inside polygon
            x, y = p.x, p.y
        ax.annotate(str(n), (x, y), ha="center", va="center",
                    fontsize=10, fontweight="bold", color="#111111", path_effects=_HALO, zorder=5)


def add_capital(ax, capital=CAPITAL):
    """Marker and label for the capital city."""
    name, lon, lat = capital
    x, y = Transformer.from_crs("EPSG:4326", WEB_MERCATOR,
                                always_xy=True).transform(lon, lat)
    ax.plot(x, y, marker=".", markersize=10, color="red",
            markeredgecolor="white", markeredgewidth=0.8, zorder=6)
    ax.annotate(name, (x, y), xytext=(7, 3), textcoords="offset points",
                fontsize=10, fontweight="bold", color="black",
                path_effects=_HALO, zorder=6)


def add_study_areas(ax, sites=STUDY_AREAS):
    """Markers and label for the areas of application"""
    to_merc = Transformer.from_crs("EPSG:4326", WEB_MERCATOR, always_xy=True)
    for name, lon, lat in sites:
        x, y = to_merc.transform(lon, lat)
        ax.plot(x, y, marker="*", markersize=9, color="#1b7837",
                markeredgecolor="white", markeredgewidth=0.8, zorder=6)
        if name:
            ax.annotate(name, (x, y), xytext=(7, 3), textcoords="offset points",
                        fontsize=9, fontweight="bold", color="black",
                        path_effects=_HALO, zorder=6)


def add_key(ax, title="Coastal Regions (1-6)", at=(0.99, 0.02)):
    """Numbered region name key, bottom-right"""
    names = {v: k for k, v in REGION_NUMBERS.items() if v in COASTAL_NUMBERS}
    lines = "\n".join(f"{n}   {names[n]}" for n in range(1, len(names) + 1))
    x, y = at
    # Leading blank line reserves room for the heading, which is drawn on top.
    ax.text(x, y, "\n" + lines, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7, fontfamily=MAP_FONT, linespacing=1, color="#222222",
            bbox={"boxstyle": "square,pad=0.5", "facecolor": "white",
                  "edgecolor": "#555555", "linewidth": 0.7, "alpha": 0.92})
    ax.annotate(title, (x, y), xycoords="axes fraction",
                xytext=(0, 5.6), textcoords="offset fontsize",
                ha="right", va="bottom", fontsize=8, fontweight="bold",
                fontfamily=MAP_FONT, color="#111111")


def add_context_labels(ax, labels=CONTEXT_LABELS):
    """Neighbouring-country and ocean (italic serif) labels."""
    to_merc = Transformer.from_crs("EPSG:4326", WEB_MERCATOR, always_xy=True)
    for text, lon, lat, kind in labels:
        x, y = to_merc.transform(lon, lat)
        if kind == "country":
            ax.text(x, y, " ".join(text), ha="center", va="center",
                    fontfamily=MAP_FONT, fontsize=11, color=COUNTRY_COLOR,
                    alpha=0.85, zorder=4, path_effects=_HALO)
        else:  # water
            spaced = "\n".join(" ".join(w) for w in text.split())
            ax.text(x, y, spaced, ha="center", va="center",
                    fontfamily=WATER_FONT, fontstyle="italic", fontsize=12,
                    color=WATER_COLOR, alpha=0.9, linespacing=1.3, zorder=4,
                    path_effects=[pe.withStroke(linewidth=1.2, foreground="white")])


def add_grid_coords(ax, extent=EXTENT_WGS84, lon_step=2, lat_step=1):
    """Degree ticks and labels."""
    w, s, e, n = extent
    to_merc = Transformer.from_crs("EPSG:4326", WEB_MERCATOR, always_xy=True)

    lons = np.arange(math.ceil(w / lon_step) * lon_step,
                     math.floor(e) + 1, lon_step)
    lats = np.arange(math.ceil(s / lat_step) * lat_step,
                     math.floor(n) + 1, lat_step)

    ax.set_xticks([to_merc.transform(lon, s)[0] for lon in lons])
    ax.set_yticks([to_merc.transform(w, lat)[1] for lat in lats])
    ax.set_xticklabels([f"{abs(int(lon))}°W" for lon in lons])
    ax.set_yticklabels([f"{abs(int(lat))}°N" for lat in lats])

    ax.tick_params(direction="out", length=4, width=0.6, labelsize=7,
                   top=True, right=True, labeltop=True, labelright=True)
    ax.tick_params(axis="y", labelrotation=90)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)


def add_north_arrow(ax, location="upper right", scale=0.4):
    """Half black / half white north arrow (fancy)."""
    north_arrow(ax, location=location, scale=scale,
                fancy=True, shadow=False,
                label={"position": "top", "text": "N",
                       "fontfamily": "sans-serif", "fontsize": 11})


def add_scalebar(ax, location="lower left"):
    """Simple two-segment (0/50/100 km) scale bar in the map font."""
    scale_bar(ax, location=location, style="boxes",
              bar={"projection": WEB_MERCATOR, "unit": "km",
                   "max": 100, "major_div": 2, "minor_div": 1},
              labels={"fontsize": 8, "fontfamily": "sans-serif", "sep": 1},
              units={"fontsize": 8, "fontfamily": "sans-serif"})


def add_inset(ax, x, y, width, height, highlight="Guyana",
              path=REPO_ROOT / "data/south_america.geojson",
              locator_box=EXTENT_WGS84):
    """
    South America locator inset with the study country (Guyana) highlighted,
    styled to sit with the main map. Rebuilds the boundaries from GEE via
    export_south_america() (needs an initialised Earth Engine session) if the
    GeoJSON is missing.

    locator_box : tuple of float or None
        (west, south, east, north) in degrees, drawn as a small box over the
        coastal north of Guyana to show where the main map sits. Defaults to
        the main map window; pass None to omit.
    """
    import geopandas as gpd
    from matplotlib.patches import Rectangle

    path = Path(path)
    if not path.exists():
        path = export_south_america(name=path.name, output_dir=path.parent)
    sa = gpd.read_file(path)

    axin = ax.inset_axes([x, y, width, height], xticks=[], yticks=[])
    # sea, echoing the ocean
    axin.set_facecolor("#79b7dc")
    sa.plot(ax=axin, color="#eeeeee", edgecolor="white",
            linewidth=0.4)  # continent
    sa[sa["country_na"] == highlight].plot(
        ax=axin, color="#b5442f", edgecolor="#7b241c", linewidth=0.5)    # study country

    if locator_box is not None:
        # Inset is unprojected (degrees), so the extent maps onto it nicely.
        w, s, e, n = locator_box
        axin.add_patch(Rectangle((w, s), e - w, n - s, facecolor="none",
                                 edgecolor="#333333", linewidth=0.6, zorder=5,
                                 path_effects=[pe.withStroke(linewidth=1.1,
                                                             foreground="white")]))

    minx, miny, maxx, maxy = sa.total_bounds
    axin.set_xlim(minx - 1.5, maxx + 1.5)
    axin.set_ylim(miny - 1.5, maxy + 1.5)
    for spine in axin.spines.values():
        spine.set_edgecolor(COUNTRY_COLOR)
        spine.set_linewidth(0.6)
    return axin


def build_figure(gdf=None, extent=EXTENT_WGS84, basemap=BASEMAP, figsize=(9, 8),
                 numbers=True, key=True, capital=True, application=True):
    """
    Draws the study-area map: region boundaries over a terrain basemap,
    framed to the coastal window, with region numbers, a key, and the capital.

    Parameters
    -----------
    gdf : geopandas.GeoDataFrame, optional
        Region polygons to draw, already in Web Mercator. Defaults to the
        output of prep_regions() when not supplied.
    extent : tuple of float, optional
        Map window as (min_lon, min_lat, max_lon, max_lat) in WGS84.
        Defaults to EXTENT_WGS84, the coastal frame.
    basemap : contextily tile provider, optional
        Background tile source passed to contextily. Defaults to BASEMAP
        (Esri WorldPhysical terrain).
    figsize : tuple of float, optional
        Figure size (width, height) in inches. Default (9, 8).
    numbers : bool, optional
        If True, label each region with its number. Default True.
    key : bool, optional
        If True, draw the key mapping region numbers to names. Default True.
    capital : bool, optional
        If True, mark and label the capital. Default True.

    Returns
    -------
    (fig, ax) : matplotlib Figure and Axes
    """
    if gdf is None:
        gdf = prep_regions()

    fig, ax = plt.subplots(figsize=figsize)

    # Region outlines only
    gdf.boundary.plot(ax=ax, color="#58351C", linewidth=0.8)

    # Frame the window and then add the basemap
    xmin, xmax, ymin, ymax = _extent_mercator(extent)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # crs=gdf.crs tells contextily the axes are already in Web Mercator,
    # so it won't warp the tiles.
    cx.add_basemap(ax, source=basemap, crs=gdf.crs)

    add_context_labels(ax)
    if numbers:
        add_region_numbers(ax, gdf)
    if capital:
        add_capital(ax)
    if application:
        add_study_areas(ax)
    if key:
        add_key(ax)

    add_grid_coords(ax, extent)
    add_north_arrow(ax)
    add_scalebar(ax)
    add_inset(ax, x=0.012, y=0.62, width=0.26, height=0.4)
    return fig, ax


if __name__ == '__main__':
    fig, ax = build_figure()
    out = Path(__file__).with_name("regions_map_fig.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"wrote {out}")
