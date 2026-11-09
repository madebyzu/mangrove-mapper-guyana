"""
Guyana administrative-region helpers: exporting the region map, tagging
samples with their region, and extracting sample coordinates.

See Also
--------
docs/references.md#ai-usage : AI assistance log
"""

import json
from pathlib import Path

import ee
import geopandas as gpd
import numpy as np
from shapely.geometry import shape

# The six coastal regions used for spatial CV. The exported GeoJSON contains
# all Guyana ADM1 regions; interior regions are intentionally excluded here.

# dataset misspelling correction
REGION_RENAMES = {'Barina-Waini': 'Barima-Waini'}

REGIONS = [
    'Barima-Waini',
    'Pomeroon-Supenaam',
    'Essequibo Islands-West Demerara',
    'Demerara-Mahaica',
    'Mahaica-Berbice',
    'East Berbice-Corentyne'
]
REPO_ROOT = Path(__file__).resolve().parents[2]


def export_regions_map(name='guyana_regions.geojson', output_dir=REPO_ROOT / 'data'):
    """
    Builds and exports a map of the administrative regions of Guyana.
    File format is a GeoJSON file.

    Parameters
    ----------
    name : str
        Name of output file, defaults to 'guyana_regions.geojson'
    output_dir : Path or str
        Path to output directory

    Returns
    -------
    out_file : Path
        Path to output file

    See Also
    --------
    docs/references.md#documentation : geoBoundaries: Political administrative boundaries at District level (ADM1)
    docs/references.md#documentation : geopandas.GeoDataFrame.to_file. GeoPandas Stable Documentation
    """
    import geemap

    geo_boundaries = ee.FeatureCollection(
        'WM/geoLab/geoBoundaries/600/ADM1').filter(ee.Filter.eq('shapeGroup', 'GUY'))
    geo_df = geemap.ee_to_gdf(geo_boundaries)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / name

    geo_df.to_file(str(out_file), driver='GeoJSON')
    print(f'Exported {name} at this path: {out_file}')
    return out_file


def export_south_america(name='south_america.geojson', output_dir=REPO_ROOT / 'data'):
    """
    Builds and exports the South America country boundaries (USDOS LSIB Simple
    2017) as a GeoJSON, for the study-area locator inset. Guyana is one feature
    within the collection (``country_na == 'Guyana'``).

    Parameters
    ----------
    name : str
        Name of output file, defaults to 'south_america.geojson'
    output_dir : Path or str
        Path to output directory

    Returns
    -------
    out_file : Path
        Path to output file

    See Also
    --------
    docs/references.md#documentation : LSIB 2017: Large Scale International Boundary Polygons, Simplified
    docs/references.md#documentation : geopandas.GeoDataFrame.to_file. GeoPandas Stable Documentation
    """
    import geemap

    countries = ee.FeatureCollection('USDOS/LSIB_SIMPLE/2017')
    south_america = countries.filter(ee.Filter.eq('wld_rgn', 'South America'))
    geo_df = geemap.ee_to_gdf(south_america)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / name

    geo_df.to_file(str(out_file), driver='GeoJSON')
    print(f'Exported {name} at this path: {out_file}')
    return out_file


def load_regions(path):
    """
    Loads regions file (GeoJSON) and corrects misspelled region name.

    Parameters
    ----------
    path : str or Path
        path to regions geojson file

    Returns
    -------
    gdf : geopandas.GeoDataFrame
        Region polygons with corrected region names.
    """
    gdf = gpd.read_file(path)[['shapeName', 'geometry']]
    gdf['shapeName'] = gdf['shapeName'].replace(REGION_RENAMES)
    return gdf


def tag_regions(data, regions_data_path=None, keep_coastal_only=True):
    """
    Tags each sample with the administrative region it falls in (or is
    nearest to), based on the sample's '.geo' GeoJSON column.

    Parameters
    ----------
    data : pandas.DataFrame
        Samples exported from GEE, with a '.geo' column of GeoJSON strings.
    regions_data_path : str or Path, optional
        Path to the regions GeoJSON. Exported via export_regions_map()
        if not provided.
    keep_coastal_only : bool
        If True, drop samples tagged with a region outside the six
        coastal CV regions in REGIONS.

    Returns
    -------
    data : pandas.DataFrame
        Copy of the input with a 'region' column added.
    """
    if regions_data_path is None:
        regions_data_path = export_regions_map()
    regions = load_regions(regions_data_path)

    data = data.copy()
    geom = data['.geo'].apply(lambda x: shape(json.loads(x)))
    points = gpd.GeoDataFrame(data, geometry=geom, crs='EPSG:4326')
    tagged = gpd.sjoin_nearest(points, regions, how='left')
    # a point equidistant to two regions produces duplicate rows
    tagged = tagged[~tagged.index.duplicated(keep='first')]

    data['region'] = tagged['shapeName'].values
    if keep_coastal_only:
        data = data[data['region'].isin(REGIONS)].reset_index(drop=True)
    return data


def extract_coords(data):
    """
    Extracts sample coordinates from the '.geo' GeoJSON column.

    Call this on the same DataFrame the labels come from (i.e. after
    tag_regions, which may drop rows) so rows stay aligned.

    Parameters
    ----------
    data : pandas.DataFrame
        Samples exported from GEE, with a '.geo' column of GeoJSON strings.

    Returns
    -------
    numpy.ndarray
        (n_samples, 2) array of [lon, lat] pairs (GeoJSON axis order).
    """
    coords = data['.geo'].apply(lambda x: json.loads(x)['coordinates'])
    return np.column_stack([coords.str[0], coords.str[1]])
