"""
Unit tests for the Guyana administrative-region export helper, not dependent on
the Google Earth Engine (GEE) api. As a result, GEE and geemap objects are
mocked out.

See Also
--------
docs/references.md#ai-usage : AI assistance log
"""

import json
import sys
from unittest.mock import MagicMock, patch

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

from mangrove_mapper.utils.regions import (
    REGION_RENAMES,
    REGIONS,
    export_regions_map,
    extract_coords,
    load_regions,
    tag_regions,
)


def _point_geo(lon, lat):
    # a '.geo' cell as GEE exports it: a GeoJSON string, [lon, lat] order
    return json.dumps({'type': 'Point', 'coordinates': [lon, lat]})


@pytest.fixture
def mock_regions():
    # geemap is imported lazily inside export_regions_map and ships in the
    # [notebook] extra, so CI installs without it. Putting the mock straight
    # into the import cache means that `import geemap` binds it, and the real
    # package never needs to be present.
    mock_geemap = MagicMock()
    with patch("mangrove_mapper.utils.regions.ee") as mock_ee, \
            patch.dict(sys.modules, {"geemap": mock_geemap}):
        yield {"ee": mock_ee, "geemap": mock_geemap}


class TestRegionsConstant:
    def test_has_six_coastal_regions(self):
        assert len(REGIONS) == 6

    def test_regions_are_unique(self):
        assert len(set(REGIONS)) == len(REGIONS)


class TestExportRegionsMap:
    def test_filters_boundaries_to_guyana(self, mock_regions, tmp_path):
        export_regions_map(output_dir=tmp_path)

        mock_regions["ee"].FeatureCollection.assert_called_once_with(
            "WM/geoLab/geoBoundaries/600/ADM1")
        mock_regions["ee"].Filter.eq.assert_called_once_with(
            "shapeGroup", "GUY")
        collection = mock_regions["ee"].FeatureCollection.return_value
        collection.filter.assert_called_once_with(
            mock_regions["ee"].Filter.eq.return_value)

    def test_converts_filtered_collection_to_gdf(self, mock_regions, tmp_path):
        export_regions_map(output_dir=tmp_path)

        filtered = mock_regions["ee"].FeatureCollection.return_value.filter.return_value
        mock_regions["geemap"].ee_to_gdf.assert_called_once_with(filtered)

    def test_writes_geojson_with_driver(self, mock_regions, tmp_path):
        export_regions_map(name="test.geojson", output_dir=tmp_path)

        geo_df = mock_regions["geemap"].ee_to_gdf.return_value
        geo_df.to_file.assert_called_once_with(
            str(tmp_path / "test.geojson"), driver="GeoJSON")

    def test_returns_output_file_path(self, mock_regions, tmp_path):
        out_file = export_regions_map(name="test.geojson", output_dir=tmp_path)

        assert out_file == tmp_path / "test.geojson"

    def test_defaults_to_geojson_filename(self, mock_regions, tmp_path):
        out_file = export_regions_map(output_dir=tmp_path)

        assert out_file.name == "guyana_regions.geojson"

    def test_creates_missing_output_dir(self, mock_regions, tmp_path):
        # output_dir is created if it does not exist (mkdir is real, not mocked)
        nested = tmp_path / "does" / "not" / "exist"

        export_regions_map(output_dir=nested)

        assert nested.exists()


class TestLoadRegions:
    def _raw_gdf(self, names):
        # minimal stand-in for the exported GeoJSON (real geopandas, no GEE)
        return gpd.GeoDataFrame(
            {"shapeName": names,
             "geometry": [Point(i, i) for i in range(len(names))]})

    def test_corrects_misspelled_region_name(self):
        raw = self._raw_gdf(["Barina-Waini", "Pomeroon-Supenaam"])

        with patch("mangrove_mapper.utils.regions.gpd.read_file", return_value=raw):
            gdf = load_regions("dummy.geojson")

        names = set(gdf["shapeName"])
        # the dataset misspelling is gone, replaced by the correct name
        assert "Barina-Waini" not in names
        assert "Barima-Waini" in names

    def test_all_coastal_regions_present_after_load(self):
        # raw file uses the dataset spelling for the misspelled region
        raw = self._raw_gdf(list(REGION_RENAMES) + REGIONS[1:])

        with patch("mangrove_mapper.utils.regions.gpd.read_file", return_value=raw):
            gdf = load_regions("dummy.geojson")

        # every coastal region in REGIONS resolves after renaming (the invariant
        # that guards the spatial-CV join from silently dropping a fold)
        assert set(REGIONS) <= set(gdf["shapeName"])


class TestTagRegions:
    def _samples(self):
        # three GEE-style samples: two near (0, 0), one near (5, 5)
        return pd.DataFrame({
            '.geo': [_point_geo(0.0, 0.0), _point_geo(0.1, 0.1),
                     _point_geo(5.0, 5.0)],
            'label': [0, 1, 1]})

    def _regions_gdf(self):
        # one coastal region near the first two samples, one interior region
        return gpd.GeoDataFrame(
            {'shapeName': ['Barima-Waini', 'Potaro-Siparuni'],
             'geometry': [Point(0, 0), Point(5, 5)]}, crs='EPSG:4326')

    def test_tags_each_sample_with_nearest_region(self):
        with patch("mangrove_mapper.utils.regions.load_regions",
                   return_value=self._regions_gdf()):
            out = tag_regions(self._samples(), 'dummy.geojson',
                              keep_coastal_only=False)

        assert list(out['region']) == [
            'Barima-Waini', 'Barima-Waini', 'Potaro-Siparuni']

    def test_drops_non_coastal_regions_by_default(self):
        with patch("mangrove_mapper.utils.regions.load_regions",
                   return_value=self._regions_gdf()):
            out = tag_regions(self._samples(), 'dummy.geojson')

        assert list(out['region']) == ['Barima-Waini', 'Barima-Waini']
        assert list(out.index) == [0, 1]  # index reset after filtering

    def test_exports_regions_map_when_no_path_given(self):
        with patch("mangrove_mapper.utils.regions.load_regions",
                   return_value=self._regions_gdf()), \
                patch("mangrove_mapper.utils.regions.export_regions_map",
                      return_value='exported.geojson') as mock_export:
            tag_regions(self._samples())

        mock_export.assert_called_once()

    def test_does_not_mutate_input(self):
        samples = self._samples()

        with patch("mangrove_mapper.utils.regions.load_regions",
                   return_value=self._regions_gdf()):
            tag_regions(samples, 'dummy.geojson')

        assert 'region' not in samples.columns


class TestExtractCoords:
    def test_returns_lon_lat_pairs_in_geojson_order(self):
        data = pd.DataFrame({
            '.geo': [_point_geo(-58.1, 6.8), _point_geo(-57.5, 6.3)]})

        coords = extract_coords(data)

        assert coords.shape == (2, 2)
        # GeoJSON order is [lon, lat]; Guyana sits around lon -58, lat 6.8
        assert np.allclose(coords[:, 0], [-58.1, -57.5])
        assert np.allclose(coords[:, 1], [6.8, 6.3])

    def test_rows_align_with_tagged_dataframe(self):
        # the intended pipeline: tag (which drops rows), then extract from
        # the same frame, so coords stay row-aligned with 'region'
        samples = pd.DataFrame({
            '.geo': [_point_geo(0.0, 0.0), _point_geo(5.0, 5.0),
                     _point_geo(0.1, 0.1)],
            'label': [0, 1, 1]})
        regions_gdf = gpd.GeoDataFrame(
            {'shapeName': ['Barima-Waini', 'Potaro-Siparuni'],
             'geometry': [Point(0, 0), Point(5, 5)]}, crs='EPSG:4326')

        with patch("mangrove_mapper.utils.regions.load_regions",
                   return_value=regions_gdf):
            tagged = tag_regions(samples, 'dummy.geojson')
        coords = extract_coords(tagged)

        # the interior (5, 5) sample is gone and coords match the survivors
        assert len(coords) == len(tagged) == 2
        assert np.allclose(coords, [[0.0, 0.0], [0.1, 0.1]])
